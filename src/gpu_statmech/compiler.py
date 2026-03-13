"""
KernelProposal → KernelSpec compiler and thermodynamic scorer (Phase 2).

This module bridges the oracle (which speaks in high-level kernel design
parameters) and the Carnot-optimality checker (which operates on the
``KernelSpec`` dataclass).  It also computes an *expressiveness proxy
score* — a hardware-grounded estimate of a kernel's representational power
— so that the optimisation loop can evaluate proposals before committing
to expensive full training runs.

Compilation pipeline
--------------------

  KernelProposal
       │
       ▼  compile()
   KernelSpec  ──► check_carnot_optimality()
       │                     │
       │                     └──► CarnotOptimalityReport
       │
       ├──► warp_occupancy()      (H100 occupancy model)
       ├──► working_set()         (register file + SMEM footprint)
       └──► expressiveness_score()

Occupancy model
~~~~~~~~~~~~~~~
H100 limits (per SM):
  • max_warps_per_sm      = 64
  • max_threads_per_sm    = 2048
  • register_file_size    = 65 536 × 32-bit regs  (= 256 KB)
  • max_smem_per_sm       = 228 × 1024 bytes

Given a ``KernelProposal``:
  1. warps_per_block       = ceil(block_size / 32)
  2. reg_limit_warps       = floor(65536 / (registers_per_thread × 32))
     (number of warps the register file can accommodate, across blocks)
  3. smem_limit_warps      = floor(smem_per_sm / smem_bytes) × warps_per_block
     (number of warps limited by SMEM, if smem_bytes > 0)
  4. actual_warps          = min(reg_limit_warps, smem_limit_warps,
                                 max_warps_per_sm)
  5. warp_occupancy        = actual_warps / max_warps_per_sm

Expressiveness proxy score
~~~~~~~~~~~~~~~~~~~~~~~~~~
A hardware-grounded proxy for the downstream representational power of a
kernel design, inspired by the roofline model and tensor-core throughput
theory.  It is NOT a loss function — it is a relative ranking signal.

  expr = w_tc  × tc_util
       + w_ai  × sat(AI / ridge)
       + w_acc × access_score(pattern)
       + w_occ × warp_occupancy

  where sat(x) = min(x, 1) clips the AI benefit at the roofline ridge
  (beyond the ridge, expressiveness is not further constrained by memory).

Default weights:  w_tc=0.35, w_ai=0.35, w_acc=0.20, w_occ=0.10.

The score is ∈ [0, 1] and is comparable across proposals and iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .carnot import (
    CarnotLimit,
    CarnotOptimalityReport,
    KernelSpec,
    check_carnot_optimality,
)
from .oracle import ACCESS_PATTERNS, KernelProposal, VALID_BLOCK_SIZES
from .partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG, MemoryLevel, SMConfig


# ---------------------------------------------------------------------------
# H100 SM hard limits
# ---------------------------------------------------------------------------

_H100_MAX_WARPS_PER_SM: int = 64
_H100_REGISTER_FILE_REGS: int = 65_536   # 32-bit registers per SM
_H100_THREADS_PER_WARP: int = 32


# ---------------------------------------------------------------------------
# Occupancy model
# ---------------------------------------------------------------------------

def warp_occupancy(
    proposal: KernelProposal,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
) -> float:
    """
    Compute achieved warp occupancy for a ``KernelProposal``.

    Uses the standard NVIDIA occupancy model:
      occupancy = min(reg_limit, smem_limit, hw_limit) / hw_limit

    Parameters
    ----------
    proposal:
        The kernel proposal to evaluate.
    sm_config:
        SM configuration; defaults to H100.
    memory_levels:
        Memory hierarchy; used to look up SMEM capacity.

    Returns
    -------
    float
        Warp occupancy ∈ [0, 1].
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    max_warps = sm_config.warps_per_sm   # typically 64 for H100
    threads_per_warp = sm_config.threads_per_warp  # 32

    warps_per_block = max(1, int(np.ceil(proposal.block_size / threads_per_warp)))

    # Register-file limit: total regs per SM / regs per warp
    regs_per_warp = proposal.registers_per_thread * threads_per_warp
    if regs_per_warp > 0:
        reg_limit_warps = _H100_REGISTER_FILE_REGS // regs_per_warp
    else:
        reg_limit_warps = max_warps

    # SMEM limit: how many blocks fit → convert to warps
    smem_level = next(
        (lvl for lvl in memory_levels if lvl.name == "smem"),
        memory_levels[1],
    )
    smem_capacity = smem_level.capacity_bytes
    if proposal.smem_bytes > 0:
        blocks_from_smem = smem_capacity // proposal.smem_bytes
        smem_limit_warps = blocks_from_smem * warps_per_block
    else:
        smem_limit_warps = max_warps

    actual_warps = min(reg_limit_warps, smem_limit_warps, max_warps)
    actual_warps = max(0, actual_warps)

    return actual_warps / max(max_warps, 1)


# ---------------------------------------------------------------------------
# Working-set extractor
# ---------------------------------------------------------------------------

def working_set(
    proposal: KernelProposal,
    sm_config: SMConfig | None = None,
) -> dict[str, int]:
    """
    Estimate the bytes resident at each memory level during kernel execution.

    Only the levels where the proposal specifies usage are included.
    HBM is omitted (any kernel can use it; capacity is not the constraint).

    Parameters
    ----------
    proposal:
        The kernel proposal.
    sm_config:
        SM configuration.

    Returns
    -------
    dict[str, int]
        Working set sizes keyed by memory level name (bytes).
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG

    ws: dict[str, int] = {}

    # Register file footprint per SM:
    #   registers_per_thread × threads_per_warp × warps_resident
    # We use warps_per_sm as the occupancy upper bound here; the actual
    # achieved occupancy is enforced separately by check_carnot_optimality.
    threads_per_warp = sm_config.threads_per_warp
    warps_per_block = max(1, int(np.ceil(proposal.block_size / threads_per_warp)))
    max_blocks = max(1, sm_config.warps_per_sm // warps_per_block)
    regs_per_sm = (
        proposal.registers_per_thread
        * threads_per_warp
        * warps_per_block
        * max_blocks
        * 4  # bytes per 32-bit register
    )
    ws["registers"] = int(regs_per_sm)

    # SMEM footprint (static allocation per block × max blocks per SM)
    if proposal.smem_bytes > 0:
        ws["smem"] = proposal.smem_bytes

    return ws


# ---------------------------------------------------------------------------
# Expressiveness proxy score
# ---------------------------------------------------------------------------

def expressiveness_score(
    proposal: KernelProposal,
    carnot_limit: CarnotLimit,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    w_tc: float = 0.35,
    w_ai: float = 0.35,
    w_acc: float = 0.20,
    w_occ: float = 0.10,
) -> float:
    """
    Compute the expressiveness proxy score for a ``KernelProposal``.

    The score is a weighted sum of four hardware-grounded sub-scores, each
    ∈ [0, 1]:

      expr = w_tc  × tc_util
           + w_ai  × min(AI / ridge, 1)
           + w_acc × access_efficiency(pattern)
           + w_occ × warp_occupancy

    Higher → the kernel is predicted to have more representational power
    per unit of hardware resource.

    Parameters
    ----------
    proposal:
        The kernel proposal to score.
    carnot_limit:
        Provides the roofline ridge point (AI threshold).
    sm_config, memory_levels:
        Hardware configuration for the occupancy model.
    w_tc, w_ai, w_acc, w_occ:
        Weights for each sub-score (must sum to 1.0).

    Returns
    -------
    float
        Expressiveness score ∈ [0, 1].
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    # Tensor-core utilisation sub-score
    s_tc = float(np.clip(proposal.tensor_core_utilisation, 0.0, 1.0))

    # Arithmetic intensity sub-score (saturates at roofline ridge)
    ridge = max(carnot_limit.roofline_intensity, 1e-9)
    s_ai = float(min(proposal.arithmetic_intensity / ridge, 1.0))

    # Memory access pattern sub-score
    s_acc = ACCESS_PATTERNS.get(proposal.memory_access_pattern, 0.0)

    # Warp occupancy sub-score
    s_occ = warp_occupancy(proposal, sm_config, memory_levels)

    score = w_tc * s_tc + w_ai * s_ai + w_acc * s_acc + w_occ * s_occ
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# CompiledKernel — full scoring result
# ---------------------------------------------------------------------------

@dataclass
class CompiledKernel:
    """
    The full evaluation result for a single ``KernelProposal``.

    Attributes
    ----------
    proposal:
        The original oracle proposal.
    kernel_spec:
        The derived ``KernelSpec`` (used by ``check_carnot_optimality``).
    optimality_report:
        Per-condition Carnot-optimality check results.
    expressiveness_score:
        Hardware-grounded proxy for representational power ∈ [0, 1].
    thermo_score:
        Achieved η_hw / η_hw,max ∈ [0, 1] from the optimality report.
    combined_score:
        ``thermo_score + expressiveness_score`` (∈ [0, 2]), used for
        ranking proposals in the optimisation loop.
    """
    proposal: KernelProposal
    kernel_spec: KernelSpec
    optimality_report: CarnotOptimalityReport
    expressiveness_score: float
    thermo_score: float

    @property
    def combined_score(self) -> float:
        """Sum of thermodynamic and expressiveness scores ∈ [0, 2]."""
        return self.thermo_score + self.expressiveness_score

    @property
    def is_carnot_optimal(self) -> bool:
        """True if all five Carnot-optimal conditions are satisfied."""
        return self.optimality_report.is_carnot_optimal

    @property
    def dominant_bottleneck(self) -> str:
        """Name of the most-violated Carnot condition (or 'none')."""
        return self.optimality_report.dominant_bottleneck


# ---------------------------------------------------------------------------
# KernelCompiler
# ---------------------------------------------------------------------------

class KernelCompiler:
    """
    Compiles ``KernelProposal`` objects into ``CompiledKernel`` results.

    Compilation maps the oracle's high-level kernel description to the
    ``KernelSpec`` dataclass and runs the full Carnot-optimality check plus
    the expressiveness proxy scorer.

    Parameters
    ----------
    carnot_limit:
        The target hardware's derived Carnot limit.
    memory_levels:
        Memory hierarchy (defaults to H100).
    sm_config:
        SM configuration (defaults to H100).
    """

    def __init__(
        self,
        carnot_limit: CarnotLimit,
        memory_levels: list[MemoryLevel] | None = None,
        sm_config: SMConfig | None = None,
    ) -> None:
        self.carnot_limit = carnot_limit
        self.memory_levels = memory_levels or H100_MEMORY_LEVELS
        self.sm_config = sm_config or H100_SM_CONFIG

    def compile(self, proposal: KernelProposal) -> CompiledKernel:
        """
        Compile and score a single ``KernelProposal``.

        Parameters
        ----------
        proposal:
            The oracle's kernel proposal.

        Returns
        -------
        CompiledKernel
            Full evaluation result.
        """
        # Derive warp occupancy using the occupancy model
        occ = warp_occupancy(proposal, self.sm_config, self.memory_levels)

        # Derive working set
        ws = working_set(proposal, self.sm_config)

        # Build KernelSpec
        spec = KernelSpec(
            name=proposal.name,
            arithmetic_intensity=proposal.arithmetic_intensity,
            working_set=ws,
            reuse_factors=dict(proposal.reuse_factors),
            warp_occupancy=occ,
            unnecessary_data_movement=proposal.unnecessary_data_movement,
        )

        # Carnot-optimality check
        report = check_carnot_optimality(
            spec,
            self.carnot_limit,
            self.memory_levels,
        )

        # Expressiveness proxy
        expr = expressiveness_score(
            proposal,
            self.carnot_limit,
            self.sm_config,
            self.memory_levels,
        )

        # Fraction of Carnot conditions satisfied (0 → 1 in steps of 1/n_cond).
        # Used as the thermodynamic quality signal for the optimisation loop.
        # The multiplicative eta_hw_fraction in the report collapses to ~0 for
        # any practical kernel (theoretical minimums are millions-fold), so we
        # use the condition-satisfaction rate as a more useful search metric.
        n_cond = len(report.conditions)
        thermo = (
            sum(1 for c in report.conditions if c.satisfied) / n_cond
            if n_cond > 0 else 0.0
        )

        return CompiledKernel(
            proposal=proposal,
            kernel_spec=spec,
            optimality_report=report,
            expressiveness_score=expr,
            thermo_score=thermo,
        )

    def batch_compile(
        self,
        proposals: list[KernelProposal],
    ) -> list[CompiledKernel]:
        """
        Compile and score a batch of ``KernelProposal`` objects.

        Parameters
        ----------
        proposals:
            List of oracle proposals.

        Returns
        -------
        list[CompiledKernel]
            One ``CompiledKernel`` per proposal, in the same order.
        """
        return [self.compile(p) for p in proposals]

    def waste_attribution(self, compiled: CompiledKernel) -> dict[str, str]:
        """
        Return a human-readable waste attribution for a compiled kernel.

        For each violated Carnot condition, describes the bottleneck and
        the remedy — suitable for feeding back to an LLM oracle as
        structured physics-grounded guidance.

        Parameters
        ----------
        compiled:
            A compiled and scored kernel.

        Returns
        -------
        dict[str, str]
            Mapping from condition name to diagnosis string.
        """
        diagnosis: dict[str, str] = {}
        for cond in compiled.optimality_report.conditions:
            if not cond.satisfied:
                pct = abs(cond.margin) / max(abs(cond.threshold), 1e-12) * 100
                diagnosis[cond.name] = (
                    f"VIOLATION ({pct:.1f}% below threshold): {cond.description}. "
                    f"Remedy: increase {cond.name.replace('_', ' ')} "
                    f"from {cond.value:.3g} to ≥ {cond.threshold:.3g}."
                )
        return diagnosis

    def feedback_message(self, compiled_batch: list[CompiledKernel]) -> str:
        """
        Synthesise a concise feedback message for the oracle from a scored batch.

        Identifies the most common bottleneck across the batch and
        the best-performing proposal.

        Parameters
        ----------
        compiled_batch:
            A scored batch of compiled kernels.

        Returns
        -------
        str
            Multi-line feedback string.
        """
        if not compiled_batch:
            return "No proposals to evaluate."

        sorted_batch = sorted(compiled_batch, key=lambda c: c.combined_score, reverse=True)
        best = sorted_batch[0]

        # Count bottleneck occurrences
        bottleneck_counts: dict[str, int] = {}
        for ck in compiled_batch:
            bn = ck.dominant_bottleneck
            if bn != "none":
                bottleneck_counts[bn] = bottleneck_counts.get(bn, 0) + 1

        lines = [
            f"Batch size: {len(compiled_batch)} proposals",
            f"Best combined score: {best.combined_score:.3f} "
            f"(η={best.thermo_score:.3f}, expr={best.expressiveness_score:.3f})",
            f"Carnot-optimal proposals: "
            f"{sum(1 for c in compiled_batch if c.is_carnot_optimal)}/{len(compiled_batch)}",
        ]

        if bottleneck_counts:
            top_bn = max(bottleneck_counts, key=lambda k: bottleneck_counts[k])
            lines.append(
                f"Most common bottleneck: {top_bn} "
                f"({bottleneck_counts[top_bn]}/{len(compiled_batch)} proposals)"
            )

        if not best.is_carnot_optimal:
            lines.append("Best proposal waste attribution:")
            for name, diag in self.waste_attribution(best).items():
                lines.append(f"  [{name}] {diag}")
        else:
            lines.append("Best proposal satisfies all Carnot conditions ✓")

        return "\n".join(lines)

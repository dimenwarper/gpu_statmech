"""
Physics-based kernel architecture oracle (Phase 2).

The oracle proposes CUDA kernel specifications constrained to the
Carnot-optimal class.  It operates as a sampling distribution over the
space of valid kernel configurations, guided by the derived Carnot
conditions and updated via feedback from the thermodynamic scorer.

Design
------
Each proposal is a ``KernelProposal`` — a lightweight description of a
kernel's resource footprint and compute characteristics:

  • Thread block configuration  (block_size, grid_size)
  • On-chip resource usage      (registers_per_thread, smem_bytes)
  • Compute profile             (arithmetic_intensity, tensor_core_utilisation)
  • Memory access pattern       (coalesced / strided / random)
  • Data reuse at each level    (reuse_factors: dict[level → float])
  • Unnecessary data movement   (unnecessary_data_movement ∈ [0, 1])

The oracle maintains a factored Gaussian prior over the continuous
parameters and a categorical distribution over the discrete ones.  After
each scored batch the prior is updated via an exponential-smoothing
evolutionary step (simple (μ, λ)-ES):

    new_mean = (1 − α) × old_mean + α × mean(top-k)
    new_std  = max(min_std, (1 − α) × old_std  + α × std(top-k))

This keeps the oracle exploratory (non-zero std) while converging toward
the Carnot-optimal region.

All sampling is deterministic given an ``np.random.Generator`` seed, so
experiments are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .carnot import CarnotLimit
from .partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG, MemoryLevel, SMConfig


# ---------------------------------------------------------------------------
# KernelProposal
# ---------------------------------------------------------------------------

# Valid block sizes on NVIDIA GPUs (must be a multiple of warp size = 32,
# and ≤ 1024 which is the H100 max threads-per-block).
VALID_BLOCK_SIZES: list[int] = [64, 128, 256, 512, 1024]

# Memory access pattern labels and their coalescing efficiency ∈ [0, 1].
ACCESS_PATTERNS: dict[str, float] = {
    "coalesced": 1.0,
    "strided":   0.6,
    "random":    0.2,
}


@dataclass
class KernelProposal:
    """
    A proposed kernel specification from the oracle.

    Fields are the minimal set needed to:
      1. Derive a ``KernelSpec`` (for Carnot-optimality checking), and
      2. Compute an expressiveness proxy score.

    Attributes
    ----------
    name:
        Human-readable identifier (e.g. ``"oracle_iter3_42"``).
    block_size:
        Threads per block.  Must be in ``VALID_BLOCK_SIZES``.
    grid_size:
        Number of thread blocks.  Determines total parallelism.
    registers_per_thread:
        Register file usage per thread.  H100 limit = 255.
    smem_bytes:
        Static shared-memory allocation per block (bytes).
        H100 limit per SM = 228 × 1024 bytes.
    arithmetic_intensity:
        FLOP per byte of HBM traffic.  The roofline ridge point for the
        H100 (≈ 0.51 FLOP/byte) is the minimum for compute-bound operation.
    tensor_core_utilisation:
        Fraction of FP16/BF16 operations dispatched to tensor cores ∈ [0, 1].
        Tensor cores deliver 8–16× throughput vs CUDA cores for GEMM.
    memory_access_pattern:
        One of ``"coalesced"``, ``"strided"``, or ``"random"``.
    reuse_factors:
        Achieved data reuse at each memory level (ops per byte loaded from
        that level).  Keys match ``MemoryLevel.name``:
        ``"smem"``, ``"L2"``, ``"HBM"``.
    unnecessary_data_movement:
        Fraction of data movement that is redundant ∈ [0, 1].
    iteration:
        Oracle iteration that produced this proposal (for provenance).
    """
    name: str
    block_size: int
    grid_size: int
    registers_per_thread: int
    smem_bytes: int
    arithmetic_intensity: float
    tensor_core_utilisation: float
    memory_access_pattern: str
    reuse_factors: dict[str, float]
    unnecessary_data_movement: float = 0.0
    iteration: int = 0

    def __post_init__(self) -> None:
        if self.memory_access_pattern not in ACCESS_PATTERNS:
            raise ValueError(
                f"memory_access_pattern must be one of {list(ACCESS_PATTERNS)}, "
                f"got {self.memory_access_pattern!r}"
            )
        if not (0.0 <= self.tensor_core_utilisation <= 1.0):
            raise ValueError("tensor_core_utilisation must be in [0, 1]")
        if not (0.0 <= self.unnecessary_data_movement <= 1.0):
            raise ValueError("unnecessary_data_movement must be in [0, 1]")


# ---------------------------------------------------------------------------
# Oracle prior state
# ---------------------------------------------------------------------------

@dataclass
class OraclePrior:
    """
    Factored Gaussian + categorical prior over kernel parameters.

    Continuous parameters are represented as (mean, std) pairs in the
    natural space of each parameter (log-space for AI and reuse factors).
    Categorical parameters are represented as probability vectors.

    Attributes
    ----------
    log_ai_mean, log_ai_std:
        Log-space mean and std for arithmetic_intensity.
    smem_mean, smem_std:
        Mean and std for smem_bytes (bytes).
    reg_mean, reg_std:
        Mean and std for registers_per_thread.
    tc_util_mean, tc_util_std:
        Mean and std for tensor_core_utilisation ∈ [0, 1]
        (sampled from a clipped normal).
    udm_mean, udm_std:
        Mean and std for unnecessary_data_movement ∈ [0, 1].
    log_smem_reuse_mean, log_smem_reuse_std:
        Log-space mean and std for reuse_factors["smem"].
    log_l2_reuse_mean, log_l2_reuse_std:
        Log-space mean and std for reuse_factors["L2"].
    block_size_probs:
        Categorical probabilities over VALID_BLOCK_SIZES.
    access_pattern_probs:
        Categorical probabilities over ACCESS_PATTERNS keys.
    """
    # log(AI) ~ N(log_ai_mean, log_ai_std)
    log_ai_mean: float
    log_ai_std: float

    # smem_bytes ~ N(smem_mean, smem_std), clipped to [0, smem_capacity]
    smem_mean: float
    smem_std: float

    # registers_per_thread ~ N(reg_mean, reg_std), clipped to [32, 255]
    reg_mean: float
    reg_std: float

    # tensor_core_utilisation ~ N(tc_util_mean, tc_util_std), clipped to [0,1]
    tc_util_mean: float
    tc_util_std: float

    # unnecessary_data_movement ~ N(udm_mean, udm_std), clipped to [0,1]
    udm_mean: float
    udm_std: float

    # log(smem_reuse) ~ N(log_smem_reuse_mean, log_smem_reuse_std)
    log_smem_reuse_mean: float
    log_smem_reuse_std: float

    # log(L2_reuse) ~ N(log_l2_reuse_mean, log_l2_reuse_std)
    log_l2_reuse_mean: float
    log_l2_reuse_std: float

    # Categorical: probabilities over VALID_BLOCK_SIZES
    block_size_probs: NDArray  # shape (len(VALID_BLOCK_SIZES),)

    # Categorical: probabilities over ACCESS_PATTERNS
    access_pattern_probs: NDArray  # shape (3,)


def _default_prior(carnot_limit: CarnotLimit, smem_capacity: int) -> OraclePrior:
    """
    Construct the default uninformative prior centred on the Carnot conditions.

    The AI prior is centred at the roofline ridge (the minimum for
    compute-bound operation) with std = 1.5 decades in log-space, giving
    broad initial coverage.

    Reuse factor priors are centred at the Carnot minimum reuse (from the
    limit's min_reuse_factors) so initial proposals cluster near feasibility.
    """
    ridge = max(carnot_limit.roofline_intensity, 1e-3)
    log_ai_mean = math.log(ridge)

    # smem: start at half capacity, ±25%
    smem_mean = smem_capacity * 0.5
    smem_std = smem_capacity * 0.25

    # registers: 64 ± 32 (reasonable H100 starting point)
    reg_mean = 64.0
    reg_std = 32.0

    # tensor core utilisation: beta-like, start at 0.5 ± 0.25
    tc_util_mean = 0.5
    tc_util_std = 0.25

    # unnecessary data movement: start low
    udm_mean = 0.1
    udm_std = 0.1

    # Reuse factors: centre at Carnot minimums (use geometric mean if available)
    smem_min = carnot_limit.min_reuse_factors.get("smem", 1.0)
    l2_min   = carnot_limit.min_reuse_factors.get("L2", 1.0)
    log_smem_reuse_mean = math.log(max(smem_min, 1.0))
    log_l2_reuse_mean   = math.log(max(l2_min, 1.0))

    return OraclePrior(
        log_ai_mean=log_ai_mean,
        log_ai_std=1.5,
        smem_mean=smem_mean,
        smem_std=smem_std,
        reg_mean=reg_mean,
        reg_std=reg_std,
        tc_util_mean=tc_util_mean,
        tc_util_std=tc_util_std,
        udm_mean=udm_mean,
        udm_std=udm_std,
        log_smem_reuse_mean=log_smem_reuse_mean,
        log_smem_reuse_std=2.0,
        log_l2_reuse_mean=log_l2_reuse_mean,
        log_l2_reuse_std=2.0,
        block_size_probs=np.ones(len(VALID_BLOCK_SIZES)) / len(VALID_BLOCK_SIZES),
        access_pattern_probs=np.ones(len(ACCESS_PATTERNS)) / len(ACCESS_PATTERNS),
    )


# ---------------------------------------------------------------------------
# Physics oracle
# ---------------------------------------------------------------------------

class PhysicsOracle:
    """
    Physics-guided kernel architecture oracle.

    Proposes ``KernelProposal`` objects sampled from a factored prior over
    kernel configuration space.  The prior is initially broad (covering the
    full feasible region) and is refined via ``feedback()`` toward the
    Carnot-optimal corner.

    Parameters
    ----------
    carnot_limit:
        The derived η_hw,max and associated Carnot conditions.
    memory_levels:
        Memory hierarchy to use for capacity / bandwidth constraints.
    sm_config:
        SM configuration for occupancy calculations.
    alpha:
        Exponential-smoothing learning rate for prior updates ∈ (0, 1].
        Larger → faster convergence, less memory of history.
    min_std_fraction:
        Minimum allowed standard deviation as a fraction of the parameter
        range; prevents the prior from collapsing (maintains exploration).
    top_k_fraction:
        Fraction of scored proposals used to update the prior.
        E.g. 0.3 → top 30% of proposals drive the update.
    """

    def __init__(
        self,
        carnot_limit: CarnotLimit,
        memory_levels: list[MemoryLevel] | None = None,
        sm_config: SMConfig | None = None,
        alpha: float = 0.4,
        min_std_fraction: float = 0.05,
        top_k_fraction: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.carnot_limit = carnot_limit
        self.memory_levels = memory_levels or H100_MEMORY_LEVELS
        self.sm_config = sm_config or H100_SM_CONFIG
        self.alpha = alpha
        self.min_std_fraction = min_std_fraction
        self.top_k_fraction = top_k_fraction
        self._seed = seed

        smem_level = next(
            (lvl for lvl in self.memory_levels if lvl.name == "smem"),
            self.memory_levels[1],
        )
        self._smem_capacity: int = smem_level.capacity_bytes
        self._iteration: int = 0
        self.prior: OraclePrior = _default_prior(carnot_limit, self._smem_capacity)

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def propose(
        self,
        n: int = 20,
        rng: np.random.Generator | None = None,
    ) -> list[KernelProposal]:
        """
        Sample ``n`` kernel proposals from the current prior.

        Parameters
        ----------
        n:
            Number of proposals to generate.
        rng:
            Random generator for reproducibility.  A fresh default-seeded
            generator is used if ``None``.

        Returns
        -------
        list[KernelProposal]
            ``n`` proposals, each drawn independently from the prior.
        """
        if rng is None:
            rng = np.random.default_rng()

        proposals: list[KernelProposal] = []
        for i in range(n):
            prop = self._sample_one(rng, index=i)
            proposals.append(prop)

        self._iteration += 1
        return proposals

    def _sample_one(self, rng: np.random.Generator, index: int) -> KernelProposal:
        """Draw a single proposal from the current prior."""
        p = self.prior

        # --- block_size (categorical) ---
        block_size = int(rng.choice(VALID_BLOCK_SIZES, p=p.block_size_probs))

        # --- grid_size: ensure enough parallelism to cover H100 SMs ---
        # Grid ≥ n_sm to keep all SMs busy; sample from 1x–8x SM count.
        n_sm = self.sm_config.n_sm
        grid_size = int(rng.integers(n_sm, 8 * n_sm + 1))

        # --- registers_per_thread (clipped normal, multiples of 8) ---
        reg_raw = rng.normal(p.reg_mean, p.reg_std)
        reg = int(np.clip(round(reg_raw / 8) * 8, 32, 255))

        # --- smem_bytes (clipped normal, multiples of 128 bytes for alignment) ---
        smem_raw = rng.normal(p.smem_mean, p.smem_std)
        smem = int(np.clip(round(smem_raw / 128) * 128, 0, self._smem_capacity))

        # --- arithmetic_intensity (log-normal) ---
        log_ai = rng.normal(p.log_ai_mean, p.log_ai_std)
        ai = float(np.clip(math.exp(log_ai), 1e-3, 1e4))

        # --- tensor_core_utilisation (clipped normal) ---
        tc_raw = rng.normal(p.tc_util_mean, p.tc_util_std)
        tc_util = float(np.clip(tc_raw, 0.0, 1.0))

        # --- memory_access_pattern (categorical) ---
        access_names = list(ACCESS_PATTERNS.keys())
        pattern = str(rng.choice(access_names, p=p.access_pattern_probs))

        # --- reuse_factors ---
        # HBM reuse = AI (by definition: FLOP / byte of HBM traffic).
        # smem and L2 reuse sampled independently (log-normal).
        log_smem_r = rng.normal(p.log_smem_reuse_mean, p.log_smem_reuse_std)
        log_l2_r   = rng.normal(p.log_l2_reuse_mean,   p.log_l2_reuse_std)
        smem_reuse = float(np.clip(math.exp(log_smem_r), 1.0, 1e7))
        l2_reuse   = float(np.clip(math.exp(log_l2_r),   1.0, 1e5))

        reuse_factors = {
            "smem": smem_reuse,
            "L2":   l2_reuse,
            "HBM":  ai,
        }

        # --- unnecessary_data_movement (clipped normal) ---
        udm_raw = rng.normal(p.udm_mean, p.udm_std)
        udm = float(np.clip(udm_raw, 0.0, 1.0))

        return KernelProposal(
            name=f"oracle_i{self._iteration}_p{index}",
            block_size=block_size,
            grid_size=grid_size,
            registers_per_thread=reg,
            smem_bytes=smem,
            arithmetic_intensity=ai,
            tensor_core_utilisation=tc_util,
            memory_access_pattern=pattern,
            reuse_factors=reuse_factors,
            unnecessary_data_movement=udm,
            iteration=self._iteration,
        )

    # ------------------------------------------------------------------
    # Prior update (feedback)
    # ------------------------------------------------------------------

    def feedback(
        self,
        proposals: Sequence[KernelProposal],
        scores: Sequence[float],
    ) -> None:
        """
        Update the proposal prior from a scored batch.

        Selects the top-k proposals (by score), computes their parameter
        statistics, and updates the prior via exponential smoothing.

        Parameters
        ----------
        proposals:
            The proposals that were scored.
        scores:
            Combined scalar score for each proposal (higher = better).
            Typically ``eta_hw_fraction + expressiveness_score``.
        """
        if len(proposals) != len(scores):
            raise ValueError(
                f"len(proposals)={len(proposals)} != len(scores)={len(scores)}"
            )
        if len(proposals) == 0:
            return

        arr_scores = np.asarray(scores, dtype=float)
        k = max(1, int(math.ceil(len(proposals) * self.top_k_fraction)))
        top_indices = np.argsort(arr_scores)[-k:]
        top_props = [proposals[i] for i in top_indices]

        α = self.alpha
        p = self.prior

        # --- Continuous parameter updates ---
        def _update_mean_std(
            old_mean: float,
            old_std: float,
            values: list[float],
            min_std: float,
        ) -> tuple[float, float]:
            arr = np.asarray(values, dtype=float)
            new_mean = (1 - α) * old_mean + α * float(arr.mean())
            new_std  = (1 - α) * old_std  + α * float(arr.std() if len(arr) > 1 else old_std)
            return new_mean, max(min_std, new_std)

        # log(AI)
        log_ais = [math.log(max(pr.arithmetic_intensity, 1e-9)) for pr in top_props]
        p.log_ai_mean, p.log_ai_std = _update_mean_std(
            p.log_ai_mean, p.log_ai_std, log_ais,
            min_std=self.min_std_fraction * 5,   # ~0.25 decades minimum
        )

        # smem_bytes
        smems = [float(pr.smem_bytes) for pr in top_props]
        p.smem_mean, p.smem_std = _update_mean_std(
            p.smem_mean, p.smem_std, smems,
            min_std=self._smem_capacity * self.min_std_fraction,
        )

        # registers_per_thread
        regs = [float(pr.registers_per_thread) for pr in top_props]
        p.reg_mean, p.reg_std = _update_mean_std(
            p.reg_mean, p.reg_std, regs,
            min_std=8.0,  # at least one step
        )

        # tensor_core_utilisation
        tcs = [pr.tensor_core_utilisation for pr in top_props]
        p.tc_util_mean, p.tc_util_std = _update_mean_std(
            p.tc_util_mean, p.tc_util_std, tcs,
            min_std=self.min_std_fraction,
        )

        # unnecessary_data_movement
        udms = [pr.unnecessary_data_movement for pr in top_props]
        p.udm_mean, p.udm_std = _update_mean_std(
            p.udm_mean, p.udm_std, udms,
            min_std=self.min_std_fraction,
        )

        # log(smem_reuse)
        log_smem_rs = [
            math.log(max(pr.reuse_factors.get("smem", 1.0), 1e-9))
            for pr in top_props
        ]
        p.log_smem_reuse_mean, p.log_smem_reuse_std = _update_mean_std(
            p.log_smem_reuse_mean, p.log_smem_reuse_std, log_smem_rs,
            min_std=0.25,
        )

        # log(L2_reuse)
        log_l2_rs = [
            math.log(max(pr.reuse_factors.get("L2", 1.0), 1e-9))
            for pr in top_props
        ]
        p.log_l2_reuse_mean, p.log_l2_reuse_std = _update_mean_std(
            p.log_l2_reuse_mean, p.log_l2_reuse_std, log_l2_rs,
            min_std=0.25,
        )

        # --- Categorical parameter updates ---

        # block_size: increment counts for top-k choices and normalise
        bs_counts = np.array(p.block_size_probs, dtype=float)
        for pr in top_props:
            idx = VALID_BLOCK_SIZES.index(pr.block_size)
            bs_counts[idx] += 1.0
        p.block_size_probs = bs_counts / bs_counts.sum()

        # access_pattern
        ap_names = list(ACCESS_PATTERNS.keys())
        ap_counts = np.array(p.access_pattern_probs, dtype=float)
        for pr in top_props:
            idx = ap_names.index(pr.memory_access_pattern)
            ap_counts[idx] += 1.0
        p.access_pattern_probs = ap_counts / ap_counts.sum()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        """Number of ``propose()`` calls made so far."""
        return self._iteration

    def reset(self) -> None:
        """Reset the prior to its initial state."""
        self._iteration = 0
        self.prior = _default_prior(self.carnot_limit, self._smem_capacity)

    def carnot_prompt(self) -> str:
        """
        Return a human-readable description of the Carnot constraints
        that an LLM-based oracle could use as a system prompt.
        """
        cl = self.carnot_limit
        lines = [
            "=== GPU Carnot-Optimal Kernel Constraints (H100) ===",
            "",
            f"  η_hw,max            = {cl.eta_hw_max * 100:.2f}%",
            f"  β_optimal           = {cl.beta_optimal:.3f}",
            f"  Roofline ridge AI   ≥ {cl.roofline_intensity:.2f} FLOP/byte",
            f"  Min warp occupancy  ≥ {cl.min_warp_occupancy:.2f}",
            "",
            "  Minimum data reuse factors:",
        ]
        for lvl, reuse in cl.min_reuse_factors.items():
            lines.append(f"    {lvl:12s}: {reuse:.2e}×")
        lines += [
            "",
            "  Effective temperatures (T_eff = latency / T_reg):",
        ]
        for lvl, t in cl.T_eff.items():
            lines.append(f"    {lvl:12s}: {t:.0f}×")
        lines += [
            "",
            "To approach η_hw,max a kernel must satisfy ALL of:",
            "  1. AI ≥ roofline ridge (compute-bound operation)",
            "  2. Working set ≤ capacity at each memory level",
            "  3. Data reuse ≥ minimum at each level",
            "  4. Warp occupancy ≥ minimum (latency hiding)",
            "  5. Unnecessary data movement = 0",
        ]
        return "\n".join(lines)

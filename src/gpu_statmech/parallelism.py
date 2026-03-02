"""
parallelism.py — Parallelism configurations as thermodynamic phases.

Different parallelism strategies correspond to different ordered phases of
the multi-GPU system, each with a characteristic symmetry and communication
pattern.  Per Section 3.7 of the project brief:

    DATA_PARALLEL     → Ferromagnetic phase
    TENSOR_PARALLEL   → Antiferromagnetic phase
    PIPELINE_PARALLEL → Domain-wall phase
    EXPERT_PARALLEL   → Spin-glass phase (MoE)
    CONTEXT_PARALLEL  → Helical phase

The key prediction is that phase transitions between strategies are sharp
functions of model size, sequence length, and interconnect bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# Thermodynamic phase labels
# ---------------------------------------------------------------------------


class ParallelismPhase(Enum):
    """
    Thermodynamic phases of the multi-GPU system.

    Each phase has a characteristic symmetry and collective communication
    pattern, analogous to ordered phases in condensed matter physics.
    """
    DATA_PARALLEL     = auto()  # Ferromagnetic: all GPUs hold same model replica
    TENSOR_PARALLEL   = auto()  # Antiferromagnetic: complementary tensor shards
    PIPELINE_PARALLEL = auto()  # Domain-wall: layers partitioned across GPUs
    EXPERT_PARALLEL   = auto()  # Spin-glass: MoE disordered token routing
    CONTEXT_PARALLEL  = auto()  # Helical: sequence axis sharded, ring attention
    HYBRID            = auto()  # Superposition of multiple phases
    NONE              = auto()  # Single-GPU; no parallelism


# ---------------------------------------------------------------------------
# Parallelism configuration
# ---------------------------------------------------------------------------


@dataclass
class ParallelismConfig:
    """
    P = (dp, tp, pp, ep, cp): Degrees of each parallelism dimension.

    Constraint:  dp × tp × pp × ep × cp ≤ G  (total GPU count)

    Attributes:
        dp  Data parallelism degree       — replica count
        tp  Tensor parallelism degree     — ops sharded across tp GPUs
        pp  Pipeline parallelism degree   — model depth split into pp stages
        ep  Expert parallelism degree     — MoE experts distributed
        cp  Context/sequence parallelism  — sequence dimension sharded

    Mapping to thermodynamic phases (Section 3.7):
        dp > 1, rest = 1  → DATA_PARALLEL    (ferromagnet)
        tp > 1, rest = 1  → TENSOR_PARALLEL  (antiferromagnet)
        pp > 1, rest = 1  → PIPELINE_PARALLEL (domain wall)
        ep > 1, rest = 1  → EXPERT_PARALLEL  (spin glass)
        cp > 1, rest = 1  → CONTEXT_PARALLEL (helical)
        multiple > 1      → HYBRID
    """
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "ep", "cp"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"ParallelismConfig.{name} must be a positive integer, got {v}")

    # ------------------------------------------------------------------
    # Resource counting
    # ------------------------------------------------------------------

    @property
    def total_gpus(self) -> int:
        """Minimum GPU count required by this configuration."""
        return self.dp * self.tp * self.pp * self.ep * self.cp

    def is_feasible(self, num_gpus: int) -> bool:
        """True if the configuration fits within the available GPU count."""
        return self.total_gpus <= num_gpus

    @property
    def active_dimensions(self) -> list[str]:
        """Names of parallelism dimensions with degree > 1."""
        return [
            name for name, val in [("dp", self.dp), ("tp", self.tp),
                                    ("pp", self.pp), ("ep", self.ep), ("cp", self.cp)]
            if val > 1
        ]

    # ------------------------------------------------------------------
    # Phase identification
    # ------------------------------------------------------------------

    @property
    def phase(self) -> ParallelismPhase:
        """
        Identify the dominant thermodynamic phase.

        Single active dimension → pure phase.
        Multiple active dimensions → HYBRID.
        All ones → NONE (single GPU).
        """
        active = self.active_dimensions
        if not active:
            return ParallelismPhase.NONE
        if len(active) > 1:
            return ParallelismPhase.HYBRID
        return {
            "dp": ParallelismPhase.DATA_PARALLEL,
            "tp": ParallelismPhase.TENSOR_PARALLEL,
            "pp": ParallelismPhase.PIPELINE_PARALLEL,
            "ep": ParallelismPhase.EXPERT_PARALLEL,
            "cp": ParallelismPhase.CONTEXT_PARALLEL,
        }[active[0]]

    @property
    def is_pure_phase(self) -> bool:
        """True if only one parallelism dimension is active."""
        return self.phase not in (ParallelismPhase.HYBRID, ParallelismPhase.NONE)

    # ------------------------------------------------------------------
    # Communication characteristics
    # ------------------------------------------------------------------

    @property
    def dominant_collective(self) -> str:
        """
        Primary collective operation required by the dominant phase.

        DATA_PARALLEL     → all-reduce  (gradient sync)
        TENSOR_PARALLEL   → all-reduce / all-gather (per-layer)
        PIPELINE_PARALLEL → point-to-point (activation passing)
        EXPERT_PARALLEL   → all-to-all   (token dispatch)
        CONTEXT_PARALLEL  → all-gather   (ring attention)
        HYBRID            → mixed
        """
        mapping = {
            ParallelismPhase.DATA_PARALLEL:     "all-reduce",
            ParallelismPhase.TENSOR_PARALLEL:   "all-reduce/all-gather",
            ParallelismPhase.PIPELINE_PARALLEL: "point-to-point",
            ParallelismPhase.EXPERT_PARALLEL:   "all-to-all",
            ParallelismPhase.CONTEXT_PARALLEL:  "all-gather",
            ParallelismPhase.HYBRID:            "mixed",
            ParallelismPhase.NONE:              "none",
        }
        return mapping[self.phase]

    @property
    def has_pipeline_bubbles(self) -> bool:
        """Pipeline parallelism introduces bubble idle time by construction."""
        return self.pp > 1

    @property
    def bubble_fraction(self) -> float:
        """
        Theoretical pipeline bubble fraction for a given pp degree.

        For a linear pipeline with pp stages and micro-batches m,
        the bubble fraction ≈ (pp - 1) / (pp - 1 + m).

        We use the worst-case (m = 1) as a conservative estimate.
        """
        if self.pp <= 1:
            return 0.0
        return (self.pp - 1) / self.pp  # m=1 worst case

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def single_gpu(cls) -> ParallelismConfig:
        return cls()

    @classmethod
    def data_only(cls, num_gpus: int) -> ParallelismConfig:
        """Pure data parallelism — ferromagnetic phase."""
        return cls(dp=num_gpus)

    @classmethod
    def tensor_only(cls, num_gpus: int) -> ParallelismConfig:
        """Pure tensor parallelism — antiferromagnetic phase."""
        return cls(tp=num_gpus)

    @classmethod
    def megatron_style(cls, dp: int, tp: int, pp: int = 1) -> ParallelismConfig:
        """
        TP within NVSwitch domain, DP across nodes, optional PP.
        Matches the Megatron-LM default strategy for large transformers.
        """
        return cls(dp=dp, tp=tp, pp=pp)

    @classmethod
    def zero3_style(cls, num_gpus: int) -> ParallelismConfig:
        """
        ZeRO-3 / FSDP: pure data parallelism with sharded optimizer state.
        Equivalent to DP at the parallelism-config level.
        """
        return cls(dp=num_gpus)


# ---------------------------------------------------------------------------
# Parallelism search space
# ---------------------------------------------------------------------------


def enumerate_configs(num_gpus: int) -> Iterator[ParallelismConfig]:
    """
    Enumerate all feasible ParallelismConfig objects for a given GPU count.

    Yields configurations where dp × tp × pp × ep × cp ≤ num_gpus.
    Excludes configurations with ep > 1 unless num_gpus ≥ 4 (MoE only
    makes sense at larger scale).

    Note: the search space can be large.  For num_gpus = 64 this yields
    O(1000+) configurations.  Use prune_configs() to reduce to a tractable set.
    """
    from math import prod

    factors = [d for d in range(1, num_gpus + 1) if num_gpus % d == 0]

    for dp in factors:
        for tp in factors:
            for pp in factors:
                ep_range = factors if num_gpus >= 4 else [1]
                for ep in ep_range:
                    for cp in factors:
                        if dp * tp * pp * ep * cp <= num_gpus:
                            yield ParallelismConfig(dp=dp, tp=tp, pp=pp, ep=ep, cp=cp)


def prune_configs(
    configs: Iterator[ParallelismConfig],
    num_gpus: int,
    max_tp: int = 8,
    max_pp: int = 16,
    require_dp_ge: int = 1,
) -> list[ParallelismConfig]:
    """
    Filter a config iterator down to a tractable set using practical priors.

    Priors (from empirical ML systems knowledge):
        - TP ≤ 8: tensor parallelism beyond one NVSwitch domain is rarely efficient
        - PP ≤ 16: deep pipelines have excessive bubble overhead
        - dp ≥ require_dp_ge: always keep some data parallelism
    """
    return [
        c for c in configs
        if c.tp <= max_tp
        and c.pp <= max_pp
        and c.dp >= require_dp_ge
        and c.is_feasible(num_gpus)
    ]

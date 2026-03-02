"""
microstate.py — Microstate σ for a single GPU.

A microstate σ is a complete specification of the GPU at a given clock cycle,
covering SM occupancy, memory hierarchy state, and bandwidth channel utilization.

Per Section 3.1 of the project brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class InstructionType(Enum):
    """Instruction types that a warp can be executing."""
    FP16 = auto()
    FP32 = auto()
    INT = auto()
    SFU = auto()          # Special Function Unit (sin, cos, rcp, sqrt, …)
    MEM = auto()          # Memory operation (load / store)
    TENSOR_CORE = auto()  # Warp-level MMA (hmma / imma)
    IDLE = auto()         # Warp is stalled or not scheduled


class PipelineStage(Enum):
    """Pipeline stage a warp is currently in."""
    FETCH = auto()
    DECODE = auto()
    ISSUE = auto()
    EXECUTE = auto()
    WRITEBACK = auto()
    STALL = auto()  # Blocked on data, resource, or memory


class MemoryLevel(Enum):
    """
    Discrete energy levels of the memory hierarchy.

    Per Section 3.3 (H100 specs):
        REGISTERS   ε₀ = 0     ~256 KB/SM,   ~0 cycle latency
        SHARED_MEM  ε₁ = 25    228 KB/SM,    ~20–30 cycle latency
        L1_CACHE    ε₁ = 25    (unified with shared mem / L1 on H100)
        L2_CACHE    ε₂ = 200   50 MB total,  ~200 cycle latency
        HBM         ε₃ = 600   80 GB,        ~400–800 cycle latency
    """
    REGISTERS  = 0
    SHARED_MEM = 1
    L1_CACHE   = 2
    L2_CACHE   = 3
    HBM        = 4


# Energy cost (in cycles) for data resident at each memory level.
MEMORY_LEVEL_ENERGY: dict[MemoryLevel, int] = {
    MemoryLevel.REGISTERS:  0,
    MemoryLevel.SHARED_MEM: 25,
    MemoryLevel.L1_CACHE:   25,
    MemoryLevel.L2_CACHE:   200,
    MemoryLevel.HBM:        600,
}


# ---------------------------------------------------------------------------
# Warp-level state
# ---------------------------------------------------------------------------


@dataclass
class WarpState:
    """State of a single warp within an SM."""
    warp_id: int
    is_active: bool
    instruction_type: InstructionType = InstructionType.IDLE
    pipeline_stage: PipelineStage = PipelineStage.IDLE

    @property
    def is_stalled(self) -> bool:
        return self.pipeline_stage == PipelineStage.STALL

    @property
    def is_computing(self) -> bool:
        return (
            self.is_active
            and self.instruction_type not in (InstructionType.IDLE, InstructionType.MEM)
            and not self.is_stalled
        )


# ---------------------------------------------------------------------------
# SM-level state
# ---------------------------------------------------------------------------


@dataclass
class SMState:
    """State of a single Streaming Multiprocessor at one clock cycle."""
    sm_id: int
    active_warps: int
    max_warps: int
    warp_states: list[WarpState] = field(default_factory=list)
    # Fraction of issued instructions per type (should sum to ≤ 1.0)
    instruction_mix: dict[InstructionType, float] = field(default_factory=dict)

    @property
    def occupancy(self) -> float:
        """Warp occupancy: active_warps / max_warps ∈ [0, 1]."""
        return self.active_warps / self.max_warps if self.max_warps > 0 else 0.0

    @property
    def stall_fraction(self) -> float:
        """Fraction of warps currently in a STALL stage."""
        if not self.warp_states:
            return 0.0
        stalled = sum(1 for w in self.warp_states if w.is_stalled)
        return stalled / len(self.warp_states)

    @property
    def compute_fraction(self) -> float:
        """Fraction of warps actively computing (non-MEM, non-stalled)."""
        if not self.warp_states:
            return 0.0
        computing = sum(1 for w in self.warp_states if w.is_computing)
        return computing / len(self.warp_states)


# ---------------------------------------------------------------------------
# Memory hierarchy state
# ---------------------------------------------------------------------------


@dataclass
class MemoryHierarchyState:
    """
    State of the GPU memory hierarchy at one clock cycle.

    Tracks utilization at each level and, optionally, the fine-grained
    occupation numbers n_i(l) = bytes of data element i resident at level l.
    """
    register_utilization: float    # Fraction of register file in use ∈ [0, 1]
    shared_mem_utilization: float  # Fraction of per-SM SMEM allocated ∈ [0, 1]
    l1_hit_rate: float             # L1 cache hit rate ∈ [0, 1]
    l2_hit_rate: float             # L2 cache hit rate ∈ [0, 1]
    l2_utilization: float          # Fraction of L2 bandwidth consumed ∈ [0, 1]
    hbm_bandwidth_utilization: float  # Fraction of peak HBM BW consumed ∈ [0, 1]

    # Fine-grained occupation numbers: (data_element_id, MemoryLevel) → bytes.
    # Optional; if empty, the energy functional falls back to coarser proxies.
    occupation_numbers: dict[tuple[int, MemoryLevel], int] = field(
        default_factory=dict
    )

    @property
    def mean_data_energy(self) -> float:
        """
        Boltzmann-weighted mean energy level of resident data.

        E_data = Σ_{i,l} n_{i,l} · ε_l  /  Σ_{i,l} n_{i,l}

        Returns 0 if no occupation numbers are available.
        """
        if not self.occupation_numbers:
            return 0.0
        total_bytes = sum(self.occupation_numbers.values())
        if total_bytes == 0:
            return 0.0
        weighted = sum(
            bytes_at * MEMORY_LEVEL_ENERGY[level]
            for (_, level), bytes_at in self.occupation_numbers.items()
        )
        return weighted / total_bytes


# ---------------------------------------------------------------------------
# Bandwidth channel state
# ---------------------------------------------------------------------------


@dataclass
class BandwidthState:
    """
    Utilization of each memory bandwidth channel ∈ [0, 1].

    Channels modeled (per Section 3.1):
        sm_to_shared_mem  — SM ↔ per-block shared memory
        sm_to_l2          — SM ↔ L2 cache (unified)
        l2_to_hbm         — L2 ↔ HBM DRAM
        nvlink             — SM ↔ SM via NVLink (multi-GPU; 0 for single-GPU)
    """
    sm_to_shared_mem: float = 0.0
    sm_to_l2: float = 0.0
    l2_to_hbm: float = 0.0
    nvlink: float = 0.0

    @property
    def mean_utilization(self) -> float:
        channels = [self.sm_to_shared_mem, self.sm_to_l2, self.l2_to_hbm]
        return sum(channels) / len(channels)

    @property
    def bottleneck(self) -> str:
        """Name of the most-utilized bandwidth channel."""
        channels = {
            "sm_to_shared_mem": self.sm_to_shared_mem,
            "sm_to_l2": self.sm_to_l2,
            "l2_to_hbm": self.l2_to_hbm,
            "nvlink": self.nvlink,
        }
        return max(channels, key=channels.__getitem__)


# ---------------------------------------------------------------------------
# Top-level Microstate
# ---------------------------------------------------------------------------


@dataclass
class Microstate:
    """
    σ: Complete specification of a single GPU at a given clock cycle.

    Covers:
        - SM occupancy vector (active warps, instruction mix, pipeline stage)
        - Memory hierarchy residency (utilization at each level + occupation numbers)
        - Bandwidth channel utilization (SM↔SMEM, SM↔L2, L2↔HBM, NVLink)

    Per Section 3.1 of the project brief.
    """
    cycle: int
    gpu_id: int
    sm_states: list[SMState]
    memory: MemoryHierarchyState
    bandwidth: BandwidthState

    # ------------------------------------------------------------------
    # Derived / aggregate properties
    # ------------------------------------------------------------------

    @property
    def num_sms(self) -> int:
        return len(self.sm_states)

    @property
    def mean_sm_occupancy(self) -> float:
        """Average warp occupancy across all SMs ∈ [0, 1]."""
        if not self.sm_states:
            return 0.0
        return sum(sm.occupancy for sm in self.sm_states) / len(self.sm_states)

    @property
    def sm_active_fraction(self) -> float:
        """Fraction of SMs with at least one active warp ∈ [0, 1]."""
        if not self.sm_states:
            return 0.0
        return sum(1 for sm in self.sm_states if sm.active_warps > 0) / len(self.sm_states)

    @property
    def pipeline_stall_fraction(self) -> float:
        """Fraction of warps across all SMs that are stalled ∈ [0, 1]."""
        total = sum(len(sm.warp_states) for sm in self.sm_states)
        if total == 0:
            return 0.0
        stalled = sum(
            1 for sm in self.sm_states for w in sm.warp_states if w.is_stalled
        )
        return stalled / total

    @property
    def mean_instruction_mix(self) -> dict[InstructionType, float]:
        """Instruction mix averaged across all SMs."""
        if not self.sm_states:
            return {}
        totals: dict[InstructionType, float] = {}
        for sm in self.sm_states:
            for itype, frac in sm.instruction_mix.items():
                totals[itype] = totals.get(itype, 0.0) + frac
        n = len(self.sm_states)
        return {itype: frac / n for itype, frac in totals.items()}

    def summary(self) -> dict:
        """Compact scalar summary suitable for logging / Pareto analysis."""
        return {
            "cycle": self.cycle,
            "gpu_id": self.gpu_id,
            "mean_sm_occupancy": self.mean_sm_occupancy,
            "sm_active_fraction": self.sm_active_fraction,
            "pipeline_stall_fraction": self.pipeline_stall_fraction,
            "hbm_bw_utilization": self.memory.hbm_bandwidth_utilization,
            "l2_hit_rate": self.memory.l2_hit_rate,
            "bw_bottleneck": self.bandwidth.bottleneck,
        }

"""
multi_gpu.py — Global microstate Σ and multi-GPU energy functional.

For a system of G GPUs the global microstate is:

    Σ = (σ₁, σ₂, …, σ_G, C)

where σ_g is the single-GPU Microstate for GPU g and C is the communication
state vector describing all active inter-device channels.

The total multi-GPU energy decomposes as:

    E(Σ) = Σ_g E_local(σ_g)
           + Σ_{(g,h)∈edges} J_{gh} · E_comm(σ_g, σ_h)

where J_{gh} is the coupling constant of the link (g, h) in the topology.

Per Sections 3.5, 3.6, 3.8 of the project brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .microstate import Microstate
from .energy import EnergyFunctional, EnergyWeights
from .topology import Topology


# ---------------------------------------------------------------------------
# Collective operations
# ---------------------------------------------------------------------------


class CollectiveOp(Enum):
    """
    Communication collective patterns, corresponding to thermodynamic phases
    (Section 3.7).

    ALL_REDUCE      — ring/tree gradient sync (Data Parallelism)
    ALL_GATHER      — shard → replica (ZeRO, FSDP)
    REDUCE_SCATTER  — replica → shard (ZeRO, FSDP)
    ALL_TO_ALL      — token dispatch/combine (Expert Parallelism)
    POINT_TO_POINT  — activation passing (Pipeline Parallelism)
    BROADCAST       — parameter broadcast
    """
    ALL_REDUCE      = auto()
    ALL_GATHER      = auto()
    REDUCE_SCATTER  = auto()
    ALL_TO_ALL      = auto()
    POINT_TO_POINT  = auto()
    BROADCAST       = auto()


# Scaling law for communication volume as a function of GPU count G and
# message size M (bytes).  Returns bytes transferred per GPU.
# Per Section 3.8 of the project brief.
def comm_volume_per_gpu(op: CollectiveOp, num_gpus: int, message_bytes: int) -> float:
    """
    Expected bytes-transferred-per-GPU for a given collective.

    ALL_REDUCE (ring):    2·(G-1)/G · M
    ALL_GATHER:           (G-1)/G · M
    REDUCE_SCATTER:       (G-1)/G · M
    ALL_TO_ALL:           (G-1) · M      (sends to every other GPU)
    POINT_TO_POINT:       M              (single send/recv pair)
    BROADCAST:            M              (one sender, G-1 receivers)
    """
    G = num_gpus
    M = message_bytes
    match op:
        case CollectiveOp.ALL_REDUCE:
            return 2.0 * (G - 1) / G * M
        case CollectiveOp.ALL_GATHER | CollectiveOp.REDUCE_SCATTER:
            return (G - 1) / G * M
        case CollectiveOp.ALL_TO_ALL:
            return (G - 1) * M
        case CollectiveOp.POINT_TO_POINT | CollectiveOp.BROADCAST:
            return float(M)
        case _:
            return float(M)


# ---------------------------------------------------------------------------
# Communication channel and state
# ---------------------------------------------------------------------------


@dataclass
class CommChannelState:
    """
    State of a single GPU-to-GPU communication channel at one cycle.

    Attributes:
        src_gpu               Source GPU index
        dst_gpu               Destination GPU index
        bytes_in_flight       Total bytes currently in transit
        bandwidth_utilization Fraction of link peak BW consumed ∈ [0, 1]
        is_active             Whether the channel is transferring data
        latency_events        Number of discrete communication initiations
                              (each incurs a fixed latency overhead)
        sync_stall_fraction   Fraction of time GPUs are idle waiting for
                              this channel (pipeline bubble cost)
    """
    src_gpu: int
    dst_gpu: int
    bytes_in_flight: int
    bandwidth_utilization: float   # ∈ [0, 1]
    is_active: bool
    latency_events: int = 0
    sync_stall_fraction: float = 0.0  # ∈ [0, 1]


@dataclass
class CommState:
    """
    C: Communication state vector — all inter-GPU channels at one cycle.

    Also records the active collective operation (if any) and the
    compute-communication overlap ratio η_overlap.

    Per Sections 3.5 and 3.9 of the project brief.
    """
    channels: list[CommChannelState] = field(default_factory=list)
    active_collective: Optional[CollectiveOp] = None
    collective_message_bytes: int = 0

    # η_overlap = T_overlapped / max(T_compute, T_comm)
    # 1.0 → "superconducting" phase (zero comm stalls)
    # 0.0 → fully serialized compute and communication
    overlap_ratio: float = 0.0

    @property
    def mean_bandwidth_utilization(self) -> float:
        if not self.channels:
            return 0.0
        active = [c for c in self.channels if c.is_active]
        if not active:
            return 0.0
        return sum(c.bandwidth_utilization for c in active) / len(active)

    @property
    def total_bytes_in_flight(self) -> int:
        return sum(c.bytes_in_flight for c in self.channels)

    @property
    def mean_sync_stall(self) -> float:
        if not self.channels:
            return 0.0
        return sum(c.sync_stall_fraction for c in self.channels) / len(self.channels)

    def channel(self, src: int, dst: int) -> Optional[CommChannelState]:
        """Look up the channel from src to dst."""
        for c in self.channels:
            if c.src_gpu == src and c.dst_gpu == dst:
                return c
        return None


# ---------------------------------------------------------------------------
# Global microstate
# ---------------------------------------------------------------------------


@dataclass
class GlobalMicrostate:
    """
    Σ = (σ₁, σ₂, …, σ_G, C): Global microstate of a multi-GPU system.

    The global microstate is the tensor product of individual GPU microstates
    plus the communication state vector.  It encodes the complete physical
    configuration of the cluster at one clock cycle.

    Per Section 3.5 of the project brief.
    """
    cycle: int
    gpu_states: list[Microstate]   # σ_g for each GPU g
    comm_state: CommState
    topology: Topology

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_states)

    @property
    def mean_local_occupancy(self) -> float:
        """Mean SM occupancy averaged across all GPUs."""
        if not self.gpu_states:
            return 0.0
        return (
            sum(s.mean_sm_occupancy for s in self.gpu_states) / len(self.gpu_states)
        )

    def summary(self) -> dict:
        return {
            "cycle": self.cycle,
            "num_gpus": self.num_gpus,
            "mean_sm_occupancy": self.mean_local_occupancy,
            "mean_bw_utilization": self.comm_state.mean_bandwidth_utilization,
            "overlap_ratio": self.comm_state.overlap_ratio,
            "active_collective": (
                self.comm_state.active_collective.name
                if self.comm_state.active_collective else None
            ),
        }


# ---------------------------------------------------------------------------
# Communication energy (pairwise)
# ---------------------------------------------------------------------------


@dataclass
class CommEnergyWeights:
    """
    β coefficients for E_comm(σ_g, σ_h).

    Per Section 3.8:
        E_comm = β₁·(bytes/bw) + β₂·latency_events + β₃·sync_stalls
    """
    bandwidth_term:  float = 1.0   # β₁
    latency_term:    float = 0.5   # β₂
    sync_stall_term: float = 1.0   # β₃


def pairwise_comm_energy(
    channel: CommChannelState,
    weights: CommEnergyWeights,
) -> float:
    """
    E_comm(σ_g, σ_h) for a single channel.

    Returns 0 if the channel is inactive.
    All terms are normalized to [0, 1] before weighting.

    bandwidth_term   = β₁ · bw_utilization
    latency_term     = β₂ · tanh(latency_events / 10)   (soft cap)
    sync_stall_term  = β₃ · sync_stall_fraction
    """
    if not channel.is_active:
        return 0.0
    import math
    bw   = weights.bandwidth_term  * channel.bandwidth_utilization
    lat  = weights.latency_term   * math.tanh(channel.latency_events / 10.0)
    sync = weights.sync_stall_term * channel.sync_stall_fraction
    return bw + lat + sync


# ---------------------------------------------------------------------------
# Multi-GPU energy functional
# ---------------------------------------------------------------------------


@dataclass
class MultiGPUEnergyFunctional:
    """
    Computes E(Σ) decomposed into local and interaction terms.

    E(Σ) = Σ_g E_local(σ_g)
           + Σ_{(g,h)∈edges} J_{gh} · E_comm(σ_g, σ_h)

    Attributes:
        local_energy_fn    Single-GPU EnergyFunctional
        comm_weights       β coefficients for pairwise communication energy
    """
    local_energy_fn: EnergyFunctional = field(
        default_factory=EnergyFunctional
    )
    comm_weights: CommEnergyWeights = field(
        default_factory=CommEnergyWeights
    )

    # ------------------------------------------------------------------
    # Local term
    # ------------------------------------------------------------------

    def local_energy(self, state: GlobalMicrostate) -> float:
        """Σ_g E_local(σ_g) — sum of single-GPU energies."""
        return sum(self.local_energy_fn.compute(sg) for sg in state.gpu_states)

    def local_energy_per_gpu(self, state: GlobalMicrostate) -> list[float]:
        """Per-GPU local energies, useful for diagnosing imbalances."""
        return [self.local_energy_fn.compute(sg) for sg in state.gpu_states]

    # ------------------------------------------------------------------
    # Interaction term
    # ------------------------------------------------------------------

    def interaction_energy(self, state: GlobalMicrostate) -> float:
        """
        Σ_{(g,h)∈edges} J_{gh} · E_comm(σ_g, σ_h)

        Iterates over undirected edges in the topology; for each edge,
        looks up the corresponding channel in CommState.  If no channel
        is found (link is idle), the contribution is zero.
        """
        total = 0.0
        for (g, h, link) in state.topology.edges():
            # Try both directions (channel may be src→dst or dst→src)
            ch = state.comm_state.channel(g, h) or state.comm_state.channel(h, g)
            e_comm = pairwise_comm_energy(ch, self.comm_weights) if ch else 0.0
            total += link.coupling_constant * e_comm
        return total

    # ------------------------------------------------------------------
    # Full energy
    # ------------------------------------------------------------------

    def compute(self, state: GlobalMicrostate) -> dict[str, float]:
        """
        Returns energy decomposed into local, interaction, and total.

        {
            "local":       Σ_g E_local(σ_g),
            "interaction": Σ J_{gh} · E_comm,
            "total":       local + interaction,
        }
        """
        e_local       = self.local_energy(state)
        e_interaction = self.interaction_energy(state)
        return {
            "local":       e_local,
            "interaction": e_interaction,
            "total":       e_local + e_interaction,
        }

    def decompose_full(self, state: GlobalMicrostate) -> dict:
        """
        Full diagnostic decomposition: per-GPU local energies + per-edge
        interaction energies + aggregate totals.
        """
        per_gpu = self.local_energy_per_gpu(state)

        per_edge: dict[tuple[int, int], float] = {}
        for (g, h, link) in state.topology.edges():
            ch = state.comm_state.channel(g, h) or state.comm_state.channel(h, g)
            e_comm = pairwise_comm_energy(ch, self.comm_weights) if ch else 0.0
            per_edge[(g, h)] = link.coupling_constant * e_comm

        totals = self.compute(state)
        return {
            "per_gpu_local":       per_gpu,
            "per_edge_interaction": per_edge,
            **totals,
        }

    # ------------------------------------------------------------------
    # Time-averaged energy
    # ------------------------------------------------------------------

    def time_average(
        self, trajectory: list[GlobalMicrostate]
    ) -> dict[str, float]:
        """
        Ē(Σ) decomposed into local and interaction, averaged over the trajectory.
        """
        if not trajectory:
            return {"local": 0.0, "interaction": 0.0, "total": 0.0}
        energies = [self.compute(s) for s in trajectory]
        T = len(trajectory)
        return {k: sum(e[k] for e in energies) / T for k in energies[0]}

    # ------------------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------------------

    def overlap_penalty(self, state: GlobalMicrostate) -> float:
        """
        Penalty term for poor compute-communication overlap.

        A system with η_overlap = 1 (perfect overlap) pays no penalty.
        A system with η_overlap = 0 (fully serialized) pays maximum penalty.

        Per Section 3.9: the overlap ratio η = T_overlapped / max(T_compute, T_comm)
        """
        return 1.0 - state.comm_state.overlap_ratio

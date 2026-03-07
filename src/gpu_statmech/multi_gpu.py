"""
Multi-GPU thermodynamic analysis: coupled-engine Carnot limit.

Extends the single-GPU partition function to N coupled GPUs connected
by a topology graph.  The full Hamiltonian factorises as:

    H({σ}) = Σ_i H_local(σ_i)  +  H_comm({σ})

Under the same mean-field factorisation used for the single-GPU treatment,
the coupled log-partition function separates:

    ln Z_multi(β) = N × ln Z_single(β)  +  ln Z_comm_topology(β)

where:
  - Z_single   is the single-GPU three-component factorisation (Z_compute ×
               Z_memory) with *no* inter-GPU communication terms.
  - Z_comm_topology = Π_{(i,j)∈E} Z_link(β, J_{ij}) is the product over all
               topology edges, each contributing the integral

                   Z_link = (1 - e^{-βJ}) / (βJ)

               which is already implemented in partition_function.z_comm.

From Z_multi all multi-GPU thermodynamic quantities follow:

    F_multi    = -(1/β) ln Z_multi / n_dof_total
    <E>_multi  = -d(ln Z_multi)/dβ  / n_dof_total    (per-DOF waste ∈ [0,1])
    S_multi    =  β (<E>_multi - F_multi)              (entropy per DOF)
    Cv_multi   =  β² × d<E>_multi/dβ                  (specific heat per DOF)
    η_multi    =  1 - <E>_multi
    η_multi,max = max_β η_multi(β)

The resonance condition measures compute–communication balance:

    η_overlap = T_overlapped / max(T_compute, T_comm)

where T_compute and T_comm are the times spent in each regime.
Perfect resonance (η_overlap → 1) requires T_compute ≈ T_comm.

Parallelism strategies map to thermodynamic phases of the communication:
    DP  →  ferromagnetic           (aligned replicas, AllReduce,         low J)
    TP  →  antiferromagnetic       (sharded tensors, AllGather+RS,       mid J)
    PP  →  domain_wall             (stage boundaries, P2P,               low-mid J)
    EP  →  spin_glass              (random routing, AllToAll,            high J)
    CP  →  quasi_antiferromagnetic (sequence sharding, AllGather on KV,  mid J)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    LINK_PRESETS,
    LinkConfig,
    MemoryLevel,
    SMConfig,
    TopologyEdge,
    log_gpu_partition_function,
)


# ---------------------------------------------------------------------------
# Thermodynamic phase labels for each parallelism strategy
# ---------------------------------------------------------------------------

THERMO_PHASE: dict[str, str] = {
    "dp": "ferromagnetic",
    "tp": "antiferromagnetic",
    "pp": "domain_wall",
    "ep": "spin_glass",
    "cp": "quasi_antiferromagnetic",
}


# ---------------------------------------------------------------------------
# Topology description
# ---------------------------------------------------------------------------

@dataclass
class TopologyGraph:
    """
    Multi-GPU topology as a directed graph of communication links.

    Each node is a GPU (0-indexed).  Each directed edge carries a
    LinkConfig with coupling constant J_ij, bandwidth, and latency.
    For an undirected link (i ↔ j), add edges in both directions.

    The coupling constant J encodes how much waste the link introduces:
      - Small J (≈ 0.1): tight NVLink coupling, low communication waste.
      - Large J (≈ 5–10): loose InfiniBand coupling, high waste.
    """
    n_gpu: int
    links: list[TopologyEdge]
    name: str = "custom"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def adjacency_J(self) -> NDArray[np.float64]:
        """Return (n_gpu, n_gpu) coupling matrix J_ij."""
        J = np.zeros((self.n_gpu, self.n_gpu), dtype=np.float64)
        for edge in self.links:
            J[edge.src, edge.dst] = edge.link.coupling_J
        return J

    def mean_J(self) -> float:
        """Mean coupling constant over all links (0.0 if no links)."""
        if not self.links:
            return 0.0
        return float(np.mean([e.link.coupling_J for e in self.links]))

    def total_bandwidth_gb_s(self) -> float:
        """Sum of all link bandwidths in GB/s."""
        return sum(e.link.bandwidth_gb_s for e in self.links)

    def bottleneck_bandwidth_gb_s(self) -> float:
        """Minimum per-link bandwidth — the bottleneck (0.0 if no links)."""
        if not self.links:
            return 0.0
        return min(e.link.bandwidth_gb_s for e in self.links)

    # ------------------------------------------------------------------
    # Factory presets
    # ------------------------------------------------------------------

    @classmethod
    def nvlink_clique(cls, n_gpu: int) -> "TopologyGraph":
        """
        All-to-all NVLink topology (e.g. a DGX H100 8-GPU node).
        Every ordered pair of GPUs is connected by an NVLink-4 link.
        """
        lc = LINK_PRESETS["nvlink4"]
        links: list[TopologyEdge] = [
            TopologyEdge(i, j, lc)
            for i in range(n_gpu)
            for j in range(n_gpu)
            if i != j
        ]
        return cls(n_gpu=n_gpu, links=links, name=f"nvlink_clique_{n_gpu}gpu")

    @classmethod
    def nvswitch_fabric(cls, n_gpu: int) -> "TopologyGraph":
        """
        NVSwitch-connected fabric.  Slightly higher J than direct NVLink
        due to the switch hop, but same bandwidth.
        """
        lc = LINK_PRESETS["nvswitch"]
        links: list[TopologyEdge] = [
            TopologyEdge(i, j, lc)
            for i in range(n_gpu)
            for j in range(n_gpu)
            if i != j
        ]
        return cls(n_gpu=n_gpu, links=links, name=f"nvswitch_{n_gpu}gpu")

    @classmethod
    def pcie_ring(cls, n_gpu: int) -> "TopologyGraph":
        """
        Bidirectional ring topology over PCIe Gen 5.
        Typical of workstation multi-GPU without NVLink.
        """
        lc = LINK_PRESETS["pcie_gen5"]
        links: list[TopologyEdge] = []
        for i in range(n_gpu):
            j = (i + 1) % n_gpu
            links.append(TopologyEdge(i, j, lc))
            links.append(TopologyEdge(j, i, lc))
        return cls(n_gpu=n_gpu, links=links, name=f"pcie_ring_{n_gpu}gpu")

    @classmethod
    def infiniband_fat_tree(cls, n_gpu: int) -> "TopologyGraph":
        """
        Fat-tree InfiniBand cluster, approximated as a full mesh for the
        partition function.  Fat-tree gives all-pairs non-blocking bandwidth
        so the full-mesh approximation is accurate for collective bandwidth.
        """
        lc = LINK_PRESETS["infiniband"]
        links: list[TopologyEdge] = [
            TopologyEdge(i, j, lc)
            for i in range(n_gpu)
            for j in range(n_gpu)
            if i != j
        ]
        return cls(n_gpu=n_gpu, links=links, name=f"ib_fat_tree_{n_gpu}gpu")

    @classmethod
    def dgx_h100(cls, n_nodes: int = 1) -> "TopologyGraph":
        """
        Standard DGX H100 topology:
          - 8 GPUs per node, all-to-all NVSwitch fabric intra-node.
          - Inter-node: InfiniBand HDR (one bidirectional link per node pair).
        """
        n_gpu = n_nodes * 8
        nvswitch = LINK_PRESETS["nvswitch"]
        ib = LINK_PRESETS["infiniband"]
        links: list[TopologyEdge] = []

        for node in range(n_nodes):
            base = node * 8
            # Intra-node all-to-all via NVSwitch
            for i in range(8):
                for j in range(8):
                    if i != j:
                        links.append(TopologyEdge(base + i, base + j, nvswitch))
            # Inter-node: bidirectional IB link per node pair
            for other_node in range(node + 1, n_nodes):
                other_base = other_node * 8
                links.append(TopologyEdge(base, other_base, ib))
                links.append(TopologyEdge(other_base, base, ib))

        return cls(n_gpu=n_gpu, links=links, name=f"dgx_h100_{n_nodes}node")


# ---------------------------------------------------------------------------
# Multi-GPU thermodynamic state
# ---------------------------------------------------------------------------

@dataclass
class MultiGPUThermodynamicState:
    """
    Thermodynamic quantities for the coupled N-GPU system at a given β.

    Intensive quantities (mean_waste, entropy, specific_heat, free_energy)
    are normalised by the total number of degrees of freedom:

        n_dof_total = n_gpu × n_sm × warps_per_sm

    so they are directly comparable to the single-GPU ThermodynamicState.
    """
    beta: float
    n_gpu: int

    log_Z_multi: float       # ln Z_multi = N × ln Z_single + ln Z_comm_topology
    log_Z_local: float       # N × ln Z_single (per-GPU local contribution)
    log_Z_comm_topo: float   # ln Z_comm_topology (inter-GPU communication)

    mean_waste: float        # <E>_multi / n_dof_total  ∈ [0, 1]
    free_energy: float       # F_multi / (β × n_dof_total)
    entropy: float           # S_multi per DOF (nats per warp)
    specific_heat: float     # Cv_multi per DOF


# ---------------------------------------------------------------------------
# Partition function for the coupled system
# ---------------------------------------------------------------------------

def _log_z_comm_topology(
    beta: float,
    topology: TopologyGraph,
) -> float:
    """
    ln Z_comm_topology(β) = Σ_{(i,j)∈E} ln Z_link(β, J_ij)

    Each link contributes independently under the mean-field factorisation:

        Z_link(β, J) = ∫₀¹ exp(−β J u) du = (1 − e^{−βJ}) / (βJ)

    Limiting cases:
      β J → 0 (free link):      Z_link → 1  → ln Z_link → 0
      β J → ∞ (costly link):    Z_link → 1/(βJ) → ln Z_link → −ln(βJ)
    """
    log_z = 0.0
    for edge in topology.links:
        bj = beta * edge.link.coupling_J
        if bj < 1e-10:
            z_link = 1.0
        else:
            z_link = (1.0 - math.exp(-bj)) / bj
        log_z += math.log(max(z_link, 1e-300))
    return log_z


def log_z_multi_gpu(
    beta: float,
    topology: TopologyGraph,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    n_bins: int = 64,
) -> tuple[float, float, float]:
    """
    Compute ln Z_multi for the N-GPU coupled system.

        ln Z_multi = N × ln Z_single + ln Z_comm_topology

    where ln Z_single uses the three-component factorisation (compute ×
    memory) with no inter-GPU communication edges (communication is
    handled at the topology level via Z_comm_topology).

    Returns:
        (log_Z_multi, log_Z_local, log_Z_comm_topo)
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    # Single-GPU log Z (no comm edges)
    log_z_single = log_gpu_partition_function(
        beta, sm_config, memory_levels, comm_edges=[], n_bins=n_bins,
    )
    log_z_local = topology.n_gpu * log_z_single
    log_z_comm_topo = _log_z_comm_topology(beta, topology)
    log_z_multi = log_z_local + log_z_comm_topo

    return log_z_multi, log_z_local, log_z_comm_topo


def multi_gpu_thermodynamic_quantities(
    beta: float,
    topology: TopologyGraph,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    n_bins: int = 64,
    d_beta: float = 1e-4,
) -> MultiGPUThermodynamicState:
    """
    Compute all multi-GPU thermodynamic quantities at inverse temperature β.

    Uses the same 4-point finite-difference stencil as the single-GPU case.
    Intensive quantities are normalised by:

        n_dof_total = n_gpu × n_sm × warps_per_sm

    Args:
        beta:          Inverse resource-pressure (high β = lightly loaded).
        topology:      Multi-GPU topology graph.
        sm_config:     SM configuration (default: H100).
        memory_levels: Memory hierarchy (default: H100).
        n_bins:        Transfer matrix resolution (default: 64).
        d_beta:        Finite-difference step size.

    Returns:
        MultiGPUThermodynamicState with all derived quantities.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    n_dof_total = topology.n_gpu * float(sm_config.n_sm * sm_config.warps_per_sm)

    def ln_z(b: float) -> float:
        lzm, _, _ = log_z_multi_gpu(b, topology, sm_config, memory_levels, n_bins)
        return lzm

    lz    = ln_z(beta)
    lz_p1 = ln_z(beta + d_beta)
    lz_m1 = ln_z(beta - d_beta)
    lz_p2 = ln_z(beta + 2 * d_beta)
    lz_m2 = ln_z(beta - 2 * d_beta)

    # <E> / n_dof = −(1/n_dof) d(ln Z)/dβ  (central difference)
    mean_waste = -(lz_p1 - lz_m1) / (2 * d_beta * n_dof_total)
    mean_waste = max(0.0, min(1.0, mean_waste))

    # Cv / n_dof = β² × (1/n_dof) × d²(ln Z)/dβ²  (4-point stencil)
    d2_ln_z = (-lz_p2 + 16*lz_p1 - 30*lz + 16*lz_m1 - lz_m2) / (12 * d_beta**2)
    specific_heat = beta**2 * d2_ln_z / n_dof_total

    free_energy = -lz / (max(beta, 1e-12) * n_dof_total)
    entropy = beta * (mean_waste - free_energy)

    _, log_z_local, log_z_comm_topo = log_z_multi_gpu(
        beta, topology, sm_config, memory_levels, n_bins,
    )

    return MultiGPUThermodynamicState(
        beta=beta,
        n_gpu=topology.n_gpu,
        log_Z_multi=lz,
        log_Z_local=log_z_local,
        log_Z_comm_topo=log_z_comm_topo,
        mean_waste=mean_waste,
        free_energy=free_energy,
        entropy=entropy,
        specific_heat=specific_heat,
    )


# ---------------------------------------------------------------------------
# Multi-GPU Carnot limit
# ---------------------------------------------------------------------------

@dataclass
class MultiGPUCarnotLimit:
    """
    The Carnot limit η_multi,max for N coupled GPUs on a given topology.

    η_multi,max is the peak of η_multi(β) = 1 − <E>_multi(β) over all β.
    It is always ≤ η_hw,max (single GPU) because the communication
    subsystem introduces additional irreversible waste.

    The resonance η_overlap measures how well compute and communication
    can be overlapped:

        η_overlap = T_overlapped / max(T_compute, T_comm)

    At resonance T_compute = T_comm → η_overlap = 1 (maximum).
    """
    n_gpu: int
    eta_multi_max: float           # ∈ [0, 1]
    eta_hw_max_single: float       # single-GPU reference ∈ [0, 1]
    beta_optimal: float            # β at which η_multi_max is achieved
    resonance_eta: float           # η_overlap ∈ [0, 1]
    comm_overhead_fraction: float  # fraction of total DOFs in comm ∈ [0, 1]
    topology: TopologyGraph
    thermo_state: MultiGPUThermodynamicState

    def scaling_efficiency(self) -> float:
        """
        η_multi,max / η_hw,max,single ∈ [0, 1].

        1.0  → the topology loses nothing vs a single GPU.
        <1.0 → communication overhead degrades multi-GPU efficiency.
        """
        return self.eta_multi_max / max(self.eta_hw_max_single, 1e-12)

    def summary(self) -> str:
        lines = [
            f"Multi-GPU Carnot Limit  [{self.topology.name}]",
            f"  n_gpu            = {self.n_gpu}",
            f"  η_multi,max      = {self.eta_multi_max:.4f}",
            f"  η_hw,max (1-GPU) = {self.eta_hw_max_single:.4f}",
            f"  scaling eff.     = {self.scaling_efficiency():.4f}",
            f"  β_optimal        = {self.beta_optimal:.4f}",
            f"  η_overlap        = {self.resonance_eta:.4f}",
            f"  comm overhead    = {self.comm_overhead_fraction * 100:.1f}%",
            f"  mean J           = {self.topology.mean_J():.3f}",
            f"  total BW (GB/s)  = {self.topology.total_bandwidth_gb_s():.0f}",
            f"  n_links          = {len(self.topology.links)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resonance condition
# ---------------------------------------------------------------------------

def resonance_condition(
    t_compute_s: float,
    t_comm_s: float,
    overlap_fraction: float = 1.0,
) -> float:
    """
    Compute the overlap efficiency η_overlap.

        η_overlap = T_overlapped / max(T_compute, T_comm)

    where T_overlapped = min(T_compute, T_comm) × overlap_fraction.

    overlap_fraction ∈ [0, 1]:
      0.0 → fully sequential (no overlap at all)
      1.0 → perfect pipelining (compute and comm fully overlap)

    At resonance T_compute = T_comm:
        η_overlap = T_compute × overlap_fraction / T_compute = overlap_fraction

    A compute-dominated kernel (T_compute >> T_comm) wastes little time
    on communication but η_overlap → 0 because little compute is overlapped.

    Args:
        t_compute_s:      Compute phase duration in seconds.
        t_comm_s:         Communication phase duration in seconds.
        overlap_fraction: Fraction of the shorter phase hidden behind the longer.

    Returns:
        η_overlap ∈ [0, 1].
    """
    t_max = max(t_compute_s, t_comm_s, 1e-15)
    t_min = min(t_compute_s, t_comm_s)
    t_overlapped = t_min * overlap_fraction
    return min(t_overlapped / t_max, 1.0)


# ---------------------------------------------------------------------------
# Carnot limit derivation
# ---------------------------------------------------------------------------

def derive_multi_gpu_carnot_limit(
    topology: TopologyGraph,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    eta_hw_max_single: float | None = None,
    beta_min: float = 0.01,
    beta_max: float = 10.0,
    n_beta: int = 200,
    n_bins: int = 64,
) -> MultiGPUCarnotLimit:
    """
    Derive η_multi,max for N coupled GPUs from the partition function.

    Strategy:
        η_multi(β) = 1 − <E>_multi(β)
        η_multi,max = max_β η_multi(β)

    The optimal β trades off two effects:
      - High β (lightly loaded): low waste per operation, but idle
        capacity dominates → η rises as β falls.
      - Low β (heavily loaded): rising stalls + communication pressure →
        η peaks then falls.

    Communication overhead fraction (fraction of DOFs in the comm graph):
        f_comm = n_links / (n_gpu × n_sm × warps_per_sm + n_links)

    Resonance η_overlap is estimated as the complement of the relative
    communication log-Z contribution:
        η_overlap ≈ 1 − |ln Z_comm| / |ln Z_multi|

    Args:
        topology:            Multi-GPU topology graph.
        sm_config:           SM configuration (default: H100).
        memory_levels:       Memory hierarchy (default: H100).
        eta_hw_max_single:   Pre-computed single-GPU limit (skips recomputation).
        beta_min / beta_max: β sweep range.
        n_beta:              Number of β points in the sweep.
        n_bins:              Transfer matrix resolution.

    Returns:
        MultiGPUCarnotLimit with all derived quantities.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    # Reference single-GPU η_hw,max
    if eta_hw_max_single is None:
        from .carnot import derive_carnot_limit
        single_limit = derive_carnot_limit(
            sm_config, memory_levels, beta_min, beta_max, 50, n_bins,
        )
        eta_hw_max_single = single_limit.eta_hw_max

    betas = np.linspace(beta_min, beta_max, n_beta).tolist()

    best_eta = -1.0
    best_state: MultiGPUThermodynamicState | None = None
    best_beta = betas[0]

    for b in betas:
        state = multi_gpu_thermodynamic_quantities(
            b, topology, sm_config, memory_levels, n_bins,
        )
        eta = 1.0 - state.mean_waste
        if eta > best_eta:
            best_eta = eta
            best_state = state
            best_beta = b

    best_eta = max(0.0, min(1.0, best_eta))
    assert best_state is not None

    # Communication overhead: edges as a fraction of total DOFs + edges
    n_dof_local = topology.n_gpu * sm_config.n_sm * sm_config.warps_per_sm
    n_edges = len(topology.links)
    comm_overhead = float(n_edges) / max(float(n_dof_local + n_edges), 1.0)

    # Resonance: complement of communication's log-Z share at optimal β
    lz_comm = best_state.log_Z_comm_topo
    lz_total = best_state.log_Z_multi
    if abs(lz_total) > 1e-12:
        comm_log_fraction = abs(lz_comm) / max(abs(lz_total), 1e-12)
        resonance_eta = max(0.0, 1.0 - min(comm_log_fraction, 1.0))
    else:
        resonance_eta = 1.0

    return MultiGPUCarnotLimit(
        n_gpu=topology.n_gpu,
        eta_multi_max=best_eta,
        eta_hw_max_single=eta_hw_max_single,
        beta_optimal=best_beta,
        resonance_eta=resonance_eta,
        comm_overhead_fraction=comm_overhead,
        topology=topology,
        thermo_state=best_state,
    )

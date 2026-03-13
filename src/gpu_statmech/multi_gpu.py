"""
Multi-GPU thermodynamic analysis: coupled-engine efficiency limit.

The coupled system factorises into local GPU thermodynamics plus an
inter-GPU communication partition function:

    ln Z_multi(β, h) = N × ln Z_single(β, h) + ln Z_comm_topology(β)

where ``h`` is the useful-work field already used by the single-GPU model.
Communication links are treated as an additional input-energy subsystem with
topology-dependent effective coupling costs derived from:

  - base disorder coupling ``J`` from the link preset
  - inverse bandwidth pressure
  - latency pressure

From this factorisation we derive:

    <E_in>_multi = <E_in>_local + <E_comm>
    <W_hw>_multi = <W_hw>_local
    η_multi      = <W_hw>_multi / <E_in>_multi

This keeps the multi-GPU path consistent with the single-GPU
``E_in - h W_hw`` formulation instead of the older ``1 - mean_waste`` proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    thermodynamic_quantities,
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
        if n_gpu <= 1:
            return cls(n_gpu=n_gpu, links=[], name=f"pcie_ring_{n_gpu}gpu")
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

_REFERENCE_LINK_BW_GB_S = max(link.bandwidth_gb_s for link in LINK_PRESETS.values())
_REFERENCE_LINK_LATENCY_US = min(link.latency_us for link in LINK_PRESETS.values())


def _effective_link_cost(link: LinkConfig) -> float:
    """
    Fold bandwidth and latency into the base coupling ``J``.

    The geometric mean keeps the penalty monotone in both slower bandwidth
    and higher latency without letting either term dominate on its own.
    """
    bw_penalty = _REFERENCE_LINK_BW_GB_S / max(link.bandwidth_gb_s, 1e-12)
    latency_penalty = link.latency_us / max(_REFERENCE_LINK_LATENCY_US, 1e-12)
    transfer_penalty = math.sqrt(bw_penalty * latency_penalty)
    return link.coupling_J * transfer_penalty


def _topology_activity_normalizer(topology: TopologyGraph) -> float:
    """
    Approximate how many outgoing links per GPU can be active in one comm phase.

    Dense fabrics expose many alternative routes. Charging every directed edge
    as simultaneously active would systematically over-penalise cliques and
    switched fabrics, so we average link contributions over the mean out-degree.
    """
    if topology.n_gpu <= 0:
        return 1.0
    mean_out_degree = len(topology.links) / max(float(topology.n_gpu), 1.0)
    return max(mean_out_degree, 1.0)


def _mean_link_input_energy(beta: float, link: LinkConfig) -> float:
    """
    Expected communication input energy for one directed link.

    With effective link cost ``J_eff`` and utilisation ``u in [0, 1]``:

        Z_link = ∫ exp(-β J_eff u) du
        <E_link> = J_eff <u>
                 = 1/β - J_eff / (exp(β J_eff) - 1)

    The small-β limit tends to ``J_eff / 2``.
    """
    j_eff = _effective_link_cost(link)
    if j_eff <= 1e-12:
        return 0.0

    beta_eff = beta * j_eff
    if beta_eff < 1e-6:
        return 0.5 * j_eff
    return max(1.0 / max(beta, 1e-12) - j_eff / math.expm1(beta_eff), 0.0)


@dataclass
class MultiGPUThermodynamicState:
    """
    Thermodynamic quantities for the coupled N-GPU system at a given β.

    Intensive quantities are normalised by the total number of local compute
    degrees of freedom:

        n_dof_total = n_gpu × n_sm × warps_per_sm

    so they are directly comparable to the single-GPU ThermodynamicState.
    """
    beta: float
    n_gpu: int

    log_Z_multi: float       # ln Z_multi = N × ln Z_single + ln Z_comm_topology
    log_Z_local: float       # N × ln Z_single (per-GPU local contribution)
    log_Z_comm_topo: float   # ln Z_comm_topology (inter-GPU communication)

    work_field: float
    memory_feed_efficiency: float
    target_activity: float | None

    mean_effective_energy: float
    mean_input_energy: float
    mean_useful_work: float
    mean_comm_input_energy: float
    mean_waste: float
    mean_activity: float
    free_energy: float       # F_multi / (β × n_dof_total)
    entropy: float           # S_multi per DOF (nats per warp)
    specific_heat: float     # Cv_multi per DOF

    @property
    def eta_multi(self) -> float:
        return self.mean_useful_work / max(self.mean_input_energy, 1e-12)


# ---------------------------------------------------------------------------
# Partition function for the coupled system
# ---------------------------------------------------------------------------

def _log_z_comm_topology(
    beta: float,
    topology: TopologyGraph,
) -> float:
    """
    ln Z_comm_topology(β) = Σ_{(i,j)∈E} ln Z_link(β, J_eff,ij)

    Each link contributes independently under the mean-field factorisation:

        Z_link(β, J_eff) = ∫₀¹ exp(−β J_eff u) du
                         = (1 − e^{−β J_eff}) / (β J_eff)

    where the effective link cost folds the base coupling, bandwidth pressure,
    and latency pressure into one dimensionless penalty.

    Limiting cases:
      β J_eff → 0 (free link):      Z_link → 1  → ln Z_link → 0
      β J_eff → ∞ (costly link):    Z_link → 1/(βJ_eff) → ln Z_link → −ln(βJ_eff)
    """
    log_z = 0.0
    for edge in topology.links:
        bj = beta * _effective_link_cost(edge.link)
        if bj < 1e-10:
            z_link = 1.0
        else:
            z_link = (1.0 - math.exp(-bj)) / bj
        log_z += math.log(max(z_link, 1e-300))
    return log_z / _topology_activity_normalizer(topology)


def log_z_multi_gpu(
    beta: float,
    topology: TopologyGraph,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    n_bins: int = 64,
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
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
        beta,
        sm_config,
        memory_levels,
        comm_edges=[],
        n_bins=n_bins,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
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
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
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

    local_state = thermodynamic_quantities(
        beta,
        sm_config,
        memory_levels,
        comm_edges=[],
        n_bins=n_bins,
        d_beta=d_beta,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
    )

    def ln_z(b: float) -> float:
        lzm, _, _ = log_z_multi_gpu(
            b,
            topology,
            sm_config,
            memory_levels,
            n_bins,
            work_field=work_field,
            activity_potential=activity_potential,
            target_activity=target_activity,
        )
        return lzm

    lz    = ln_z(beta)
    lz_p1 = ln_z(beta + d_beta)
    lz_m1 = ln_z(beta - d_beta)
    lz_p2 = ln_z(beta + 2 * d_beta)
    lz_m2 = ln_z(beta - 2 * d_beta)

    # <E_eff> / n_dof = −(1/n_dof) d(ln Z)/dβ  (central difference)
    mean_effective_energy = -(lz_p1 - lz_m1) / (2 * d_beta * n_dof_total)

    # Cv / n_dof = β² × (1/n_dof) × d²(ln Z)/dβ²  (4-point stencil)
    d2_ln_z = (-lz_p2 + 16*lz_p1 - 30*lz + 16*lz_m1 - lz_m2) / (12 * d_beta**2)
    specific_heat = beta**2 * d2_ln_z / n_dof_total

    free_energy = -lz / (max(beta, 1e-12) * n_dof_total)
    mean_comm_input_energy = sum(
        _mean_link_input_energy(beta, edge.link) for edge in topology.links
    ) / max(n_dof_total * _topology_activity_normalizer(topology), 1.0)
    mean_input_energy = local_state.mean_input_energy + mean_comm_input_energy
    mean_useful_work = local_state.mean_useful_work
    mean_waste = max(mean_input_energy - mean_useful_work, 0.0)
    entropy = beta * (mean_effective_energy - free_energy)

    _, log_z_local, log_z_comm_topo = log_z_multi_gpu(
        beta,
        topology,
        sm_config,
        memory_levels,
        n_bins,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
    )

    return MultiGPUThermodynamicState(
        beta=beta,
        n_gpu=topology.n_gpu,
        log_Z_multi=lz,
        log_Z_local=log_z_local,
        log_Z_comm_topo=log_z_comm_topo,
        work_field=local_state.work_field,
        memory_feed_efficiency=local_state.memory_feed_efficiency,
        target_activity=local_state.target_activity,
        mean_effective_energy=mean_effective_energy,
        mean_input_energy=mean_input_energy,
        mean_useful_work=mean_useful_work,
        mean_comm_input_energy=mean_comm_input_energy,
        mean_waste=mean_waste,
        mean_activity=local_state.mean_activity,
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

    η_multi,max is the peak of η_multi(β) = <W_hw>_multi / <E_in>_multi
    over all β. It should remain ≤ η_hw,max (single GPU) because the
    communication subsystem adds input energy without adding useful work.

    ``resonance_eta`` remains a coarse proxy here; the physically grounded
    quantity in this dataclass is ``comm_overhead_fraction``, which now means
    the fraction of total input energy spent in communication.
    """
    n_gpu: int
    eta_multi_max: float           # ∈ [0, 1]
    eta_hw_max_single: float       # single-GPU reference ∈ [0, 1]
    beta_optimal: float            # β at which η_multi_max is achieved
    resonance_eta: float           # η_overlap ∈ [0, 1]
    comm_overhead_fraction: float  # fraction of total input energy in comm ∈ [0, 1]
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
            f"  η_overlap proxy  = {self.resonance_eta:.4f}",
            f"  comm energy      = {self.comm_overhead_fraction * 100:.1f}%",
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
    work_field: float | None = None,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> MultiGPUCarnotLimit:
    """
    Derive η_multi,max for N coupled GPUs from the partition function.

    Strategy:
        η_multi(β) = <W_hw>_multi(β) / <E_in>_multi(β)
        η_multi,max = max_β η_multi(β)

    The multi-GPU sweep is evaluated at the same fixed-load closure as the
    single-GPU model when ``target_activity`` is provided.

    Communication overhead fraction is the input-energy share spent on
    communication at the optimal operating point:

        f_comm = <E_comm> / <E_in>_multi

    ``resonance_eta`` remains a coarse balance proxy:

        1 - |E_work - E_comm| / (E_work + E_comm)

    Args:
        topology:            Multi-GPU topology graph.
        sm_config:           SM configuration (default: H100).
        memory_levels:       Memory hierarchy (default: H100).
        eta_hw_max_single:   Pre-computed single-GPU limit (skips recomputation).
        beta_min / beta_max: β sweep range.
        n_beta:              Number of β points in the sweep.
        n_bins:              Transfer matrix resolution.
        work_field:          Fixed useful-work field ``h``.
        activity_potential:  Backwards-compatible alias for ``work_field``.
        target_activity:     Fixed-load closure for solving ``h(β)``.

    Returns:
        MultiGPUCarnotLimit with all derived quantities.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if target_activity is None and work_field is None and activity_potential is None:
        target_activity = 0.20

    # Reference single-GPU η_hw,max
    if eta_hw_max_single is None:
        from .carnot import derive_carnot_limit
        single_limit = derive_carnot_limit(
            sm_config=sm_config,
            memory_levels=memory_levels,
            beta_min=beta_min,
            beta_max=beta_max,
            n_beta=n_beta,
            n_bins=n_bins,
            work_field=work_field,
            activity_potential=activity_potential,
            target_activity=target_activity,
        )
        eta_hw_max_single = single_limit.eta_hw_max

    betas = np.linspace(beta_min, beta_max, n_beta).tolist()

    best_eta = -1.0
    best_state: MultiGPUThermodynamicState | None = None
    best_beta = betas[0]

    for b in betas:
        state = multi_gpu_thermodynamic_quantities(
            b,
            topology,
            sm_config,
            memory_levels,
            n_bins,
            work_field=work_field or 0.0,
            activity_potential=activity_potential,
            target_activity=target_activity,
        )
        eta = max(0.0, min(1.0, state.eta_multi))
        if eta > best_eta:
            best_eta = eta
            best_state = state
            best_beta = b

    assert best_state is not None

    comm_overhead = best_state.mean_comm_input_energy / max(best_state.mean_input_energy, 1e-12)
    work_plus_comm = best_state.mean_useful_work + best_state.mean_comm_input_energy
    if work_plus_comm > 1e-12:
        imbalance = abs(best_state.mean_useful_work - best_state.mean_comm_input_energy) / work_plus_comm
        resonance_eta = max(0.0, 1.0 - min(imbalance, 1.0))
    else:
        resonance_eta = 1.0

    return MultiGPUCarnotLimit(
        n_gpu=topology.n_gpu,
        eta_multi_max=max(0.0, min(best_eta, 1.0)),
        eta_hw_max_single=eta_hw_max_single,
        beta_optimal=best_beta,
        resonance_eta=resonance_eta,
        comm_overhead_fraction=comm_overhead,
        topology=topology,
        thermo_state=best_state,
    )

"""
Partition function approximation for a GPU thermodynamic system.

Z ≈ Z_compute × Z_memory × Z_comm

- Z_compute : mean-field factorisation over SMs
              Z_compute ≈ Z_SM^N_SM, where Z_SM = Z_warp^warps_per_SM
              with a shared-bandwidth Lagrange-multiplier correction.

- Z_memory  : exact transfer-matrix solution over the 4-level hierarchy
              reg → smem → L2 → HBM (1-D chain → analytically tractable).

- Z_comm    : mean-field on the cluster graph with topology-dependent
              coupling constants J_gh.

From Z all standard thermodynamic quantities follow:
  F  = -kT ln Z          (free energy / maximum extractable work)
  <E> = -d(ln Z)/dβ      (expected waste energy)
  S  = -dF/dT            (entropy of execution-state degeneracy)
  Cv = d<E>/dT           (specific heat / sensitivity to resource pressure)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Hardware parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MemoryLevel:
    """One level of the GPU memory hierarchy."""
    name: str
    capacity_bytes: int          # total capacity visible to one SM (or shared)
    bandwidth_bytes_per_cycle: float  # bytes/cycle between this level and next-warmer
    latency_cycles: float        # round-trip access latency in cycles
    energy_per_byte_pj: float    # pico-Joules per byte accessed at this level


# H100 memory hierarchy (per-SM values for reg/smem; global for L2/HBM)
H100_MEMORY_LEVELS: list[MemoryLevel] = [
    MemoryLevel("registers", capacity_bytes=256 * 1024,
                bandwidth_bytes_per_cycle=128 * 4,   # ~128 threads × 4B/thread/cycle
                latency_cycles=1.0,
                energy_per_byte_pj=0.1),
    MemoryLevel("smem",      capacity_bytes=228 * 1024,
                bandwidth_bytes_per_cycle=128 * 4,
                latency_cycles=23.0,
                energy_per_byte_pj=0.5),
    MemoryLevel("L2",        capacity_bytes=50 * 1024 * 1024,
                bandwidth_bytes_per_cycle=6400,      # 6.4 TB/s / ~1.6 GHz ≈ 4000 B/cyc
                latency_cycles=200.0,
                energy_per_byte_pj=2.0),
    MemoryLevel("HBM",       capacity_bytes=80 * 1024 * 1024 * 1024,
                bandwidth_bytes_per_cycle=4000,      # 3.35 TB/s / ~1.6 GHz ≈ 2094 B/cyc
                latency_cycles=600.0,
                energy_per_byte_pj=20.0),
]


@dataclass
class SMConfig:
    """Streaming-multiprocessor configuration."""
    n_sm: int                    # number of SMs
    warps_per_sm: int            # max resident warps per SM
    threads_per_warp: int = 32
    peak_flops_per_cycle: float = 2048.0   # FP16 tensor-core ops/cycle per SM (H100)


H100_SM_CONFIG = SMConfig(n_sm=132, warps_per_sm=64, peak_flops_per_cycle=2048.0)


@dataclass
class LinkConfig:
    """One inter-GPU link for the communication partition function."""
    name: str
    bandwidth_gb_s: float
    latency_us: float
    coupling_J: float            # dimensionless waste-energy coupling constant


LINK_PRESETS: dict[str, LinkConfig] = {
    "nvlink4":    LinkConfig("nvlink4",    bandwidth_gb_s=900,  latency_us=1.0,  coupling_J=0.1),
    "nvswitch":   LinkConfig("nvswitch",   bandwidth_gb_s=900,  latency_us=1.0,  coupling_J=0.1),
    "pcie_gen5":  LinkConfig("pcie_gen5",  bandwidth_gb_s=64,   latency_us=3.5,  coupling_J=1.0),
    "infiniband": LinkConfig("infiniband", bandwidth_gb_s=50,   latency_us=2.0,  coupling_J=5.0),
    "roce":       LinkConfig("roce",       bandwidth_gb_s=50,   latency_us=2.5,  coupling_J=10.0),
}


# ---------------------------------------------------------------------------
# Warp-state energy table
# ---------------------------------------------------------------------------

# Fractional waste energy for each warp state: 0 = fully useful, 1 = fully wasted.
# Based on NVIDIA Nsight stall taxonomy.
WARP_STATE_WASTE: dict[str, float] = {
    "eligible":         0.0,   # executing useful work
    "long_scoreboard":  1.0,   # waiting on long-latency memory (HBM)
    "short_scoreboard": 0.6,   # waiting on L2
    "barrier":          0.9,   # synchronisation stall
    "exec_dep":         0.3,   # short data dependency
    "mem_throttle":     0.8,   # too many outstanding memory requests
    "fetch":            0.2,   # instruction-cache miss / fetch stall
    "idle":             1.0,   # no active warp
}

N_WARP_STATES = len(WARP_STATE_WASTE)
_WARP_WASTE_VALUES = np.array(list(WARP_STATE_WASTE.values()), dtype=np.float64)


# ---------------------------------------------------------------------------
# Z_compute  (mean-field over SMs, warp-level factorisation)
# ---------------------------------------------------------------------------

def z_warp(beta: float) -> float:
    """
    Single-warp partition function.

    Z_warp(β) = Σ_{s ∈ warp_states} exp(-β · waste(s))

    β is the inverse resource-pressure (high β → low temperature → lightly loaded).
    """
    return float(np.sum(np.exp(-beta * _WARP_WASTE_VALUES)))


def z_sm(beta: float, warps_per_sm: int) -> float:
    """
    Single-SM partition function under warp independence assumption.

    Z_SM(β) = Z_warp(β)^warps_per_sm
    """
    return z_warp(beta) ** warps_per_sm


def mean_field_bandwidth_correction(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
) -> float:
    """
    Lagrange-multiplier correction for shared HBM bandwidth constraint.

    SMs are not truly independent — they compete for L2→HBM bandwidth.
    In mean-field theory this introduces a self-consistency equation:

        <BW_used> = BW_capacity

    We solve for the correction factor λ such that the effective single-SM
    partition function becomes Z_SM_eff = Z_SM × exp(-β λ × <BW_per_SM>),
    where λ is chosen so that N_SM × <BW_per_SM>(λ) = BW_capacity.

    Returns the multiplicative correction factor to Z_compute.
    """
    # Expected fraction of warps in memory-stalled states (long + mem_throttle)
    waste_vals = _WARP_WASTE_VALUES
    state_names = list(WARP_STATE_WASTE.keys())
    mem_stall_mask = np.array(
        [1.0 if s in ("long_scoreboard", "mem_throttle") else 0.0 for s in state_names]
    )

    # Mean-field: iterate to self-consistency
    lambda_mf = 0.0
    for _ in range(100):
        weights = np.exp(-beta * (waste_vals + lambda_mf * mem_stall_mask))
        weights /= weights.sum()
        mean_mem_stall_frac = float(np.dot(weights, mem_stall_mask))
        # Each stalled warp demands ~1 HBM transaction per latency window
        demand = sm_config.n_sm * sm_config.warps_per_sm * mean_mem_stall_frac
        # Normalise to bandwidth capacity
        supply = hbm_bandwidth_bytes_per_cycle * sm_config.n_sm  # rough proxy
        excess = max(0.0, demand - supply) / max(supply, 1e-12)
        lambda_new = excess
        if abs(lambda_new - lambda_mf) < 1e-8:
            break
        lambda_mf = 0.5 * lambda_mf + 0.5 * lambda_new   # damped update

    # log(correction) = N_SM × (-β λ <mem_stall>)  — keep in log space to avoid overflow
    weights = np.exp(-beta * (waste_vals + lambda_mf * mem_stall_mask))
    weights /= weights.sum()
    mean_mem_stall_frac = float(np.dot(weights, mem_stall_mask))
    log_correction_per_sm = -beta * lambda_mf * mean_mem_stall_frac
    return log_correction_per_sm * sm_config.n_sm   # log of total correction


def log_z_compute(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
) -> float:
    """
    Log of the full compute partition function — always in log space to avoid overflow.

    ln Z_compute(β) = N_SM × warps_per_SM × ln Z_warp(β)  +  ln(bandwidth_correction)

    Z_SM^N_SM overflows float64 even at moderate N (e.g. 8^(64×132) ≈ 10^7500),
    so we never materialise Z itself.
    """
    log_z = sm_config.n_sm * sm_config.warps_per_sm * math.log(max(z_warp(beta), 1e-300))
    if apply_bandwidth_correction:
        log_z += mean_field_bandwidth_correction(beta, sm_config, hbm_bandwidth_bytes_per_cycle)
    return log_z


def z_compute(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
) -> float:
    """
    Compute partition function (raw value).  Will overflow for large N_SM / low β —
    use log_z_compute() for all thermodynamic calculations.
    """
    return math.exp(log_z_compute(beta, sm_config, hbm_bandwidth_bytes_per_cycle,
                                   apply_bandwidth_correction))


# ---------------------------------------------------------------------------
# Z_memory  (exact transfer-matrix over the 4-level 1-D chain)
# ---------------------------------------------------------------------------

def _transfer_matrix(
    level_from: MemoryLevel,
    level_to: MemoryLevel,
    beta: float,
    n_bins: int = 64,
) -> NDArray[np.float64]:
    """
    Build the transfer matrix T[i, j] between adjacent memory levels.

    State = discrete occupancy fraction u ∈ [0, 1] (binned into n_bins).

    T[i, j] = exp(-β · cost(u_i, u_j, level_from, level_to))

    cost encodes:
      - data-movement waste: bytes moved × energy_per_byte of the warmer level
      - latency waste: fraction of cycles spent waiting (proportional to occupancy
        of the warmer level times the latency ratio)
    """
    u = np.linspace(0.0, 1.0, n_bins)
    ui, uj = np.meshgrid(u, u, indexing="ij")  # shape (n_bins, n_bins)

    # Bytes moved between levels is proportional to the *change* in occupancy
    # weighted by the capacity of the warmer level.
    bytes_moved = np.abs(uj - ui) * level_to.capacity_bytes

    # Energy cost of moving those bytes (paid at the warmer level's rate)
    energy_cost = bytes_moved * level_to.energy_per_byte_pj * 1e-12  # → Joules (tiny)

    # Normalise to a dimensionless "waste fraction" ∈ [0, 1]
    # by dividing by the cost of saturating the link (max possible movement)
    max_energy = level_to.capacity_bytes * level_to.energy_per_byte_pj * 1e-12
    waste = energy_cost / max(max_energy, 1e-30)

    # Latency penalty: occupancy at the warmer level drives stall fraction
    latency_ratio = level_to.latency_cycles / max(level_from.latency_cycles, 1.0)
    latency_waste = uj * min(latency_ratio / 600.0, 1.0)  # normalise to HBM latency

    total_waste = waste + latency_waste
    T = np.exp(-beta * total_waste)
    return T.astype(np.float64)


def z_memory(
    beta: float,
    memory_levels: list[MemoryLevel] | None = None,
    n_bins: int = 64,
) -> float:
    """
    Exact partition function for the memory hierarchy via transfer matrices.

    For a 4-level chain (reg → smem → L2 → HBM):

        Z_memory = 1^T · T_{reg→smem} · T_{smem→L2} · T_{L2→HBM} · 1

    where 1 is the all-ones vector (sum over all boundary occupancy states).

    This is exact for the 1-D chain topology — no approximation.
    """
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    # Initial distribution: uniform over occupancy states of the coldest level
    vec = np.ones(n_bins, dtype=np.float64)

    # Propagate through each adjacent pair in the chain (cold → hot)
    for i in range(len(memory_levels) - 1):
        T = _transfer_matrix(memory_levels[i], memory_levels[i + 1], beta, n_bins)
        vec = T.T @ vec   # sum over the "from" states

    return float(vec.sum())


# ---------------------------------------------------------------------------
# Z_comm  (mean-field on the cluster graph)
# ---------------------------------------------------------------------------

@dataclass
class TopologyEdge:
    """One directed edge in the inter-GPU communication graph."""
    src: int
    dst: int
    link: LinkConfig


def z_comm(
    beta: float,
    edges: list[TopologyEdge],
    n_gpus: int,
) -> float:
    """
    Communication partition function via mean-field on the cluster graph.

    Each link (g, h) contributes an independent factor:

        Z_link = Σ_{utilisation u ∈ [0,1]} exp(-β · J_gh · u)
               ≈ ∫₀¹ exp(-β J_gh u) du
               = (1 - exp(-β J_gh)) / (β J_gh)

    The full Z_comm is the product over all edges.

    For β J_gh → 0 (free communication): Z_link → 1.
    For β J_gh → ∞ (expensive communication): Z_link → 1/(β J_gh).
    """
    if not edges:
        return 1.0

    z = 1.0
    for edge in edges:
        bj = beta * edge.link.coupling_J
        if bj < 1e-10:
            z_link = 1.0
        else:
            z_link = (1.0 - math.exp(-bj)) / bj
        z *= z_link
    return z


def dgx_h100_edges(n_nodes: int = 1) -> list[TopologyEdge]:
    """
    Return topology edges for an n_nodes × 8-GPU DGX H100 cluster.
    Intra-node: all-to-all NVSwitch (28 pairs per node).
    Inter-node: each node connects to every other via InfiniBand.
    """
    edges: list[TopologyEdge] = []
    nvswitch = LINK_PRESETS["nvswitch"]
    ib = LINK_PRESETS["infiniband"]

    for node in range(n_nodes):
        base = node * 8
        # Intra-node all-to-all
        for i in range(8):
            for j in range(8):
                if i != j:
                    edges.append(TopologyEdge(base + i, base + j, nvswitch))
        # Inter-node: one representative link per node pair
        for other_node in range(n_nodes):
            if other_node != node:
                edges.append(TopologyEdge(base, other_node * 8, ib))

    return edges


# ---------------------------------------------------------------------------
# Combined partition function and derived thermodynamic quantities
# ---------------------------------------------------------------------------

@dataclass
class ThermodynamicState:
    """All thermodynamic quantities derived from Z at a given β."""
    beta: float
    log_Z: float

    # Primary quantities (natural units: waste energy normalised to [0,1])
    free_energy: float           # F = -kT ln Z  (in units of kT)
    mean_waste: float            # <E> = -d(ln Z)/dβ
    entropy: float               # S = β(<E> - F)  [dimensionless, in units of k]
    specific_heat: float         # Cv = β² d<E>/dβ  [dimensionless]

    # Decomposed contributions
    log_Z_compute: float
    log_Z_memory: float
    log_Z_comm: float


def log_gpu_partition_function(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
) -> float:
    """
    ln Z(β) = ln Z_compute(β) + ln Z_memory(β) + ln Z_comm(β)

    Always returns the log — Z itself overflows float64 for realistic GPU configs.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if comm_edges is None:
        comm_edges = []

    hbm_bw = memory_levels[-1].bandwidth_bytes_per_cycle
    log_zc = log_z_compute(beta, sm_config, hbm_bw)
    log_zm = math.log(max(z_memory(beta, memory_levels, n_bins), 1e-300))
    log_zk = math.log(max(z_comm(beta, comm_edges, sm_config.n_sm), 1e-300))
    return log_zc + log_zm + log_zk


def gpu_partition_function(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
) -> float:
    """
    Z(β) = exp(ln Z_compute + ln Z_memory + ln Z_comm).

    Will overflow for realistic GPU configs — use log_gpu_partition_function()
    for thermodynamic calculations.  Retained for small/toy hardware configs.
    """
    return math.exp(log_gpu_partition_function(
        beta, sm_config, memory_levels, comm_edges, n_bins))


def thermodynamic_quantities(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
    d_beta: float = 1e-4,
) -> ThermodynamicState:
    """
    Compute all thermodynamic quantities at inverse temperature β by finite
    differences of ln Z.

        F   = -ln Z / β          (free energy in units of kT)
        <E> = -d(ln Z)/dβ        (mean waste, numerical derivative)
        S   = β(<E> - F)         (entropy in units of k_B)
        Cv  = β² × d<E>/dβ       (specific heat, dimensionless)
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if comm_edges is None:
        comm_edges = []

    hbm_bw = memory_levels[-1].bandwidth_bytes_per_cycle

    def ln_z(b: float) -> float:
        return log_gpu_partition_function(b, sm_config, memory_levels, comm_edges, n_bins)

    def ln_z_component(b: float) -> tuple[float, float, float]:
        log_zc = log_z_compute(b, sm_config, hbm_bw)
        log_zm = math.log(max(z_memory(b, memory_levels, n_bins), 1e-300))
        log_zk = math.log(max(z_comm(b, comm_edges, sm_config.n_sm), 1e-300))
        return log_zc, log_zm, log_zk

    lz = ln_z(beta)
    lz_plus  = ln_z(beta + d_beta)
    lz_minus = ln_z(beta - d_beta)
    lz_plus2 = ln_z(beta + 2 * d_beta)
    lz_minus2 = ln_z(beta - 2 * d_beta)

    # Number of independent degrees of freedom (warps across all SMs).
    # ln Z is extensive: ln Z ≈ n_dof × ln Z_warp + ...
    # Dividing by n_dof gives intensive (per-DOF) quantities in [0, 1].
    n_dof = float(sm_config.n_sm * sm_config.warps_per_sm)

    # <E>/n_dof = -(1/n_dof) d(ln Z)/dβ  (central difference) → ∈ [0, 1]
    mean_waste = -(lz_plus - lz_minus) / (2 * d_beta * n_dof)

    # Cv/n_dof = β² × (1/n_dof) × d²(ln Z)/dβ² = β² × var(E_per_dof) ≥ 0
    # Sign: d²(ln Z)/dβ² = var(E) ≥ 0, so Cv = +β² × d²(ln Z)/dβ²
    d2_ln_z = (-lz_plus2 + 16*lz_plus - 30*lz + 16*lz_minus - lz_minus2) / (12 * d_beta**2)
    specific_heat = beta**2 * d2_ln_z / n_dof

    free_energy = -lz / (max(beta, 1e-12) * n_dof)
    entropy = beta * (mean_waste - free_energy)   # per-DOF entropy (nats per warp)

    lzc, lzm, lzk = ln_z_component(beta)

    return ThermodynamicState(
        beta=beta,
        log_Z=lz,
        free_energy=free_energy,
        mean_waste=mean_waste,
        entropy=entropy,
        specific_heat=specific_heat,
        log_Z_compute=lzc,
        log_Z_memory=lzm,
        log_Z_comm=lzk,
    )


def beta_sweep(
    betas: Sequence[float],
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
) -> list[ThermodynamicState]:
    """Compute thermodynamic quantities over a range of β values."""
    return [
        thermodynamic_quantities(b, sm_config, memory_levels, comm_edges, n_bins)
        for b in betas
    ]

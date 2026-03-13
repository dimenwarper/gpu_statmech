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
  <E_eff> = -d(ln Z)/dβ  (expected effective energy E_in - h W_hw)
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

# Productive issue fraction for each scheduler-visible warp state.
# This gates both the dynamic energy spent on issued work and the useful
# hardware work extracted from the cycle.
WARP_STATE_ACTIVITY: dict[str, float] = {
    "eligible":         1.00,
    "long_scoreboard":  0.00,
    "short_scoreboard": 0.00,
    "barrier":          0.00,
    "exec_dep":         0.35,
    "mem_throttle":     0.00,
    "fetch":            0.10,
    "idle":             0.00,
}

# Baseline input energy burned per warp-cycle in each scheduler state.  This
# captures leakage plus state-specific overheads from front-end, scoreboard,
# synchronisation, and issue logic.  Values are dimensionless and normalised so
# one warp-cycle stays O(1).
WARP_STATE_BASE_INPUT_ENERGY: dict[str, float] = {
    "eligible":         0.08,
    "long_scoreboard":  0.05,
    "short_scoreboard": 0.06,
    "barrier":          0.04,
    "exec_dep":         0.07,
    "mem_throttle":     0.06,
    "fetch":            0.05,
    "idle":             0.02,
}

# Dynamic input energy spent when a productive instruction issues in a warp
# cycle.  Tensor-core issue slots define the upper end of the scale.
INSTRUCTION_CLASS_INPUT_ENERGY: dict[str, float] = {
    "tensor_core": 0.24,
    "fp16":        0.16,
    "fp32":        0.20,
    "int":         0.12,
    "sfu":         0.18,
    "mem":         0.14,
}

# Useful hardware work extracted from a productive issue of each instruction
# class, in the same dimensionless energy units as the input-energy model.
INSTRUCTION_CLASS_USEFUL_WORK: dict[str, float] = {
    "tensor_core": 0.22,
    "fp16":        0.12,
    "fp32":        0.09,
    "int":         0.06,
    "sfu":         0.07,
    "mem":         0.02,
}

# Coarse prior over instruction classes within an issued warp cycle.  Using a
# prior (rather than exploding the raw state count) preserves Z_warp(β=0)=N_states.
INSTRUCTION_CLASS_PRIOR: dict[str, float] = {
    "tensor_core": 0.12,
    "fp16":        0.18,
    "fp32":        0.30,
    "int":         0.20,
    "sfu":         0.08,
    "mem":         0.12,
}

N_WARP_STATES = len(WARP_STATE_WASTE)
_STATE_NAMES = tuple(WARP_STATE_WASTE.keys())
_WARP_WASTE_VALUES = np.array([WARP_STATE_WASTE[s] for s in _STATE_NAMES], dtype=np.float64)
_WARP_ACTIVITY_VALUES = np.array([WARP_STATE_ACTIVITY[s] for s in _STATE_NAMES], dtype=np.float64)
_WARP_BASE_INPUT_VALUES = np.array(
    [WARP_STATE_BASE_INPUT_ENERGY[s] for s in _STATE_NAMES], dtype=np.float64
)
_INSTR_NAMES = tuple(INSTRUCTION_CLASS_INPUT_ENERGY.keys())
_INSTR_INPUT_VALUES = np.array(
    [INSTRUCTION_CLASS_INPUT_ENERGY[i] for i in _INSTR_NAMES], dtype=np.float64
)
_INSTR_WORK_VALUES = np.array(
    [INSTRUCTION_CLASS_USEFUL_WORK[i] for i in _INSTR_NAMES], dtype=np.float64
)
_INSTR_PRIOR_VALUES = np.array([INSTRUCTION_CLASS_PRIOR[i] for i in _INSTR_NAMES], dtype=np.float64)
_INSTR_PRIOR_VALUES /= max(float(_INSTR_PRIOR_VALUES.sum()), 1e-12)
_MEM_STALL_MASK = np.array(
    [1.0 if s in ("long_scoreboard", "mem_throttle") else 0.0 for s in _STATE_NAMES],
    dtype=np.float64,
)
_INPUT_ENERGY_GRID = (
    _WARP_BASE_INPUT_VALUES[:, None]
    + _WARP_ACTIVITY_VALUES[:, None] * _INSTR_INPUT_VALUES[None, :]
)
_USEFUL_WORK_GRID = _WARP_ACTIVITY_VALUES[:, None] * _INSTR_WORK_VALUES[None, :]


def _resolve_work_field(
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Backwards-compatible alias: ``activity_potential`` now maps to ``work_field``.
    """
    if activity_potential is not None:
        return activity_potential
    return work_field


def _warp_microstate_weights(
    beta: float,
    work_field: float = 0.0,
    bandwidth_penalty: float = 0.0,
) -> NDArray[np.float64]:
    """
    Unnormalised Boltzmann weights over (warp_state, instruction_class).

    The effective energy is:

        E_eff = E_in(state, instr) + bandwidth_penalty * mem_stall(state)
                - h * W_hw(state, instr)

    where h is the useful-work field conjugate to productive hardware work.
    """
    eff_energy = (
        _INPUT_ENERGY_GRID
        + bandwidth_penalty * _MEM_STALL_MASK[:, None]
        - work_field * _USEFUL_WORK_GRID
    )
    return _INSTR_PRIOR_VALUES[None, :] * np.exp(-beta * eff_energy)


def _mean_warp_activity_from_weights(weights: NDArray[np.float64]) -> float:
    total = max(float(weights.sum()), 1e-300)
    return float(np.sum(weights * _WARP_ACTIVITY_VALUES[:, None]) / total)


def _mean_input_energy_from_weights(
    weights: NDArray[np.float64],
    bandwidth_penalty: float = 0.0,
) -> float:
    total = max(float(weights.sum()), 1e-300)
    input_grid = _INPUT_ENERGY_GRID + bandwidth_penalty * _MEM_STALL_MASK[:, None]
    return float(np.sum(weights * input_grid) / total)


def _mean_useful_work_from_weights(weights: NDArray[np.float64]) -> float:
    total = max(float(weights.sum()), 1e-300)
    return float(np.sum(weights * _USEFUL_WORK_GRID) / total)


def _mean_mem_stall_from_weights(weights: NDArray[np.float64]) -> float:
    state_weights = weights.sum(axis=1)
    total = max(float(state_weights.sum()), 1e-300)
    return float(np.dot(state_weights, _MEM_STALL_MASK) / total)


def _solve_bandwidth_penalty(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    work_field: float = 0.0,
) -> tuple[float, NDArray[np.float64]]:
    """
    Solve the mean-field bandwidth penalty λ and return (λ, equilibrium weights).
    """
    lambda_mf = 0.0
    for _ in range(100):
        weights = _warp_microstate_weights(
            beta,
            work_field=work_field,
            bandwidth_penalty=lambda_mf,
        )
        mean_mem_stall_frac = _mean_mem_stall_from_weights(weights)
        # Each stalled warp demands ~1 HBM transaction per latency window.
        demand = sm_config.n_sm * sm_config.warps_per_sm * mean_mem_stall_frac
        # Normalise to bandwidth capacity.
        supply = hbm_bandwidth_bytes_per_cycle * sm_config.n_sm  # rough proxy
        excess = max(0.0, demand - supply) / max(supply, 1e-12)
        lambda_new = excess
        if abs(lambda_new - lambda_mf) < 1e-8:
            lambda_mf = lambda_new
            break
        lambda_mf = 0.5 * lambda_mf + 0.5 * lambda_new

    weights = _warp_microstate_weights(
        beta,
        work_field=work_field,
        bandwidth_penalty=lambda_mf,
    )
    return lambda_mf, weights


# ---------------------------------------------------------------------------
# Z_compute  (mean-field over SMs, warp-level factorisation)
# ---------------------------------------------------------------------------

def z_warp(
    beta: float,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Single-warp partition function.

    Z_warp(β, h) = Σ_{s, i} p(i) exp[-β (E_in(s, i) - h · W_hw(s, i))]

    β is the inverse resource-pressure (high β → low temperature → lightly loaded).
    h is the useful-work field rewarding productive hardware work.
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    return float(np.sum(_warp_microstate_weights(beta, work_field=resolved_work_field)))


def mean_warp_activity(
    beta: float,
    work_field: float = 0.0,
    bandwidth_penalty: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Mean productive activity per warp-cycle, normalised to the tensor-core scale.
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    weights = _warp_microstate_weights(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )
    return _mean_warp_activity_from_weights(weights)


def mean_warp_input_energy(
    beta: float,
    work_field: float = 0.0,
    bandwidth_penalty: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """Mean input energy per warp-cycle for the compute subsystem."""
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    weights = _warp_microstate_weights(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )
    return _mean_input_energy_from_weights(weights, bandwidth_penalty=bandwidth_penalty)


def mean_warp_useful_work(
    beta: float,
    work_field: float = 0.0,
    bandwidth_penalty: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """Mean useful hardware work per warp-cycle for the compute subsystem."""
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    weights = _warp_microstate_weights(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )
    return _mean_useful_work_from_weights(weights)


def z_sm(
    beta: float,
    warps_per_sm: int,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Single-SM partition function under warp independence assumption.

    Z_SM(β) = Z_warp(β)^warps_per_sm
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    return z_warp(beta, work_field=resolved_work_field) ** warps_per_sm


def mean_field_bandwidth_correction(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    work_field: float = 0.0,
    activity_potential: float | None = None,
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
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    lambda_mf, weights = _solve_bandwidth_penalty(
        beta,
        sm_config,
        hbm_bandwidth_bytes_per_cycle,
        work_field=resolved_work_field,
    )
    mean_mem_stall_frac = _mean_mem_stall_from_weights(weights)
    log_correction_per_sm = -beta * lambda_mf * mean_mem_stall_frac
    return log_correction_per_sm * sm_config.n_sm   # log of total correction


def log_z_compute(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Log of the full compute partition function — always in log space to avoid overflow.

    ln Z_compute(β) = N_SM × warps_per_SM × ln Z_warp(β)  +  ln(bandwidth_correction)

    Z_SM^N_SM overflows float64 even at moderate N (e.g. 8^(64×132) ≈ 10^7500),
    so we never materialise Z itself.
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    log_z = sm_config.n_sm * sm_config.warps_per_sm * math.log(
        max(z_warp(beta, work_field=resolved_work_field), 1e-300)
    )
    if apply_bandwidth_correction:
        log_z += mean_field_bandwidth_correction(
            beta,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
            work_field=resolved_work_field,
        )
    return log_z


def z_compute(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Compute partition function (raw value).  Will overflow for large N_SM / low β —
    use log_z_compute() for all thermodynamic calculations.
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    return math.exp(log_z_compute(
        beta,
        sm_config,
        hbm_bandwidth_bytes_per_cycle,
        apply_bandwidth_correction,
        work_field=resolved_work_field,
    ))


def mean_compute_activity(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """
    Mean productive compute activity per warp under the compute partition function.
    """
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    bandwidth_penalty = 0.0
    if apply_bandwidth_correction:
        bandwidth_penalty, _ = _solve_bandwidth_penalty(
            beta,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
            work_field=resolved_work_field,
        )
    return mean_warp_activity(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )


def mean_compute_input_energy(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """Mean compute-side input energy per warp, including bandwidth pressure."""
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    bandwidth_penalty = 0.0
    if apply_bandwidth_correction:
        bandwidth_penalty, _ = _solve_bandwidth_penalty(
            beta,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
            work_field=resolved_work_field,
        )
    return mean_warp_input_energy(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )


def mean_compute_useful_work(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    work_field: float = 0.0,
    activity_potential: float | None = None,
) -> float:
    """Mean compute-side useful hardware work per warp."""
    resolved_work_field = _resolve_work_field(work_field, activity_potential)
    bandwidth_penalty = 0.0
    if apply_bandwidth_correction:
        bandwidth_penalty, _ = _solve_bandwidth_penalty(
            beta,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
            work_field=resolved_work_field,
        )
    return mean_warp_useful_work(
        beta,
        work_field=resolved_work_field,
        bandwidth_penalty=bandwidth_penalty,
    )


def solve_work_field(
    beta: float,
    target_activity: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    apply_bandwidth_correction: bool = True,
    tol: float = 1e-6,
    max_iter: int = 64,
) -> float:
    """
    Solve for the useful-work field h that yields a target compute activity.

    This closes the ensemble with the constraint:

        <A>(beta, h) = target_activity

    using a monotone bisection solve over h.
    """
    if not 0.0 <= target_activity <= 1.0:
        raise ValueError("target_activity must be in [0, 1]")

    def activity_at(work_field: float) -> float:
        return mean_compute_activity(
            beta,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
            apply_bandwidth_correction=apply_bandwidth_correction,
            work_field=work_field,
        )

    lo = -4.0
    hi = 4.0
    act_lo = activity_at(lo)
    act_hi = activity_at(hi)

    for _ in range(24):
        if act_lo <= target_activity <= act_hi:
            break
        if target_activity < act_lo:
            hi = lo
            act_hi = act_lo
            lo *= 2.0
            act_lo = activity_at(lo)
        else:
            lo = hi
            act_lo = act_hi
            hi *= 2.0
            act_hi = activity_at(hi)
    else:
        raise ValueError(
            "target_activity is outside the attainable range "
            f"[{act_lo:.6f}, {act_hi:.6f}] at beta={beta:.6f}"
        )

    if abs(act_lo - target_activity) <= tol:
        return lo
    if abs(act_hi - target_activity) <= tol:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        act_mid = activity_at(mid)
        if abs(act_mid - target_activity) <= tol:
            return mid
        if act_mid < target_activity:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _resolve_operating_work_field(
    beta: float,
    sm_config: SMConfig,
    hbm_bandwidth_bytes_per_cycle: float,
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> float:
    """
    Resolve the operating work field for a state.

    ``target_activity`` takes precedence and closes the ensemble by solving for
    the field that yields the requested mean activity at the current β.
    """
    if target_activity is not None:
        return solve_work_field(
            beta,
            target_activity,
            sm_config,
            hbm_bandwidth_bytes_per_cycle,
        )
    return _resolve_work_field(work_field, activity_potential)


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
    work_field: float

    # Primary quantities (all intensive, normalised per warp DOF)
    free_energy: float           # F = -kT ln Z  (in units of kT)
    mean_effective_energy: float # <E_in - h W_hw> from -d(ln Z)/dβ
    mean_input_energy: float     # <E_in>
    mean_useful_work: float      # <W_hw>
    mean_waste: float            # <E_in - W_hw>
    mean_activity: float         # productive issue activity per warp-cycle
    target_activity: float | None
    entropy: float               # S = β(<E_eff> - F)  [dimensionless]
    specific_heat: float         # Cv = β² d<E_eff>/dβ  [dimensionless]

    # Decomposed contributions
    log_Z_compute: float
    log_Z_memory: float
    log_Z_comm: float

    @property
    def eta_hw(self) -> float:
        """Hardware efficiency η_hw = <W_hw> / <E_in>."""
        return self.mean_useful_work / max(self.mean_input_energy, 1e-12)


def log_gpu_partition_function(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
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
    resolved_work_field = _resolve_operating_work_field(
        beta,
        sm_config,
        hbm_bw,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
    )
    log_zc = log_z_compute(
        beta,
        sm_config,
        hbm_bw,
        work_field=resolved_work_field,
    )
    log_zm = math.log(max(z_memory(beta, memory_levels, n_bins), 1e-300))
    log_zk = math.log(max(z_comm(beta, comm_edges, sm_config.n_sm), 1e-300))
    return log_zc + log_zm + log_zk


def gpu_partition_function(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> float:
    """
    Z(β) = exp(ln Z_compute + ln Z_memory + ln Z_comm).

    Will overflow for realistic GPU configs — use log_gpu_partition_function()
    for thermodynamic calculations.  Retained for small/toy hardware configs.
    """
    return math.exp(log_gpu_partition_function(
        beta,
        sm_config,
        memory_levels,
        comm_edges,
        n_bins,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
    ))


def thermodynamic_quantities(
    beta: float,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    comm_edges: list[TopologyEdge] | None = None,
    n_bins: int = 64,
    d_beta: float = 1e-4,
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> ThermodynamicState:
    """
    Compute all thermodynamic quantities at inverse temperature β by finite
    differences of ln Z.

        F   = -ln Z / β          (free energy in units of kT)
        <E_eff> = -d(ln Z)/dβ     (mean effective energy, numerical derivative)
        S       = β(<E_eff> - F)  (entropy in units of k_B)
        Cv      = β² × d<E_eff>/dβ (specific heat, dimensionless)
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if comm_edges is None:
        comm_edges = []

    hbm_bw = memory_levels[-1].bandwidth_bytes_per_cycle
    resolved_work_field = _resolve_operating_work_field(
        beta,
        sm_config,
        hbm_bw,
        work_field=work_field,
        activity_potential=activity_potential,
        target_activity=target_activity,
    )

    def ln_z(b: float) -> float:
        return log_gpu_partition_function(
            b,
            sm_config,
            memory_levels,
            comm_edges,
            n_bins,
            work_field=resolved_work_field,
        )

    def ln_z_component(b: float) -> tuple[float, float, float]:
        log_zc = log_z_compute(
            b,
            sm_config,
            hbm_bw,
            work_field=resolved_work_field,
        )
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

    # <E_eff>/n_dof = -(1/n_dof) d(ln Z)/dβ
    mean_effective_energy = -(lz_plus - lz_minus) / (2 * d_beta * n_dof)

    # Cv/n_dof = β² × (1/n_dof) × d²(ln Z)/dβ² = β² × var(E_per_dof) ≥ 0
    # Sign: d²(ln Z)/dβ² = var(E) ≥ 0, so Cv = +β² × d²(ln Z)/dβ²
    d2_ln_z = (-lz_plus2 + 16*lz_plus - 30*lz + 16*lz_minus - lz_minus2) / (12 * d_beta**2)
    specific_heat = beta**2 * d2_ln_z / n_dof

    free_energy = -lz / (max(beta, 1e-12) * n_dof)
    lzc, lzm, lzk = ln_z_component(beta)
    lzc_plus, lzm_plus, lzk_plus = ln_z_component(beta + d_beta)
    lzc_minus, lzm_minus, lzk_minus = ln_z_component(beta - d_beta)

    mean_memory_input_energy = -(lzm_plus - lzm_minus) / (2 * d_beta * n_dof)
    mean_comm_input_energy = -(lzk_plus - lzk_minus) / (2 * d_beta * n_dof)
    mean_compute_input = mean_compute_input_energy(
        beta,
        sm_config,
        hbm_bw,
        work_field=resolved_work_field,
    )
    mean_compute_work = mean_compute_useful_work(
        beta,
        sm_config,
        hbm_bw,
        work_field=resolved_work_field,
    )
    mean_input_energy = mean_compute_input + mean_memory_input_energy + mean_comm_input_energy
    mean_useful_work = mean_compute_work
    mean_waste = max(mean_input_energy - mean_useful_work, 0.0)
    mean_activity = mean_compute_activity(
        beta,
        sm_config,
        hbm_bw,
        work_field=resolved_work_field,
    )
    entropy = beta * (mean_effective_energy - free_energy)

    return ThermodynamicState(
        beta=beta,
        log_Z=lz,
        work_field=resolved_work_field,
        free_energy=free_energy,
        mean_effective_energy=mean_effective_energy,
        mean_input_energy=mean_input_energy,
        mean_useful_work=mean_useful_work,
        mean_waste=mean_waste,
        mean_activity=mean_activity,
        target_activity=target_activity,
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
    work_field: float = 0.0,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> list[ThermodynamicState]:
    """Compute thermodynamic quantities over a range of β values."""
    return [
        thermodynamic_quantities(
            b,
            sm_config,
            memory_levels,
            comm_edges,
            n_bins,
            work_field=work_field,
            activity_potential=activity_potential,
            target_activity=target_activity,
        )
        for b in betas
    ]

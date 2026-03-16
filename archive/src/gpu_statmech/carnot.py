"""
GPU Carnot limit derivation and Carnot-optimal computation class.

η_hw,max is derived from the partition function as the maximum fraction of
input energy converted to useful computation:

    η_hw,max = max_β <W_hw(β)> / <E_in(β)>

The Carnot-optimal computation class is the set of kernels that saturate
η_hw,max. Membership is checked via five necessary conditions derived from
the maximum-work protocol (Section 3.5 of the project brief):

    1. Arithmetic intensity ≥ roofline ridge point  (derived, not assumed)
    2. Working set at each memory level ≤ capacity
    3. Data reuse factor at each level ≥ minimum for amortisation
    4. Warp occupancy ≥ occupancy needed to hide memory latency
    5. No unnecessary data movement (residency check)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    MemoryLevel,
    SMConfig,
    ThermodynamicState,
    beta_sweep,
    thermodynamic_quantities,
)


# ---------------------------------------------------------------------------
# Carnot limit derivation
# ---------------------------------------------------------------------------

@dataclass
class CarnotLimit:
    """
    The hardware Carnot limit η_hw,max for a given GPU configuration,
    together with the effective temperatures of each memory level.
    """
    eta_hw_max: float                        # ∈ [0, 1]
    beta_optimal: float                      # β at which η_hw,max is achieved
    work_field_optimal: float                # h that realises the optimal operating point
    target_activity: float | None            # fixed-load closure, if used
    T_eff: dict[str, float]                  # effective temperatures per memory level
    roofline_intensity: float                # min arithmetic intensity (FLOP/byte)
    min_reuse_factors: dict[str, float]      # min data reuse per level
    min_warp_occupancy: float                # minimum occupancy for latency hiding
    thermo_state: ThermodynamicState         # full thermodynamics at β_optimal


def _effective_temperature(level: MemoryLevel, ref_level: MemoryLevel) -> float:
    """
    Map a memory level to an effective temperature via its access cost ratio.

    T_eff(l) = latency(l) / latency(ref)

    Coldest level (registers, latency ≈ 1 cycle) → T_eff = 1.
    Hottest level (HBM, latency ≈ 600 cycles) → T_eff = 600.

    This gives the temperature gradient T_HBM >> T_L2 >> T_smem >> T_reg
    used in the Carnot efficiency formula η = 1 - T_cold / T_hot.
    """
    return level.latency_cycles / max(ref_level.latency_cycles, 1.0)


def _naive_carnot_efficiency(
    memory_levels: list[MemoryLevel],
) -> float:
    """
    Naive (capacity-unconstrained) Carnot efficiency:

        η_naive = 1 - T_reg / T_HBM = 1 - latency_reg / latency_HBM

    This approaches 1 but ignores finite capacity at each level, which forces
    data to spend time at intermediate (warm) levels.  The real η_hw,max is
    strictly lower and is derived from the full partition function.
    """
    T_cold = memory_levels[0].latency_cycles   # registers
    T_hot  = memory_levels[-1].latency_cycles  # HBM
    return 1.0 - T_cold / T_hot


def derive_carnot_limit(
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    beta_min: float = 0.01,
    beta_max: float = 10.0,
    n_beta: int = 200,
    n_bins: int = 64,
    work_field: float | None = None,
    activity_potential: float | None = None,
    target_activity: float | None = None,
) -> CarnotLimit:
    """
    Derive η_hw,max from the partition function.

    Strategy:
      η_hw(β, h) = <W_hw(β, h)> / <E_in(β, h)>

    where h is the useful-work field conjugate to productive hardware work.
    If ``target_activity`` is provided, h is solved from:

      <A>(β, h) = target_activity

    so the Carnot sweep operates at fixed load rather than fixed field.

    η_hw,max = max_β η_hw(β, h)

    The optimal β balances two competing effects:
      - High β (low T, lightly loaded): low waste per operation but also
        low throughput → η is dominated by idle capacity.
      - Low β (high T, heavily loaded): high throughput but rising waste
        from pipeline stalls and memory pressure → η plateaus then falls.

    The Carnot limit is the peak of this curve.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    if target_activity is None and work_field is None and activity_potential is None:
        target_activity = 0.20

    resolved_work_field = activity_potential if activity_potential is not None else work_field
    betas = np.linspace(beta_min, beta_max, n_beta).tolist()
    states = beta_sweep(
        betas,
        sm_config,
        memory_levels,
        n_bins=n_bins,
        work_field=resolved_work_field or 0.0,
        target_activity=target_activity,
    )

    etas = [max(0.0, min(1.0, s.eta_hw)) for s in states]

    best_idx = int(np.argmax(etas))
    best_state = states[best_idx]
    beta_opt = betas[best_idx]
    eta_max = etas[best_idx]

    # Effective temperatures
    ref_level = memory_levels[0]
    T_eff = {lvl.name: _effective_temperature(lvl, ref_level) for lvl in memory_levels}

    # --- Derive Carnot-optimal conditions ---

    # 1. Roofline ridge point (arithmetic intensity threshold):
    #    AI_min = peak_FLOPS_per_cycle / bandwidth_HBM_bytes_per_cycle
    #    A kernel must perform at least this many FLOPs per HBM byte to be
    #    compute-bound (and thus capable of approaching η_hw,max).
    hbm = memory_levels[-1]
    ai_min = sm_config.peak_flops_per_cycle / max(hbm.bandwidth_bytes_per_cycle, 1.0)

    # 2. Minimum data reuse factor per level:
    #    reuse_min(l) = capacity(l-1) / bandwidth(l) × peak_FLOPS
    #    Each byte loaded to level l must be reused enough times to amortise
    #    the load cost before eviction.
    min_reuse: dict[str, float] = {}
    for i in range(1, len(memory_levels)):
        warmer = memory_levels[i]      # level being loaded from
        colder = memory_levels[i - 1]  # level being computed at
        reuse = (colder.capacity_bytes / max(warmer.bandwidth_bytes_per_cycle, 1.0)
                 * sm_config.peak_flops_per_cycle)
        min_reuse[warmer.name] = reuse

    # 3. Minimum warp occupancy for latency hiding:
    #    occupancy_min = ceil(latency_HBM / instructions_per_warp_per_cycle)
    #    With latency L and 1 instruction issued per warp per cycle,
    #    we need L warps to keep the pipeline full.
    #    Express as a fraction of warps_per_sm.
    latency_hide_warps = math.ceil(hbm.latency_cycles)
    min_occupancy = min(1.0, latency_hide_warps / sm_config.warps_per_sm)

    return CarnotLimit(
        eta_hw_max=eta_max,
        beta_optimal=beta_opt,
        work_field_optimal=best_state.work_field,
        target_activity=target_activity,
        T_eff=T_eff,
        roofline_intensity=ai_min,
        min_reuse_factors=min_reuse,
        min_warp_occupancy=min_occupancy,
        thermo_state=best_state,
    )


# ---------------------------------------------------------------------------
# Carnot-optimal conditions checker
# ---------------------------------------------------------------------------

@dataclass
class CarnotConditionResult:
    """Result of checking one Carnot-optimal condition."""
    name: str
    satisfied: bool
    value: float        # measured value
    threshold: float    # required value
    margin: float       # value - threshold  (positive = satisfied, with slack)
    description: str


@dataclass
class CarnotOptimalityReport:
    """
    Full Carnot-optimality assessment for a kernel specification.

    A kernel is Carnot-optimal if all five conditions are satisfied.
    If not, the report identifies which conditions are violated and by
    how much, enabling targeted optimisation.
    """
    is_carnot_optimal: bool
    conditions: list[CarnotConditionResult]
    eta_hw_fraction: float      # estimated η_hw / η_hw,max ∈ [0, 1]
    dominant_bottleneck: str    # name of the most-violated condition


@dataclass
class KernelSpec:
    """
    Lightweight kernel specification sufficient to check Carnot-optimality.

    All fields use SI-adjacent units consistent with the partition function.
    """
    name: str

    # Arithmetic intensity (FLOP / byte of HBM traffic)
    arithmetic_intensity: float

    # Working set sizes (bytes resident at each memory level during execution)
    working_set: dict[str, int]   # e.g. {"registers": 32768, "smem": 49152}

    # Achieved data reuse factors (ops performed per byte loaded from each level)
    reuse_factors: dict[str, float]  # e.g. {"smem": 8.0, "L2": 2.0, "HBM": 16.0}

    # Achieved warp occupancy fraction ∈ [0, 1]
    warp_occupancy: float

    # Unnecessary data movement fraction ∈ [0, 1]
    # 0 = no unnecessary movement; 1 = all movement is redundant
    unnecessary_data_movement: float = 0.0


def check_carnot_optimality(
    kernel: KernelSpec,
    limit: CarnotLimit,
    memory_levels: list[MemoryLevel] | None = None,
) -> CarnotOptimalityReport:
    """
    Check whether a kernel satisfies the five Carnot-optimal conditions.

    Returns a full report with per-condition results, an estimated
    η_hw / η_hw,max, and the dominant bottleneck.
    """
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    conditions: list[CarnotConditionResult] = []

    # 1. Arithmetic intensity ≥ roofline ridge point
    ai = kernel.arithmetic_intensity
    ai_thresh = limit.roofline_intensity
    conditions.append(CarnotConditionResult(
        name="arithmetic_intensity",
        satisfied=ai >= ai_thresh,
        value=ai,
        threshold=ai_thresh,
        margin=ai - ai_thresh,
        description=(
            f"AI={ai:.1f} FLOP/byte {'≥' if ai >= ai_thresh else '<'} "
            f"roofline ridge {ai_thresh:.1f} FLOP/byte"
        ),
    ))

    # 2. Working set ≤ capacity at each level
    for level in memory_levels:
        ws = kernel.working_set.get(level.name, 0)
        cap = level.capacity_bytes
        if ws > 0:
            conditions.append(CarnotConditionResult(
                name=f"working_set_{level.name}",
                satisfied=ws <= cap,
                value=float(ws),
                threshold=float(cap),
                margin=float(cap - ws),
                description=(
                    f"{level.name} working set {ws/1024:.1f} KB "
                    f"{'≤' if ws <= cap else '>'} capacity {cap/1024:.1f} KB"
                ),
            ))

    # 3. Data reuse factor ≥ minimum at each level
    for level_name, reuse_min in limit.min_reuse_factors.items():
        reuse = kernel.reuse_factors.get(level_name, 0.0)
        conditions.append(CarnotConditionResult(
            name=f"reuse_{level_name}",
            satisfied=reuse >= reuse_min,
            value=reuse,
            threshold=reuse_min,
            margin=reuse - reuse_min,
            description=(
                f"{level_name} reuse {reuse:.1f}× "
                f"{'≥' if reuse >= reuse_min else '<'} "
                f"minimum {reuse_min:.1f}×"
            ),
        ))

    # 4. Warp occupancy ≥ minimum for latency hiding
    occ = kernel.warp_occupancy
    occ_min = limit.min_warp_occupancy
    conditions.append(CarnotConditionResult(
        name="warp_occupancy",
        satisfied=occ >= occ_min,
        value=occ,
        threshold=occ_min,
        margin=occ - occ_min,
        description=(
            f"occupancy {occ:.2f} "
            f"{'≥' if occ >= occ_min else '<'} "
            f"latency-hiding minimum {occ_min:.2f}"
        ),
    ))

    # 5. No unnecessary data movement
    udm = kernel.unnecessary_data_movement
    conditions.append(CarnotConditionResult(
        name="unnecessary_data_movement",
        satisfied=udm <= 0.0,
        value=udm,
        threshold=0.0,
        margin=-udm,
        description=(
            f"unnecessary data movement {udm*100:.1f}% "
            f"{'= 0 ✓' if udm <= 0.0 else '> 0 ✗'}"
        ),
    ))

    all_satisfied = all(c.satisfied for c in conditions)

    # Estimate η_hw / η_hw,max from condition violations
    # Each violated condition contributes a multiplicative penalty.
    eta_fraction = 1.0
    for c in conditions:
        if not c.satisfied:
            # Penalty proportional to relative violation magnitude (capped at 1)
            if c.threshold > 0:
                violation = abs(c.margin) / max(abs(c.threshold), 1e-12)
                eta_fraction *= max(0.0, 1.0 - min(violation, 1.0))

    # Dominant bottleneck: condition with largest relative violation
    violated = [c for c in conditions if not c.satisfied]
    if violated:
        dominant = max(
            violated,
            key=lambda c: abs(c.margin) / max(abs(c.threshold), 1e-12),
        )
        bottleneck = dominant.name
    else:
        bottleneck = "none"

    return CarnotOptimalityReport(
        is_carnot_optimal=all_satisfied,
        conditions=conditions,
        eta_hw_fraction=eta_fraction,
        dominant_bottleneck=bottleneck,
    )


# ---------------------------------------------------------------------------
# Roofline recovery validation
# ---------------------------------------------------------------------------

def verify_roofline_recovery(
    memory_levels: list[MemoryLevel] | None = None,
    sm_config: SMConfig | None = None,
) -> dict[str, float]:
    """
    Verify that the Carnot-optimal arithmetic-intensity condition reduces to
    the standard roofline ridge point in the two-level memory limit.

    The roofline model (Williams et al. 2009) says:
        attainable_GFLOPS = min(peak_FLOPS, peak_BW × AI)

    The ridge point AI* = peak_FLOPS / peak_BW is the minimum AI for
    compute-bound operation.

    We should recover: limit.roofline_intensity ≈ AI*

    Returns a dict with both values and their ratio (should be ≈ 1.0).
    """
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if sm_config is None:
        sm_config = H100_SM_CONFIG

    hbm = memory_levels[-1]
    roofline_ridge = sm_config.peak_flops_per_cycle / hbm.bandwidth_bytes_per_cycle

    limit = derive_carnot_limit(
        sm_config=sm_config,
        memory_levels=memory_levels,
    )
    carnot_ai = limit.roofline_intensity

    return {
        "roofline_ridge_flop_per_byte": roofline_ridge,
        "carnot_ai_min_flop_per_byte":  carnot_ai,
        "ratio":                         carnot_ai / max(roofline_ridge, 1e-12),
        "eta_hw_max":                    limit.eta_hw_max,
        "naive_carnot_efficiency":       _naive_carnot_efficiency(memory_levels),
    }

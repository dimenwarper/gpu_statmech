"""
Thermodynamic analysis module (Phase 1).

Takes gpusim MicrostateSnapshot traces and produces:

  - η_hw and η_hw / η_hw,max  per kernel and per full protocol
  - Waste decomposition by source (stall, idle, unnecessary movement)
  - Bottleneck attribution: which constraint is responsible for each
    percentage point of efficiency lost below η_max
  - Phase identification: compute-bound / memory-bound / latency-bound
  - Execution entropy: degeneracy of microstate trajectories at a given
    utilisation level (estimated via perturbation analysis)
  - Distance from Carnot limit, decomposed by waste source
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .carnot import CarnotLimit, derive_carnot_limit
from .energy import EnergyDecomposition, EnergyParams, aggregate_energy, compute_energy
from .observables import (
    TraceObservables,
    aggregate_trace_observables,
    canonicalize_snapshot,
    warp_state_family_fractions,
)
from .partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    MemoryLevel,
    SMConfig,
    ThermodynamicState,
    thermodynamic_quantities,
)


# ---------------------------------------------------------------------------
# Execution phase classification
# ---------------------------------------------------------------------------

class ExecutionPhase:
    COMPUTE_BOUND  = "compute_bound"
    MEMORY_BOUND   = "memory_bound"
    LATENCY_BOUND  = "latency_bound"
    MIXED          = "mixed"


class BetaInferenceMethod:
    OBSERVABLE_MATCH = "observable_match"
    CRUDE_WASTE_LOGIT = "crude_waste_logit"


_STATE_FAMILY_MATCH_WEIGHTS = {
    "productive": 2.0,
    "dependency": 1.5,
    "memory": 2.5,
    "sync_frontend": 1.0,
    "idle": 1.0,
}


def _warp_state_family_match_error(
    observed: dict[str, float],
    predicted: dict[str, float],
) -> float:
    err = 0.0
    for family, weight in _STATE_FAMILY_MATCH_WEIGHTS.items():
        err += weight * (float(predicted.get(family, 0.0)) - float(observed.get(family, 0.0))) ** 2
    return err


def classify_phase(
    snapshot: dict[str, Any],
    carnot_limit: CarnotLimit,
) -> str:
    """
    Classify the execution phase of a single snapshot.

    Prefer warp-state observables when available:
      - Memory-bound: dominant long-scoreboard / mem-throttle pressure
      - Latency-bound: dominant dependency / barrier / fetch pressure
      - Compute-bound: high eligible fraction with low stall pressure
      - Mixed: no single mechanism dominates

    Fall back to the older occupancy heuristic for legacy flat snapshots that
    do not expose warp-state fractions.
    """
    snap = canonicalize_snapshot(snapshot)
    state_frac = snap.get("warp_state_frac", {})
    if isinstance(state_frac, dict) and state_frac:
        eligible = float(state_frac.get("eligible", 0.0))
        idle = float(state_frac.get("idle", 0.0))
        short_scoreboard = float(state_frac.get("short_scoreboard", 0.0))
        memory_pressure = (
            float(state_frac.get("long_scoreboard", 0.0))
            + float(state_frac.get("mem_throttle", 0.0))
            + 0.5 * short_scoreboard
        )
        latency_pressure = (
            float(state_frac.get("exec_dep", 0.0))
            + float(state_frac.get("barrier", 0.0))
            + float(state_frac.get("fetch", 0.0))
            + 0.5 * short_scoreboard
        )
        issue = float(snap.get("issue_activity", 0.0))
        memory_stall = float(snap.get("memory_stall_fraction", 0.0))
        bw = float(snap.get("hbm_bw_util", 0.0))

        if memory_pressure >= 0.45 or memory_stall >= 0.35 or bw >= 0.7:
            return ExecutionPhase.MEMORY_BOUND
        if latency_pressure >= 0.20 and latency_pressure > memory_pressure + 0.05 and issue < 0.45:
            return ExecutionPhase.LATENCY_BOUND
        if (
            eligible >= 0.45
            and issue >= 0.35
            and memory_pressure < 0.25
            and latency_pressure < 0.15
            and idle < 0.25
        ):
            return ExecutionPhase.COMPUTE_BOUND
        return ExecutionPhase.MIXED

    stall = float(snap.get("stall_fraction", 0.0))
    bw = float(snap.get("hbm_bw_util", 0.0))
    occ = float(snap.get("active_warps", 1.0))
    min_occ = carnot_limit.min_warp_occupancy

    if bw > 0.7:
        return ExecutionPhase.MEMORY_BOUND
    if stall > 0.4 and occ < min_occ:
        return ExecutionPhase.LATENCY_BOUND
    if stall < 0.2 and bw < 0.7:
        return ExecutionPhase.COMPUTE_BOUND
    return ExecutionPhase.MIXED


# ---------------------------------------------------------------------------
# Bottleneck attribution
# ---------------------------------------------------------------------------

@dataclass
class BottleneckAttribution:
    """
    Attribution of efficiency loss below η_hw,max to specific hardware constraints.

    Each entry gives the fraction of total input energy attributed to that waste source.
    """
    # Efficiency gap
    eta_hw: float
    eta_hw_max: float
    gap: float                    # η_hw,max - η_hw

    # Waste sources (each as fraction of E_total)
    stall_fraction: float         # pipeline stalls
    idle_fraction: float          # idle SM capacity
    dram_movement_fraction: float # unnecessary HBM traffic
    sram_overhead_fraction: float # SRAM access overhead

    # Dominant constraint
    dominant_source: str
    dominant_fraction: float

    # Human-readable explanation
    explanation: str


def attribute_bottleneck(
    energy: EnergyDecomposition,
    phase: str,
    carnot_limit: CarnotLimit,
) -> BottleneckAttribution:
    """
    Attribute efficiency loss to specific hardware constraints.
    """
    eta_hw     = energy.eta_hw
    eta_hw_max = carnot_limit.eta_hw_max
    gap        = max(0.0, eta_hw_max - eta_hw)

    wb = energy.waste_breakdown()
    stall_frac = wb["stall"]
    idle_frac  = wb["idle"]
    dram_frac  = wb.get("unnecessary_movement", 0.0)

    # SRAM overhead: fraction of E_total spent on register + SMEM access
    sram_frac = energy.E_sram_nj / max(energy.E_total_nj, 1e-12)

    sources = {
        "pipeline_stalls":          stall_frac,
        "idle_sm_capacity":         idle_frac,
        "unnecessary_dram_traffic": dram_frac,
        "sram_overhead":            sram_frac,
    }
    dominant_src = max(sources, key=sources.__getitem__)
    dominant_frac = sources[dominant_src]

    # Build a human-readable explanation
    pct = lambda f: f"{f * 100:.1f}%"
    lines = [
        f"η_hw = {pct(eta_hw)} vs η_hw,max = {pct(eta_hw_max)} "
        f"(gap = {pct(gap)}, phase = {phase})",
        f"  Waste breakdown:",
        f"    pipeline stalls         : {pct(stall_frac)} of E_total",
        f"    idle SM capacity        : {pct(idle_frac)} of E_total",
        f"    unnecessary DRAM traffic: {pct(dram_frac)} of E_total",
        f"    SRAM access overhead    : {pct(sram_frac)} of E_total",
        f"  Dominant bottleneck: {dominant_src} ({pct(dominant_frac)})",
    ]

    if dominant_src == "pipeline_stalls":
        lines.append(
            "  Recommendation: increase warp occupancy to hide latency; "
            "check for long-scoreboard stalls (HBM-latency-bound)."
        )
    elif dominant_src == "idle_sm_capacity":
        lines.append(
            "  Recommendation: increase parallelism (more thread blocks) "
            "or fuse kernels to expose more independent work per SM."
        )
    elif dominant_src == "unnecessary_dram_traffic":
        lines.append(
            "  Recommendation: improve data reuse — increase tile size "
            "or use shared memory to cache reused data; check for redundant loads."
        )
    elif dominant_src == "sram_overhead":
        lines.append(
            "  Recommendation: reduce shared memory pressure; "
            "consolidate register usage or reduce smem bank conflicts."
        )

    return BottleneckAttribution(
        eta_hw=eta_hw,
        eta_hw_max=eta_hw_max,
        gap=gap,
        stall_fraction=stall_frac,
        idle_fraction=idle_frac,
        dram_movement_fraction=dram_frac,
        sram_overhead_fraction=sram_frac,
        dominant_source=dominant_src,
        dominant_fraction=dominant_frac,
        explanation="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Execution entropy
# ---------------------------------------------------------------------------

def estimate_entropy(
    snapshots: list[dict[str, Any]],
    beta: float | None = None,
    sm_config: SMConfig | None = None,
) -> float:
    """
    Estimate the execution entropy S from a set of MicrostateSnapshots.

    Strategy: treat the distribution of (active_warps, stall_fraction) pairs
    across snapshots as an empirical microstate distribution.  The entropy is:

        S_empirical = -Σ_i p_i ln p_i

    where p_i is the empirical probability of observing macrostate i.

    This is a lower bound on the true entropy (since we're coarse-graining),
    but it captures the degeneracy structure visible from outside the kernel.

    Higher entropy = more ways to achieve the same utilisation level = more
    flexibility to optimise without disrupting macroscopic performance.
    Lower entropy = the utilisation pattern is fragile and highly constrained.
    """
    if not snapshots:
        return 0.0

    # Coarse-grain to 10×10 grid of (active_warps, stall_fraction)
    n_bins = 10
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    for s in snapshots:
        snap = canonicalize_snapshot(s)
        aw = float(snap.get("active_warps", 0.5))
        sf = float(snap.get("stall_fraction", 0.2))
        i = min(int(aw * n_bins), n_bins - 1)
        j = min(int(sf * n_bins), n_bins - 1)
        counts[i, j] += 1.0

    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    # Shannon entropy
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


# ---------------------------------------------------------------------------
# Full thermodynamic analysis
# ---------------------------------------------------------------------------

@dataclass
class KernelThermoAnalysis:
    """Complete thermodynamic analysis for a single kernel execution."""
    kernel_name: str

    # Efficiency
    eta_hw: float
    eta_hw_max: float
    eta_hw_fraction: float        # η_hw / η_hw,max

    # Energy decomposition
    energy: EnergyDecomposition

    # Phase
    dominant_phase: str
    phase_distribution: dict[str, float]   # fraction of snapshots in each phase

    # Bottleneck
    bottleneck: BottleneckAttribution

    # Entropy
    execution_entropy: float
    observables: TraceObservables

    # Thermodynamic state at the observed operating point
    thermo_state: ThermodynamicState
    beta_inference_method: str
    beta_inference_error: float


@dataclass
class ProtocolThermoAnalysis:
    """Thermodynamic analysis for a full computation protocol (sequence of kernels)."""
    kernel_analyses: list[KernelThermoAnalysis]

    @property
    def eta_hw(self) -> float:
        """Protocol-level η_hw = total W_hw / total E_total."""
        w = sum(k.energy.W_hw_nj for k in self.kernel_analyses)
        e = sum(k.energy.E_total_nj for k in self.kernel_analyses)
        return w / max(e, 1e-12)

    @property
    def eta_hw_max(self) -> float:
        """Protocol-level η_hw,max = min over kernels (weakest link)."""
        if not self.kernel_analyses:
            return 0.0
        return min(k.eta_hw_max for k in self.kernel_analyses)

    @property
    def eta_hw_fraction(self) -> float:
        return self.eta_hw / max(self.eta_hw_max, 1e-12)

    @property
    def total_energy(self) -> EnergyDecomposition:
        return aggregate_energy([])   # placeholder — use sum_energy()

    def dominant_bottleneck(self) -> str:
        """The bottleneck source responsible for the most waste across the protocol."""
        from collections import Counter
        sources = [k.bottleneck.dominant_source for k in self.kernel_analyses]
        if not sources:
            return "none"
        return Counter(sources).most_common(1)[0][0]

    def summary(self) -> str:
        lines = [
            f"Protocol: {len(self.kernel_analyses)} kernel(s)",
            f"  η_hw          = {self.eta_hw * 100:.1f}%",
            f"  η_hw,max      = {self.eta_hw_max * 100:.1f}%",
            f"  η_hw/η_hw,max = {self.eta_hw_fraction * 100:.1f}%",
            f"  Dominant bottleneck: {self.dominant_bottleneck()}",
        ]
        for ka in self.kernel_analyses:
            lines.append(
                f"  [{ka.kernel_name}] η={ka.eta_hw*100:.1f}%  "
                f"phase={ka.dominant_phase}  "
                f"bottleneck={ka.bottleneck.dominant_source}"
            )
        return "\n".join(lines)


def analyse_kernel(
    kernel_name: str,
    snapshots: list[dict[str, Any]],
    carnot_limit: CarnotLimit | None = None,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    energy_params: EnergyParams | None = None,
    beta_inference_method: str = BetaInferenceMethod.OBSERVABLE_MATCH,
    beta_min: float = 0.01,
    beta_max: float = 10.0,
    n_beta: int = 200,
) -> KernelThermoAnalysis:
    """
    Full thermodynamic analysis for a single kernel from its snapshot trace.

    Parameters
    ----------
    kernel_name : str
    snapshots   : list of MicrostateSnapshot dicts from gpusim
    carnot_limit: pre-computed CarnotLimit (derived once and reused)
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if carnot_limit is None:
        carnot_limit = derive_carnot_limit(sm_config, memory_levels)

    observables = aggregate_trace_observables(snapshots)

    # Energy
    energy = aggregate_energy(snapshots, n_sm=sm_config.n_sm,
                               params=energy_params)

    # Phase classification
    phase_counts: dict[str, int] = {}
    for s in snapshots:
        p = classify_phase(s, carnot_limit)
        phase_counts[p] = phase_counts.get(p, 0) + 1
    total_snaps = max(len(snapshots), 1)
    phase_dist = {p: c / total_snaps for p, c in phase_counts.items()}
    dominant_phase = max(phase_counts, key=phase_counts.__getitem__) if phase_counts else ExecutionPhase.MIXED

    # Bottleneck
    bottleneck = attribute_bottleneck(energy, dominant_phase, carnot_limit)

    # Entropy
    entropy = estimate_entropy(snapshots)

    # Thermodynamic state at the observed operating point
    if beta_inference_method == BetaInferenceMethod.CRUDE_WASTE_LOGIT:
        wf = energy.waste_fraction
        beta_obs = math.log(max(1e-6, (1.0 - wf) / max(wf, 1e-6)))
        thermo_state = thermodynamic_quantities(
            beta=max(beta_min, beta_obs),
            sm_config=sm_config,
            memory_levels=memory_levels,
        )
        beta_inference_error = 0.0
    elif beta_inference_method == BetaInferenceMethod.OBSERVABLE_MATCH:
        if observables.n_snapshots == 0 or observables.mean_issue_activity <= 1e-9:
            thermo_state = thermodynamic_quantities(
                beta=beta_min,
                sm_config=sm_config,
                memory_levels=memory_levels,
                work_field=0.0,
            )
            beta_inference_error = 0.0
            return KernelThermoAnalysis(
                kernel_name=kernel_name,
                eta_hw=energy.eta_hw,
                eta_hw_max=carnot_limit.eta_hw_max,
                eta_hw_fraction=energy.eta_hw / max(carnot_limit.eta_hw_max, 1e-12),
                energy=energy,
                dominant_phase=dominant_phase,
                phase_distribution=phase_dist,
                bottleneck=bottleneck,
                execution_entropy=entropy,
                observables=observables,
                thermo_state=thermo_state,
                beta_inference_method=beta_inference_method,
                beta_inference_error=beta_inference_error,
            )
        betas = np.linspace(beta_min, beta_max, n_beta).tolist()
        best_score = float("inf")
        best_state: ThermodynamicState | None = None
        for beta in betas:
            state = thermodynamic_quantities(
                beta=beta,
                sm_config=sm_config,
                memory_levels=memory_levels,
                target_activity=observables.mean_issue_activity,
            )
            predicted_families = warp_state_family_fractions(state.warp_state_fractions)
            family_err = _warp_state_family_match_error(
                observables.mean_warp_state_family_fractions,
                predicted_families,
            )
            memory_err = (
                predicted_families.get("memory", 0.0)
                - observables.mean_warp_state_family_fractions.get("memory", 0.0)
            )
            productive_err = (
                predicted_families.get("productive", 0.0)
                - observables.mean_warp_state_family_fractions.get("productive", 0.0)
            )
            feed_err = state.memory_feed_efficiency - observables.memory_feed_efficiency_proxy
            score = (
                4.0 * family_err
                + 1.0 * memory_err * memory_err
                + 0.5 * productive_err * productive_err
                + 0.25 * feed_err * feed_err
            )
            if score < best_score:
                best_score = score
                best_state = state
        assert best_state is not None
        thermo_state = best_state
        beta_inference_error = best_score
    else:
        raise ValueError(f"unknown beta_inference_method: {beta_inference_method}")

    return KernelThermoAnalysis(
        kernel_name=kernel_name,
        eta_hw=energy.eta_hw,
        eta_hw_max=carnot_limit.eta_hw_max,
        eta_hw_fraction=energy.eta_hw / max(carnot_limit.eta_hw_max, 1e-12),
        energy=energy,
        dominant_phase=dominant_phase,
        phase_distribution=phase_dist,
        bottleneck=bottleneck,
        execution_entropy=entropy,
        observables=observables,
        thermo_state=thermo_state,
        beta_inference_method=beta_inference_method,
        beta_inference_error=beta_inference_error,
    )


def analyse_protocol(
    kernel_traces: dict[str, list[dict[str, Any]]],
    carnot_limit: CarnotLimit | None = None,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    beta_inference_method: str = BetaInferenceMethod.OBSERVABLE_MATCH,
    beta_min: float = 0.01,
    beta_max: float = 10.0,
    n_beta: int = 200,
) -> ProtocolThermoAnalysis:
    """
    Analyse a full computation protocol as a sequence of named kernels.

    Parameters
    ----------
    kernel_traces : {kernel_name: [snapshot, ...], ...}
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if carnot_limit is None:
        carnot_limit = derive_carnot_limit(sm_config, memory_levels)

    analyses = [
        analyse_kernel(
            name,
            snaps,
            carnot_limit,
            sm_config,
            memory_levels,
            beta_inference_method=beta_inference_method,
            beta_min=beta_min,
            beta_max=beta_max,
            n_beta=n_beta,
        )
        for name, snaps in kernel_traces.items()
    ]
    return ProtocolThermoAnalysis(kernel_analyses=analyses)

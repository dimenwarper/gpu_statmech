"""
energy.py — Energy functional E(σ) for a single GPU microstate.

Defines the mapping from a Microstate to a scalar energy value representing
wasted GPU capacity, and the time-averaged energy Ē over a trajectory.

Per Section 3.2 of the project brief:

    E(σ) = α₁·(1 – SM_util) + α₂·(1 – mem_bandwidth_util)
           + α₃·pipeline_stalls + α₄·data_movement_cost

Ground state E = 0 corresponds to perfect utilization: every SM fully
occupied, every memory channel saturated, zero pipeline bubbles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .microstate import Microstate, MemoryLevel, MEMORY_LEVEL_ENERGY


# ---------------------------------------------------------------------------
# Energy weights
# ---------------------------------------------------------------------------


@dataclass
class EnergyWeights:
    """
    α coefficients weighting each waste term in E(σ).

    Defaults are equal-weight (1.0) except stall and data-movement terms,
    which are set to 0.5 to reflect that they are secondary to raw utilization.
    These can be calibrated from hardware power-modeling data.

    Attributes:
        sm_utilization   α₁ — penalty for idle SM capacity
        mem_bandwidth    α₂ — penalty for unused HBM bandwidth
        pipeline_stalls  α₃ — penalty for warp pipeline stalls
        data_movement    α₄ — penalty for data residing in high-energy memory levels
    """
    sm_utilization:  float = 1.0
    mem_bandwidth:   float = 1.0
    pipeline_stalls: float = 0.5
    data_movement:   float = 0.5

    def __post_init__(self) -> None:
        for name in ("sm_utilization", "mem_bandwidth", "pipeline_stalls", "data_movement"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"EnergyWeights.{name} must be ≥ 0, got {v}")


# ---------------------------------------------------------------------------
# Roofline helpers
# ---------------------------------------------------------------------------


@dataclass
class RooflinePoint:
    """
    A single point on the roofline model.

    arithmetic_intensity  — FLOPs / byte (x-axis)
    achieved_flops        — actual FLOP/s attained by the kernel
    peak_compute_flops    — device peak FLOP/s (e.g. 989e12 for H100 FP16)
    peak_memory_bandwidth — device peak BW in bytes/s (e.g. 3.35e12 for H100)
    """
    arithmetic_intensity: float   # FLOPs / byte
    achieved_flops: float         # FLOP/s
    peak_compute_flops: float     # FLOP/s
    peak_memory_bandwidth: float  # bytes/s

    @property
    def roofline_flops(self) -> float:
        """Predicted FLOP/s from the roofline model."""
        compute_roof = self.peak_compute_flops
        memory_roof  = self.arithmetic_intensity * self.peak_memory_bandwidth
        return min(compute_roof, memory_roof)

    @property
    def mfu(self) -> float:
        """Model FLOP Utilization: achieved / peak_compute ∈ [0, 1]."""
        if self.peak_compute_flops == 0:
            return 0.0
        return min(self.achieved_flops / self.peak_compute_flops, 1.0)

    @property
    def roofline_efficiency(self) -> float:
        """achieved_flops / roofline_flops ∈ [0, 1]."""
        roof = self.roofline_flops
        if roof == 0:
            return 0.0
        return min(self.achieved_flops / roof, 1.0)

    @property
    def is_compute_bound(self) -> bool:
        ridge = self.peak_compute_flops / self.peak_memory_bandwidth
        return self.arithmetic_intensity >= ridge


# ---------------------------------------------------------------------------
# Energy functional
# ---------------------------------------------------------------------------


@dataclass
class EnergyFunctional:
    """
    Computes E(σ) for a single GPU Microstate.

    E(σ) = α₁·(1 – SM_util) + α₂·(1 – mem_bw_util)
           + α₃·pipeline_stalls + α₄·data_movement_cost

    All terms are dimensionless and lie in [0, 1] before weighting,
    so E(σ) ∈ [0, α₁ + α₂ + α₃ + α₄].

    The ground state E = 0 is achieved only when:
        - All SMs are fully occupied (mean_sm_occupancy = 1)
        - HBM bandwidth is fully saturated (hbm_bandwidth_utilization = 1)
        - No pipeline stalls (pipeline_stall_fraction = 0)
        - All data resides in registers (data_movement_cost = 0)
    """
    weights: EnergyWeights = field(default_factory=EnergyWeights)

    # ------------------------------------------------------------------
    # Individual waste terms
    # ------------------------------------------------------------------

    def sm_waste(self, state: Microstate) -> float:
        """α₁ · (1 – SM_util): wasted SM capacity."""
        return self.weights.sm_utilization * (1.0 - state.mean_sm_occupancy)

    def bandwidth_waste(self, state: Microstate) -> float:
        """α₂ · (1 – mem_bw_util): unused HBM bandwidth."""
        return self.weights.mem_bandwidth * (
            1.0 - state.memory.hbm_bandwidth_utilization
        )

    def stall_cost(self, state: Microstate) -> float:
        """α₃ · pipeline_stalls: cost of warp pipeline stalls."""
        return self.weights.pipeline_stalls * state.pipeline_stall_fraction

    def data_movement_cost(self, state: Microstate) -> float:
        """
        α₄ · data_movement_cost: penalty for data in high-energy memory levels.

        Uses fine-grained occupation numbers when available; falls back to
        a coarse proxy (HBM bandwidth utilization as memory pressure signal)
        when they are absent.

        Normalized to [0, 1]: cost = 0 when all data is in registers,
        cost = 1 when all data is in HBM.
        """
        mem = state.memory
        if mem.occupation_numbers:
            total_bytes = sum(mem.occupation_numbers.values())
            if total_bytes == 0:
                return 0.0
            weighted = sum(
                bytes_at * MEMORY_LEVEL_ENERGY[level]
                for (_, level), bytes_at in mem.occupation_numbers.items()
            )
            max_energy = total_bytes * MEMORY_LEVEL_ENERGY[MemoryLevel.HBM]
            normalized = weighted / max_energy if max_energy > 0 else 0.0
        else:
            # Coarse proxy: data pressure correlates with HBM bandwidth usage.
            # If the kernel is memory-bound (high HBM BW) and has low L2 hit
            # rate, data is predominantly in HBM — the highest energy level.
            l2_miss_rate = 1.0 - mem.l2_hit_rate
            normalized = l2_miss_rate * mem.hbm_bandwidth_utilization

        return self.weights.data_movement * normalized

    # ------------------------------------------------------------------
    # Full energy
    # ------------------------------------------------------------------

    def compute(self, state: Microstate) -> float:
        """
        Compute E(σ) for a single microstate.

        Returns a non-negative scalar; lower is better.
        E = 0 ↔ ground state (perfect utilization, no waste).
        """
        return (
            self.sm_waste(state)
            + self.bandwidth_waste(state)
            + self.stall_cost(state)
            + self.data_movement_cost(state)
        )

    def decompose(self, state: Microstate) -> dict[str, float]:
        """
        Return E(σ) broken down by term — useful for diagnostics and
        feeding detailed feedback into the LLM oracle.
        """
        return {
            "sm_waste":           self.sm_waste(state),
            "bandwidth_waste":    self.bandwidth_waste(state),
            "stall_cost":         self.stall_cost(state),
            "data_movement_cost": self.data_movement_cost(state),
            "total":              self.compute(state),
        }

    def time_average(self, trajectory: list[Microstate]) -> float:
        """
        Ē = (1/T) Σ_t E(σ(t))

        Time-averaged energy over a complete kernel trajectory.
        This is the primary figure of merit for hardware utilization.
        """
        if not trajectory:
            return 0.0
        return sum(self.compute(s) for s in trajectory) / len(trajectory)

    def time_average_decomposed(
        self, trajectory: list[Microstate]
    ) -> dict[str, float]:
        """
        Ē decomposed by term, averaged over the trajectory.
        Useful for identifying the dominant bottleneck.
        """
        if not trajectory:
            return {k: 0.0 for k in ("sm_waste", "bandwidth_waste",
                                      "stall_cost", "data_movement_cost", "total")}
        decomps = [self.decompose(s) for s in trajectory]
        T = len(trajectory)
        return {k: sum(d[k] for d in decomps) / T for k in decomps[0]}

    # ------------------------------------------------------------------
    # Roofline recovery
    # ------------------------------------------------------------------

    def from_roofline(self, point: RooflinePoint) -> float:
        """
        Compute an approximate E from a RooflinePoint.

        This provides a bridge between the roofline model (which is a special
        case of E when only α₁ and α₂ are nonzero) and the full energy
        functional.  Useful for validation (Section 7.1).

        The roofline model identifies the *active bottleneck*:
          - Compute-bound  (AI ≥ ridge): memory bandwidth is NOT the bottleneck;
            penalise only wasted compute capacity.
          - Memory-bound   (AI <  ridge): compute is NOT the bottleneck;
            penalise only wasted bandwidth.

        This mirrors the physical interpretation — if a kernel is running at
        peak compute, E_hardware ≈ 0 even though bandwidth utilisation is low.
        """
        compute_eff = min(point.achieved_flops / point.peak_compute_flops, 1.0)

        if point.peak_memory_bandwidth > 0 and point.arithmetic_intensity > 0:
            bw_eff = min(
                point.achieved_flops
                / (point.arithmetic_intensity * point.peak_memory_bandwidth),
                1.0,
            )
        else:
            bw_eff = 0.0

        if point.is_compute_bound:
            # Memory bandwidth is not the bottleneck — only penalise idle compute.
            return self.weights.sm_utilization * (1.0 - compute_eff)
        else:
            # Compute is not the bottleneck — only penalise idle bandwidth.
            return self.weights.mem_bandwidth * (1.0 - bw_eff)

"""Tests for gpu_statmech.energy."""

import pytest
from gpu_statmech.energy import EnergyWeights, EnergyFunctional, RooflinePoint
from gpu_statmech.microstate import (
    Microstate, SMState, MemoryHierarchyState, BandwidthState,
    WarpState, InstructionType, PipelineStage, MemoryLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_microstate(
    occupancy: float = 1.0,
    hbm_util: float = 1.0,
    stall_frac: float = 0.0,
    l2_hit: float = 1.0,
    occupation_numbers: dict | None = None,
    num_sms: int = 1,
    num_warps: int = 4,
    max_warps: int = 64,
) -> Microstate:
    """
    Build a synthetic Microstate with controllable utilization values.
    occupancy      → active_warps = occupancy * max_warps per SM
    stall_frac     → that fraction of warps set to PipelineStage.STALL
    """
    active_per_sm = round(occupancy * max_warps)
    n_stalled = round(stall_frac * num_warps)
    warps = (
        [WarpState(i, True, InstructionType.FP16, PipelineStage.STALL)
         for i in range(n_stalled)] +
        [WarpState(i + n_stalled, True, InstructionType.FP16, PipelineStage.EXECUTE)
         for i in range(num_warps - n_stalled)]
    )
    sm_states = [
        SMState(sm_id=i, active_warps=active_per_sm,
                max_warps=max_warps, warp_states=warps)
        for i in range(num_sms)
    ]
    mem = MemoryHierarchyState(
        register_utilization=0.8,
        shared_mem_utilization=0.5,
        l1_hit_rate=0.9,
        l2_hit_rate=l2_hit,
        l2_utilization=0.5,
        hbm_bandwidth_utilization=hbm_util,
        occupation_numbers=occupation_numbers or {},
    )
    bw = BandwidthState(sm_to_shared_mem=0.5, sm_to_l2=0.5,
                        l2_to_hbm=hbm_util)
    return Microstate(cycle=0, gpu_id=0, sm_states=sm_states,
                      memory=mem, bandwidth=bw)


# ---------------------------------------------------------------------------
# EnergyWeights
# ---------------------------------------------------------------------------

class TestEnergyWeights:
    def test_defaults_non_negative(self):
        w = EnergyWeights()
        assert w.sm_utilization >= 0
        assert w.mem_bandwidth >= 0
        assert w.pipeline_stalls >= 0
        assert w.data_movement >= 0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            EnergyWeights(sm_utilization=-0.1)


# ---------------------------------------------------------------------------
# Ground state: E = 0 at perfect utilization
# ---------------------------------------------------------------------------

class TestGroundState:
    def test_ground_state_zero(self):
        """E(σ) = 0 when SM is fully occupied, HBM saturated, no stalls,
        all data in registers."""
        state = make_microstate(
            occupancy=1.0,
            hbm_util=1.0,
            stall_frac=0.0,
            occupation_numbers={(0, MemoryLevel.REGISTERS): 1000},
        )
        ef = EnergyFunctional()
        assert ef.compute(state) == pytest.approx(0.0, abs=1e-9)

    def test_ground_state_zero_no_occ_numbers(self):
        """Ground state with l2_hit=1 → data_movement proxy also 0."""
        state = make_microstate(
            occupancy=1.0, hbm_util=1.0, stall_frac=0.0, l2_hit=1.0
        )
        ef = EnergyFunctional()
        # data_movement = w.data_movement * l2_miss_rate * hbm_util
        # l2_miss_rate = 0 → data_movement term = 0
        assert ef.compute(state) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Worst state: all waste → E = sum(all alpha weights)
# ---------------------------------------------------------------------------

class TestWorstState:
    def test_max_energy_components(self):
        """Idle SM, zero BW, all stalled, all data in HBM."""
        state = make_microstate(
            occupancy=0.0,
            hbm_util=0.0,
            stall_frac=1.0,
            l2_hit=0.0,
            occupation_numbers={(0, MemoryLevel.HBM): 1000},
        )
        w = EnergyWeights(
            sm_utilization=1.0,
            mem_bandwidth=1.0,
            pipeline_stalls=0.5,
            data_movement=0.5,
        )
        ef = EnergyFunctional(weights=w)
        e = ef.compute(state)
        # sm_waste=1.0, bw_waste=1.0, stall_cost=0.5, data_movement=0.5 → 3.0
        assert e == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Individual waste terms
# ---------------------------------------------------------------------------

class TestWasteTerms:
    def setup_method(self):
        self.ef = EnergyFunctional()

    def test_sm_waste_half_occupancy(self):
        state = make_microstate(occupancy=0.5, hbm_util=1.0, stall_frac=0.0)
        # α₁=1.0, (1 - 0.5) = 0.5
        assert self.ef.sm_waste(state) == pytest.approx(0.5)

    def test_bandwidth_waste_half_utilization(self):
        state = make_microstate(occupancy=1.0, hbm_util=0.5, stall_frac=0.0)
        assert self.ef.bandwidth_waste(state) == pytest.approx(0.5)

    def test_stall_cost_half_stalled(self):
        state = make_microstate(occupancy=1.0, hbm_util=1.0, stall_frac=0.5)
        # α₃=0.5, stall_frac=0.5 → 0.25
        assert self.ef.stall_cost(state) == pytest.approx(0.25)

    def test_data_movement_all_registers(self):
        state = make_microstate(
            occupation_numbers={(0, MemoryLevel.REGISTERS): 1000}
        )
        assert self.ef.data_movement_cost(state) == pytest.approx(0.0)

    def test_data_movement_all_hbm(self):
        state = make_microstate(
            occupation_numbers={(0, MemoryLevel.HBM): 1000}
        )
        # normalized cost = 1.0, then * α₄=0.5
        assert self.ef.data_movement_cost(state) == pytest.approx(0.5)

    def test_decompose_keys(self):
        state = make_microstate()
        d = self.ef.decompose(state)
        for k in ("sm_waste", "bandwidth_waste", "stall_cost",
                  "data_movement_cost", "total"):
            assert k in d

    def test_decompose_total_matches_compute(self):
        state = make_microstate(occupancy=0.7, hbm_util=0.6, stall_frac=0.2)
        d = self.ef.decompose(state)
        assert d["total"] == pytest.approx(self.ef.compute(state))


# ---------------------------------------------------------------------------
# Time-averaging
# ---------------------------------------------------------------------------

class TestTimeAverage:
    def test_empty_trajectory(self):
        ef = EnergyFunctional()
        assert ef.time_average([]) == 0.0

    def test_single_state(self):
        ef = EnergyFunctional()
        state = make_microstate(occupancy=0.5, hbm_util=0.5, stall_frac=0.0)
        assert ef.time_average([state]) == pytest.approx(ef.compute(state))

    def test_average_two_states(self):
        ef = EnergyFunctional()
        s1 = make_microstate(occupancy=1.0, hbm_util=1.0)  # E ≈ 0
        s2 = make_microstate(occupancy=0.0, hbm_util=0.0)  # E = large
        e1 = ef.compute(s1)
        e2 = ef.compute(s2)
        assert ef.time_average([s1, s2]) == pytest.approx((e1 + e2) / 2)

    def test_decomposed_average_matches(self):
        ef = EnergyFunctional()
        states = [make_microstate(occupancy=o) for o in [0.2, 0.5, 0.8, 1.0]]
        avg = ef.time_average(states)
        avg_decomp = ef.time_average_decomposed(states)
        assert avg_decomp["total"] == pytest.approx(avg)


# ---------------------------------------------------------------------------
# Roofline recovery
# ---------------------------------------------------------------------------

class TestRooflineRecovery:
    # H100 specs
    PEAK_FLOPS = 989e12   # FP16 tensor core
    PEAK_BW    = 3.35e12  # HBM bandwidth (bytes/s)

    def _point(self, ai: float, achieved_frac: float) -> RooflinePoint:
        peak = min(self.PEAK_FLOPS, ai * self.PEAK_BW)
        return RooflinePoint(
            arithmetic_intensity=ai,
            achieved_flops=achieved_frac * peak,
            peak_compute_flops=self.PEAK_FLOPS,
            peak_memory_bandwidth=self.PEAK_BW,
        )

    def test_compute_bound_vs_memory_bound(self):
        ridge = self.PEAK_FLOPS / self.PEAK_BW  # ~295 FLOPs/byte
        compute_bound = self._point(ridge * 2, 1.0)
        memory_bound  = self._point(ridge / 2, 1.0)
        assert compute_bound.is_compute_bound
        assert not memory_bound.is_compute_bound

    def test_mfu_at_peak(self):
        p = RooflinePoint(
            arithmetic_intensity=1000.0,
            achieved_flops=self.PEAK_FLOPS,
            peak_compute_flops=self.PEAK_FLOPS,
            peak_memory_bandwidth=self.PEAK_BW,
        )
        assert p.mfu == pytest.approx(1.0)

    def test_roofline_energy_at_peak_is_zero(self):
        """At perfect roofline efficiency, hardware energy should be 0."""
        ridge = self.PEAK_FLOPS / self.PEAK_BW
        p = RooflinePoint(
            arithmetic_intensity=ridge * 10,   # compute bound
            achieved_flops=self.PEAK_FLOPS,
            peak_compute_flops=self.PEAK_FLOPS,
            peak_memory_bandwidth=self.PEAK_BW,
        )
        ef = EnergyFunctional(
            weights=EnergyWeights(pipeline_stalls=0.0, data_movement=0.0)
        )
        assert ef.from_roofline(p) == pytest.approx(0.0, abs=1e-6)

    def test_roofline_energy_at_half_utilization(self):
        ridge = self.PEAK_FLOPS / self.PEAK_BW
        p = RooflinePoint(
            arithmetic_intensity=ridge * 10,
            achieved_flops=0.5 * self.PEAK_FLOPS,
            peak_compute_flops=self.PEAK_FLOPS,
            peak_memory_bandwidth=self.PEAK_BW,
        )
        ef = EnergyFunctional(
            weights=EnergyWeights(sm_utilization=1.0, mem_bandwidth=1.0,
                                  pipeline_stalls=0.0, data_movement=0.0)
        )
        # compute_eff = 0.5  → sm_waste = 0.5
        # bw_eff: achieved/(AI*bw) = 0.5*peak_compute/(ridge*10*peak_bw)
        #       = 0.5*peak_compute/(10*peak_compute) = 0.05 → bw_waste ≈ 0.95
        e = ef.from_roofline(p)
        assert e > 0.0

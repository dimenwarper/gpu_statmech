"""Tests for gpu_statmech.free_energy."""

import math
import pytest

from gpu_statmech.free_energy import (
    GeometricAnnealing,
    CosineAnnealing,
    LinearAnnealing,
    FreeEnergy,
)
from gpu_statmech.microstate import (
    Microstate, SMState, MemoryHierarchyState, BandwidthState,
)
from gpu_statmech.multi_gpu import (
    GlobalMicrostate, CommState, CommChannelState,
)
from gpu_statmech.topology import Topology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_microstate(occupancy=1.0, hbm_util=1.0) -> Microstate:
    active = round(occupancy * 64)
    sm = SMState(sm_id=0, active_warps=active, max_warps=64)
    mem = MemoryHierarchyState(
        register_utilization=0.8, shared_mem_utilization=0.5,
        l1_hit_rate=0.9, l2_hit_rate=0.8,
        l2_utilization=0.5, hbm_bandwidth_utilization=hbm_util,
    )
    bw = BandwidthState(l2_to_hbm=hbm_util)
    return Microstate(cycle=0, gpu_id=0, sm_states=[sm], memory=mem, bandwidth=bw)


def make_global(num_gpus=2, occupancy=1.0) -> GlobalMicrostate:
    gpu_states = [make_microstate(occupancy) for _ in range(num_gpus)]
    return GlobalMicrostate(
        cycle=0,
        gpu_states=gpu_states,
        comm_state=CommState(),
        topology=Topology.pcie_cluster(num_gpus),
    )


# ---------------------------------------------------------------------------
# Annealing schedules
# ---------------------------------------------------------------------------

class TestGeometricAnnealing:
    def test_step_zero_is_initial(self):
        sched = GeometricAnnealing(initial_temperature=2.0, factor=0.9)
        assert sched.temperature_at(0) == pytest.approx(2.0)

    def test_decreasing(self):
        sched = GeometricAnnealing(initial_temperature=1.0, factor=0.9)
        temps = [sched.temperature_at(k) for k in range(10)]
        assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

    def test_clamped_at_min(self):
        sched = GeometricAnnealing(initial_temperature=1.0, factor=0.5,
                                   min_temperature=0.1)
        for k in range(100):
            assert sched.temperature_at(k) >= 0.1

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            GeometricAnnealing(factor=1.5)

    def test_invalid_min_raises(self):
        with pytest.raises(ValueError):
            GeometricAnnealing(min_temperature=0.0)


class TestCosineAnnealing:
    def test_step_zero_is_max(self):
        sched = CosineAnnealing(max_temperature=1.0, min_temperature=0.01,
                                total_steps=20)
        assert sched.temperature_at(0) == pytest.approx(1.0)

    def test_final_step_is_min(self):
        sched = CosineAnnealing(max_temperature=1.0, min_temperature=0.01,
                                total_steps=20)
        assert sched.temperature_at(20) == pytest.approx(0.01)

    def test_decreasing(self):
        sched = CosineAnnealing(max_temperature=1.0, min_temperature=0.01,
                                total_steps=10)
        temps = [sched.temperature_at(k) for k in range(11)]
        assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

    def test_clamps_beyond_total_steps(self):
        sched = CosineAnnealing(max_temperature=1.0, min_temperature=0.01,
                                total_steps=5)
        assert sched.temperature_at(100) == pytest.approx(0.01)


class TestLinearAnnealing:
    def test_step_zero_is_max(self):
        sched = LinearAnnealing(max_temperature=1.0, min_temperature=0.0,
                                total_steps=10)
        assert sched.temperature_at(0) == pytest.approx(1.0)

    def test_final_step_is_min(self):
        sched = LinearAnnealing(max_temperature=1.0, min_temperature=0.0,
                                total_steps=10)
        assert sched.temperature_at(10) == pytest.approx(0.0)

    def test_midpoint(self):
        sched = LinearAnnealing(max_temperature=1.0, min_temperature=0.0,
                                total_steps=10)
        assert sched.temperature_at(5) == pytest.approx(0.5)

    def test_decreasing(self):
        sched = LinearAnnealing(max_temperature=1.0, min_temperature=0.0,
                                total_steps=10)
        temps = [sched.temperature_at(k) for k in range(11)]
        assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))


# ---------------------------------------------------------------------------
# FreeEnergy.compute
# ---------------------------------------------------------------------------

class TestFreeEnergyCompute:
    def test_high_expressiveness_lowers_F(self):
        """Higher S_model → lower free energy (more favoured)."""
        fe = FreeEnergy(temperature=1.0)
        state = make_microstate(occupancy=0.5, hbm_util=0.5)
        f_low  = fe.compute(state, expressiveness_score=0.1)
        f_high = fe.compute(state, expressiveness_score=0.9)
        assert f_high < f_low

    def test_high_hardware_energy_raises_F(self):
        """Worse hardware state (lower occupancy) → higher free energy."""
        fe = FreeEnergy(temperature=1.0)
        good = make_microstate(occupancy=1.0, hbm_util=1.0)
        bad  = make_microstate(occupancy=0.0, hbm_util=0.0)
        assert fe.compute(bad, 0.5) > fe.compute(good, 0.5)

    def test_zero_temperature_ignores_expressiveness(self):
        """At T=0, F = E_hardware regardless of S_model."""
        fe = FreeEnergy(temperature=0.0)
        state = make_microstate(occupancy=0.5)
        f1 = fe.compute(state, expressiveness_score=0.1)
        f2 = fe.compute(state, expressiveness_score=0.9)
        assert f1 == pytest.approx(f2)

    def test_global_state_accepted(self):
        fe = FreeEnergy(temperature=1.0)
        state = make_global(num_gpus=2, occupancy=0.8)
        f = fe.compute(state, expressiveness_score=0.5)
        assert isinstance(f, float)

    def test_decompose_keys(self):
        fe = FreeEnergy(temperature=1.0)
        state = make_microstate()
        d = fe.decompose(state, 0.5)
        for key in ("E_hardware", "T", "S_model", "T_times_S", "F"):
            assert key in d

    def test_decompose_f_matches_compute(self):
        fe = FreeEnergy(temperature=0.8)
        state = make_microstate(occupancy=0.6)
        d = fe.decompose(state, 0.7)
        assert d["F"] == pytest.approx(fe.compute(state, 0.7))

    def test_t_times_s_correct(self):
        fe = FreeEnergy(temperature=0.5)
        state = make_microstate()
        d = fe.decompose(state, 0.4)
        assert d["T_times_S"] == pytest.approx(0.5 * 0.4)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_empty_input(self):
        fe = FreeEnergy()
        assert fe.pareto_frontier([]) == []

    def test_single_candidate(self):
        fe = FreeEnergy()
        c = {"E_hardware": 0.5, "S_model": 0.8, "name": "A"}
        result = fe.pareto_frontier([c])
        assert result == [c]

    def test_dominated_excluded(self):
        fe = FreeEnergy()
        # B dominates A (lower E, higher S)
        a = {"E_hardware": 0.8, "S_model": 0.4}
        b = {"E_hardware": 0.3, "S_model": 0.9}
        result = fe.pareto_frontier([a, b])
        assert b in result
        assert a not in result

    def test_non_dominated_both_kept(self):
        fe = FreeEnergy()
        # A is better on hardware, B is better on expressiveness
        a = {"E_hardware": 0.1, "S_model": 0.4}
        b = {"E_hardware": 0.9, "S_model": 0.95}
        result = fe.pareto_frontier([a, b])
        assert a in result
        assert b in result

    def test_sorted_by_hardware_energy(self):
        fe = FreeEnergy()
        candidates = [
            {"E_hardware": 0.9, "S_model": 0.9},
            {"E_hardware": 0.1, "S_model": 0.5},
            {"E_hardware": 0.5, "S_model": 0.7},
        ]
        result = fe.pareto_frontier(candidates)
        hw_vals = [c["E_hardware"] for c in result]
        assert hw_vals == sorted(hw_vals)

    def test_three_candidates_one_dominated(self):
        fe = FreeEnergy()
        # a: most expressive, poor hardware  → non-dominated
        # b: best hardware, moderate express → non-dominated
        # c: worse than b on both axes       → dominated by b
        a = {"E_hardware": 0.8, "S_model": 0.95}
        b = {"E_hardware": 0.1, "S_model": 0.5}
        c = {"E_hardware": 0.2, "S_model": 0.4}   # b.E < c.E and b.S > c.S
        result = fe.pareto_frontier([a, b, c])
        assert a in result
        assert b in result
        assert c not in result


# ---------------------------------------------------------------------------
# Temperature annealing
# ---------------------------------------------------------------------------

class TestFreeEnergyAnnealing:
    def test_anneal_decreases_temperature(self):
        fe = FreeEnergy(temperature=1.0,
                        annealing_schedule=GeometricAnnealing(
                            initial_temperature=1.0, factor=0.9))
        t0 = fe.temperature
        fe.anneal()
        assert fe.temperature < t0

    def test_anneal_increments_step(self):
        fe = FreeEnergy()
        assert fe.step == 0
        fe.anneal()
        assert fe.step == 1
        fe.anneal()
        assert fe.step == 2

    def test_reset_restores_initial(self):
        sched = GeometricAnnealing(initial_temperature=2.0, factor=0.8)
        fe = FreeEnergy(temperature=2.0, annealing_schedule=sched)
        for _ in range(5):
            fe.anneal()
        fe.reset()
        assert fe.step == 0
        assert fe.temperature == pytest.approx(sched.temperature_at(0))

    def test_is_cold(self):
        fe = FreeEnergy(temperature=1e-4)
        assert fe.is_cold

    def test_is_not_cold(self):
        fe = FreeEnergy(temperature=1.0)
        assert not fe.is_cold

    def test_is_hot(self):
        fe = FreeEnergy(temperature=0.8)
        assert fe.is_hot

    def test_is_not_hot(self):
        fe = FreeEnergy(temperature=0.1)
        assert not fe.is_hot

    def test_anneal_returns_new_temperature(self):
        fe = FreeEnergy(temperature=1.0,
                        annealing_schedule=GeometricAnnealing(
                            initial_temperature=1.0, factor=0.9))
        new_t = fe.anneal()
        assert new_t == pytest.approx(fe.temperature)

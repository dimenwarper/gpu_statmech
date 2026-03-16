"""
Tests for the energy model.

Key invariants:
  - E_total > 0 for any non-trivial snapshot
  - η_hw ∈ (0, 1)
  - W_hw + Q_waste ≈ E_total
  - Waste breakdown fractions sum to ≤ 1
  - Higher stall_fraction → lower η_hw
  - Higher active_warps → higher η_hw
  - aggregate_energy sums correctly over snapshots
"""

import pytest

from gpu_statmech.energy import (
    EnergyDecomposition,
    EnergyParams,
    aggregate_energy,
    compute_energy,
)


def _snapshot(**overrides) -> dict:
    base = {
        "cycle":             1000,
        "active_warps":      0.6,
        "stall_fraction":    0.2,
        "instr_mix":         {"fp32": 0.8, "mem": 0.2},
        "l2_hit_rate":       0.8,
        "hbm_bw_util":       0.4,
        "smem_util":         0.5,
        "blocks_executed":   256,
        "threads_per_block": 128,
    }
    base.update(overrides)
    return base


def _gpusim_snapshot(**overrides) -> dict:
    base = {
        "cycle": 1,
        "gpu_id": 0,
        "active_sm_id": 0,
        "total_virtual_cycles": 24,
        "warp_state_cycles": {
            "eligible": 192,
            "long_scoreboard": 144,
            "short_scoreboard": 48,
            "barrier": 0,
            "exec_dep": 48,
            "mem_throttle": 48,
            "fetch": 0,
            "idle": 96,
        },
        "sm_active_warps": [24, 0, 0, 0],
        "sm_max_warps": 64,
        "sm_instr_mix": [
            {"fp16": 0.1, "fp32": 0.2, "int": 0.0, "sfu": 0.0, "mem": 0.2, "tensor_core": 0.5}
        ],
        "sm_stall_frac": [0.5, 0.0, 0.0, 0.0],
        "reg_utilization": 0.5,
        "smem_utilization": 0.6,
        "l2_hit_rate": 0.75,
        "hbm_bw_utilization": 0.2,
        "bw_nvlink": 0.0,
    }
    base.update(overrides)
    return base


class TestComputeEnergy:
    def test_total_energy_positive(self):
        e = compute_energy(_snapshot())
        assert e.E_total_nj > 0.0

    def test_eta_in_unit_interval(self):
        e = compute_energy(_snapshot())
        assert 0.0 <= e.eta_hw <= 1.0

    def test_components_sum_to_total(self):
        e = compute_energy(_snapshot())
        component_sum = e.E_compute_nj + e.E_sram_nj + e.E_dram_nj + e.E_leakage_nj
        assert abs(component_sum - e.E_total_nj) < 1e-6 * e.E_total_nj

    def test_waste_breakdown_sums_to_leq_one(self):
        e = compute_energy(_snapshot())
        bd = e.waste_breakdown()
        total = sum(bd.values())
        assert total <= 1.0 + 1e-6

    def test_useful_plus_waste_equals_one(self):
        e = compute_energy(_snapshot())
        bd = e.waste_breakdown()
        assert abs(bd["useful"] + e.waste_fraction - 1.0) < 1e-6

    def test_high_stall_lowers_eta(self):
        e_low  = compute_energy(_snapshot(stall_fraction=0.05))
        e_high = compute_energy(_snapshot(stall_fraction=0.80))
        assert e_low.eta_hw >= e_high.eta_hw

    def test_high_active_warps_raises_eta(self):
        e_low  = compute_energy(_snapshot(active_warps=0.1))
        e_high = compute_energy(_snapshot(active_warps=0.95))
        assert e_high.eta_hw >= e_low.eta_hw

    def test_tensor_core_mix(self):
        snap = _snapshot(instr_mix={"tensor_core": 1.0})
        e = compute_energy(snap)
        assert e.E_total_nj > 0.0
        assert 0.0 <= e.eta_hw <= 1.0

    def test_custom_params(self):
        params = EnergyParams(fp32_mac_pj=1.0)   # artificially expensive compute
        e = compute_energy(_snapshot(), params=params)
        assert e.E_compute_nj > 0.0

    def test_empty_instr_mix_uses_default(self):
        snap = _snapshot(instr_mix={})
        e = compute_energy(snap)
        assert e.E_total_nj > 0.0

    def test_raw_gpusim_snapshot_produces_nonzero_eta(self):
        e = compute_energy(_gpusim_snapshot())
        assert e.E_total_nj > 0.0
        assert e.W_hw_nj > 0.0
        assert 0.0 < e.eta_hw < 1.0


class TestAggregateEnergy:
    def test_empty_returns_zeros(self):
        e = aggregate_energy([])
        assert e.E_total_nj == 0.0
        assert e.W_hw_nj == 0.0

    def test_single_snapshot_matches_compute_energy(self):
        snap = _snapshot()
        e_agg    = aggregate_energy([snap])
        e_single = compute_energy(snap)
        assert abs(e_agg.E_total_nj - e_single.E_total_nj) < 1e-9

    def test_two_identical_snapshots_doubles_energy(self):
        snap = _snapshot()
        e1 = aggregate_energy([snap])
        e2 = aggregate_energy([snap, snap])
        assert abs(e2.E_total_nj - 2 * e1.E_total_nj) < 1e-9

    def test_aggregated_eta_in_unit_interval(self):
        snaps = [_snapshot(active_warps=0.3), _snapshot(active_warps=0.7)]
        e = aggregate_energy(snaps)
        assert 0.0 <= e.eta_hw <= 1.0

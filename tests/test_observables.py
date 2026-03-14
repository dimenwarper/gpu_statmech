import pytest

from gpu_statmech.observables import (
    TraceObservables,
    aggregate_trace_observables,
    canonicalize_snapshot,
)


def _gpusim_snapshot(**overrides) -> dict:
    base = {
        "cycle": 3,
        "gpu_id": 0,
        "sm_active_warps": [16, 32],
        "sm_max_warps": 64,
        "sm_instr_mix": [
            {"fp16": 0.0, "fp32": 1.0, "int": 0.0, "sfu": 0.0, "mem": 0.0, "tensor_core": 0.0},
            {"fp16": 0.5, "fp32": 0.5, "int": 0.0, "sfu": 0.0, "mem": 0.0, "tensor_core": 0.0},
        ],
        "sm_stall_frac": [0.25, 0.50],
        "reg_utilization": 0.40,
        "smem_utilization": 0.60,
        "l2_hit_rate": 0.75,
        "hbm_bw_utilization": 0.20,
        "bw_nvlink": 0.10,
    }
    base.update(overrides)
    return base


class TestCanonicalizeSnapshot:
    def test_flattens_gpusim_snapshot(self):
        snap = canonicalize_snapshot(_gpusim_snapshot())
        assert snap["cycle"] == 1.0
        assert snap["active_warps"] == pytest.approx((16 + 32) / 2 / 64)
        assert snap["stall_fraction"] == pytest.approx((0.25 + 0.50) / 2)
        assert snap["instr_mix"]["fp32"] == pytest.approx(0.75)
        assert snap["instr_mix"]["fp16"] == pytest.approx(0.25)
        assert snap["hbm_bw_util"] == pytest.approx(0.20)
        assert snap["smem_util"] == pytest.approx(0.60)

    def test_passes_through_flat_snapshot(self):
        snap = canonicalize_snapshot({"active_warps": 0.5, "stall_fraction": 0.2})
        assert snap["active_warps"] == pytest.approx(0.5)
        assert snap["stall_fraction"] == pytest.approx(0.2)


class TestAggregateTraceObservables:
    def test_aggregates_flat_snapshots(self):
        obs = aggregate_trace_observables([
            {"active_warps": 0.5, "stall_fraction": 0.2, "l2_hit_rate": 0.8, "hbm_bw_util": 0.3},
            {"active_warps": 0.7, "stall_fraction": 0.1, "l2_hit_rate": 0.6, "hbm_bw_util": 0.5},
        ])
        assert isinstance(obs, TraceObservables)
        assert obs.mean_active_warp_fraction == pytest.approx(0.6)
        assert obs.mean_stall_fraction == pytest.approx(0.15)
        assert obs.mean_issue_activity == pytest.approx(0.6 * 0.85)

    def test_aggregates_gpusim_snapshots(self):
        obs = aggregate_trace_observables([_gpusim_snapshot(), _gpusim_snapshot()])
        assert obs.mean_active_warp_fraction == pytest.approx((16 + 32) / 2 / 64)
        assert obs.mean_stall_fraction == pytest.approx((0.25 + 0.50) / 2)
        assert 0.0 <= obs.memory_feed_efficiency_proxy <= 1.0

    def test_empty_trace(self):
        obs = aggregate_trace_observables([])
        assert obs.n_snapshots == 0
        assert obs.mean_issue_activity == 0.0

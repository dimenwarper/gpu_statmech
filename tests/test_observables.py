import pytest

from gpu_statmech.observables import (
    TraceObservables,
    aggregate_trace_observables,
    canonicalize_snapshot,
    warp_state_family_fractions,
)


def _gpusim_snapshot(**overrides) -> dict:
    base = {
        "cycle": 3,
        "gpu_id": 0,
        "active_sm_id": 0,
        "total_virtual_cycles": 20,
        "warp_state_cycles": {
            "eligible": 96,
            "long_scoreboard": 64,
            "short_scoreboard": 32,
            "barrier": 0,
            "exec_dep": 16,
            "mem_throttle": 48,
            "fetch": 0,
            "idle": 64,
        },
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
        assert snap["cycle"] == 20.0
        assert snap["active_warps"] == pytest.approx(16 / 64)
        assert snap["stall_fraction"] == pytest.approx((64 + 32 + 16 + 48) / (320 - 64))
        assert snap["memory_stall_fraction"] == pytest.approx((64 + 48) / 320)
        assert snap["issue_activity"] == pytest.approx((96 + 0.35 * 16) / 320)
        assert snap["total_warp_cycles"] == pytest.approx(320.0)
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
        assert obs.mean_issue_activity == pytest.approx((0.5 * 0.8 + 0.7 * 0.9) / 2)

    def test_aggregates_gpusim_snapshots(self):
        obs = aggregate_trace_observables([_gpusim_snapshot(), _gpusim_snapshot()])
        assert obs.mean_active_warp_fraction == pytest.approx(16 / 64)
        assert obs.mean_stall_fraction == pytest.approx((64 + 32 + 16 + 48) / (320 - 64))
        assert obs.mean_memory_stall_fraction == pytest.approx((64 + 48) / 320)
        assert obs.mean_issue_activity == pytest.approx((96 + 0.35 * 16) / 320)
        assert obs.mean_warp_state_fractions["eligible"] == pytest.approx(96 / 320)
        assert obs.mean_warp_state_family_fractions["productive"] == pytest.approx(96 / 320)
        assert obs.mean_warp_state_family_fractions["memory"] == pytest.approx((64 + 48) / 320)
        assert 0.0 <= obs.memory_feed_efficiency_proxy <= 1.0

    def test_weights_by_snapshot_duration(self):
        short = _gpusim_snapshot(total_virtual_cycles=10)
        long = _gpusim_snapshot(
            total_virtual_cycles=30,
            warp_state_cycles={
                "eligible": 72,
                "long_scoreboard": 96,
                "short_scoreboard": 0,
                "barrier": 0,
                "exec_dep": 0,
                "mem_throttle": 48,
                "fetch": 0,
                "idle": 24,
            },
        )
        obs = aggregate_trace_observables([short, long])
        expected = ((10 * ((64 + 48) / 320)) + (30 * ((96 + 48) / 240))) / 40
        assert obs.mean_memory_stall_fraction == pytest.approx(expected)

    def test_empty_trace(self):
        obs = aggregate_trace_observables([])
        assert obs.n_snapshots == 0
        assert obs.mean_issue_activity == 0.0


class TestWarpStateFamilies:
    def test_families_normalize_and_collapse_states(self):
        families = warp_state_family_fractions({
            "eligible": 0.3,
            "long_scoreboard": 0.2,
            "short_scoreboard": 0.1,
            "barrier": 0.05,
            "exec_dep": 0.15,
            "mem_throttle": 0.05,
            "fetch": 0.05,
            "idle": 0.1,
        })
        assert abs(sum(families.values()) - 1.0) < 1e-9
        assert families["productive"] == pytest.approx(0.3)
        assert families["dependency"] == pytest.approx(0.25)
        assert families["memory"] == pytest.approx(0.25)
        assert families["sync_frontend"] == pytest.approx(0.10)
        assert families["idle"] == pytest.approx(0.1)

"""
Tests for the thermodynamic analysis module.

Key invariants:
  - analyse_kernel returns η_hw ∈ (0, 1) and η_hw_fraction ∈ (0, 1]
  - Phase classification is one of the four known phases
  - Bottleneck attribution dominant_source is a known waste source
  - Entropy is non-negative
  - High stall → latency-bound phase identified
  - High HBM utilisation → memory-bound phase identified
  - ProtocolThermoAnalysis aggregates correctly
"""

import pytest

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.thermo import (
    BetaInferenceMethod,
    ExecutionPhase,
    KernelThermoAnalysis,
    ProtocolThermoAnalysis,
    analyse_kernel,
    analyse_protocol,
    attribute_bottleneck,
    classify_phase,
    estimate_entropy,
)
from gpu_statmech.energy import compute_energy


KNOWN_PHASES = {
    ExecutionPhase.COMPUTE_BOUND,
    ExecutionPhase.MEMORY_BOUND,
    ExecutionPhase.LATENCY_BOUND,
    ExecutionPhase.MIXED,
}

KNOWN_BOTTLENECKS = {
    "pipeline_stalls",
    "idle_sm_capacity",
    "unnecessary_dram_traffic",
    "sram_overhead",
}


def _snap(**overrides) -> dict:
    base = {
        "cycle":             500,
        "active_warps":      0.5,
        "stall_fraction":    0.3,
        "instr_mix":         {"fp32": 1.0},
        "l2_hit_rate":       0.7,
        "hbm_bw_util":       0.4,
        "smem_util":         0.4,
        "blocks_executed":   128,
        "threads_per_block": 256,
    }
    base.update(overrides)
    return base


def _gpusim_snap(**overrides) -> dict:
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
        "sm_active_warps": [24, 20, 16, 12],
        "sm_max_warps": 64,
        "sm_instr_mix": [{"fp32": 1.0, "fp16": 0.0, "int": 0.0, "sfu": 0.0, "mem": 0.0, "tensor_core": 0.0}] * 4,
        "sm_stall_frac": [0.2, 0.3, 0.2, 0.3],
        "reg_utilization": 0.4,
        "smem_utilization": 0.5,
        "l2_hit_rate": 0.7,
        "hbm_bw_utilization": 0.3,
        "bw_nvlink": 0.0,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def limit():
    return derive_carnot_limit()


# ---------------------------------------------------------------------------
# classify_phase
# ---------------------------------------------------------------------------

class TestClassifyPhase:
    def test_memory_bound(self, limit):
        snap = _snap(hbm_bw_util=0.9)
        assert classify_phase(snap, limit) == ExecutionPhase.MEMORY_BOUND

    def test_latency_bound(self, limit):
        snap = _snap(stall_fraction=0.8, active_warps=0.01)
        assert classify_phase(snap, limit) == ExecutionPhase.LATENCY_BOUND

    def test_compute_bound(self, limit):
        snap = _snap(stall_fraction=0.05, hbm_bw_util=0.1)
        assert classify_phase(snap, limit) == ExecutionPhase.COMPUTE_BOUND

    def test_returns_known_phase(self, limit):
        snap = _snap()
        assert classify_phase(snap, limit) in KNOWN_PHASES


# ---------------------------------------------------------------------------
# estimate_entropy
# ---------------------------------------------------------------------------

class TestEstimateEntropy:
    def test_empty_returns_zero(self):
        assert estimate_entropy([]) == 0.0

    def test_nonneg(self):
        snaps = [_snap(active_warps=0.3), _snap(active_warps=0.7)]
        assert estimate_entropy(snaps) >= 0.0

    def test_single_snap_zero_entropy(self):
        # All snapshots in same bin → zero entropy
        assert estimate_entropy([_snap()]) == 0.0

    def test_diverse_snaps_higher_entropy(self):
        uniform = [_snap(active_warps=0.5, stall_fraction=0.5)] * 8
        diverse = [
            _snap(active_warps=0.1 * i, stall_fraction=0.1 * (9 - i))
            for i in range(1, 9)
        ]
        assert estimate_entropy(diverse) >= estimate_entropy(uniform)


# ---------------------------------------------------------------------------
# analyse_kernel
# ---------------------------------------------------------------------------

class TestAnalyseKernel:
    @pytest.fixture
    def analysis(self, limit) -> KernelThermoAnalysis:
        snaps = [_snap() for _ in range(5)]
        return analyse_kernel("test_kernel", snaps, carnot_limit=limit)

    def test_eta_hw_in_unit_interval(self, analysis):
        assert 0.0 <= analysis.eta_hw <= 1.0

    def test_eta_fraction_in_unit_interval(self, analysis):
        assert 0.0 <= analysis.eta_hw_fraction <= 1.0 + 1e-6

    def test_eta_hw_max_equals_limit(self, analysis, limit):
        assert abs(analysis.eta_hw_max - limit.eta_hw_max) < 1e-9

    def test_dominant_phase_is_known(self, analysis):
        assert analysis.dominant_phase in KNOWN_PHASES

    def test_bottleneck_source_is_known(self, analysis):
        assert analysis.bottleneck.dominant_source in KNOWN_BOTTLENECKS

    def test_entropy_nonneg(self, analysis):
        assert analysis.execution_entropy >= 0.0

    def test_high_stall_lowers_eta(self, limit):
        snaps_low  = [_snap(stall_fraction=0.05)] * 4
        snaps_high = [_snap(stall_fraction=0.90)] * 4
        a_low  = analyse_kernel("low_stall",  snaps_low,  carnot_limit=limit)
        a_high = analyse_kernel("high_stall", snaps_high, carnot_limit=limit)
        assert a_low.eta_hw >= a_high.eta_hw

    def test_empty_snapshots(self, limit):
        analysis = analyse_kernel("empty", [], carnot_limit=limit)
        assert analysis.eta_hw == 0.0

    def test_phase_distribution_sums_to_one(self, analysis):
        total = sum(analysis.phase_distribution.values())
        assert abs(total - 1.0) < 1e-9

    def test_observable_match_is_default(self, analysis):
        assert analysis.beta_inference_method == BetaInferenceMethod.OBSERVABLE_MATCH
        assert analysis.beta_inference_error >= 0.0

    def test_crude_beta_inference_still_available(self, limit):
        analysis = analyse_kernel(
            "crude",
            [_snap()] * 4,
            carnot_limit=limit,
            beta_inference_method=BetaInferenceMethod.CRUDE_WASTE_LOGIT,
        )
        assert analysis.beta_inference_method == BetaInferenceMethod.CRUDE_WASTE_LOGIT
        assert analysis.thermo_state.beta >= 0.01

    def test_accepts_raw_gpusim_snapshot_schema(self, limit):
        analysis = analyse_kernel("gpusim", [_gpusim_snap()] * 4, carnot_limit=limit)
        assert 0.0 <= analysis.observables.mean_active_warp_fraction <= 1.0
        assert 0.0 <= analysis.observables.mean_stall_fraction <= 1.0
        assert 0.0 <= analysis.observables.mean_memory_stall_fraction <= 1.0
        assert analysis.thermo_state.target_activity == pytest.approx(
            analysis.observables.mean_issue_activity,
            abs=1e-3,
        )


# ---------------------------------------------------------------------------
# analyse_protocol
# ---------------------------------------------------------------------------

class TestAnalyseProtocol:
    @pytest.fixture
    def protocol(self, limit) -> ProtocolThermoAnalysis:
        traces = {
            "matmul":    [_snap(hbm_bw_util=0.2, stall_fraction=0.1)] * 4,
            "attention": [_snap(hbm_bw_util=0.8)] * 4,
            "softmax":   [_snap(stall_fraction=0.5, active_warps=0.2)] * 4,
        }
        return analyse_protocol(traces, carnot_limit=limit)

    def test_correct_number_of_kernels(self, protocol):
        assert len(protocol.kernel_analyses) == 3

    def test_protocol_eta_in_unit_interval(self, protocol):
        assert 0.0 <= protocol.eta_hw <= 1.0

    def test_protocol_eta_max_is_min_of_kernels(self, protocol):
        expected = min(k.eta_hw_max for k in protocol.kernel_analyses)
        assert abs(protocol.eta_hw_max - expected) < 1e-9

    def test_dominant_bottleneck_is_known(self, protocol):
        assert protocol.dominant_bottleneck() in KNOWN_BOTTLENECKS

    def test_summary_contains_kernel_names(self, protocol):
        summary = protocol.summary()
        for name in ["matmul", "attention", "softmax"]:
            assert name in summary

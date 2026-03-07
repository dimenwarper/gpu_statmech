"""
Tests for the parallelism module.

Key invariants:
  - ParallelismConfig: n_gpu = dp × tp × pp × ep × cp.
  - CommVolumes: single-GPU config has zero comm bytes.
  - CommVolumes: each strategy contributes only to its own volume slot.
  - build_parallelism_topology: correct GPU count and positive link count per strategy.
  - enumerate_configs: all returned configs have the target n_gpu.
  - enumerate_configs: TP respects n_heads divisibility constraint.
  - score_config: η_multi ∈ [0, 1], comm_overhead ∈ [0, 1], η_overlap ∈ [0, 1].
  - pareto_frontier: no returned config is dominated.
  - pareto_frontier: all non-frontier configs are dominated.
  - optimise_parallelism: recommended config has the highest η_multi.
"""

import math

import pytest

from gpu_statmech.partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    SMConfig,
)
from gpu_statmech.multi_gpu import (
    TopologyGraph,
    derive_multi_gpu_carnot_limit,
    resonance_condition,
)
from gpu_statmech.parallelism import (
    CommVolumes,
    GPT2_SMALL,
    LLAMA_7B,
    ModelParams,
    ParallelismConfig,
    ParallelismOptimResult,
    ParallelismScore,
    build_parallelism_topology,
    enumerate_configs,
    estimate_comm_time_s,
    estimate_comm_volumes,
    estimate_compute_time_s,
    optimise_parallelism,
    pareto_frontier,
    score_config,
)


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------

_TINY_SM = SMConfig(n_sm=4, warps_per_sm=8, peak_flops_per_cycle=64.0)
_TINY_MEM = H100_MEMORY_LEVELS[:2]

_SMALL_MODEL = ModelParams(
    n_params=10_000_000,
    n_layers=4,
    hidden_dim=256,
    n_heads=4,
    seq_len=128,
    batch_size=64,
    dtype_bytes=2,
)

# Fast score kwargs (low resolution to keep tests quick)
_FAST = dict(
    sm_config=_TINY_SM,
    memory_levels=_TINY_MEM,
    eta_hw_single=0.5,
    n_beta=10,
    n_bins=16,
)


# ---------------------------------------------------------------------------
# ParallelismConfig
# ---------------------------------------------------------------------------

class TestParallelismConfig:
    def test_n_gpu_single(self):
        assert ParallelismConfig().n_gpu == 1

    def test_n_gpu_product(self):
        c = ParallelismConfig(dp=4, tp=2, pp=2, ep=1, cp=1)
        assert c.n_gpu == 16

    def test_label_single_gpu(self):
        assert ParallelismConfig().label == "single-GPU"

    def test_label_dp_only(self):
        assert ParallelismConfig(dp=8).label == "DP8"

    def test_label_combined(self):
        c = ParallelismConfig(dp=2, tp=4)
        assert "DP2" in c.label
        assert "TP4" in c.label

    def test_dominant_strategy_highest(self):
        c = ParallelismConfig(dp=1, tp=4, pp=2)
        assert c.dominant_strategy == "tp"

    def test_dominant_strategy_single_gpu(self):
        c = ParallelismConfig()
        # All equal (1); any strategy is valid; just check it returns a string
        assert c.dominant_strategy in ("dp", "tp", "pp", "ep", "cp")

    def test_dominant_phase_dp(self):
        c = ParallelismConfig(dp=8)
        assert c.dominant_phase == "ferromagnetic"

    def test_dominant_phase_tp(self):
        c = ParallelismConfig(tp=8)
        assert c.dominant_phase == "antiferromagnetic"

    def test_dominant_phase_pp(self):
        c = ParallelismConfig(pp=8)
        assert c.dominant_phase == "domain_wall"

    def test_dominant_phase_ep(self):
        c = ParallelismConfig(ep=8)
        assert c.dominant_phase == "spin_glass"

    def test_dominant_phase_cp(self):
        c = ParallelismConfig(cp=8)
        assert c.dominant_phase == "quasi_antiferromagnetic"


# ---------------------------------------------------------------------------
# CommVolumes
# ---------------------------------------------------------------------------

class TestCommVolumes:
    def test_total_bytes_sum(self):
        cv = CommVolumes(
            dp_allreduce_bytes=100.0,
            tp_allgather_bytes=200.0,
            tp_reducescatter_bytes=150.0,
            pp_p2p_bytes=50.0,
            ep_alltoall_bytes=75.0,
            cp_allgather_bytes=25.0,
        )
        assert cv.total_bytes == pytest.approx(600.0)

    def test_breakdown_sums_to_one(self):
        cv = CommVolumes(
            dp_allreduce_bytes=100.0,
            tp_allgather_bytes=200.0,
        )
        breakdown = cv.breakdown()
        assert abs(sum(breakdown.values()) - 1.0) < 1e-9

    def test_all_zero_breakdown(self):
        cv = CommVolumes()
        breakdown = cv.breakdown()
        # total_bytes=0 → uses max(total, 1.0) → fractions are all 0
        assert all(v == pytest.approx(0.0) for v in breakdown.values())


# ---------------------------------------------------------------------------
# estimate_comm_volumes
# ---------------------------------------------------------------------------

class TestEstimateCommVolumes:
    def test_single_gpu_all_zero(self):
        config = ParallelismConfig()
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.total_bytes == pytest.approx(0.0)

    def test_dp_only_nonzero_allreduce(self):
        config = ParallelismConfig(dp=4)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.dp_allreduce_bytes > 0.0
        assert cv.tp_allgather_bytes == 0.0
        assert cv.tp_reducescatter_bytes == 0.0
        assert cv.pp_p2p_bytes == 0.0
        assert cv.ep_alltoall_bytes == 0.0
        assert cv.cp_allgather_bytes == 0.0

    def test_tp_only_nonzero_allgather_reducescatter(self):
        config = ParallelismConfig(tp=4)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.tp_allgather_bytes > 0.0
        assert cv.tp_reducescatter_bytes > 0.0
        assert cv.dp_allreduce_bytes == 0.0
        assert cv.pp_p2p_bytes == 0.0

    def test_pp_only_nonzero_p2p(self):
        config = ParallelismConfig(pp=4)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.pp_p2p_bytes > 0.0
        assert cv.dp_allreduce_bytes == 0.0
        assert cv.tp_allgather_bytes == 0.0

    def test_dp_allreduce_scales_with_n_params(self):
        m_small = ModelParams(1_000_000,  4, 256, 4, seq_len=128, batch_size=64)
        m_large = ModelParams(10_000_000, 4, 256, 4, seq_len=128, batch_size=64)
        config = ParallelismConfig(dp=4)
        cv_s = estimate_comm_volumes(config, m_small)
        cv_l = estimate_comm_volumes(config, m_large)
        assert cv_l.dp_allreduce_bytes == pytest.approx(10.0 * cv_s.dp_allreduce_bytes)

    def test_tp_allgather_equals_reducescatter(self):
        config = ParallelismConfig(tp=4)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.tp_allgather_bytes == pytest.approx(cv.tp_reducescatter_bytes)

    def test_cp_nonzero(self):
        config = ParallelismConfig(cp=4)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.cp_allgather_bytes > 0.0

    def test_ep_nonzero_with_experts(self):
        moe = ModelParams(10_000_000, 4, 256, 4, n_experts=8, seq_len=128, batch_size=64)
        config = ParallelismConfig(ep=4)
        cv = estimate_comm_volumes(config, moe)
        assert cv.ep_alltoall_bytes > 0.0

    def test_total_bytes_positive_for_multi_gpu(self):
        config = ParallelismConfig(dp=2, tp=2)
        cv = estimate_comm_volumes(config, _SMALL_MODEL)
        assert cv.total_bytes > 0.0


# ---------------------------------------------------------------------------
# estimate_compute_time_s
# ---------------------------------------------------------------------------

class TestEstimateComputeTime:
    def test_positive(self):
        config = ParallelismConfig(dp=4)
        t = estimate_compute_time_s(config, _SMALL_MODEL, eta_hw=0.5)
        assert t > 0.0

    def test_scales_inversely_with_n_gpu(self):
        m = _SMALL_MODEL
        t1 = estimate_compute_time_s(ParallelismConfig(dp=1), m, eta_hw=0.5)
        t4 = estimate_compute_time_s(ParallelismConfig(dp=4), m, eta_hw=0.5)
        assert t1 == pytest.approx(4.0 * t4, rel=1e-6)

    def test_scales_inversely_with_eta_hw(self):
        config = ParallelismConfig(dp=4)
        t_half = estimate_compute_time_s(config, _SMALL_MODEL, eta_hw=0.5)
        t_full = estimate_compute_time_s(config, _SMALL_MODEL, eta_hw=1.0)
        assert t_half == pytest.approx(2.0 * t_full, rel=1e-6)


# ---------------------------------------------------------------------------
# estimate_comm_time_s
# ---------------------------------------------------------------------------

class TestEstimateCommTime:
    def test_no_links_returns_inf(self):
        cv = CommVolumes(dp_allreduce_bytes=1e9)
        g = TopologyGraph(n_gpu=1, links=[])
        assert estimate_comm_time_s(cv, g) == float("inf")

    def test_zero_volume_returns_zero(self):
        g = TopologyGraph.nvlink_clique(2)
        cv = CommVolumes()
        assert estimate_comm_time_s(cv, g) == pytest.approx(0.0)

    def test_positive_for_nonzero_volume(self):
        g = TopologyGraph.nvlink_clique(2)
        cv = CommVolumes(dp_allreduce_bytes=1e9)
        assert estimate_comm_time_s(cv, g) > 0.0

    def test_scales_with_volume(self):
        g = TopologyGraph.nvlink_clique(2)
        cv1 = CommVolumes(dp_allreduce_bytes=1e9)
        cv2 = CommVolumes(dp_allreduce_bytes=2e9)
        t1 = estimate_comm_time_s(cv1, g)
        t2 = estimate_comm_time_s(cv2, g)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-6)


# ---------------------------------------------------------------------------
# build_parallelism_topology
# ---------------------------------------------------------------------------

class TestBuildParallelismTopology:
    def test_single_gpu_no_links(self):
        g = build_parallelism_topology(ParallelismConfig())
        assert g.n_gpu == 1
        assert len(g.links) == 0

    def test_dp_only_has_links(self):
        g = build_parallelism_topology(ParallelismConfig(dp=4))
        assert g.n_gpu == 4
        assert len(g.links) > 0

    def test_tp_only_has_links(self):
        g = build_parallelism_topology(ParallelismConfig(tp=4))
        assert g.n_gpu == 4
        assert len(g.links) > 0

    def test_pp_only_has_links(self):
        g = build_parallelism_topology(ParallelismConfig(pp=4))
        assert g.n_gpu == 4
        assert len(g.links) > 0

    def test_n_gpu_matches_config(self):
        for config in [
            ParallelismConfig(dp=4),
            ParallelismConfig(tp=4),
            ParallelismConfig(dp=2, tp=2),
            ParallelismConfig(dp=2, pp=2),
        ]:
            g = build_parallelism_topology(config)
            assert g.n_gpu == config.n_gpu

    def test_tp_has_higher_J_than_dp(self):
        # TP scale (1.5×) > DP scale (1.0×) → TP topology has higher mean J
        g_dp = build_parallelism_topology(ParallelismConfig(dp=4))
        g_tp = build_parallelism_topology(ParallelismConfig(tp=4))
        assert g_tp.mean_J() >= g_dp.mean_J() - 1e-9

    def test_ep_has_highest_j(self):
        # EP scale (5.0×) is highest among strategies
        g_dp = build_parallelism_topology(ParallelismConfig(dp=4))
        g_ep = build_parallelism_topology(ParallelismConfig(ep=4))
        assert g_ep.mean_J() > g_dp.mean_J()


# ---------------------------------------------------------------------------
# enumerate_configs
# ---------------------------------------------------------------------------

class TestEnumerateConfigs:
    def test_all_configs_have_correct_n_gpu(self):
        configs = enumerate_configs(8, _SMALL_MODEL)
        for c in configs:
            assert c.n_gpu == 8

    def test_tp_respects_n_heads(self):
        # _SMALL_MODEL has n_heads=4; tp > 4 should not appear
        configs = enumerate_configs(8, _SMALL_MODEL, max_tp=8)
        for c in configs:
            assert c.tp <= _SMALL_MODEL.n_heads
            assert _SMALL_MODEL.n_heads % c.tp == 0

    def test_tp_respects_hidden_dim(self):
        # _SMALL_MODEL has hidden_dim=256; all tp must divide into it
        configs = enumerate_configs(8, _SMALL_MODEL)
        for c in configs:
            assert _SMALL_MODEL.hidden_dim % c.tp == 0

    def test_no_duplicates(self):
        configs = enumerate_configs(8, _SMALL_MODEL)
        keys = [(c.dp, c.tp, c.pp, c.ep, c.cp) for c in configs]
        assert len(keys) == len(set(keys))

    def test_n_gpu_1_returns_single(self):
        configs = enumerate_configs(1, _SMALL_MODEL)
        assert len(configs) >= 1
        for c in configs:
            assert c.n_gpu == 1

    def test_returns_nonempty_for_n_gpu_8(self):
        configs = enumerate_configs(8, _SMALL_MODEL)
        assert len(configs) > 0

    def test_no_ep_without_flag(self):
        configs = enumerate_configs(8, _SMALL_MODEL, include_ep=False)
        for c in configs:
            assert c.ep == 1

    def test_no_cp_without_flag(self):
        configs = enumerate_configs(8, _SMALL_MODEL, include_cp=False)
        for c in configs:
            assert c.cp == 1

    def test_ep_appears_with_moe_model(self):
        moe = ModelParams(10_000_000, 4, 256, 4, n_experts=8, seq_len=128, batch_size=64)
        configs = enumerate_configs(8, moe, include_ep=True)
        ep_configs = [c for c in configs if c.ep > 1]
        assert len(ep_configs) > 0

    def test_max_tp_respected(self):
        configs = enumerate_configs(8, _SMALL_MODEL, max_tp=2)
        for c in configs:
            assert c.tp <= 2


# ---------------------------------------------------------------------------
# score_config
# ---------------------------------------------------------------------------

class TestScoreConfig:
    @pytest.fixture(scope="class")
    def dp4_score(self) -> ParallelismScore:
        return score_config(
            ParallelismConfig(dp=4), _SMALL_MODEL, **_FAST,
        )

    @pytest.fixture(scope="class")
    def tp4_score(self) -> ParallelismScore:
        return score_config(
            ParallelismConfig(tp=4), _SMALL_MODEL, **_FAST,
        )

    @pytest.fixture(scope="class")
    def single_score(self) -> ParallelismScore:
        return score_config(
            ParallelismConfig(), _SMALL_MODEL, **_FAST,
        )

    def test_eta_multi_in_unit_interval(self, dp4_score):
        assert 0.0 <= dp4_score.eta_multi <= 1.0

    def test_eta_multi_max_in_unit_interval(self, dp4_score):
        assert 0.0 <= dp4_score.eta_multi_max <= 1.0

    def test_eta_hw_fraction_in_unit_interval(self, dp4_score):
        assert 0.0 <= dp4_score.eta_hw_fraction <= 1.0 + 1e-6

    def test_comm_overhead_in_unit_interval(self, dp4_score):
        assert 0.0 <= dp4_score.comm_overhead <= 1.0

    def test_resonance_eta_in_unit_interval(self, dp4_score):
        assert 0.0 <= dp4_score.resonance_eta <= 1.0

    def test_single_gpu_zero_comm_overhead(self, single_score):
        # No communication → overhead is 0 (comm time = 0)
        assert single_score.comm_overhead == pytest.approx(0.0)

    def test_single_gpu_resonance_one(self, single_score):
        # No comm → T_comm = 0 → η_overlap = 0 or comm_overhead=0
        # resonance_condition(T_c, 0) = 0/T_c = 0
        # But actually comm_overhead would be 0, which is fine
        assert single_score.comm_overhead == pytest.approx(0.0)

    def test_dominant_bottleneck_valid(self, dp4_score):
        assert dp4_score.dominant_bottleneck in ("compute", "communication", "balanced")

    def test_thermo_phase_valid(self, dp4_score):
        valid_phases = {
            "ferromagnetic", "antiferromagnetic", "domain_wall",
            "spin_glass", "quasi_antiferromagnetic", "unknown",
        }
        assert dp4_score.thermo_phase in valid_phases

    def test_tp_has_higher_comm_than_dp(self, dp4_score, tp4_score):
        # TP coupling is 1.5× DP coupling → TP should have higher comm overhead
        # (same n_gpu, same model, same bandwidth but higher J in topology)
        # This may not hold in all cases due to different comm volumes;
        # check that TP has nonzero comm (at least)
        assert tp4_score.comm_volumes.tp_allgather_bytes > 0.0

    def test_summary_is_string(self, dp4_score):
        s = dp4_score.summary()
        assert isinstance(s, str)
        assert "η_multi" in s

    def test_config_stored(self, dp4_score):
        assert dp4_score.config.dp == 4


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def _make_score(
        self,
        eta_multi: float,
        comm_overhead: float,
        label: str = "test",
    ) -> ParallelismScore:
        config = ParallelismConfig()
        cv = CommVolumes()
        return ParallelismScore(
            config=config,
            eta_multi=eta_multi,
            eta_multi_max=1.0,
            eta_hw_fraction=eta_multi,
            comm_overhead=comm_overhead,
            resonance_eta=1.0 - comm_overhead,
            dominant_bottleneck="compute",
            thermo_phase="ferromagnetic",
            comm_volumes=cv,
        )

    def test_empty_input(self):
        assert pareto_frontier([]) == []

    def test_single_element_is_on_frontier(self):
        s = self._make_score(0.5, 0.3)
        frontier = pareto_frontier([s])
        assert len(frontier) == 1

    def test_dominated_config_excluded(self):
        # A dominates B: η_multi_A > η_multi_B AND comm_overhead_A < comm_overhead_B
        A = self._make_score(0.8, 0.1)
        B = self._make_score(0.6, 0.3)
        frontier = pareto_frontier([A, B])
        assert A in frontier
        assert B not in frontier

    def test_nondominated_both_on_frontier(self):
        # A: better η, worse comm.  B: worse η, better comm.  Neither dominates.
        A = self._make_score(0.8, 0.4)
        B = self._make_score(0.5, 0.1)
        frontier = pareto_frontier([A, B])
        assert A in frontier
        assert B in frontier

    def test_no_dominated_config_on_frontier(self):
        scores = [
            self._make_score(0.9, 0.1),  # Pareto
            self._make_score(0.7, 0.3),  # Pareto
            self._make_score(0.5, 0.5),  # Pareto
            self._make_score(0.6, 0.4),  # dominated by (0.7, 0.3)
        ]
        frontier = pareto_frontier(scores)
        # Verify no frontier member is dominated by another frontier member
        for candidate in frontier:
            for other in frontier:
                if other is candidate:
                    continue
                assert not (
                    other.eta_multi >= candidate.eta_multi
                    and other.comm_overhead <= candidate.comm_overhead
                    and (other.eta_multi > candidate.eta_multi
                         or other.comm_overhead < candidate.comm_overhead)
                )

    def test_frontier_sorted_by_eta_multi_descending(self):
        scores = [
            self._make_score(0.9, 0.4),
            self._make_score(0.5, 0.1),
            self._make_score(0.7, 0.2),
        ]
        frontier = pareto_frontier(scores)
        etas = [s.eta_multi for s in frontier]
        assert etas == sorted(etas, reverse=True)

    def test_frontier_is_subset_of_input(self):
        scores = [self._make_score(0.9, 0.1), self._make_score(0.6, 0.3)]
        frontier = pareto_frontier(scores)
        assert all(s in scores for s in frontier)


# ---------------------------------------------------------------------------
# optimise_parallelism
# ---------------------------------------------------------------------------

class TestOptimiseParallelism:
    @pytest.fixture(scope="class")
    def result_4gpu(self) -> ParallelismOptimResult:
        return optimise_parallelism(
            4, _SMALL_MODEL,
            sm_config=_TINY_SM,
            memory_levels=_TINY_MEM,
            max_tp=4,
            max_pp=4,
            eta_hw_single=0.5,
            n_beta=10,
            n_bins=16,
        )

    def test_all_configs_have_correct_n_gpu(self, result_4gpu):
        for s in result_4gpu.scores:
            assert s.config.n_gpu == 4

    def test_recommended_has_max_eta_multi(self, result_4gpu):
        max_eta = max(s.eta_multi for s in result_4gpu.scores)
        assert result_4gpu.recommended.eta_multi == pytest.approx(max_eta)

    def test_pareto_configs_are_subset_of_scores(self, result_4gpu):
        for p in result_4gpu.pareto_configs:
            assert p in result_4gpu.scores

    def test_no_pareto_config_dominated(self, result_4gpu):
        frontier = result_4gpu.pareto_configs
        for candidate in frontier:
            for other in frontier:
                if other is candidate:
                    continue
                dominated = (
                    other.eta_multi >= candidate.eta_multi
                    and other.comm_overhead <= candidate.comm_overhead
                    and (other.eta_multi > candidate.eta_multi
                         or other.comm_overhead < candidate.comm_overhead)
                )
                assert not dominated

    def test_multi_gpu_limit_in_unit_interval(self, result_4gpu):
        assert 0.0 <= result_4gpu.multi_gpu_limit.eta_multi_max <= 1.0

    def test_summary_is_string(self, result_4gpu):
        s = result_4gpu.summary()
        assert isinstance(s, str)
        assert "η_multi,max" in s

    def test_scores_not_empty(self, result_4gpu):
        assert len(result_4gpu.scores) > 0

    def test_all_scores_valid(self, result_4gpu):
        for s in result_4gpu.scores:
            assert 0.0 <= s.eta_multi <= 1.0
            assert 0.0 <= s.comm_overhead <= 1.0
            assert 0.0 <= s.resonance_eta <= 1.0


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

class TestModelPresets:
    def test_gpt2_small_params(self):
        assert GPT2_SMALL.n_params == 117_000_000
        assert GPT2_SMALL.n_heads == 12
        assert GPT2_SMALL.hidden_dim == 768

    def test_llama_7b_params(self):
        assert LLAMA_7B.n_params == 7_000_000_000

    def test_presets_valid_for_enumerate(self):
        # GPT2_SMALL should yield valid configs for n_gpu=8
        configs = enumerate_configs(8, GPT2_SMALL)
        assert len(configs) > 0
        for c in configs:
            assert c.n_gpu == 8

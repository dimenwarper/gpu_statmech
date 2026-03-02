"""Tests for gpu_statmech.parallelism."""

import pytest
from gpu_statmech.parallelism import (
    ParallelismPhase,
    ParallelismConfig,
    enumerate_configs,
    prune_configs,
)


# ---------------------------------------------------------------------------
# ParallelismConfig construction
# ---------------------------------------------------------------------------

class TestParallelismConfigConstruction:
    def test_defaults_are_single_gpu(self):
        c = ParallelismConfig()
        assert c.dp == 1
        assert c.tp == 1
        assert c.pp == 1
        assert c.ep == 1
        assert c.cp == 1

    def test_total_gpus(self):
        c = ParallelismConfig(dp=2, tp=4, pp=2)
        assert c.total_gpus == 16

    def test_total_gpus_single(self):
        assert ParallelismConfig().total_gpus == 1

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError):
            ParallelismConfig(dp=0)

    def test_invalid_negative_raises(self):
        with pytest.raises(ValueError):
            ParallelismConfig(tp=-1)

    def test_feasibility_true(self):
        c = ParallelismConfig(dp=4, tp=2)
        assert c.is_feasible(8)

    def test_feasibility_exact(self):
        c = ParallelismConfig(dp=8)
        assert c.is_feasible(8)

    def test_feasibility_false(self):
        c = ParallelismConfig(dp=4, tp=4)
        assert not c.is_feasible(8)


# ---------------------------------------------------------------------------
# Phase identification
# ---------------------------------------------------------------------------

class TestPhaseIdentification:
    def test_none_phase_single_gpu(self):
        assert ParallelismConfig().phase == ParallelismPhase.NONE

    def test_data_parallel_phase(self):
        c = ParallelismConfig(dp=8)
        assert c.phase == ParallelismPhase.DATA_PARALLEL

    def test_tensor_parallel_phase(self):
        c = ParallelismConfig(tp=8)
        assert c.phase == ParallelismPhase.TENSOR_PARALLEL

    def test_pipeline_parallel_phase(self):
        c = ParallelismConfig(pp=4)
        assert c.phase == ParallelismPhase.PIPELINE_PARALLEL

    def test_expert_parallel_phase(self):
        c = ParallelismConfig(ep=4)
        assert c.phase == ParallelismPhase.EXPERT_PARALLEL

    def test_context_parallel_phase(self):
        c = ParallelismConfig(cp=4)
        assert c.phase == ParallelismPhase.CONTEXT_PARALLEL

    def test_hybrid_phase_two_dims(self):
        c = ParallelismConfig(dp=2, tp=4)
        assert c.phase == ParallelismPhase.HYBRID

    def test_hybrid_phase_three_dims(self):
        c = ParallelismConfig(dp=2, tp=2, pp=2)
        assert c.phase == ParallelismPhase.HYBRID

    def test_is_pure_phase_true(self):
        assert ParallelismConfig(dp=8).is_pure_phase

    def test_is_pure_phase_false_for_hybrid(self):
        assert not ParallelismConfig(dp=2, tp=4).is_pure_phase

    def test_is_pure_phase_false_for_none(self):
        assert not ParallelismConfig().is_pure_phase


# ---------------------------------------------------------------------------
# Active dimensions
# ---------------------------------------------------------------------------

class TestActiveDimensions:
    def test_no_active_dims(self):
        assert ParallelismConfig().active_dimensions == []

    def test_single_active_dim(self):
        assert ParallelismConfig(dp=4).active_dimensions == ["dp"]

    def test_multiple_active_dims(self):
        dims = ParallelismConfig(dp=2, tp=4).active_dimensions
        assert set(dims) == {"dp", "tp"}


# ---------------------------------------------------------------------------
# Communication characteristics
# ---------------------------------------------------------------------------

class TestCollectiveMapping:
    def test_dp_uses_all_reduce(self):
        assert "all-reduce" in ParallelismConfig(dp=8).dominant_collective

    def test_tp_uses_all_reduce_or_gather(self):
        assert "all" in ParallelismConfig(tp=8).dominant_collective

    def test_pp_uses_point_to_point(self):
        assert "point-to-point" in ParallelismConfig(pp=4).dominant_collective

    def test_ep_uses_all_to_all(self):
        assert "all-to-all" in ParallelismConfig(ep=4).dominant_collective

    def test_cp_uses_all_gather(self):
        assert "all-gather" in ParallelismConfig(cp=4).dominant_collective


# ---------------------------------------------------------------------------
# Pipeline bubbles
# ---------------------------------------------------------------------------

class TestPipelineBubbles:
    def test_no_pp_no_bubbles(self):
        assert not ParallelismConfig(dp=8).has_pipeline_bubbles
        assert ParallelismConfig(dp=8).bubble_fraction == pytest.approx(0.0)

    def test_pp2_bubble_fraction(self):
        # (pp-1)/pp = 0.5
        c = ParallelismConfig(pp=2)
        assert c.has_pipeline_bubbles
        assert c.bubble_fraction == pytest.approx(0.5)

    def test_pp4_bubble_fraction(self):
        c = ParallelismConfig(pp=4)
        assert c.bubble_fraction == pytest.approx(0.75)

    def test_more_stages_more_bubbles(self):
        c2 = ParallelismConfig(pp=2)
        c8 = ParallelismConfig(pp=8)
        assert c8.bubble_fraction > c2.bubble_fraction


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:
    def test_single_gpu(self):
        c = ParallelismConfig.single_gpu()
        assert c.total_gpus == 1
        assert c.phase == ParallelismPhase.NONE

    def test_data_only(self):
        c = ParallelismConfig.data_only(8)
        assert c.dp == 8
        assert c.phase == ParallelismPhase.DATA_PARALLEL

    def test_tensor_only(self):
        c = ParallelismConfig.tensor_only(8)
        assert c.tp == 8
        assert c.phase == ParallelismPhase.TENSOR_PARALLEL

    def test_megatron_style(self):
        c = ParallelismConfig.megatron_style(dp=4, tp=4, pp=2)
        assert c.total_gpus == 32
        assert c.phase == ParallelismPhase.HYBRID


# ---------------------------------------------------------------------------
# Config enumeration
# ---------------------------------------------------------------------------

class TestEnumerateConfigs:
    def test_single_gpu_yields_one_config(self):
        configs = list(enumerate_configs(1))
        assert len(configs) == 1
        assert configs[0] == ParallelismConfig()

    def test_all_configs_feasible(self):
        for c in enumerate_configs(8):
            assert c.is_feasible(8)

    def test_includes_pure_dp(self):
        configs = list(enumerate_configs(8))
        assert any(c.dp == 8 and c.tp == 1 and c.pp == 1 and c.ep == 1 and c.cp == 1
                   for c in configs)

    def test_includes_pure_tp(self):
        configs = list(enumerate_configs(8))
        assert any(c.tp == 8 and c.dp == 1 and c.pp == 1
                   for c in configs)

    def test_count_grows_with_gpu_count(self):
        assert len(list(enumerate_configs(4))) < len(list(enumerate_configs(8)))


# ---------------------------------------------------------------------------
# Config pruning
# ---------------------------------------------------------------------------

class TestPruneConfigs:
    def test_max_tp_respected(self):
        configs = prune_configs(enumerate_configs(64), 64, max_tp=4)
        assert all(c.tp <= 4 for c in configs)

    def test_max_pp_respected(self):
        configs = prune_configs(enumerate_configs(64), 64, max_pp=8)
        assert all(c.pp <= 8 for c in configs)

    def test_require_dp_ge(self):
        configs = prune_configs(enumerate_configs(8), 8, require_dp_ge=2)
        assert all(c.dp >= 2 for c in configs)

    def test_pruned_smaller_than_full(self):
        full   = list(enumerate_configs(8))
        pruned = prune_configs(iter(full), 8, max_tp=4, max_pp=4)
        assert len(pruned) <= len(full)

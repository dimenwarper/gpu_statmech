"""Tests for gpu_statmech.loop — thermodynamic optimisation loop."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.loop import LoopConfig, LoopState, OptimisationLoop
from gpu_statmech.pareto import ParetoPoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carnot_limit():
    return derive_carnot_limit()


def _small_config(**kwargs) -> LoopConfig:
    defaults = dict(
        n_proposals_per_iter=15,
        eta_threshold=0.1,
        max_iterations=5,
        convergence_tolerance=1e-4,
        patience=3,
        seed=42,
        verbose=False,
    )
    defaults.update(kwargs)
    return LoopConfig(**defaults)


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------

class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.n_proposals_per_iter == 30
        assert cfg.max_iterations == 20
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = LoopConfig(n_proposals_per_iter=5, seed=7)
        assert cfg.n_proposals_per_iter == 5
        assert cfg.seed == 7


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------

class TestLoopState:
    def _dummy_state(self, **kwargs):
        defaults = dict(
            iteration=2,
            all_kernels=[],
            pareto_front=[],
            hypervolume=0.12,
            hypervolume_history=[0.05, 0.10, 0.12],
            best_eta=0.8,
            best_expressiveness=0.7,
            best_combined=1.5,
            converged=False,
        )
        defaults.update(kwargs)
        return LoopState(**defaults)

    def test_summary_string(self):
        s = self._dummy_state()
        summary = s.summary()
        assert "Iter" in summary
        assert "frontier" in summary
        assert "HV" in summary

    def test_n_carnot_optimal_empty(self):
        s = self._dummy_state()
        assert s.n_carnot_optimal == 0

    def test_frontier_size(self):
        s = self._dummy_state()
        assert s.frontier_size == 0

    def test_converged_flag_in_summary(self):
        s = self._dummy_state(converged=True)
        assert "CONVERGED" in s.summary()


# ---------------------------------------------------------------------------
# OptimisationLoop construction
# ---------------------------------------------------------------------------

class TestLoopConstruction:
    def test_constructs_without_error(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        assert loop is not None

    def test_default_config(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit)
        assert loop.config.n_proposals_per_iter == 30

    def test_uses_h100_by_default(self, carnot_limit):
        from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG
        loop = OptimisationLoop(carnot_limit)
        assert loop.memory_levels is H100_MEMORY_LEVELS
        assert loop.sm_config is H100_SM_CONFIG


# ---------------------------------------------------------------------------
# OptimisationLoop.step()
# ---------------------------------------------------------------------------

class TestLoopStep:
    def test_step_returns_loop_state(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config())
        state = loop.step()
        assert isinstance(state, LoopState)

    def test_step_accumulates_kernels(self, carnot_limit):
        cfg  = _small_config(n_proposals_per_iter=10)
        loop = OptimisationLoop(carnot_limit, cfg)
        s1 = loop.step()
        s2 = loop.step()
        assert len(s2.all_kernels) == 20
        assert s2.iteration == 1

    def test_iteration_index_increments(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        s0 = loop.step()
        s1 = loop.step()
        assert s0.iteration == 0
        assert s1.iteration == 1

    def test_hypervolume_nonnegative(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        state = loop.step()
        assert state.hypervolume >= 0.0

    def test_hypervolume_history_grows(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        for i in range(3):
            state = loop.step()
        assert len(state.hypervolume_history) == 3

    def test_pareto_front_is_list_of_pareto_points(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config())
        state = loop.step()
        for pt in state.pareto_front:
            assert isinstance(pt, ParetoPoint)

    def test_best_eta_nondecreasing(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        prev_eta = 0.0
        for _ in range(4):
            state = loop.step()
            assert state.best_eta >= prev_eta
            prev_eta = state.best_eta

    def test_feedback_message_non_empty(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config())
        state = loop.step()
        assert len(state.feedback_message) > 0

    def test_on_iteration_callback(self, carnot_limit):
        called_with = []
        def cb(state):
            called_with.append(state.iteration)

        loop = OptimisationLoop(carnot_limit, _small_config(max_iterations=3), on_iteration=cb)
        loop.run()
        assert len(called_with) >= 1
        assert called_with[0] == 0

    def test_eta_threshold_filters_candidates(self, carnot_limit):
        """With very high threshold, Pareto frontier may be empty."""
        cfg  = _small_config(eta_threshold=0.9999, n_proposals_per_iter=10)
        loop = OptimisationLoop(carnot_limit, cfg)
        state = loop.step()
        # All proposals below 0.9999 → frontier empty (or very small)
        assert state.hypervolume >= 0.0   # should not crash


# ---------------------------------------------------------------------------
# OptimisationLoop.run()
# ---------------------------------------------------------------------------

class TestLoopRun:
    def test_run_returns_loop_state(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config(max_iterations=3))
        state = loop.run()
        assert isinstance(state, LoopState)

    def test_run_respects_max_iterations(self, carnot_limit):
        cfg   = _small_config(max_iterations=4, patience=100)
        loop  = OptimisationLoop(carnot_limit, cfg)
        state = loop.run()
        assert state.iteration == 3   # 0-indexed

    def test_run_n_iterations_override(self, carnot_limit):
        cfg   = _small_config(max_iterations=10, patience=100)
        loop  = OptimisationLoop(carnot_limit, cfg)
        state = loop.run(n_iterations=2)
        assert state.iteration == 1

    def test_convergence_stops_early(self, carnot_limit):
        """Very tight tolerance + low patience should converge quickly."""
        cfg = _small_config(
            max_iterations=20,
            convergence_tolerance=1.0,  # any change < 1.0 triggers convergence
            patience=1,
        )
        loop  = OptimisationLoop(carnot_limit, cfg)
        state = loop.run()
        assert state.converged
        assert state.iteration < 19  # stopped before max

    def test_zero_iterations_returns_empty_state(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config())
        state = loop.run(n_iterations=0)
        assert state.iteration == -1
        assert len(state.all_kernels) == 0

    def test_reproducible_with_same_seed(self, carnot_limit):
        cfg = _small_config(seed=123, max_iterations=3)
        loop1 = OptimisationLoop(carnot_limit, cfg)
        loop2 = OptimisationLoop(carnot_limit, cfg)
        s1 = loop1.run()
        s2 = loop2.run()
        assert s1.hypervolume == pytest.approx(s2.hypervolume, rel=1e-9)
        assert s1.best_eta    == pytest.approx(s2.best_eta,    rel=1e-9)


# ---------------------------------------------------------------------------
# OptimisationLoop.reset()
# ---------------------------------------------------------------------------

class TestLoopReset:
    def test_reset_clears_kernels(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        loop.run(n_iterations=2)
        loop.reset()
        state = loop.step()
        cfg = _small_config()
        assert len(state.all_kernels) == cfg.n_proposals_per_iter

    def test_reset_restores_hv_history(self, carnot_limit):
        loop = OptimisationLoop(carnot_limit, _small_config())
        loop.run(n_iterations=3)
        loop.reset()
        state = loop.step()
        assert len(state.hypervolume_history) == 1


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestLoopDiagnostics:
    def test_convergence_report(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config(max_iterations=3))
        loop.run()
        report = loop.convergence_report()
        assert "Iterations completed" in report
        assert "hypervolume" in report.lower() or "HV" in report

    def test_convergence_report_before_run(self, carnot_limit):
        loop   = OptimisationLoop(carnot_limit, _small_config())
        report = loop.convergence_report()
        assert "No iterations" in report

    def test_best_kernels(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config(max_iterations=3))
        loop.run()
        top5  = loop.best_kernels(n=5)
        assert len(top5) <= 5
        scores = [ck.combined_score for ck in top5]
        assert scores == sorted(scores, reverse=True)

    def test_pareto_report(self, carnot_limit):
        loop  = OptimisationLoop(carnot_limit, _small_config(max_iterations=2))
        loop.run()
        report = loop.pareto_report()
        assert isinstance(report, str)
        assert len(report) > 0

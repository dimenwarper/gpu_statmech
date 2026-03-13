"""Tests for gpu_statmech.pareto — Pareto frontier utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.compiler import KernelCompiler
from gpu_statmech.oracle import PhysicsOracle
from gpu_statmech.pareto import (
    ParetoPoint,
    crowding_distance,
    hypervolume_2d,
    is_dominated,
    pareto_frontier,
    pareto_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carnot_limit():
    return derive_carnot_limit()


@pytest.fixture(scope="module")
def compiled_batch(carnot_limit):
    oracle    = PhysicsOracle(carnot_limit, seed=7)
    compiler  = KernelCompiler(carnot_limit)
    proposals = oracle.propose(n=40, rng=np.random.default_rng(7))
    return compiler.batch_compile(proposals)


def _point(eta: float, expr: float) -> ParetoPoint:
    """Build a minimal ParetoPoint with mock kernel for testing geometry."""
    from gpu_statmech.carnot import check_carnot_optimality, KernelSpec
    from gpu_statmech.compiler import CompiledKernel

    carnot_limit = derive_carnot_limit()
    spec = KernelSpec(
        name="mock",
        arithmetic_intensity=eta * 10,
        working_set={},
        reuse_factors={},
        warp_occupancy=0.5,
        unnecessary_data_movement=0.0,
    )
    from gpu_statmech.carnot import CarnotOptimalityReport, CarnotConditionResult
    report = CarnotOptimalityReport(
        is_carnot_optimal=False,
        conditions=[],
        eta_hw_fraction=eta,
        dominant_bottleneck="none",
    )
    from gpu_statmech.oracle import KernelProposal
    proposal = KernelProposal(
        name="mock",
        block_size=256, grid_size=132,
        registers_per_thread=64, smem_bytes=0,
        arithmetic_intensity=1.0, tensor_core_utilisation=0.5,
        memory_access_pattern="coalesced",
        reuse_factors={"HBM": 1.0},
    )
    ck = CompiledKernel(
        proposal=proposal,
        kernel_spec=spec,
        optimality_report=report,
        expressiveness_score=expr,
        thermo_score=eta,
    )
    return ParetoPoint(kernel=ck, eta_fraction=eta, expressiveness=expr)


# ---------------------------------------------------------------------------
# ParetoPoint
# ---------------------------------------------------------------------------

class TestParetoPoint:
    def test_objectives_tuple(self):
        pt = _point(0.7, 0.6)
        assert pt.objectives == (0.7, 0.6)

    def test_from_compiled(self, compiled_batch):
        ck = compiled_batch[0]
        pt = ParetoPoint.from_compiled(ck)
        assert pt.eta_fraction == pytest.approx(ck.thermo_score)
        assert pt.expressiveness == pytest.approx(ck.expressiveness_score)
        assert pt.kernel is ck


# ---------------------------------------------------------------------------
# is_dominated
# ---------------------------------------------------------------------------

class TestIsDominated:
    def test_clearly_dominated(self):
        good = _point(0.9, 0.9)
        bad  = _point(0.5, 0.5)
        assert is_dominated(bad, good)

    def test_clearly_not_dominated(self):
        good = _point(0.9, 0.9)
        bad  = _point(0.5, 0.5)
        assert not is_dominated(good, bad)

    def test_tradeoff_not_dominated(self):
        """Neither dominates when they trade off objectives."""
        a = _point(0.9, 0.3)
        b = _point(0.3, 0.9)
        assert not is_dominated(a, b)
        assert not is_dominated(b, a)

    def test_equal_points_not_dominated(self):
        """Identical points do not dominate each other."""
        a = _point(0.5, 0.5)
        b = _point(0.5, 0.5)
        assert not is_dominated(a, b)
        assert not is_dominated(b, a)

    def test_strictly_better_one_objective(self):
        """Better on one, equal on other → dominates."""
        a = _point(0.5, 0.6)
        b = _point(0.5, 0.5)
        assert is_dominated(b, a)
        assert not is_dominated(a, b)


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_empty_input(self):
        assert pareto_frontier([]) == []

    def test_single_point(self):
        pts = [_point(0.5, 0.5)]
        front = pareto_frontier(pts)
        assert len(front) == 1

    def test_two_non_dominated(self):
        pts = [_point(0.9, 0.3), _point(0.3, 0.9)]
        front = pareto_frontier(pts)
        assert len(front) == 2

    def test_one_dominated_removed(self):
        good  = _point(0.9, 0.9)
        bad   = _point(0.5, 0.5)
        front = pareto_frontier([good, bad])
        assert len(front) == 1
        assert front[0].eta_fraction == pytest.approx(0.9)

    def test_sorted_by_eta(self):
        pts   = [_point(0.8, 0.4), _point(0.2, 0.9), _point(0.5, 0.6)]
        front = pareto_frontier(pts)
        etas  = [p.eta_fraction for p in front]
        assert etas == sorted(etas)

    def test_all_dominated_except_one(self):
        best = _point(1.0, 1.0)
        rest = [_point(0.1 * i, 0.1 * i) for i in range(9)]
        front = pareto_frontier([best] + rest)
        assert len(front) == 1
        assert front[0].eta_fraction == pytest.approx(1.0)

    def test_compiled_batch_frontier_non_empty(self, compiled_batch):
        pts   = [ParetoPoint.from_compiled(ck) for ck in compiled_batch]
        front = pareto_frontier(pts)
        assert len(front) >= 1
        assert len(front) <= len(pts)

    def test_frontier_subset_of_inputs(self, compiled_batch):
        pts   = [ParetoPoint.from_compiled(ck) for ck in compiled_batch]
        front = pareto_frontier(pts)
        all_etas  = {p.eta_fraction  for p in pts}
        front_etas = {p.eta_fraction for p in front}
        assert front_etas <= all_etas


# ---------------------------------------------------------------------------
# hypervolume_2d
# ---------------------------------------------------------------------------

class TestHypervolume2d:
    def test_empty_frontier(self):
        assert hypervolume_2d([]) == 0.0

    def test_single_point(self):
        pt = _point(0.8, 0.6)
        hv = hypervolume_2d([pt], reference=(0.0, 0.0))
        # staircase: width = 0.8, height = 0.6 → area = 0.48
        assert hv == pytest.approx(0.8 * 0.6, rel=1e-6)

    def test_two_non_dominated(self):
        a  = _point(0.9, 0.3)
        b  = _point(0.3, 0.9)
        hv = hypervolume_2d([a, b], reference=(0.0, 0.0))
        # Exact: sort by eta desc → (0.9, 0.3), (0.3, 0.9)
        # Rectangle 1: width=0.9, height=0.3  → 0.27
        # Rectangle 2: width=0.3, height=(0.9-0.3)=0.6 → 0.18
        expected = 0.9 * 0.3 + 0.3 * 0.6
        assert hv == pytest.approx(expected, rel=1e-6)

    def test_dominated_point_doesnt_increase_hv(self):
        good = _point(0.9, 0.9)
        bad  = _point(0.5, 0.5)
        hv_good_only  = hypervolume_2d([good])
        hv_both       = hypervolume_2d(pareto_frontier([good, bad]))
        assert hv_good_only == pytest.approx(hv_both, rel=1e-9)

    def test_monotone_with_better_point(self):
        front1 = [_point(0.8, 0.6)]
        front2 = [_point(0.8, 0.6), _point(0.3, 0.95)]
        hv1 = hypervolume_2d(front1)
        hv2 = hypervolume_2d(front2)
        assert hv2 >= hv1

    def test_nonnegative(self, compiled_batch):
        pts   = [ParetoPoint.from_compiled(ck) for ck in compiled_batch]
        front = pareto_frontier(pts)
        hv    = hypervolume_2d(front)
        assert hv >= 0.0

    def test_reference_above_front_gives_zero(self):
        pt = _point(0.5, 0.5)
        hv = hypervolume_2d([pt], reference=(1.0, 1.0))
        assert hv == 0.0


# ---------------------------------------------------------------------------
# crowding_distance
# ---------------------------------------------------------------------------

class TestCrowdingDistance:
    def test_empty(self):
        d = crowding_distance([])
        assert len(d) == 0

    def test_single_point_is_inf(self):
        d = crowding_distance([_point(0.5, 0.5)])
        assert d[0] == np.inf

    def test_two_points_both_inf(self):
        pts = [_point(0.3, 0.8), _point(0.8, 0.3)]
        d   = crowding_distance(pts)
        assert all(np.isinf(d))

    def test_boundary_points_are_inf(self):
        pts = [_point(0.1, 0.9), _point(0.5, 0.5), _point(0.9, 0.1)]
        d   = crowding_distance(pts)
        # Boundary in eta: indices 0 and 2 (sorted)
        # At least 2 points should have inf distance
        assert np.sum(np.isinf(d)) >= 2

    def test_interior_point_finite(self):
        pts = [_point(0.1, 0.9), _point(0.5, 0.5), _point(0.9, 0.1)]
        d   = crowding_distance(pts)
        assert any(np.isfinite(d))

    def test_shape_matches_input(self, compiled_batch):
        pts = [ParetoPoint.from_compiled(ck) for ck in compiled_batch[:10]]
        d   = crowding_distance(pts)
        assert d.shape == (10,)
        assert np.all(d >= 0)


# ---------------------------------------------------------------------------
# pareto_summary
# ---------------------------------------------------------------------------

class TestParetoSummary:
    def test_empty_frontier(self):
        s = pareto_summary([])
        assert "empty" in s.lower()

    def test_non_empty_summary(self, compiled_batch):
        pts   = [ParetoPoint.from_compiled(ck) for ck in compiled_batch]
        front = pareto_frontier(pts)
        s     = pareto_summary(front)
        assert "Pareto frontier" in s
        assert "Hypervolume" in s
        assert "Best η" in s

    def test_summary_lists_all_points(self):
        pts   = [_point(0.9, 0.3), _point(0.3, 0.9), _point(0.6, 0.6)]
        front = pareto_frontier(pts)
        s     = pareto_summary(front)
        # All three should appear (none dominates the others)
        assert str(len(front)) in s

    def test_precomputed_hv_used(self):
        pts   = [_point(0.5, 0.5)]
        front = pareto_frontier(pts)
        s     = pareto_summary(front, hypervolume=0.9999)
        assert "0.9999" in s

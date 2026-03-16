"""
Tests for the Carnot limit derivation and optimality checker.

Key invariants:
  - η_hw,max ∈ (0, 1)
  - Roofline ridge point is recovered as a special case (ratio ≈ 1)
  - Naive Carnot efficiency (capacity-unconstrained) ≥ η_hw,max
  - A kernel that satisfies all conditions is marked Carnot-optimal
  - A kernel that violates a condition is not, and the correct bottleneck is named
  - η_hw_fraction is monotone in violation severity
"""

import math

import pytest

from gpu_statmech.carnot import (
    CarnotLimit,
    CarnotOptimalityReport,
    KernelSpec,
    check_carnot_optimality,
    derive_carnot_limit,
    verify_roofline_recovery,
    _naive_carnot_efficiency,
)
from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG


# ---------------------------------------------------------------------------
# derive_carnot_limit
# ---------------------------------------------------------------------------

class TestDeriveCarnotLimit:
    @pytest.fixture(scope="class")
    def limit(self) -> CarnotLimit:
        return derive_carnot_limit()

    def test_eta_in_unit_interval(self, limit):
        assert 0.0 < limit.eta_hw_max < 1.0

    def test_naive_carnot_upper_bound(self, limit):
        # The capacity-constrained η_hw,max must be ≤ the naive (unconstrained) limit
        naive = _naive_carnot_efficiency(H100_MEMORY_LEVELS)
        assert limit.eta_hw_max <= naive + 1e-6

    def test_effective_temperatures_ordered(self, limit):
        # T_reg < T_smem < T_L2 < T_HBM
        temps = [limit.T_eff[lvl.name] for lvl in H100_MEMORY_LEVELS]
        assert all(temps[i] <= temps[i + 1] for i in range(len(temps) - 1))

    def test_roofline_intensity_positive(self, limit):
        assert limit.roofline_intensity > 0.0

    def test_min_occupancy_in_unit_interval(self, limit):
        assert 0.0 < limit.min_warp_occupancy <= 1.0

    def test_beta_optimal_positive(self, limit):
        assert limit.beta_optimal > 0.0

    def test_min_reuse_factors_positive(self, limit):
        for name, reuse in limit.min_reuse_factors.items():
            assert reuse > 0.0, f"Reuse factor for {name} should be positive"

    def test_load_closure_metadata_present(self, limit):
        assert limit.target_activity is not None
        assert math.isfinite(limit.work_field_optimal)
        assert limit.thermo_state.mean_activity == pytest.approx(limit.target_activity, abs=1e-4)


# ---------------------------------------------------------------------------
# verify_roofline_recovery
# ---------------------------------------------------------------------------

class TestRooflineRecovery:
    def test_ratio_close_to_one(self):
        result = verify_roofline_recovery()
        # The Carnot-derived AI minimum should agree with the roofline ridge
        # to within 10% (they are derived from the same hardware parameters)
        assert abs(result["ratio"] - 1.0) < 0.10, (
            f"Roofline recovery ratio = {result['ratio']:.3f}, expected ≈ 1.0"
        )

    def test_eta_hw_max_reported(self):
        result = verify_roofline_recovery()
        assert 0.0 < result["eta_hw_max"] < 1.0

    def test_naive_geq_carnot(self):
        result = verify_roofline_recovery()
        assert result["naive_carnot_efficiency"] >= result["eta_hw_max"] - 1e-6


# ---------------------------------------------------------------------------
# check_carnot_optimality
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def limit() -> CarnotLimit:
    return derive_carnot_limit()


def _ideal_kernel(limit: CarnotLimit) -> KernelSpec:
    """A kernel that satisfies all five Carnot-optimal conditions."""
    hbm = H100_MEMORY_LEVELS[-1]
    smem = H100_MEMORY_LEVELS[1]
    return KernelSpec(
        name="ideal",
        arithmetic_intensity=limit.roofline_intensity * 2.0,   # well above ridge
        working_set={
            "registers": H100_MEMORY_LEVELS[0].capacity_bytes // 2,
            "smem":      smem.capacity_bytes // 2,
        },
        reuse_factors={
            name: val * 2.0
            for name, val in limit.min_reuse_factors.items()
        },
        warp_occupancy=limit.min_warp_occupancy * 1.5,
        unnecessary_data_movement=0.0,
    )


class TestCheckCarnotOptimality:
    def test_ideal_kernel_is_optimal(self, limit):
        kernel = _ideal_kernel(limit)
        report = check_carnot_optimality(kernel, limit)
        assert report.is_carnot_optimal

    def test_ideal_kernel_eta_fraction_high(self, limit):
        kernel = _ideal_kernel(limit)
        report = check_carnot_optimality(kernel, limit)
        assert report.eta_hw_fraction > 0.9

    def test_low_arithmetic_intensity_fails(self, limit):
        kernel = _ideal_kernel(limit)
        kernel.arithmetic_intensity = limit.roofline_intensity * 0.1
        report = check_carnot_optimality(kernel, limit)
        assert not report.is_carnot_optimal
        assert "arithmetic_intensity" in report.dominant_bottleneck

    def test_working_set_overflow_fails(self, limit):
        kernel = _ideal_kernel(limit)
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        kernel.working_set["smem"] = smem_cap * 3   # 3× capacity → spill
        report = check_carnot_optimality(kernel, limit)
        assert not report.is_carnot_optimal
        assert "smem" in report.dominant_bottleneck

    def test_low_reuse_fails(self, limit):
        kernel = _ideal_kernel(limit)
        for k in kernel.reuse_factors:
            kernel.reuse_factors[k] = 0.01   # far below minimum
        report = check_carnot_optimality(kernel, limit)
        assert not report.is_carnot_optimal

    def test_low_occupancy_fails(self, limit):
        kernel = _ideal_kernel(limit)
        kernel.warp_occupancy = limit.min_warp_occupancy * 0.1
        report = check_carnot_optimality(kernel, limit)
        assert not report.is_carnot_optimal
        assert "warp_occupancy" in report.dominant_bottleneck

    def test_unnecessary_movement_fails(self, limit):
        kernel = _ideal_kernel(limit)
        kernel.unnecessary_data_movement = 0.5
        report = check_carnot_optimality(kernel, limit)
        assert not report.is_carnot_optimal
        assert "unnecessary_data_movement" in report.dominant_bottleneck

    def test_eta_fraction_decreases_with_violation(self, limit):
        """Larger violations should produce lower η_hw_fraction."""
        kernel_mild = _ideal_kernel(limit)
        kernel_mild.arithmetic_intensity = limit.roofline_intensity * 0.8

        kernel_severe = _ideal_kernel(limit)
        kernel_severe.arithmetic_intensity = limit.roofline_intensity * 0.1

        r_mild   = check_carnot_optimality(kernel_mild,   limit)
        r_severe = check_carnot_optimality(kernel_severe, limit)
        assert r_mild.eta_hw_fraction >= r_severe.eta_hw_fraction

    def test_all_conditions_present(self, limit):
        kernel = _ideal_kernel(limit)
        report = check_carnot_optimality(kernel, limit)
        names = {c.name for c in report.conditions}
        assert "arithmetic_intensity" in names
        assert "warp_occupancy" in names
        assert "unnecessary_data_movement" in names

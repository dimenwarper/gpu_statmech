"""Tests for gpu_statmech.compiler — KernelProposal → KernelSpec compiler."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.compiler import (
    CompiledKernel,
    KernelCompiler,
    expressiveness_score,
    warp_occupancy,
    working_set,
)
from gpu_statmech.oracle import KernelProposal, PhysicsOracle
from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carnot_limit():
    return derive_carnot_limit()


@pytest.fixture(scope="module")
def compiler(carnot_limit):
    return KernelCompiler(carnot_limit)


def _make_proposal(**kwargs) -> KernelProposal:
    defaults = dict(
        name="test",
        block_size=256,
        grid_size=132,
        registers_per_thread=64,
        smem_bytes=49_152,
        arithmetic_intensity=10.0,
        tensor_core_utilisation=0.8,
        memory_access_pattern="coalesced",
        reuse_factors={"smem": 16.0, "L2": 4.0, "HBM": 10.0},
        unnecessary_data_movement=0.0,
    )
    defaults.update(kwargs)
    return KernelProposal(**defaults)


# ---------------------------------------------------------------------------
# warp_occupancy
# ---------------------------------------------------------------------------

class TestWarpOccupancy:
    def test_returns_fraction(self, carnot_limit):
        p = _make_proposal(block_size=256, registers_per_thread=64, smem_bytes=0)
        occ = warp_occupancy(p)
        assert 0.0 < occ <= 1.0

    def test_high_register_pressure_lowers_occupancy(self):
        """High register usage should yield lower occupancy."""
        p_low  = _make_proposal(registers_per_thread=32,  smem_bytes=0)
        p_high = _make_proposal(registers_per_thread=128, smem_bytes=0)
        occ_low  = warp_occupancy(p_low)
        occ_high = warp_occupancy(p_high)
        assert occ_low >= occ_high

    def test_high_smem_lowers_occupancy(self):
        """Large SMEM allocation should yield lower occupancy."""
        p_small = _make_proposal(smem_bytes=4_096,   registers_per_thread=32)
        p_large = _make_proposal(smem_bytes=114_688, registers_per_thread=32)
        occ_small = warp_occupancy(p_small)
        occ_large = warp_occupancy(p_large)
        assert occ_small >= occ_large

    def test_zero_smem_not_penalised(self):
        """Zero SMEM should not penalise occupancy (uses register limit only)."""
        p = _make_proposal(smem_bytes=0, registers_per_thread=32)
        occ = warp_occupancy(p)
        assert occ > 0.5   # should be near max with 32 regs/thread

    def test_max_occupancy_leq_one(self):
        p = _make_proposal(registers_per_thread=32, smem_bytes=0, block_size=64)
        occ = warp_occupancy(p)
        assert occ <= 1.0

    def test_small_block_can_reach_full_occupancy(self):
        """With small block + 32 regs, occupancy should equal 1.0."""
        # 32 regs/thread × 32 threads/warp = 1024 regs/warp
        # 65536 / 1024 = 64 warps (H100 max) → occupancy = 1.0
        p = _make_proposal(block_size=64, registers_per_thread=32, smem_bytes=0)
        occ = warp_occupancy(p)
        assert occ == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# working_set
# ---------------------------------------------------------------------------

class TestWorkingSet:
    def test_returns_dict(self):
        p = _make_proposal()
        ws = working_set(p)
        assert isinstance(ws, dict)

    def test_registers_key_present(self):
        p = _make_proposal()
        ws = working_set(p)
        assert "registers" in ws
        assert ws["registers"] > 0

    def test_smem_key_present_when_nonzero(self):
        p = _make_proposal(smem_bytes=32_768)
        ws = working_set(p)
        assert "smem" in ws
        assert ws["smem"] == 32_768

    def test_smem_key_absent_when_zero(self):
        p = _make_proposal(smem_bytes=0)
        ws = working_set(p)
        assert "smem" not in ws

    def test_register_footprint_scales_with_regs(self):
        p32  = _make_proposal(registers_per_thread=32,  smem_bytes=0)
        p128 = _make_proposal(registers_per_thread=128, smem_bytes=0)
        ws32  = working_set(p32)
        ws128 = working_set(p128)
        assert ws128["registers"] > ws32["registers"]


# ---------------------------------------------------------------------------
# expressiveness_score
# ---------------------------------------------------------------------------

class TestExpressivenessScore:
    def test_in_range(self, carnot_limit):
        p = _make_proposal()
        s = expressiveness_score(p, carnot_limit)
        assert 0.0 <= s <= 1.0

    def test_coalesced_beats_random(self, carnot_limit):
        p_coal = _make_proposal(memory_access_pattern="coalesced")
        p_rand = _make_proposal(memory_access_pattern="random")
        assert expressiveness_score(p_coal, carnot_limit) > expressiveness_score(p_rand, carnot_limit)

    def test_high_tc_util_raises_score(self, carnot_limit):
        p_low  = _make_proposal(tensor_core_utilisation=0.0)
        p_high = _make_proposal(tensor_core_utilisation=1.0)
        assert expressiveness_score(p_high, carnot_limit) > expressiveness_score(p_low, carnot_limit)

    def test_high_ai_raises_score(self, carnot_limit):
        p_lo = _make_proposal(arithmetic_intensity=0.1, reuse_factors={"smem": 1.0, "L2": 1.0, "HBM": 0.1})
        p_hi = _make_proposal(arithmetic_intensity=100.0, reuse_factors={"smem": 100.0, "L2": 50.0, "HBM": 100.0})
        assert expressiveness_score(p_hi, carnot_limit) > expressiveness_score(p_lo, carnot_limit)

    def test_low_reuse_lowers_score(self, carnot_limit):
        p_lo = _make_proposal(reuse_factors={"smem": 1.0, "L2": 1.0, "HBM": 1.0})
        p_hi = _make_proposal(reuse_factors={"smem": 64.0, "L2": 16.0, "HBM": 16.0})
        assert expressiveness_score(p_hi, carnot_limit) > expressiveness_score(p_lo, carnot_limit)

    def test_unnecessary_movement_lowers_score(self, carnot_limit):
        p_clean = _make_proposal(unnecessary_data_movement=0.0)
        p_dirty = _make_proposal(unnecessary_data_movement=0.8)
        assert expressiveness_score(p_clean, carnot_limit) > expressiveness_score(p_dirty, carnot_limit)

    def test_saturates_above_ridge(self, carnot_limit):
        """AI score should saturate once AI >> ridge."""
        ridge = carnot_limit.roofline_intensity
        p_at  = _make_proposal(arithmetic_intensity=ridge,      reuse_factors={"smem": 1.0, "L2": 1.0, "HBM": ridge})
        p_way = _make_proposal(arithmetic_intensity=ridge * 100, reuse_factors={"smem": 1.0, "L2": 1.0, "HBM": ridge * 100})
        # Both should have the same AI sub-score (saturated at 1)
        s_at  = expressiveness_score(p_at,  carnot_limit)
        s_way = expressiveness_score(p_way, carnot_limit)
        # s_way should equal s_at (both fully compute-bound)
        assert abs(s_way - s_at) < 1e-9

    def test_custom_weights_sum_to_one(self, carnot_limit):
        """Custom weights that don't sum to 1 still return a valid score."""
        p = _make_proposal()
        s = expressiveness_score(p, carnot_limit, w_tc=0.25, w_ai=0.25, w_acc=0.25, w_occ=0.25)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# KernelCompiler.compile()
# ---------------------------------------------------------------------------

class TestKernelCompilerCompile:
    def test_returns_compiled_kernel(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert isinstance(ck, CompiledKernel)

    def test_proposal_preserved(self, compiler):
        p = _make_proposal(name="my_kernel")
        ck = compiler.compile(p)
        assert ck.proposal is p

    def test_kernel_spec_ai(self, compiler):
        p = _make_proposal(arithmetic_intensity=42.0)
        ck = compiler.compile(p)
        assert ck.kernel_spec.arithmetic_intensity == pytest.approx(42.0)

    def test_thermo_score_in_range(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert 0.0 <= ck.thermo_score <= 1.0

    def test_expressiveness_in_range(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert 0.0 <= ck.expressiveness_score <= 1.0

    def test_combined_score_is_sum(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert ck.combined_score == pytest.approx(ck.thermo_score + ck.expressiveness_score)

    def test_architecture_score_alias(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert ck.architecture_score == pytest.approx(ck.expressiveness_score)

    def test_is_carnot_optimal_property(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert isinstance(ck.is_carnot_optimal, bool)

    def test_dominant_bottleneck_is_string(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert isinstance(ck.dominant_bottleneck, str)

    def test_warp_occupancy_in_kernel_spec(self, compiler):
        p = _make_proposal()
        ck = compiler.compile(p)
        assert 0.0 <= ck.kernel_spec.warp_occupancy <= 1.0

    def test_udm_passed_through(self, compiler):
        p = _make_proposal(unnecessary_data_movement=0.25)
        ck = compiler.compile(p)
        assert ck.kernel_spec.unnecessary_data_movement == pytest.approx(0.25)

    def test_carnot_optimal_high_ai(self, carnot_limit):
        """A kernel far above the roofline with good reuse should score well."""
        ridge = carnot_limit.roofline_intensity
        p = _make_proposal(
            arithmetic_intensity=ridge * 20,
            tensor_core_utilisation=1.0,
            memory_access_pattern="coalesced",
            reuse_factors={"smem": 1e6, "L2": 1e4, "HBM": ridge * 20},
            unnecessary_data_movement=0.0,
            registers_per_thread=32,
            smem_bytes=0,
        )
        compiler = KernelCompiler(carnot_limit)
        ck = compiler.compile(p)
        assert ck.thermo_score > 0.0
        assert ck.expressiveness_score > 0.0


# ---------------------------------------------------------------------------
# KernelCompiler.batch_compile()
# ---------------------------------------------------------------------------

class TestBatchCompile:
    def test_batch_length(self, compiler, carnot_limit):
        oracle = PhysicsOracle(carnot_limit, seed=1)
        proposals = oracle.propose(n=15, rng=np.random.default_rng(1))
        results = compiler.batch_compile(proposals)
        assert len(results) == 15

    def test_batch_order_preserved(self, compiler, carnot_limit):
        oracle = PhysicsOracle(carnot_limit, seed=2)
        proposals = oracle.propose(n=5, rng=np.random.default_rng(2))
        results = compiler.batch_compile(proposals)
        for p, ck in zip(proposals, results):
            assert ck.proposal is p


# ---------------------------------------------------------------------------
# Waste attribution and feedback message
# ---------------------------------------------------------------------------

class TestWasteAttribution:
    def test_perfect_kernel_no_attribution(self, carnot_limit):
        """A Carnot-optimal kernel should have empty waste attribution."""
        ridge = carnot_limit.roofline_intensity
        # Build a proposal that should pass all five conditions
        p = _make_proposal(
            arithmetic_intensity=ridge * 50,
            reuse_factors={
                "smem": 1e7, "L2": 1e5, "HBM": ridge * 50,
            },
            unnecessary_data_movement=0.0,
            registers_per_thread=32,
            smem_bytes=0,
        )
        compiler = KernelCompiler(carnot_limit)
        ck = compiler.compile(p)
        if ck.is_carnot_optimal:
            diag = compiler.waste_attribution(ck)
            assert len(diag) == 0

    def test_bad_kernel_has_attribution(self, compiler):
        """A clearly sub-optimal kernel should have non-empty attribution."""
        p = _make_proposal(
            arithmetic_intensity=0.001,  # far below roofline
            reuse_factors={"smem": 0.01, "L2": 0.01, "HBM": 0.001},
            unnecessary_data_movement=0.9,
            registers_per_thread=255,
            smem_bytes=H100_MEMORY_LEVELS[1].capacity_bytes,
        )
        ck = compiler.compile(p)
        diag = compiler.waste_attribution(ck)
        assert len(diag) > 0
        for name, msg in diag.items():
            assert "VIOLATION" in msg
            assert "Remedy" in msg

    def test_feedback_message_non_empty(self, compiler, carnot_limit):
        oracle = PhysicsOracle(carnot_limit, seed=3)
        proposals = oracle.propose(n=10, rng=np.random.default_rng(3))
        compiled = compiler.batch_compile(proposals)
        msg = compiler.feedback_message(compiled)
        assert len(msg) > 0
        assert "Batch size" in msg

    def test_feedback_message_empty_batch(self, compiler):
        msg = compiler.feedback_message([])
        assert "No proposals" in msg

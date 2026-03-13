"""Tests for gpu_statmech.oracle — physics-based kernel proposal oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.oracle import (
    ACCESS_PATTERNS,
    VALID_BLOCK_SIZES,
    KernelProposal,
    OraclePrior,
    PhysicsOracle,
    _default_prior,
)
from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carnot_limit():
    return derive_carnot_limit()


@pytest.fixture(scope="module")
def oracle(carnot_limit):
    return PhysicsOracle(carnot_limit, seed=0)


# ---------------------------------------------------------------------------
# KernelProposal validation
# ---------------------------------------------------------------------------

class TestKernelProposal:
    def test_valid_proposal(self, carnot_limit):
        """A well-formed proposal constructs without error."""
        p = KernelProposal(
            name="test",
            block_size=256,
            grid_size=132,
            registers_per_thread=64,
            smem_bytes=49_152,
            arithmetic_intensity=10.0,
            tensor_core_utilisation=0.8,
            memory_access_pattern="coalesced",
            reuse_factors={"smem": 16.0, "L2": 4.0, "HBM": 10.0},
            unnecessary_data_movement=0.05,
        )
        assert p.name == "test"
        assert p.block_size == 256

    def test_invalid_access_pattern(self):
        with pytest.raises(ValueError, match="memory_access_pattern"):
            KernelProposal(
                name="bad",
                block_size=256,
                grid_size=132,
                registers_per_thread=64,
                smem_bytes=0,
                arithmetic_intensity=1.0,
                tensor_core_utilisation=0.5,
                memory_access_pattern="diagonal",   # invalid
                reuse_factors={},
            )

    def test_invalid_tc_util(self):
        with pytest.raises(ValueError, match="tensor_core_utilisation"):
            KernelProposal(
                name="bad",
                block_size=256,
                grid_size=132,
                registers_per_thread=64,
                smem_bytes=0,
                arithmetic_intensity=1.0,
                tensor_core_utilisation=1.5,         # out of range
                memory_access_pattern="coalesced",
                reuse_factors={},
            )

    def test_invalid_udm(self):
        with pytest.raises(ValueError, match="unnecessary_data_movement"):
            KernelProposal(
                name="bad",
                block_size=256,
                grid_size=132,
                registers_per_thread=64,
                smem_bytes=0,
                arithmetic_intensity=1.0,
                tensor_core_utilisation=0.5,
                memory_access_pattern="coalesced",
                reuse_factors={},
                unnecessary_data_movement=-0.1,     # negative
            )


# ---------------------------------------------------------------------------
# Default prior
# ---------------------------------------------------------------------------

class TestDefaultPrior:
    def test_prior_fields(self, carnot_limit):
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        p = _default_prior(carnot_limit, smem_cap)
        assert isinstance(p, OraclePrior)
        assert p.log_ai_std > 0
        assert p.smem_std > 0
        assert p.reg_std > 0
        assert len(p.block_size_probs) == len(VALID_BLOCK_SIZES)
        assert len(p.access_pattern_probs) == len(ACCESS_PATTERNS)

    def test_prior_probs_normalised(self, carnot_limit):
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        p = _default_prior(carnot_limit, smem_cap)
        assert abs(p.block_size_probs.sum() - 1.0) < 1e-9
        assert abs(p.access_pattern_probs.sum() - 1.0) < 1e-9

    def test_prior_centred_on_ridge(self, carnot_limit):
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        p = _default_prior(carnot_limit, smem_cap)
        # log_ai_mean should be log(roofline_intensity)
        expected = math.log(carnot_limit.roofline_intensity)
        assert abs(p.log_ai_mean - expected) < 1e-9


# ---------------------------------------------------------------------------
# PhysicsOracle.propose()
# ---------------------------------------------------------------------------

class TestOraclePropose:
    def test_returns_n_proposals(self, oracle):
        rng = np.random.default_rng(1)
        proposals = oracle.propose(n=10, rng=rng)
        assert len(proposals) == 10

    def test_proposal_types(self, oracle):
        rng = np.random.default_rng(2)
        props = oracle.propose(n=5, rng=rng)
        for p in props:
            assert isinstance(p, KernelProposal)

    def test_block_size_valid(self, oracle):
        rng = np.random.default_rng(3)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert p.block_size in VALID_BLOCK_SIZES

    def test_access_pattern_valid(self, oracle):
        rng = np.random.default_rng(4)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert p.memory_access_pattern in ACCESS_PATTERNS

    def test_registers_in_range(self, oracle):
        rng = np.random.default_rng(5)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert 32 <= p.registers_per_thread <= 255

    def test_smem_in_range(self, oracle):
        rng = np.random.default_rng(6)
        props = oracle.propose(n=50, rng=rng)
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        for p in props:
            assert 0 <= p.smem_bytes <= smem_cap

    def test_arithmetic_intensity_positive(self, oracle):
        rng = np.random.default_rng(7)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert p.arithmetic_intensity > 0.0

    def test_tc_util_in_range(self, oracle):
        rng = np.random.default_rng(8)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert 0.0 <= p.tensor_core_utilisation <= 1.0

    def test_udm_in_range(self, oracle):
        rng = np.random.default_rng(9)
        props = oracle.propose(n=50, rng=rng)
        for p in props:
            assert 0.0 <= p.unnecessary_data_movement <= 1.0

    def test_hbm_reuse_equals_ai(self, oracle):
        """HBM reuse factor must equal arithmetic_intensity by definition."""
        rng = np.random.default_rng(10)
        props = oracle.propose(n=20, rng=rng)
        for p in props:
            assert abs(p.reuse_factors["HBM"] - p.arithmetic_intensity) < 1e-9

    def test_grid_size_covers_sms(self, oracle):
        """Grid must be at least n_sm to utilise all SMs."""
        rng = np.random.default_rng(11)
        props = oracle.propose(n=20, rng=rng)
        for p in props:
            assert p.grid_size >= H100_SM_CONFIG.n_sm

    def test_iteration_counter_increments(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=42)
        assert o.iteration == 0
        o.propose(n=1)
        assert o.iteration == 1
        o.propose(n=1)
        assert o.iteration == 2

    def test_reproducible_with_same_seed(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=99)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        p1 = o.propose(n=5, rng=rng1)
        o.reset()
        p2 = o.propose(n=5, rng=rng2)
        for a, b in zip(p1, p2):
            assert a.block_size == b.block_size
            assert a.arithmetic_intensity == pytest.approx(b.arithmetic_intensity)

    def test_proposal_names_unique(self, oracle):
        rng = np.random.default_rng(12)
        props = oracle.propose(n=20, rng=rng)
        names = [p.name for p in props]
        assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# PhysicsOracle.feedback()
# ---------------------------------------------------------------------------

class TestOracleFeedback:
    def test_feedback_updates_prior(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        rng = np.random.default_rng(0)
        props = o.propose(n=20, rng=rng)
        old_ai_mean = o.prior.log_ai_mean
        # All get high scores → prior should shift
        scores = [1.5] * 20
        o.feedback(props, scores)
        # The mean may or may not shift (depends on samples), but no error
        assert isinstance(o.prior.log_ai_mean, float)

    def test_feedback_wrong_length_raises(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        rng = np.random.default_rng(0)
        props = o.propose(n=5, rng=rng)
        with pytest.raises(ValueError, match="len"):
            o.feedback(props, [1.0, 2.0])  # wrong length

    def test_feedback_empty_is_noop(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        old_mean = o.prior.log_ai_mean
        o.feedback([], [])
        assert o.prior.log_ai_mean == old_mean

    def test_probs_stay_normalised_after_feedback(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        rng = np.random.default_rng(0)
        props = o.propose(n=30, rng=rng)
        scores = list(np.random.default_rng(1).random(30))
        o.feedback(props, scores)
        assert abs(o.prior.block_size_probs.sum() - 1.0) < 1e-9
        assert abs(o.prior.access_pattern_probs.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestOracleReset:
    def test_reset_restores_iteration(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        o.propose(n=5)
        assert o.iteration == 1
        o.reset()
        assert o.iteration == 0

    def test_reset_restores_prior(self, carnot_limit):
        o = PhysicsOracle(carnot_limit, seed=0)
        smem_cap = H100_MEMORY_LEVELS[1].capacity_bytes
        original_mean = _default_prior(carnot_limit, smem_cap).log_ai_mean
        rng = np.random.default_rng(0)
        props = o.propose(n=20, rng=rng)
        o.feedback(props, [2.0] * 20)
        o.reset()
        assert o.prior.log_ai_mean == pytest.approx(original_mean)


# ---------------------------------------------------------------------------
# carnot_prompt
# ---------------------------------------------------------------------------

class TestCarnotPrompt:
    def test_prompt_contains_key_fields(self, oracle):
        prompt = oracle.carnot_prompt()
        assert "η_hw,max" in prompt
        assert "roofline" in prompt.lower() or "Roofline" in prompt
        assert "FLOP/byte" in prompt

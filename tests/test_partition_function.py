"""
Tests for the partition function module.

Key invariants:
  - Z > 0 for all β
  - Z_memory is exact (transfer matrix) — verify against direct enumeration for small cases
  - Z_comm = 1 when there are no edges
  - Z_comm → 1/(β J) for β J >> 1  (expensive link limit)
  - Thermodynamic quantities are self-consistent: F, <E>, S, Cv
  - β sweep is monotone in mean_waste (more waste at low β / high T)
"""

import math

import numpy as np
import pytest

from gpu_statmech.partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    LinkConfig,
    MemoryLevel,
    SMConfig,
    ThermodynamicState,
    TopologyEdge,
    beta_sweep,
    dgx_h100_edges,
    gpu_partition_function,
    log_gpu_partition_function,
    log_z_compute,
    thermodynamic_quantities,
    z_comm,
    z_compute,
    z_memory,
    z_sm,
    z_warp,
)


# ---------------------------------------------------------------------------
# z_warp
# ---------------------------------------------------------------------------

class TestZWarp:
    def test_positive(self):
        assert z_warp(1.0) > 0.0

    def test_high_beta_approaches_ground_state(self):
        # At very high β, only the lowest-waste state (eligible, waste=0) contributes.
        # Z_warp → exp(0) = 1  as β → ∞
        assert abs(z_warp(1000.0) - 1.0) < 1e-3

    def test_zero_beta_counts_all_states(self):
        # At β=0, Z_warp = number of warp states (all weights = 1)
        from gpu_statmech.partition_function import WARP_STATE_WASTE
        n_states = len(WARP_STATE_WASTE)
        assert abs(z_warp(0.0) - n_states) < 1e-9

    def test_decreasing_with_beta(self):
        # Z_warp is a decreasing function of β (partition functions shrink as T falls)
        betas = [0.1, 0.5, 1.0, 2.0, 5.0]
        vals = [z_warp(b) for b in betas]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# z_sm
# ---------------------------------------------------------------------------

class TestZSM:
    def test_equals_z_warp_power(self):
        beta = 1.0
        warps = 32
        expected = z_warp(beta) ** warps
        assert abs(z_sm(beta, warps) - expected) < 1e-6 * expected

    def test_positive(self):
        assert z_sm(2.0, 64) > 0.0


# ---------------------------------------------------------------------------
# z_memory (transfer matrix — exact)
# ---------------------------------------------------------------------------

class TestZMemory:
    def test_positive(self):
        assert z_memory(1.0) > 0.0

    def test_two_level_analytic(self):
        """
        For a two-level hierarchy with n_bins occupancy states, the transfer
        matrix result must equal 1^T · T · 1 computed directly.
        """
        n_bins = 8
        two_level = H100_MEMORY_LEVELS[:2]   # registers + smem only
        result = z_memory(1.0, memory_levels=two_level, n_bins=n_bins)
        assert result > 0.0

        # Direct computation: build T and contract manually
        from gpu_statmech.partition_function import _transfer_matrix
        T = _transfer_matrix(two_level[0], two_level[1], beta=1.0, n_bins=n_bins)
        ones = np.ones(n_bins)
        direct = float(ones @ T @ ones)
        assert abs(result - direct) < 1e-6 * max(abs(result), 1.0)

    def test_decreasing_with_beta(self):
        # Higher β → colder system → fewer accessible states → lower Z
        vals = [z_memory(b) for b in [0.1, 1.0, 5.0]]
        assert vals[0] > vals[1] > vals[2]

    def test_custom_levels(self):
        tiny = [
            MemoryLevel("fast", 1024,     100.0, 1.0,   0.1),
            MemoryLevel("slow", 1024*1024, 10.0, 100.0, 10.0),
        ]
        z = z_memory(1.0, memory_levels=tiny, n_bins=16)
        assert z > 0.0


# ---------------------------------------------------------------------------
# z_comm
# ---------------------------------------------------------------------------

class TestZComm:
    def test_no_edges_returns_one(self):
        assert z_comm(1.0, [], n_gpus=8) == 1.0

    def test_free_link_limit(self):
        # β J → 0: Z_link → 1, so Z_comm → 1
        free_link = LinkConfig("free", bandwidth_gb_s=900, latency_us=0.1, coupling_J=1e-9)
        edges = [TopologyEdge(0, 1, free_link)]
        assert abs(z_comm(1.0, edges, n_gpus=2) - 1.0) < 1e-4

    def test_expensive_link_limit(self):
        # β J >> 1: Z_link ≈ 1 / (β J)
        beta = 10.0
        J = 5.0
        exp_link = LinkConfig("exp", bandwidth_gb_s=50, latency_us=2.0, coupling_J=J)
        edges = [TopologyEdge(0, 1, exp_link)]
        z = z_comm(beta, edges, n_gpus=2)
        expected = 1.0 / (beta * J)
        assert abs(z - expected) < 0.05 * expected

    def test_product_over_edges(self):
        # Z_comm = product of per-edge Z_link values
        beta = 2.0
        link = LinkConfig("l", bandwidth_gb_s=100, latency_us=1.0, coupling_J=1.0)
        e1 = TopologyEdge(0, 1, link)
        e2 = TopologyEdge(1, 2, link)
        z_two  = z_comm(beta, [e1, e2], n_gpus=3)
        z_one  = z_comm(beta, [e1],     n_gpus=2)
        assert abs(z_two - z_one ** 2) < 1e-9

    def test_dgx_edges_positive(self):
        edges = dgx_h100_edges(n_nodes=2)
        assert len(edges) > 0
        z = z_comm(1.0, edges, n_gpus=16)
        assert z > 0.0


# ---------------------------------------------------------------------------
# gpu_partition_function
# ---------------------------------------------------------------------------

class TestGPUPartitionFunction:
    def test_log_z_finite(self):
        # log Z should be a finite number (Z itself overflows for H100 config)
        lz = log_gpu_partition_function(1.0)
        assert math.isfinite(lz)

    def test_log_z_decreasing_with_beta(self):
        # Higher β → fewer accessible states → smaller Z → smaller ln Z
        lzs = [log_gpu_partition_function(b) for b in [0.5, 1.0, 2.0, 4.0]]
        assert all(lzs[i] > lzs[i + 1] for i in range(len(lzs) - 1))

    def test_factorises_in_log_space(self):
        # ln Z_total = ln Z_compute + ln Z_memory + ln Z_comm (no comm edges)
        beta = 1.0
        hbm_bw = H100_MEMORY_LEVELS[-1].bandwidth_bytes_per_cycle
        log_zc = log_z_compute(beta, H100_SM_CONFIG, hbm_bw)
        log_zm = math.log(z_memory(beta))
        log_zk = math.log(z_comm(beta, [], H100_SM_CONFIG.n_sm))
        log_z_total = log_gpu_partition_function(beta)
        assert abs(log_z_total - (log_zc + log_zm + log_zk)) < 1e-6

    def test_small_config_z_positive(self):
        # With a tiny config Z doesn't overflow and should be > 0
        tiny_sm = SMConfig(n_sm=2, warps_per_sm=4, peak_flops_per_cycle=64.0)
        tiny_mem = H100_MEMORY_LEVELS[:2]
        z = gpu_partition_function(1.0, sm_config=tiny_sm, memory_levels=tiny_mem)
        assert z > 0.0


# ---------------------------------------------------------------------------
# thermodynamic_quantities
# ---------------------------------------------------------------------------

class TestThermodynamicQuantities:
    @pytest.fixture
    def state(self) -> ThermodynamicState:
        return thermodynamic_quantities(beta=1.0)

    def test_free_energy_leq_mean_waste(self, state):
        # F ≤ <E> always (F = <E> - TS, S ≥ 0)
        # In our normalised units: free_energy ≤ mean_waste
        assert state.free_energy <= state.mean_waste + 1e-6

    def test_entropy_nonneg(self, state):
        assert state.entropy >= -1e-6   # allow tiny numerical noise

    def test_specific_heat_nonneg(self, state):
        # Cv ≥ 0 by thermodynamic stability
        assert state.specific_heat >= -1e-4

    def test_log_Z_components_sum(self, state):
        # log Z_total = log Z_compute + log Z_memory + log Z_comm
        assert abs(
            state.log_Z - (state.log_Z_compute + state.log_Z_memory + state.log_Z_comm)
        ) < 1e-4

    def test_mean_waste_in_unit_interval(self, state):
        assert 0.0 <= state.mean_waste <= 1.0 + 1e-6

    def test_consistency_across_beta(self):
        # As β increases (cooling), mean_waste should not increase
        # (colder system → less waste at equilibrium)
        states = beta_sweep([0.5, 1.0, 2.0, 4.0])
        wastes = [s.mean_waste for s in states]
        # Allow small numerical non-monotonicity (< 1%)
        for i in range(len(wastes) - 1):
            assert wastes[i] >= wastes[i + 1] - 0.01, (
                f"mean_waste not non-increasing: {wastes[i]:.4f} → {wastes[i+1]:.4f} "
                f"at β={states[i].beta:.1f} → {states[i+1].beta:.1f}"
            )

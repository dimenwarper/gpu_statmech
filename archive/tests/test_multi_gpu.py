"""
Tests for the multi_gpu module.

Key invariants:
  - TopologyGraph factories produce graphs with the correct number of GPUs and links.
  - log_z_comm_topology ≤ 0 for all β > 0 (communication adds waste).
  - log_z_multi_gpu = log_Z_local + log_Z_comm_topo (exact decomposition).
  - For N=1 with no links, multi-GPU quantities match single-GPU.
  - Mean waste increases with topology coupling (higher J → more waste).
  - MultiGPUCarnotLimit: η_multi_max ∈ [0, 1], β_optimal > 0.
  - scaling_efficiency ≤ 1 (comm overhead can only reduce η).
  - NVLink topology has higher scaling efficiency than InfiniBand.
  - resonance_condition returns η_overlap ∈ [0, 1].
  - Resonance peaks at T_compute = T_comm.
"""

import math

import numpy as np
import pytest

from gpu_statmech.partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    LINK_PRESETS,
    LinkConfig,
    SMConfig,
    TopologyEdge,
    log_gpu_partition_function,
    thermodynamic_quantities,
)
from gpu_statmech.multi_gpu import (
    MultiGPUCarnotLimit,
    MultiGPUThermodynamicState,
    THERMO_PHASE,
    TopologyGraph,
    _log_z_comm_topology,
    derive_multi_gpu_carnot_limit,
    log_z_multi_gpu,
    multi_gpu_thermodynamic_quantities,
    resonance_condition,
)


# ---------------------------------------------------------------------------
# Tiny hardware config for fast tests
# ---------------------------------------------------------------------------

_TINY_SM = SMConfig(n_sm=4, warps_per_sm=8, peak_flops_per_cycle=64.0)
_TINY_MEM = H100_MEMORY_LEVELS[:2]   # registers + smem only


# ---------------------------------------------------------------------------
# TopologyGraph factories
# ---------------------------------------------------------------------------

class TestTopologyGraph:
    def test_nvlink_clique_n_gpu(self):
        g = TopologyGraph.nvlink_clique(4)
        assert g.n_gpu == 4

    def test_nvlink_clique_n_links(self):
        # All-to-all: N × (N-1) directed edges
        g = TopologyGraph.nvlink_clique(4)
        assert len(g.links) == 4 * 3

    def test_nvswitch_fabric_n_links(self):
        g = TopologyGraph.nvswitch_fabric(4)
        assert len(g.links) == 4 * 3

    def test_pcie_ring_n_links(self):
        # Bidirectional ring: 2N edges
        g = TopologyGraph.pcie_ring(4)
        assert len(g.links) == 2 * 4

    def test_infiniband_fat_tree_n_links(self):
        g = TopologyGraph.infiniband_fat_tree(4)
        assert len(g.links) == 4 * 3

    def test_dgx_h100_single_node(self):
        g = TopologyGraph.dgx_h100(n_nodes=1)
        assert g.n_gpu == 8
        # 8×7 = 56 intra-node edges, 0 inter-node
        assert len(g.links) == 56

    def test_dgx_h100_two_nodes(self):
        g = TopologyGraph.dgx_h100(n_nodes=2)
        assert g.n_gpu == 16
        # 2 × 56 intra-node + 2 inter-node (bidirectional)
        assert len(g.links) == 2 * 56 + 2

    def test_single_gpu_no_links(self):
        g = TopologyGraph.nvlink_clique(1)
        assert g.n_gpu == 1
        assert len(g.links) == 0

    def test_single_gpu_pcie_ring_has_no_self_loops(self):
        g = TopologyGraph.pcie_ring(1)
        assert g.n_gpu == 1
        assert len(g.links) == 0

    def test_adjacency_J_shape(self):
        g = TopologyGraph.nvlink_clique(3)
        J = g.adjacency_J()
        assert J.shape == (3, 3)
        assert J[0, 0] == 0.0   # no self-loops

    def test_adjacency_J_values(self):
        g = TopologyGraph.nvlink_clique(3)
        J = g.adjacency_J()
        expected_J = LINK_PRESETS["nvlink4"].coupling_J
        assert J[0, 1] == pytest.approx(expected_J)
        assert J[1, 0] == pytest.approx(expected_J)

    def test_mean_J_nvlink(self):
        g = TopologyGraph.nvlink_clique(4)
        assert g.mean_J() == pytest.approx(LINK_PRESETS["nvlink4"].coupling_J)

    def test_mean_J_empty(self):
        g = TopologyGraph(n_gpu=1, links=[], name="empty")
        assert g.mean_J() == 0.0

    def test_total_bandwidth_nvlink(self):
        g = TopologyGraph.nvlink_clique(4)
        expected = 4 * 3 * LINK_PRESETS["nvlink4"].bandwidth_gb_s
        assert g.total_bandwidth_gb_s() == pytest.approx(expected)

    def test_bottleneck_bandwidth(self):
        # Mixed topology: bottleneck is the slower link
        fast_lc = LINK_PRESETS["nvlink4"]
        slow_lc = LINK_PRESETS["infiniband"]
        links = [
            TopologyEdge(0, 1, fast_lc),
            TopologyEdge(1, 2, slow_lc),
        ]
        g = TopologyGraph(n_gpu=3, links=links)
        assert g.bottleneck_bandwidth_gb_s() == pytest.approx(slow_lc.bandwidth_gb_s)

    def test_bottleneck_bandwidth_empty(self):
        g = TopologyGraph(n_gpu=1, links=[])
        assert g.bottleneck_bandwidth_gb_s() == 0.0

    def test_name_set(self):
        g = TopologyGraph.nvlink_clique(4)
        assert "nvlink" in g.name


# ---------------------------------------------------------------------------
# _log_z_comm_topology
# ---------------------------------------------------------------------------

class TestLogZCommTopology:
    def test_empty_topology_returns_zero(self):
        g = TopologyGraph(n_gpu=1, links=[])
        assert _log_z_comm_topology(1.0, g) == 0.0

    def test_nonpositive_for_any_beta(self):
        # Z_link = (1 - e^{-βJ}) / (βJ) ≤ 1 → ln Z_link ≤ 0 → sum ≤ 0
        g = TopologyGraph.nvlink_clique(4)
        for beta in [0.1, 1.0, 5.0, 10.0]:
            assert _log_z_comm_topology(beta, g) <= 1e-9

    def test_approaches_zero_at_small_beta(self):
        # β → 0: Z_link → 1, ln Z_link → 0
        g = TopologyGraph.nvlink_clique(2)
        val = _log_z_comm_topology(1e-6, g)
        assert abs(val) < 1e-3

    def test_more_negative_at_larger_beta(self):
        # Higher β → each Z_link smaller → more negative log
        g = TopologyGraph.nvlink_clique(2)
        v1 = _log_z_comm_topology(0.1, g)
        v5 = _log_z_comm_topology(5.0, g)
        assert v5 < v1

    def test_more_negative_with_higher_J(self):
        # Higher J link → larger βJ → smaller Z_link → more negative log
        beta = 2.0
        low_J  = LinkConfig("low",  bandwidth_gb_s=900, latency_us=1.0, coupling_J=0.1)
        high_J = LinkConfig("high", bandwidth_gb_s=900, latency_us=1.0, coupling_J=5.0)
        g_low  = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, low_J)])
        g_high = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, high_J)])
        assert _log_z_comm_topology(beta, g_high) < _log_z_comm_topology(beta, g_low)

    def test_more_negative_with_slower_bandwidth(self):
        beta = 2.0
        fast = LinkConfig("fast", bandwidth_gb_s=900, latency_us=1.0, coupling_J=1.0)
        slow = LinkConfig("slow", bandwidth_gb_s=64, latency_us=1.0, coupling_J=1.0)
        g_fast = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, fast)])
        g_slow = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, slow)])
        assert _log_z_comm_topology(beta, g_slow) < _log_z_comm_topology(beta, g_fast)

    def test_more_negative_with_higher_latency(self):
        beta = 2.0
        low_lat = LinkConfig("low_lat", bandwidth_gb_s=900, latency_us=1.0, coupling_J=1.0)
        high_lat = LinkConfig("high_lat", bandwidth_gb_s=900, latency_us=3.5, coupling_J=1.0)
        g_low = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, low_lat)])
        g_high = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, high_lat)])
        assert _log_z_comm_topology(beta, g_high) < _log_z_comm_topology(beta, g_low)

    def test_proportional_to_n_links_same_J(self):
        # For identical links, log Z = n_links × log Z_single_link
        lc = LinkConfig("lc", bandwidth_gb_s=100, latency_us=1.0, coupling_J=1.0)
        g1 = TopologyGraph(n_gpu=2, links=[TopologyEdge(0, 1, lc)])
        g2 = TopologyGraph(n_gpu=3, links=[TopologyEdge(0, 1, lc), TopologyEdge(1, 2, lc)])
        v1 = _log_z_comm_topology(2.0, g1)
        v2 = _log_z_comm_topology(2.0, g2)
        assert abs(v2 - 2 * v1) < 1e-9


# ---------------------------------------------------------------------------
# log_z_multi_gpu
# ---------------------------------------------------------------------------

class TestLogZMultiGpu:
    def test_decomposition(self):
        # log_Z_multi = log_Z_local + log_Z_comm_topo (exact)
        g = TopologyGraph.nvlink_clique(2)
        lz_multi, lz_local, lz_comm = log_z_multi_gpu(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16,
        )
        assert abs(lz_multi - (lz_local + lz_comm)) < 1e-9

    def test_single_gpu_no_links_matches_single_gpu(self):
        # N=1, no links: log_Z_multi = log_Z_single
        g = TopologyGraph(n_gpu=1, links=[])
        lz_multi, lz_local, lz_comm = log_z_multi_gpu(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16,
        )
        lz_single = log_gpu_partition_function(1.0, _TINY_SM, _TINY_MEM, [], n_bins=16)
        assert abs(lz_multi - lz_single) < 1e-9
        assert lz_comm == 0.0

    def test_n_gpu_scales_local(self):
        # N × log_Z_single = log_Z_local (by construction)
        g = TopologyGraph(n_gpu=3, links=[])
        _, lz_local, _ = log_z_multi_gpu(1.0, g, _TINY_SM, _TINY_MEM, n_bins=16)
        lz_single = log_gpu_partition_function(1.0, _TINY_SM, _TINY_MEM, [], n_bins=16)
        assert abs(lz_local - 3 * lz_single) < 1e-9

    def test_comm_reduces_log_z(self):
        # Adding comm links should make log_Z_multi ≤ log_Z_local (comm adds waste)
        g_no_comm = TopologyGraph(n_gpu=2, links=[])
        g_with_comm = TopologyGraph.nvlink_clique(2)
        lz_no, _, _ = log_z_multi_gpu(2.0, g_no_comm, _TINY_SM, _TINY_MEM, n_bins=16)
        lz_with, _, _ = log_z_multi_gpu(2.0, g_with_comm, _TINY_SM, _TINY_MEM, n_bins=16)
        assert lz_with <= lz_no + 1e-9

    def test_finite(self):
        g = TopologyGraph.dgx_h100(n_nodes=1)
        lz, _, _ = log_z_multi_gpu(1.0, g, n_bins=16)
        assert math.isfinite(lz)


# ---------------------------------------------------------------------------
# multi_gpu_thermodynamic_quantities
# ---------------------------------------------------------------------------

class TestMultiGPUThermodynamicQuantities:
    @pytest.fixture
    def state_2gpu(self) -> MultiGPUThermodynamicState:
        g = TopologyGraph.nvlink_clique(2)
        return multi_gpu_thermodynamic_quantities(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16,
        )

    def test_mean_waste_in_unit_interval(self, state_2gpu):
        assert 0.0 <= state_2gpu.mean_waste <= 1.0 + 1e-6

    def test_eta_multi_in_unit_interval(self, state_2gpu):
        assert 0.0 <= state_2gpu.eta_multi <= 1.0 + 1e-6

    def test_entropy_nonneg(self, state_2gpu):
        assert state_2gpu.entropy >= -1e-4

    def test_specific_heat_nonneg(self, state_2gpu):
        assert state_2gpu.specific_heat >= -1e-4

    def test_free_energy_leq_mean_waste(self, state_2gpu):
        # F ≤ <E> (second law: F = <E> - TS, S ≥ 0)
        assert state_2gpu.free_energy <= state_2gpu.mean_waste + 1e-6

    def test_log_z_decomposition(self, state_2gpu):
        assert abs(state_2gpu.log_Z_multi
                   - (state_2gpu.log_Z_local + state_2gpu.log_Z_comm_topo)) < 1e-9

    def test_n_gpu_stored(self, state_2gpu):
        assert state_2gpu.n_gpu == 2

    def test_single_gpu_matches_reference(self):
        # Single GPU, no links should match the single-GPU thermodynamics.
        g = TopologyGraph(n_gpu=1, links=[])
        state = multi_gpu_thermodynamic_quantities(1.0, g, _TINY_SM, _TINY_MEM, n_bins=16)
        reference = thermodynamic_quantities(1.0, _TINY_SM, _TINY_MEM, comm_edges=[], n_bins=16)
        assert state.mean_input_energy == pytest.approx(reference.mean_input_energy)
        assert state.mean_useful_work == pytest.approx(reference.mean_useful_work)
        assert state.mean_waste == pytest.approx(reference.mean_waste)
        assert state.eta_multi == pytest.approx(reference.eta_hw)

    def test_mean_waste_increases_with_more_links(self):
        # Adding higher-J links raises mean waste
        beta = 1.0
        g_nvlink = TopologyGraph.nvlink_clique(2)
        g_ib = TopologyGraph.infiniband_fat_tree(2)
        s_nvlink = multi_gpu_thermodynamic_quantities(beta, g_nvlink, _TINY_SM, _TINY_MEM, n_bins=16)
        s_ib = multi_gpu_thermodynamic_quantities(beta, g_ib, _TINY_SM, _TINY_MEM, n_bins=16)
        # IB has higher J → more comm waste → higher total mean waste
        assert s_ib.mean_waste >= s_nvlink.mean_waste - 1e-6
        assert s_ib.mean_comm_input_energy >= s_nvlink.mean_comm_input_energy - 1e-6

    def test_mean_waste_decreases_with_beta(self):
        # Colder system (higher β) → less waste
        g = TopologyGraph.nvlink_clique(2)
        wastes = [
            multi_gpu_thermodynamic_quantities(b, g, _TINY_SM, _TINY_MEM, n_bins=16).mean_waste
            for b in [0.5, 1.0, 3.0, 6.0]
        ]
        for i in range(len(wastes) - 1):
            assert wastes[i] >= wastes[i + 1] - 0.01

    def test_comm_energy_fraction_is_bounded(self, state_2gpu):
        frac = state_2gpu.mean_comm_input_energy / max(state_2gpu.mean_input_energy, 1e-12)
        assert 0.0 <= frac <= 1.0 + 1e-6

    def test_target_comm_load_closure_matches_target(self):
        g = TopologyGraph.nvlink_clique(2)
        state = multi_gpu_thermodynamic_quantities(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16, target_comm_load=0.20,
        )
        assert state.mean_comm_load == pytest.approx(0.20, abs=1e-4)

    def test_comm_input_energy_increases_with_target_comm_load(self):
        g = TopologyGraph.nvlink_clique(2)
        s_low = multi_gpu_thermodynamic_quantities(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16, target_comm_load=0.05,
        )
        s_high = multi_gpu_thermodynamic_quantities(
            1.0, g, _TINY_SM, _TINY_MEM, n_bins=16, target_comm_load=0.40,
        )
        assert s_high.mean_comm_input_energy > s_low.mean_comm_input_energy

    def test_slower_links_require_larger_comm_field(self):
        beta = 1.0
        target_comm_load = 0.05
        g_nvlink = TopologyGraph.nvlink_clique(2)
        g_pcie = TopologyGraph.pcie_ring(2)
        s_nvlink = multi_gpu_thermodynamic_quantities(
            beta, g_nvlink, _TINY_SM, _TINY_MEM, n_bins=16, target_comm_load=target_comm_load,
        )
        s_pcie = multi_gpu_thermodynamic_quantities(
            beta, g_pcie, _TINY_SM, _TINY_MEM, n_bins=16, target_comm_load=target_comm_load,
        )
        assert s_pcie.comm_field > s_nvlink.comm_field


# ---------------------------------------------------------------------------
# resonance_condition
# ---------------------------------------------------------------------------

class TestResonanceCondition:
    def test_equal_times_returns_overlap_fraction(self):
        # T_compute = T_comm → η = overlap_fraction
        eta = resonance_condition(1.0, 1.0, overlap_fraction=1.0)
        assert eta == pytest.approx(1.0)

    def test_equal_times_partial_overlap(self):
        eta = resonance_condition(1.0, 1.0, overlap_fraction=0.5)
        assert eta == pytest.approx(0.5)

    def test_compute_dominated(self):
        # T_compute >> T_comm: η_overlap = T_comm / T_compute
        eta = resonance_condition(100.0, 1.0, overlap_fraction=1.0)
        assert eta == pytest.approx(0.01)

    def test_comm_dominated(self):
        # T_comm >> T_compute: η_overlap = T_compute / T_comm
        eta = resonance_condition(1.0, 100.0, overlap_fraction=1.0)
        assert eta == pytest.approx(0.01)

    def test_no_overlap(self):
        # overlap_fraction=0 → no compute is hidden → η = 0
        eta = resonance_condition(1.0, 1.0, overlap_fraction=0.0)
        assert eta == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        for tc in [0.1, 1.0, 10.0]:
            for tk in [0.1, 1.0, 10.0]:
                eta = resonance_condition(tc, tk)
                assert 0.0 <= eta <= 1.0 + 1e-9

    def test_symmetric(self):
        # η_overlap(T_c, T_k) = η_overlap(T_k, T_c) for overlap_fraction=1
        assert resonance_condition(3.0, 7.0) == pytest.approx(resonance_condition(7.0, 3.0))

    def test_resonance_is_global_maximum(self):
        # For fixed sum T_c + T_k, η_overlap is maximised when T_c = T_k
        t_total = 4.0
        etas = [resonance_condition(t, t_total - t) for t in np.linspace(0.1, 3.9, 20)]
        max_eta_idx = int(np.argmax(etas))
        # Maximum should be near the midpoint
        t_at_max = np.linspace(0.1, 3.9, 20)[max_eta_idx]
        assert abs(t_at_max - t_total / 2) < 0.5


# ---------------------------------------------------------------------------
# MultiGPUCarnotLimit
# ---------------------------------------------------------------------------

class TestMultiGPUCarnotLimit:
    @pytest.fixture(scope="class")
    def nvlink_limit(self) -> MultiGPUCarnotLimit:
        g = TopologyGraph.nvlink_clique(2)
        # Let eta_hw_max_single be computed from the same hardware config so
        # that the scaling_efficiency() ≤ 1 invariant is guaranteed.
        return derive_multi_gpu_carnot_limit(
            g, _TINY_SM, _TINY_MEM,
            n_beta=20, n_bins=16,
        )

    @pytest.fixture(scope="class")
    def ib_limit(self) -> MultiGPUCarnotLimit:
        g = TopologyGraph.infiniband_fat_tree(2)
        return derive_multi_gpu_carnot_limit(
            g, _TINY_SM, _TINY_MEM,
            n_beta=20, n_bins=16,
        )

    def test_eta_multi_max_in_unit_interval(self, nvlink_limit):
        assert 0.0 <= nvlink_limit.eta_multi_max <= 1.0

    def test_beta_optimal_positive(self, nvlink_limit):
        assert nvlink_limit.beta_optimal > 0.0

    def test_resonance_eta_in_unit_interval(self, nvlink_limit):
        assert 0.0 <= nvlink_limit.resonance_eta <= 1.0

    def test_comm_overhead_in_unit_interval(self, nvlink_limit):
        assert 0.0 <= nvlink_limit.comm_overhead_fraction <= 1.0

    def test_scaling_efficiency_leq_one(self, nvlink_limit):
        # η_multi ≤ η_hw,single (comm overhead can only reduce efficiency)
        assert nvlink_limit.scaling_efficiency() <= 1.0 + 1e-6

    def test_scaling_efficiency_positive(self, nvlink_limit):
        assert nvlink_limit.scaling_efficiency() > 0.0

    def test_nvlink_higher_scaling_than_ib(self, nvlink_limit, ib_limit):
        # NVLink (J=0.1) has lower comm waste than IB (J=5.0)
        # → NVLink scaling efficiency should be higher
        assert nvlink_limit.scaling_efficiency() >= ib_limit.scaling_efficiency() - 1e-6

    def test_n_gpu_stored(self, nvlink_limit):
        assert nvlink_limit.n_gpu == 2

    def test_summary_contains_key_fields(self, nvlink_limit):
        s = nvlink_limit.summary()
        assert "η_multi,max" in s
        assert "scaling" in s
        assert "β_optimal" in s
        assert "g_optimal" in s

    def test_thermo_state_is_consistent(self, nvlink_limit):
        st = nvlink_limit.thermo_state
        assert st.beta == pytest.approx(nvlink_limit.beta_optimal)
        assert 0.0 <= st.mean_waste <= 1.0
        assert nvlink_limit.eta_multi_max == pytest.approx(st.eta_multi, abs=1e-6)
        assert st.comm_field == pytest.approx(nvlink_limit.comm_field_optimal)


# ---------------------------------------------------------------------------
# THERMO_PHASE mapping
# ---------------------------------------------------------------------------

class TestThermoPhaseMappings:
    def test_dp_is_ferromagnetic(self):
        assert THERMO_PHASE["dp"] == "ferromagnetic"

    def test_tp_is_antiferromagnetic(self):
        assert THERMO_PHASE["tp"] == "antiferromagnetic"

    def test_pp_is_domain_wall(self):
        assert THERMO_PHASE["pp"] == "domain_wall"

    def test_ep_is_spin_glass(self):
        assert THERMO_PHASE["ep"] == "spin_glass"

    def test_cp_is_quasi_antiferromagnetic(self):
        assert THERMO_PHASE["cp"] == "quasi_antiferromagnetic"

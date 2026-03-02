"""Tests for gpu_statmech.topology."""

import pytest
from gpu_statmech.topology import (
    InterconnectType,
    COUPLING_CONSTANTS,
    LINK_SPECS,
    LinkConfig,
    Topology,
)


# ---------------------------------------------------------------------------
# Coupling constant ordering (cheaper links have lower J)
# ---------------------------------------------------------------------------

class TestCouplingConstants:
    def test_nvlink_cheapest(self):
        assert COUPLING_CONSTANTS[InterconnectType.NVLINK] <= 0.2

    def test_ethernet_most_expensive(self):
        j_eth = COUPLING_CONSTANTS[InterconnectType.ETH_ROCE]
        for itype, j in COUPLING_CONSTANTS.items():
            assert j <= j_eth

    def test_nvlink_nvswitch_equal(self):
        assert (COUPLING_CONSTANTS[InterconnectType.NVLINK] ==
                COUPLING_CONSTANTS[InterconnectType.NVSWITCH])

    def test_nvswitch_cheaper_than_pcie(self):
        assert (COUPLING_CONSTANTS[InterconnectType.NVSWITCH] <
                COUPLING_CONSTANTS[InterconnectType.PCIE_GEN5])

    def test_pcie_cheaper_than_ib(self):
        assert (COUPLING_CONSTANTS[InterconnectType.PCIE_GEN5] <
                COUPLING_CONSTANTS[InterconnectType.IB_NDR])

    def test_ib_cheaper_than_roce(self):
        assert (COUPLING_CONSTANTS[InterconnectType.IB_NDR] <
                COUPLING_CONSTANTS[InterconnectType.ETH_ROCE])


# ---------------------------------------------------------------------------
# LinkConfig
# ---------------------------------------------------------------------------

class TestLinkConfig:
    def test_from_type_nvswitch(self):
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        assert lnk.bandwidth_gbps == LINK_SPECS[InterconnectType.NVSWITCH]["bandwidth_gbps"]
        assert lnk.latency_us == LINK_SPECS[InterconnectType.NVSWITCH]["latency_us"]
        assert lnk.coupling_constant == COUPLING_CONSTANTS[InterconnectType.NVSWITCH]

    def test_transfer_time_zero_bytes(self):
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        assert lnk.transfer_time_us(0) == pytest.approx(lnk.latency_us)

    def test_transfer_time_positive(self):
        lnk = LinkConfig.from_type(InterconnectType.IB_NDR)
        t = lnk.transfer_time_us(1 << 30)  # 1 GiB
        assert t > lnk.latency_us

    def test_ib_slower_than_nvswitch(self):
        nvswitch = LinkConfig.from_type(InterconnectType.NVSWITCH)
        ib = LinkConfig.from_type(InterconnectType.IB_NDR)
        msg = 1 << 27  # 128 MiB
        assert ib.transfer_time_us(msg) > nvswitch.transfer_time_us(msg)


# ---------------------------------------------------------------------------
# Topology construction and queries
# ---------------------------------------------------------------------------

class TestTopologyBasics:
    def test_add_link_bidirectional(self):
        topo = Topology(num_gpus=2)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        topo.add_link(0, 1, lnk)
        assert topo.link(0, 1) is not None
        assert topo.link(1, 0) is not None

    def test_num_edges(self):
        topo = Topology(num_gpus=4)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        topo.add_link(0, 1, lnk)
        topo.add_link(1, 2, lnk)
        assert topo.num_edges == 2

    def test_neighbors(self):
        topo = Topology(num_gpus=4)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        topo.add_link(0, 1, lnk)
        topo.add_link(0, 2, lnk)
        assert set(topo.neighbors(0)) == {1, 2}
        assert topo.neighbors(3) == []

    def test_self_link_raises(self):
        topo = Topology(num_gpus=4)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        with pytest.raises(ValueError):
            topo.add_link(0, 0, lnk)

    def test_out_of_range_raises(self):
        topo = Topology(num_gpus=4)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        with pytest.raises(ValueError):
            topo.add_link(0, 5, lnk)

    def test_coupling_constant_no_link(self):
        topo = Topology(num_gpus=4)
        assert topo.coupling_constant(0, 3) == float("inf")

    def test_coupling_constant_with_link(self):
        topo = Topology(num_gpus=2)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        topo.add_link(0, 1, lnk)
        assert topo.coupling_constant(0, 1) == pytest.approx(0.1)

    def test_edges_undirected(self):
        topo = Topology(num_gpus=3)
        lnk = LinkConfig.from_type(InterconnectType.NVSWITCH)
        topo.add_link(0, 1, lnk)
        topo.add_link(1, 2, lnk)
        edges = list(topo.edges())
        assert len(edges) == 2
        for a, b, _ in edges:
            assert a < b


# ---------------------------------------------------------------------------
# Preset topologies
# ---------------------------------------------------------------------------

class TestPresetTopologies:
    def test_dgx_h100_8_gpus(self):
        topo = Topology.dgx_h100()
        assert topo.num_gpus == 8

    def test_dgx_h100_fully_connected(self):
        topo = Topology.dgx_h100()
        assert topo.is_fully_connected()

    def test_dgx_h100_28_edges(self):
        # C(8,2) = 28
        topo = Topology.dgx_h100()
        assert topo.num_edges == 28

    def test_dgx_h100_all_nvswitch(self):
        topo = Topology.dgx_h100()
        for _, _, lnk in topo.edges():
            assert lnk.interconnect_type == InterconnectType.NVSWITCH

    def test_dgx_h100_low_mean_coupling(self):
        topo = Topology.dgx_h100()
        assert topo.mean_coupling() == pytest.approx(0.1)

    def test_dgx_a100_8_gpus(self):
        topo = Topology.dgx_a100()
        assert topo.num_gpus == 8
        assert topo.is_fully_connected()

    def test_dgx_a100_lower_bandwidth(self):
        topo = Topology.dgx_a100()
        for _, _, lnk in topo.edges():
            assert lnk.bandwidth_gbps == pytest.approx(600.0)

    def test_dgx_superpod_gpu_count(self):
        topo = Topology.dgx_superpod(num_nodes=4)
        assert topo.num_gpus == 32

    def test_dgx_superpod_intra_node_edges(self):
        topo = Topology.dgx_superpod(num_nodes=2)
        # Intra-node: 2 × C(8,2) = 56; inter-node: 1 IB link
        assert topo.num_edges == 56 + 1

    def test_dgx_superpod_has_both_link_types(self):
        topo = Topology.dgx_superpod(num_nodes=2)
        types = {lnk.interconnect_type for _, _, lnk in topo.edges()}
        assert InterconnectType.NVSWITCH in types
        assert InterconnectType.IB_NDR in types

    def test_pcie_cluster_fully_connected(self):
        topo = Topology.pcie_cluster(4)
        assert topo.is_fully_connected()
        assert topo.mean_coupling() == pytest.approx(
            COUPLING_CONSTANTS[InterconnectType.PCIE_GEN5]
        )

    def test_from_adjacency(self):
        adj = {(0, 1): InterconnectType.NVSWITCH,
               (1, 2): InterconnectType.IB_NDR}
        topo = Topology.from_adjacency(3, adj)
        assert topo.num_edges == 2
        assert topo.link(0, 1).interconnect_type == InterconnectType.NVSWITCH
        assert topo.link(1, 2).interconnect_type == InterconnectType.IB_NDR

    def test_repr(self):
        topo = Topology.dgx_h100()
        r = repr(topo)
        assert "num_gpus=8" in r
        assert "num_edges=28" in r

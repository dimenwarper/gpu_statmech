"""Tests for gpu_statmech.multi_gpu."""

import pytest
from gpu_statmech.multi_gpu import (
    CollectiveOp,
    comm_volume_per_gpu,
    CommChannelState,
    CommState,
    GlobalMicrostate,
    CommEnergyWeights,
    pairwise_comm_energy,
    MultiGPUEnergyFunctional,
)
from gpu_statmech.microstate import (
    Microstate, SMState, MemoryHierarchyState, BandwidthState,
)
from gpu_statmech.topology import Topology, LinkConfig, InterconnectType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_microstate(gpu_id=0, occupancy=1.0, hbm_util=1.0) -> Microstate:
    max_warps = 64
    active = round(occupancy * max_warps)
    sm = SMState(sm_id=0, active_warps=active, max_warps=max_warps)
    mem = MemoryHierarchyState(
        register_utilization=0.8,
        shared_mem_utilization=0.5,
        l1_hit_rate=0.9,
        l2_hit_rate=0.8,
        l2_utilization=0.5,
        hbm_bandwidth_utilization=hbm_util,
    )
    bw = BandwidthState(l2_to_hbm=hbm_util)
    return Microstate(cycle=0, gpu_id=gpu_id, sm_states=[sm],
                      memory=mem, bandwidth=bw)


def make_channel(src=0, dst=1, bw_util=0.8, active=True,
                 latency_events=0, sync_stall=0.0) -> CommChannelState:
    return CommChannelState(
        src_gpu=src, dst_gpu=dst,
        bytes_in_flight=1 << 20,
        bandwidth_utilization=bw_util,
        is_active=active,
        latency_events=latency_events,
        sync_stall_fraction=sync_stall,
    )


def make_global(num_gpus=2, occupancies=None, channels=None,
                topology=None, overlap=0.0) -> GlobalMicrostate:
    occupancies = occupancies or [1.0] * num_gpus
    gpu_states = [make_microstate(i, occupancies[i]) for i in range(num_gpus)]
    comm = CommState(
        channels=channels or [],
        overlap_ratio=overlap,
    )
    topo = topology or Topology.dgx_h100() if num_gpus == 8 else Topology.pcie_cluster(num_gpus)
    return GlobalMicrostate(cycle=0, gpu_states=gpu_states,
                            comm_state=comm, topology=topo)


# ---------------------------------------------------------------------------
# comm_volume_per_gpu
# ---------------------------------------------------------------------------

class TestCommVolumePerGpu:
    M = 1 << 20   # 1 MiB message

    def test_all_reduce_less_than_all_to_all(self):
        G = 8
        ar = comm_volume_per_gpu(CollectiveOp.ALL_REDUCE, G, self.M)
        a2a = comm_volume_per_gpu(CollectiveOp.ALL_TO_ALL, G, self.M)
        assert ar < a2a

    def test_all_reduce_approaches_2M_large_G(self):
        # Ring all-reduce: 2*(G-1)/G * M → 2M as G → ∞
        vol = comm_volume_per_gpu(CollectiveOp.ALL_REDUCE, 1024, self.M)
        assert vol == pytest.approx(2 * self.M, rel=0.01)

    def test_all_gather_half_of_all_to_all(self):
        G = 8
        ag = comm_volume_per_gpu(CollectiveOp.ALL_GATHER, G, self.M)
        a2a = comm_volume_per_gpu(CollectiveOp.ALL_TO_ALL, G, self.M)
        # all_gather = (G-1)/G * M, all_to_all = (G-1)*M → ratio = 1/G
        assert ag == pytest.approx(a2a / G, rel=1e-6)

    def test_point_to_point_equals_message_size(self):
        vol = comm_volume_per_gpu(CollectiveOp.POINT_TO_POINT, 8, self.M)
        assert vol == pytest.approx(self.M)

    def test_reduce_scatter_equals_all_gather(self):
        G, M = 8, self.M
        assert comm_volume_per_gpu(CollectiveOp.REDUCE_SCATTER, G, M) == \
               comm_volume_per_gpu(CollectiveOp.ALL_GATHER, G, M)


# ---------------------------------------------------------------------------
# CommChannelState
# ---------------------------------------------------------------------------

class TestCommChannelState:
    def test_inactive_channel(self):
        ch = make_channel(active=False)
        assert not ch.is_active

    def test_active_channel(self):
        ch = make_channel(active=True)
        assert ch.is_active


# ---------------------------------------------------------------------------
# CommState
# ---------------------------------------------------------------------------

class TestCommState:
    def test_mean_bw_util_no_channels(self):
        cs = CommState()
        assert cs.mean_bandwidth_utilization == 0.0

    def test_mean_bw_util_active_only(self):
        channels = [
            make_channel(0, 1, bw_util=0.6, active=True),
            make_channel(1, 2, bw_util=0.4, active=True),
            make_channel(2, 3, bw_util=0.9, active=False),  # excluded
        ]
        cs = CommState(channels=channels)
        # Average of active channels: (0.6 + 0.4) / 2 = 0.5
        assert cs.mean_bandwidth_utilization == pytest.approx(0.5)

    def test_total_bytes_in_flight(self):
        channels = [make_channel(0, 1), make_channel(1, 2)]
        cs = CommState(channels=channels)
        assert cs.total_bytes_in_flight == 2 * (1 << 20)

    def test_channel_lookup(self):
        ch = make_channel(src=2, dst=5)
        cs = CommState(channels=[ch])
        assert cs.channel(2, 5) is ch
        assert cs.channel(5, 2) is None

    def test_mean_sync_stall(self):
        channels = [
            make_channel(0, 1, sync_stall=0.2),
            make_channel(1, 2, sync_stall=0.6),
        ]
        cs = CommState(channels=channels)
        assert cs.mean_sync_stall == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# GlobalMicrostate
# ---------------------------------------------------------------------------

class TestGlobalMicrostate:
    def test_num_gpus(self):
        state = make_global(num_gpus=4, topology=Topology.pcie_cluster(4))
        assert state.num_gpus == 4

    def test_mean_local_occupancy_uniform(self):
        state = make_global(
            num_gpus=4,
            occupancies=[0.5, 0.5, 0.5, 0.5],
            topology=Topology.pcie_cluster(4),
        )
        assert state.mean_local_occupancy == pytest.approx(0.5)

    def test_mean_local_occupancy_mixed(self):
        state = make_global(
            num_gpus=2,
            occupancies=[1.0, 0.0],
            topology=Topology.pcie_cluster(2),
        )
        assert state.mean_local_occupancy == pytest.approx(0.5)

    def test_summary_keys(self):
        state = make_global(num_gpus=2, topology=Topology.pcie_cluster(2))
        s = state.summary()
        for key in ("cycle", "num_gpus", "mean_sm_occupancy",
                    "mean_bw_utilization", "overlap_ratio", "active_collective"):
            assert key in s


# ---------------------------------------------------------------------------
# pairwise_comm_energy
# ---------------------------------------------------------------------------

class TestPairwiseCommEnergy:
    def test_inactive_channel_zero_energy(self):
        ch = make_channel(active=False)
        w = CommEnergyWeights()
        assert pairwise_comm_energy(ch, w) == pytest.approx(0.0)

    def test_active_channel_positive_energy(self):
        ch = make_channel(active=True, bw_util=0.8)
        w = CommEnergyWeights()
        assert pairwise_comm_energy(ch, w) > 0.0

    def test_higher_bw_util_higher_energy(self):
        w = CommEnergyWeights()
        low  = pairwise_comm_energy(make_channel(bw_util=0.2, active=True), w)
        high = pairwise_comm_energy(make_channel(bw_util=0.9, active=True), w)
        assert high > low

    def test_latency_events_increase_energy(self):
        w = CommEnergyWeights()
        e0 = pairwise_comm_energy(make_channel(active=True, latency_events=0), w)
        e1 = pairwise_comm_energy(make_channel(active=True, latency_events=5), w)
        assert e1 > e0

    def test_sync_stall_increases_energy(self):
        w = CommEnergyWeights()
        e0 = pairwise_comm_energy(make_channel(active=True, sync_stall=0.0), w)
        e1 = pairwise_comm_energy(make_channel(active=True, sync_stall=0.5), w)
        assert e1 > e0


# ---------------------------------------------------------------------------
# MultiGPUEnergyFunctional
# ---------------------------------------------------------------------------

class TestMultiGPUEnergyFunctional:
    def setup_method(self):
        self.fn = MultiGPUEnergyFunctional()

    def test_compute_keys(self):
        state = make_global(num_gpus=2, topology=Topology.pcie_cluster(2))
        result = self.fn.compute(state)
        assert set(result.keys()) == {"local", "interaction", "total"}

    def test_total_equals_local_plus_interaction(self):
        topo = Topology.pcie_cluster(2)
        channels = [make_channel(0, 1, bw_util=0.5, active=True)]
        state = make_global(num_gpus=2, channels=channels, topology=topo)
        result = self.fn.compute(state)
        assert result["total"] == pytest.approx(
            result["local"] + result["interaction"]
        )

    def test_local_energy_sums_per_gpu(self):
        topo = Topology.pcie_cluster(2)
        state = make_global(num_gpus=2,
                            occupancies=[0.5, 0.5],
                            topology=topo)
        per_gpu = self.fn.local_energy_per_gpu(state)
        total_local = self.fn.local_energy(state)
        assert sum(per_gpu) == pytest.approx(total_local)

    def test_no_active_channels_zero_interaction(self):
        topo = Topology.pcie_cluster(2)
        # Channels present but all inactive
        channels = [make_channel(0, 1, active=False)]
        state = make_global(num_gpus=2, channels=channels, topology=topo)
        assert self.fn.interaction_energy(state) == pytest.approx(0.0)

    def test_high_coupling_amplifies_comm_cost(self):
        """IB topology (J=5) should yield higher interaction energy than
        NVSwitch topology (J=0.1) for the same communication pattern."""
        channels = [make_channel(0, 1, bw_util=0.8, active=True)]
        comm = CommState(channels=channels)

        gpu_states = [make_microstate(i, occupancy=1.0) for i in range(2)]

        topo_fast = Topology.pcie_cluster(2)  # J_pcie=1.0
        # Manually create a slower topology
        topo_slow = Topology(num_gpus=2)
        ib_link = LinkConfig.from_type(InterconnectType.IB_NDR)  # J=5.0
        topo_slow.add_link(0, 1, ib_link)

        state_fast = GlobalMicrostate(cycle=0, gpu_states=gpu_states,
                                      comm_state=comm, topology=topo_fast)
        state_slow = GlobalMicrostate(cycle=0, gpu_states=gpu_states,
                                      comm_state=comm, topology=topo_slow)

        e_fast = self.fn.interaction_energy(state_fast)
        e_slow = self.fn.interaction_energy(state_slow)
        assert e_slow > e_fast

    def test_time_average_empty(self):
        result = self.fn.time_average([])
        assert result == {"local": 0.0, "interaction": 0.0, "total": 0.0}

    def test_time_average_single(self):
        topo = Topology.pcie_cluster(2)
        state = make_global(num_gpus=2, topology=topo)
        single = self.fn.time_average([state])
        direct = self.fn.compute(state)
        assert single["total"] == pytest.approx(direct["total"])

    def test_overlap_penalty_perfect(self):
        topo = Topology.pcie_cluster(2)
        state = make_global(num_gpus=2, overlap=1.0, topology=topo)
        assert self.fn.overlap_penalty(state) == pytest.approx(0.0)

    def test_overlap_penalty_zero_overlap(self):
        topo = Topology.pcie_cluster(2)
        state = make_global(num_gpus=2, overlap=0.0, topology=topo)
        assert self.fn.overlap_penalty(state) == pytest.approx(1.0)

    def test_decompose_full_keys(self):
        topo = Topology.pcie_cluster(2)
        channels = [make_channel(0, 1, bw_util=0.5, active=True)]
        state = make_global(num_gpus=2, channels=channels, topology=topo)
        d = self.fn.decompose_full(state)
        assert "per_gpu_local" in d
        assert "per_edge_interaction" in d
        assert "total" in d

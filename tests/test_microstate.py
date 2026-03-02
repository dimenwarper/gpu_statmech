"""Tests for gpu_statmech.microstate."""

import pytest
from gpu_statmech.microstate import (
    InstructionType,
    MemoryLevel,
    MEMORY_LEVEL_ENERGY,
    PipelineStage,
    WarpState,
    SMState,
    MemoryHierarchyState,
    BandwidthState,
    Microstate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_warp(warp_id: int, active: bool = True,
              itype=InstructionType.FP16,
              stage=PipelineStage.EXECUTE) -> WarpState:
    return WarpState(warp_id=warp_id, is_active=active,
                     instruction_type=itype, pipeline_stage=stage)


def make_sm(sm_id: int, active: int, max_w: int,
            warps: list[WarpState] | None = None) -> SMState:
    return SMState(sm_id=sm_id, active_warps=active, max_warps=max_w,
                   warp_states=warps or [])


def make_memory(hbm_util=0.8, l2_hit=0.6, l2_util=0.5,
                reg_util=0.9, smem_util=0.4, l1_hit=0.7) -> MemoryHierarchyState:
    return MemoryHierarchyState(
        register_utilization=reg_util,
        shared_mem_utilization=smem_util,
        l1_hit_rate=l1_hit,
        l2_hit_rate=l2_hit,
        l2_utilization=l2_util,
        hbm_bandwidth_utilization=hbm_util,
    )


def make_bw(**kwargs) -> BandwidthState:
    return BandwidthState(**kwargs)


def make_microstate(
    cycle=0, gpu_id=0,
    sm_states=None, memory=None, bandwidth=None,
) -> Microstate:
    return Microstate(
        cycle=cycle,
        gpu_id=gpu_id,
        sm_states=sm_states or [],
        memory=memory or make_memory(),
        bandwidth=bandwidth or BandwidthState(),
    )


# ---------------------------------------------------------------------------
# MemoryLevel energy ordering
# ---------------------------------------------------------------------------

class TestMemoryLevelEnergy:
    def test_registers_cheapest(self):
        assert MEMORY_LEVEL_ENERGY[MemoryLevel.REGISTERS] == 0

    def test_hbm_most_expensive(self):
        assert MEMORY_LEVEL_ENERGY[MemoryLevel.HBM] == 600

    def test_strict_ordering(self):
        costs = [
            MEMORY_LEVEL_ENERGY[MemoryLevel.REGISTERS],
            MEMORY_LEVEL_ENERGY[MemoryLevel.SHARED_MEM],
            MEMORY_LEVEL_ENERGY[MemoryLevel.L2_CACHE],
            MEMORY_LEVEL_ENERGY[MemoryLevel.HBM],
        ]
        assert costs == sorted(costs)

    def test_shared_and_l1_same_cost(self):
        assert (MEMORY_LEVEL_ENERGY[MemoryLevel.SHARED_MEM] ==
                MEMORY_LEVEL_ENERGY[MemoryLevel.L1_CACHE])


# ---------------------------------------------------------------------------
# WarpState
# ---------------------------------------------------------------------------

class TestWarpState:
    def test_is_stalled(self):
        w = make_warp(0, stage=PipelineStage.STALL)
        assert w.is_stalled

    def test_not_stalled_when_executing(self):
        w = make_warp(0, stage=PipelineStage.EXECUTE)
        assert not w.is_stalled

    def test_is_computing(self):
        w = make_warp(0, active=True, itype=InstructionType.FP16,
                      stage=PipelineStage.EXECUTE)
        assert w.is_computing

    def test_not_computing_when_mem(self):
        w = make_warp(0, active=True, itype=InstructionType.MEM,
                      stage=PipelineStage.EXECUTE)
        assert not w.is_computing

    def test_not_computing_when_idle(self):
        w = make_warp(0, active=False, itype=InstructionType.IDLE)
        assert not w.is_computing


# ---------------------------------------------------------------------------
# SMState
# ---------------------------------------------------------------------------

class TestSMState:
    def test_occupancy_full(self):
        sm = make_sm(0, active=64, max_w=64)
        assert sm.occupancy == pytest.approx(1.0)

    def test_occupancy_half(self):
        sm = make_sm(0, active=32, max_w=64)
        assert sm.occupancy == pytest.approx(0.5)

    def test_occupancy_zero_max_warps(self):
        sm = make_sm(0, active=0, max_w=0)
        assert sm.occupancy == 0.0

    def test_stall_fraction_all_stalled(self):
        warps = [make_warp(i, stage=PipelineStage.STALL) for i in range(4)]
        sm = make_sm(0, active=4, max_w=64, warps=warps)
        assert sm.stall_fraction == pytest.approx(1.0)

    def test_stall_fraction_none_stalled(self):
        warps = [make_warp(i, stage=PipelineStage.EXECUTE) for i in range(4)]
        sm = make_sm(0, active=4, max_w=64, warps=warps)
        assert sm.stall_fraction == pytest.approx(0.0)

    def test_stall_fraction_half(self):
        warps = (
            [make_warp(i, stage=PipelineStage.STALL) for i in range(2)] +
            [make_warp(i + 2, stage=PipelineStage.EXECUTE) for i in range(2)]
        )
        sm = make_sm(0, active=4, max_w=64, warps=warps)
        assert sm.stall_fraction == pytest.approx(0.5)

    def test_stall_fraction_no_warps(self):
        sm = make_sm(0, active=0, max_w=64, warps=[])
        assert sm.stall_fraction == 0.0


# ---------------------------------------------------------------------------
# MemoryHierarchyState
# ---------------------------------------------------------------------------

class TestMemoryHierarchyState:
    def test_mean_data_energy_no_occupation(self):
        mem = make_memory()
        assert mem.mean_data_energy == 0.0

    def test_mean_data_energy_all_registers(self):
        mem = make_memory()
        mem.occupation_numbers = {
            (0, MemoryLevel.REGISTERS): 1024,
            (1, MemoryLevel.REGISTERS): 512,
        }
        assert mem.mean_data_energy == 0.0

    def test_mean_data_energy_all_hbm(self):
        mem = make_memory()
        mem.occupation_numbers = {(0, MemoryLevel.HBM): 1000}
        assert mem.mean_data_energy == pytest.approx(600.0)

    def test_mean_data_energy_mixed(self):
        # 50% in registers (ε=0), 50% in HBM (ε=600) → mean = 300
        mem = make_memory()
        mem.occupation_numbers = {
            (0, MemoryLevel.REGISTERS): 500,
            (1, MemoryLevel.HBM): 500,
        }
        assert mem.mean_data_energy == pytest.approx(300.0)

    def test_mean_data_energy_zero_bytes(self):
        mem = make_memory()
        mem.occupation_numbers = {(0, MemoryLevel.HBM): 0}
        assert mem.mean_data_energy == 0.0


# ---------------------------------------------------------------------------
# BandwidthState
# ---------------------------------------------------------------------------

class TestBandwidthState:
    def test_mean_utilization(self):
        bw = BandwidthState(sm_to_shared_mem=0.6, sm_to_l2=0.9, l2_to_hbm=0.3)
        assert bw.mean_utilization == pytest.approx(0.6)

    def test_bottleneck(self):
        bw = BandwidthState(sm_to_shared_mem=0.3, sm_to_l2=0.95, l2_to_hbm=0.4)
        assert bw.bottleneck == "sm_to_l2"

    def test_bottleneck_nvlink(self):
        bw = BandwidthState(sm_to_shared_mem=0.1, sm_to_l2=0.1,
                            l2_to_hbm=0.1, nvlink=0.99)
        assert bw.bottleneck == "nvlink"


# ---------------------------------------------------------------------------
# Microstate aggregate properties
# ---------------------------------------------------------------------------

class TestMicrostate:
    def _make_full_state(self) -> Microstate:
        warps = [make_warp(i, stage=PipelineStage.EXECUTE) for i in range(4)]
        sms = [make_sm(i, active=32, max_w=64, warps=warps) for i in range(4)]
        return make_microstate(sm_states=sms)

    def test_num_sms(self):
        state = self._make_full_state()
        assert state.num_sms == 4

    def test_mean_sm_occupancy(self):
        state = self._make_full_state()
        assert state.mean_sm_occupancy == pytest.approx(0.5)

    def test_sm_active_fraction_all_active(self):
        warps = [make_warp(0)]
        sms = [make_sm(i, active=1, max_w=64, warps=warps) for i in range(4)]
        state = make_microstate(sm_states=sms)
        assert state.sm_active_fraction == pytest.approx(1.0)

    def test_sm_active_fraction_half_active(self):
        warps = [make_warp(0)]
        active_sms  = [make_sm(i,     active=1, max_w=64, warps=warps) for i in range(2)]
        idle_sms    = [make_sm(i + 2, active=0, max_w=64, warps=[])    for i in range(2)]
        state = make_microstate(sm_states=active_sms + idle_sms)
        assert state.sm_active_fraction == pytest.approx(0.5)

    def test_pipeline_stall_fraction_all_stalled(self):
        warps = [make_warp(i, stage=PipelineStage.STALL) for i in range(4)]
        sms   = [make_sm(0, active=4, max_w=64, warps=warps)]
        state = make_microstate(sm_states=sms)
        assert state.pipeline_stall_fraction == pytest.approx(1.0)

    def test_pipeline_stall_fraction_no_warps(self):
        state = make_microstate(sm_states=[make_sm(0, active=0, max_w=64)])
        assert state.pipeline_stall_fraction == 0.0

    def test_summary_keys(self):
        state = self._make_full_state()
        s = state.summary()
        for key in ("cycle", "gpu_id", "mean_sm_occupancy", "sm_active_fraction",
                    "pipeline_stall_fraction", "hbm_bw_utilization",
                    "l2_hit_rate", "bw_bottleneck"):
            assert key in s

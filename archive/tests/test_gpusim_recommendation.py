from types import SimpleNamespace

import pytest

from gpu_statmech.gpusim_driver import GpuSimKernelProfile
from gpu_statmech.gpusim_recommendation import (
    apply_intervention,
    fit_statmech_response_model,
    generate_recommendation_baselines,
    oracle_attainment_ratio,
    predict_intervention_gains_statmech,
    recommend_intervention_raw_counter,
    recommend_intervention_roofline,
    recommend_intervention_statmech,
    recommend_intervention_statmech_response,
)


def _profile() -> GpuSimKernelProfile:
    return GpuSimKernelProfile(
        name="toy",
        description="toy",
        threads_per_block=256,
        regs_per_thread=96,
        smem_per_block=64 * 1024,
        grid=(4096, 1, 1),
        instr_mix={
            "fp16": 0.20,
            "fp32": 0.20,
            "int": 0.00,
            "sfu": 0.00,
            "mem": 0.40,
            "tensor_core": 0.20,
        },
    )


def _analysis(
    *,
    families: dict[str, float],
    phase: str = "mixed",
    memory_feed_efficiency: float = 0.5,
    mean_memory_stall_fraction: float = 0.2,
    mean_hbm_bw_utilization: float = 0.2,
    mean_active_warp_fraction: float = 0.4,
    mean_issue_activity: float = 0.3,
    mean_reg_utilization: float = 0.4,
    mean_smem_utilization: float = 0.4,
    mean_instr_mix: dict[str, float] | None = None,
    stall_fraction: float = 0.1,
    idle_fraction: float = 0.1,
    dram_movement_fraction: float = 0.1,
    sram_overhead_fraction: float = 0.1,
):
    return SimpleNamespace(
        observables=SimpleNamespace(
            mean_warp_state_family_fractions=families,
            mean_memory_stall_fraction=mean_memory_stall_fraction,
            mean_hbm_bw_utilization=mean_hbm_bw_utilization,
            mean_active_warp_fraction=mean_active_warp_fraction,
            mean_issue_activity=mean_issue_activity,
            mean_reg_utilization=mean_reg_utilization,
            mean_smem_utilization=mean_smem_utilization,
            mean_instr_mix=mean_instr_mix or {
                "fp16": 0.2,
                "fp32": 0.4,
                "int": 0.0,
                "sfu": 0.0,
                "mem": 0.3,
                "tensor_core": 0.1,
            },
        ),
        bottleneck=SimpleNamespace(
            stall_fraction=stall_fraction,
            idle_fraction=idle_fraction,
            dram_movement_fraction=dram_movement_fraction,
            sram_overhead_fraction=sram_overhead_fraction,
        ),
        thermo_state=SimpleNamespace(memory_feed_efficiency=memory_feed_efficiency),
        dominant_phase=phase,
    )


def test_generate_recommendation_baselines_emits_four_per_family():
    baselines = generate_recommendation_baselines([_profile()])
    assert [b.stress for b in baselines] == [
        "base",
        "memory_stressed",
        "footprint_stressed",
        "compute_unoptimized",
    ]
    assert len({b.profile.name for b in baselines}) == 4


def test_apply_locality_intervention_reduces_mem_share():
    profile = _profile()
    improved = apply_intervention(profile, "locality")
    assert improved.instr_mix["mem"] < profile.instr_mix["mem"]
    assert improved.smem_per_block > profile.smem_per_block


def test_apply_tensorize_intervention_shifts_mix_to_tensor_core():
    profile = _profile()
    improved = apply_intervention(profile, "tensorize")
    assert improved.instr_mix["tensor_core"] > profile.instr_mix["tensor_core"]
    assert improved.instr_mix["mem"] < profile.instr_mix["mem"]


def test_apply_occupancy_intervention_reduces_resource_pressure():
    profile = _profile()
    improved = apply_intervention(profile, "occupancy")
    assert improved.regs_per_thread < profile.regs_per_thread
    assert improved.smem_per_block < profile.smem_per_block
    assert improved.threads_per_block <= profile.threads_per_block


def test_statmech_recommendation_prefers_locality_for_memory_pressure():
    analysis = _analysis(
        families={"productive": 0.2, "dependency": 0.1, "memory": 0.5, "sync_frontend": 0.05, "idle": 0.15},
        phase="memory_bound",
        mean_memory_stall_fraction=0.5,
        dram_movement_fraction=0.4,
        memory_feed_efficiency=0.2,
    )
    assert recommend_intervention_statmech(analysis) == "locality"


def test_statmech_recommendation_prefers_occupancy_for_dependency_pressure():
    analysis = _analysis(
        families={"productive": 0.2, "dependency": 0.4, "memory": 0.1, "sync_frontend": 0.15, "idle": 0.15},
        phase="mixed",
        mean_reg_utilization=0.8,
        mean_smem_utilization=0.8,
        stall_fraction=0.4,
    )
    assert recommend_intervention_statmech(analysis) == "occupancy"


def test_statmech_recommendation_prefers_tensorize_for_scalar_compute():
    analysis = _analysis(
        families={"productive": 0.5, "dependency": 0.1, "memory": 0.1, "sync_frontend": 0.05, "idle": 0.25},
        phase="compute_bound",
        mean_issue_activity=0.5,
        mean_instr_mix={
            "fp16": 0.2,
            "fp32": 0.6,
            "int": 0.0,
            "sfu": 0.0,
            "mem": 0.1,
            "tensor_core": 0.1,
        },
    )
    assert recommend_intervention_statmech(analysis) == "tensorize"


def test_raw_counter_and_roofline_heuristics_map_to_expected_levers():
    analysis = _analysis(
        families={"productive": 0.6, "dependency": 0.1, "memory": 0.1, "sync_frontend": 0.05, "idle": 0.15},
        mean_memory_stall_fraction=0.1,
        mean_hbm_bw_utilization=0.2,
        mean_issue_activity=0.4,
        mean_instr_mix={
            "fp16": 0.1,
            "fp32": 0.7,
            "int": 0.0,
            "sfu": 0.0,
            "mem": 0.1,
            "tensor_core": 0.1,
        },
    )
    assert recommend_intervention_raw_counter(analysis) == "tensorize"
    assert recommend_intervention_roofline(analysis) == "tensorize"


def test_response_model_predicts_larger_gain_for_memory_pressure():
    analysis = _analysis(
        families={"productive": 0.2, "dependency": 0.1, "memory": 0.55, "sync_frontend": 0.05, "idle": 0.1},
        mean_memory_stall_fraction=0.45,
        dram_movement_fraction=0.35,
        memory_feed_efficiency=0.25,
    )
    params = fit_statmech_response_model(
        [analysis],
        [{"locality": 0.12, "occupancy": 0.01, "tensorize": 0.02}],
    )
    gains = predict_intervention_gains_statmech(analysis, params)
    assert gains["locality"] > gains["occupancy"]
    assert gains["locality"] > gains["tensorize"]
    assert recommend_intervention_statmech_response(analysis, params) == "locality"


@pytest.mark.parametrize(
    ("chosen_gain", "oracle_gain", "expected"),
    [
        (0.5, 1.0, 0.5),
        (2.0, 1.0, 1.0),
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
)
def test_oracle_attainment_ratio_is_stable(chosen_gain: float, oracle_gain: float, expected: float):
    assert oracle_attainment_ratio(chosen_gain, oracle_gain) == pytest.approx(expected)

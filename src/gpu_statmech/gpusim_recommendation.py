"""
Simulator-side intervention generation and recommendation helpers.

This module supports counterfactual recommendation experiments on top of the
canonical `gpusim` kernel profiles. It does two things:

  1. generate controlled baseline stresses plus intervention variants
  2. map a thermodynamic analysis to a recommended optimization lever
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .gpusim_driver import GpuSimKernelProfile


INTERVENTION_KEYS = ("locality", "occupancy", "tensorize")
BASELINE_STRESS_KEYS = (
    "base",
    "memory_stressed",
    "footprint_stressed",
    "compute_unoptimized",
)
COMPUTE_KEYS = ("tensor_core", "fp16", "fp32", "int", "sfu")
MAX_SMEM_PER_BLOCK_BYTES = 96 * 1024


@dataclass(frozen=True)
class RecommendationBaseline:
    """A named stressed baseline used in the simulator recommendation study."""

    family: str
    stress: str
    profile: GpuSimKernelProfile

    @property
    def key(self) -> str:
        return f"{self.family}:{self.stress}"


@dataclass(frozen=True)
class StatMechResponseParams:
    """Interpretable lever-response coefficients for first-order gain prediction."""

    locality_feed: float = 1.0
    locality_memory_pressure: float = 1.0
    locality_movement: float = 1.0

    occupancy_dependency: float = 1.0
    occupancy_resource: float = 1.0
    occupancy_issue_gap: float = 1.0

    tensorize_scalar_math: float = 1.0
    tensorize_tensor_gap: float = 1.0
    tensorize_productive_surplus: float = 1.0


def _renormalize_instr_mix(instr_mix: Mapping[str, float]) -> dict[str, float]:
    keys = ("fp16", "fp32", "int", "sfu", "mem", "tensor_core")
    norm = {key: max(float(instr_mix.get(key, 0.0)), 0.0) for key in keys}
    total = sum(norm.values())
    if total <= 0.0:
        return {"fp16": 0.0, "fp32": 1.0, "int": 0.0, "sfu": 0.0, "mem": 0.0, "tensor_core": 0.0}
    return {key: value / total for key, value in norm.items()}


def _move_instr_mass(
    instr_mix: Mapping[str, float],
    *,
    source_keys: tuple[str, ...],
    dest_keys: tuple[str, ...],
    delta: float,
) -> dict[str, float]:
    mix = _renormalize_instr_mix(instr_mix)
    available = sum(mix.get(key, 0.0) for key in source_keys)
    moved = min(max(delta, 0.0), available)
    if moved <= 0.0:
        return mix

    source_total = max(available, 1e-12)
    for key in source_keys:
        mix[key] = max(mix.get(key, 0.0) - moved * mix.get(key, 0.0) / source_total, 0.0)

    dest_total = sum(mix.get(key, 0.0) for key in dest_keys)
    if dest_total <= 0.0:
        for key in dest_keys:
            mix[key] = mix.get(key, 0.0) + moved / len(dest_keys)
    else:
        for key in dest_keys:
            mix[key] = mix.get(key, 0.0) + moved * mix.get(key, 0.0) / dest_total
    return _renormalize_instr_mix(mix)


def _with_profile_name(profile: GpuSimKernelProfile, *, suffix: str, description: str) -> GpuSimKernelProfile:
    return replace(
        profile,
        name=f"{profile.name}__{suffix}",
        description=f"{profile.description} [{description}]",
    )


def _cap_smem_bytes(value: float) -> int:
    return min(max(int(round(value)), 0), MAX_SMEM_PER_BLOCK_BYTES)


def make_memory_stressed(profile: GpuSimKernelProfile) -> GpuSimKernelProfile:
    """Construct a baseline with weaker locality and higher memory pressure."""

    stressed = replace(
        profile,
        instr_mix=_move_instr_mass(
            profile.instr_mix,
            source_keys=COMPUTE_KEYS,
            dest_keys=("mem",),
            delta=0.20,
        ),
        smem_per_block=_cap_smem_bytes(profile.smem_per_block * 0.5),
        regs_per_thread=min(profile.regs_per_thread + 8, 255),
    )
    return _with_profile_name(stressed, suffix="memory_stressed", description="memory stressed")


def make_footprint_stressed(profile: GpuSimKernelProfile) -> GpuSimKernelProfile:
    """Construct a baseline with heavier per-block resource footprint."""

    stressed = replace(
        profile,
        regs_per_thread=min(int(round(profile.regs_per_thread * 1.5)) + 8, 255),
        smem_per_block=_cap_smem_bytes(profile.smem_per_block * 1.5 + 16 * 1024),
    )
    return _with_profile_name(stressed, suffix="footprint_stressed", description="footprint stressed")


def make_compute_unoptimized(profile: GpuSimKernelProfile) -> GpuSimKernelProfile:
    """Construct a baseline that gives up specialized math throughput."""

    stressed = replace(
        profile,
        instr_mix=_move_instr_mass(
            profile.instr_mix,
            source_keys=("tensor_core", "fp16"),
            dest_keys=("fp32", "mem"),
            delta=0.25,
        ),
        regs_per_thread=min(profile.regs_per_thread + 12, 255),
    )
    return _with_profile_name(stressed, suffix="compute_unoptimized", description="compute unoptimized")


def generate_recommendation_baselines(
    profiles: list[GpuSimKernelProfile],
) -> list[RecommendationBaseline]:
    """Expand canonical families into a baseline set with controlled stresses."""

    baselines: list[RecommendationBaseline] = []
    for profile in profiles:
        baselines.extend(
            [
                RecommendationBaseline(
                    family=profile.name,
                    stress="base",
                    profile=_with_profile_name(profile, suffix="base", description="baseline"),
                ),
                RecommendationBaseline(
                    family=profile.name,
                    stress="memory_stressed",
                    profile=make_memory_stressed(profile),
                ),
                RecommendationBaseline(
                    family=profile.name,
                    stress="footprint_stressed",
                    profile=make_footprint_stressed(profile),
                ),
                RecommendationBaseline(
                    family=profile.name,
                    stress="compute_unoptimized",
                    profile=make_compute_unoptimized(profile),
                ),
            ]
        )
    return baselines


def apply_intervention(profile: GpuSimKernelProfile, lever: str) -> GpuSimKernelProfile:
    """Return a counterfactual profile after applying the named optimization lever."""

    if lever not in INTERVENTION_KEYS:
        available = ", ".join(INTERVENTION_KEYS)
        raise ValueError(f"unknown intervention '{lever}' (available: {available})")

    if lever == "locality":
        improved = replace(
            profile,
            instr_mix=_move_instr_mass(
                profile.instr_mix,
                source_keys=("mem",),
                dest_keys=COMPUTE_KEYS,
                delta=0.20,
            ),
            smem_per_block=_cap_smem_bytes(profile.smem_per_block * 1.5 + 16 * 1024),
        )
    elif lever == "occupancy":
        improved = replace(
            profile,
            regs_per_thread=max(24, int(round(profile.regs_per_thread * 0.65))),
            smem_per_block=_cap_smem_bytes(profile.smem_per_block * 0.5),
            threads_per_block=min(profile.threads_per_block, 128),
        )
    else:  # tensorize
        improved = replace(
            profile,
            instr_mix=_move_instr_mass(
                profile.instr_mix,
                source_keys=("fp32", "mem", "int", "sfu"),
                dest_keys=("tensor_core", "fp16"),
                delta=0.25,
            ),
            regs_per_thread=max(24, int(round(profile.regs_per_thread * 0.9))),
        )

    return _with_profile_name(improved, suffix=f"iv_{lever}", description=f"intervention={lever}")


def oracle_attainment_ratio(chosen_gain: float, oracle_gain: float) -> float:
    """Return realized gain / oracle gain, with a stable definition at zero gain."""

    if oracle_gain <= 1e-12:
        return 1.0
    return max(0.0, min(chosen_gain / oracle_gain, 1.0))


def _response_features(analysis: object) -> dict[str, np.ndarray]:
    observables = analysis.observables
    families = observables.mean_warp_state_family_fractions
    instr_mix = observables.mean_instr_mix
    bottleneck = analysis.bottleneck

    locality = np.asarray(
        [
            max(0.0, 1.0 - analysis.thermo_state.memory_feed_efficiency),
            0.5 * float(families.get("memory", 0.0))
            + 0.5 * float(observables.mean_memory_stall_fraction),
            0.5 * float(instr_mix.get("mem", 0.0))
            + 0.5 * float(bottleneck.dram_movement_fraction),
        ],
        dtype=float,
    )
    occupancy = np.asarray(
        [
            float(families.get("dependency", 0.0))
            + 0.5 * float(families.get("sync_frontend", 0.0)),
            0.5 * float(observables.mean_reg_utilization)
            + 0.5 * float(observables.mean_smem_utilization),
            max(0.0, 1.0 - float(observables.mean_issue_activity)),
        ],
        dtype=float,
    )
    tensorize = np.asarray(
        [
            float(instr_mix.get("fp32", 0.0)) + 0.5 * float(instr_mix.get("fp16", 0.0)),
            max(0.0, 1.0 - float(instr_mix.get("tensor_core", 0.0))),
            max(0.0, float(families.get("productive", 0.0)) - float(families.get("memory", 0.0)))
            * float(observables.mean_issue_activity),
        ],
        dtype=float,
    )
    return {
        "locality": locality,
        "occupancy": occupancy,
        "tensorize": tensorize,
    }


def fit_statmech_response_model(
    analyses: list[object],
    realized_gains: list[dict[str, float]],
    *,
    ridge: float = 1e-3,
) -> StatMechResponseParams:
    """
    Fit first-order intervention effect sizes from simulator counterfactuals.

    Each lever is fit independently against the clipped positive gain so the
    learned coefficients remain interpretable as "how much upside this signal
    typically converts into efficiency improvement".
    """

    if len(analyses) != len(realized_gains):
        raise ValueError("analyses and realized_gains must have the same length")
    if not analyses:
        return StatMechResponseParams()

    fitted: dict[str, np.ndarray] = {}
    for lever in INTERVENTION_KEYS:
        X = np.vstack([_response_features(analysis)[lever] for analysis in analyses])
        y = np.asarray([max(0.0, float(gains.get(lever, 0.0))) for gains in realized_gains], dtype=float)
        xtx = X.T @ X + ridge * np.eye(X.shape[1], dtype=float)
        xty = X.T @ y
        coeff = np.linalg.solve(xtx, xty)
        fitted[lever] = np.clip(coeff, 0.0, None)

    return StatMechResponseParams(
        locality_feed=float(fitted["locality"][0]),
        locality_memory_pressure=float(fitted["locality"][1]),
        locality_movement=float(fitted["locality"][2]),
        occupancy_dependency=float(fitted["occupancy"][0]),
        occupancy_resource=float(fitted["occupancy"][1]),
        occupancy_issue_gap=float(fitted["occupancy"][2]),
        tensorize_scalar_math=float(fitted["tensorize"][0]),
        tensorize_tensor_gap=float(fitted["tensorize"][1]),
        tensorize_productive_surplus=float(fitted["tensorize"][2]),
    )


def predict_intervention_gains_statmech(
    analysis: object,
    params: StatMechResponseParams,
) -> dict[str, float]:
    """Predict per-lever efficiency gain from the inferred baseline state."""

    features = _response_features(analysis)
    return {
        "locality": float(
            params.locality_feed * features["locality"][0]
            + params.locality_memory_pressure * features["locality"][1]
            + params.locality_movement * features["locality"][2]
        ),
        "occupancy": float(
            params.occupancy_dependency * features["occupancy"][0]
            + params.occupancy_resource * features["occupancy"][1]
            + params.occupancy_issue_gap * features["occupancy"][2]
        ),
        "tensorize": float(
            params.tensorize_scalar_math * features["tensorize"][0]
            + params.tensorize_tensor_gap * features["tensorize"][1]
            + params.tensorize_productive_surplus * features["tensorize"][2]
        ),
    }


def recommend_intervention_statmech_response(
    analysis: object,
    params: StatMechResponseParams,
) -> str:
    gains = predict_intervention_gains_statmech(analysis, params)
    return max(INTERVENTION_KEYS, key=lambda lever: (gains[lever], -INTERVENTION_KEYS.index(lever)))


def statmech_intervention_scores(analysis: object) -> dict[str, float]:
    """
    Score each intervention lever from a thermodynamic trace analysis.

    The scores are intentionally coarse. The experiment tests whether the
    thermodynamic view is useful as a recommendation policy, not whether this
    hand-coded mapping is final.
    """

    observables = analysis.observables
    families = observables.mean_warp_state_family_fractions
    instr_mix = observables.mean_instr_mix
    bottleneck = analysis.bottleneck
    productive = float(families.get("productive", 0.0))
    memory = float(families.get("memory", 0.0))
    dependency = float(families.get("dependency", 0.0))
    sync_frontend = float(families.get("sync_frontend", 0.0))
    tensorizable = float(instr_mix.get("fp16", 0.0)) + float(instr_mix.get("fp32", 0.0))
    tensor_core = float(instr_mix.get("tensor_core", 0.0))

    locality = (
        2.0 * memory
        + 1.5 * observables.mean_memory_stall_fraction
        + 1.0 * (1.0 - analysis.thermo_state.memory_feed_efficiency)
        + 1.0 * bottleneck.dram_movement_fraction
    )
    occupancy = (
        1.75 * dependency
        + 0.75 * sync_frontend
        + 1.5 * bottleneck.stall_fraction
        + 0.75 * bottleneck.sram_overhead_fraction
        + 0.75 * max(0.0, observables.mean_reg_utilization - 0.55)
        + 0.75 * max(0.0, observables.mean_smem_utilization - 0.55)
    )
    tensorize = (
        1.25 * tensorizable
        + 1.0 * max(0.0, productive - memory)
        + 0.75 * max(0.0, 0.30 - tensor_core)
        + 0.5 * max(0.0, observables.mean_issue_activity - observables.mean_memory_stall_fraction)
    )

    if analysis.dominant_phase == "memory_bound":
        locality += 0.25
    elif analysis.dominant_phase == "latency_bound":
        occupancy += 0.25
    elif analysis.dominant_phase == "compute_bound":
        tensorize += 0.25
    elif analysis.dominant_phase == "mixed":
        occupancy += 0.10
        locality += 0.10
        tensorize += 0.10

    return {
        "locality": locality,
        "occupancy": occupancy,
        "tensorize": tensorize,
    }


def recommend_intervention_statmech(analysis: object) -> str:
    scores = statmech_intervention_scores(analysis)
    return max(INTERVENTION_KEYS, key=lambda lever: (scores[lever], -INTERVENTION_KEYS.index(lever)))


def recommend_intervention_raw_counter(analysis: object) -> str:
    families = analysis.observables.mean_warp_state_family_fractions
    coarse = {
        "locality": float(families.get("memory", 0.0)),
        "occupancy": float(families.get("dependency", 0.0)) + float(families.get("sync_frontend", 0.0)),
        "tensorize": float(families.get("productive", 0.0)),
    }
    return max(INTERVENTION_KEYS, key=lambda lever: (coarse[lever], -INTERVENTION_KEYS.index(lever)))


def recommend_intervention_roofline(analysis: object) -> str:
    observables = analysis.observables
    if observables.mean_memory_stall_fraction >= 0.30 or observables.mean_hbm_bw_utilization >= 0.55:
        return "locality"
    if observables.mean_issue_activity >= 0.30 and observables.mean_instr_mix.get("fp32", 0.0) >= 0.30:
        return "tensorize"
    return "occupancy"


def recommend_intervention_occupancy_only(_analysis: object) -> str:
    return "occupancy"

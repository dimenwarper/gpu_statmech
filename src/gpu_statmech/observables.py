"""
Simulator-trace observable extraction and normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .partition_function import WARP_STATE_ACTIVITY


WARP_STATE_KEYS = tuple(WARP_STATE_ACTIVITY)


def _mean_or(default: float, values: Any) -> float:
    if isinstance(values, (list, tuple)) and values:
        return float(np.mean(np.asarray(values, dtype=float)))
    if values is None:
        return default
    return float(values)


def _average_instr_mix(mixes: Any) -> dict[str, float]:
    keys = ("fp16", "fp32", "int", "sfu", "mem", "tensor_core")
    if not isinstance(mixes, list) or not mixes:
        return {}

    totals = {k: 0.0 for k in keys}
    for mix in mixes:
        for key in keys:
            totals[key] += float(mix.get(key, 0.0))

    n = float(len(mixes))
    return {k: totals[k] / n for k in keys}


def _normalize_warp_state_fractions(snapshot: dict[str, Any]) -> dict[str, float]:
    fractions = snapshot.get("warp_state_frac")
    if isinstance(fractions, dict) and fractions:
        total = sum(max(float(fractions.get(key, 0.0)), 0.0) for key in WARP_STATE_KEYS)
        if total > 0.0:
            return {
                key: max(float(fractions.get(key, 0.0)), 0.0) / total
                for key in WARP_STATE_KEYS
            }

    cycles = snapshot.get("warp_state_cycles")
    if isinstance(cycles, dict) and cycles:
        total = sum(max(float(cycles.get(key, 0.0)), 0.0) for key in WARP_STATE_KEYS)
        if total > 0.0:
            return {
                key: max(float(cycles.get(key, 0.0)), 0.0) / total
                for key in WARP_STATE_KEYS
            }

    return {}


def _stall_fraction_from_state_fractions(state_frac: dict[str, float]) -> float | None:
    if not state_frac:
        return None
    non_idle = max(1.0 - float(state_frac.get("idle", 0.0)), 1e-12)
    stalled = sum(
        float(state_frac.get(key, 0.0))
        for key in WARP_STATE_KEYS
        if key not in {"eligible", "idle"}
    )
    return float(np.clip(stalled / non_idle, 0.0, 1.0))


def _memory_stall_fraction_from_state_fractions(state_frac: dict[str, float]) -> float:
    if not state_frac:
        return 0.0
    return float(
        np.clip(
            float(state_frac.get("long_scoreboard", 0.0))
            + float(state_frac.get("mem_throttle", 0.0)),
            0.0,
            1.0,
        )
    )


def _issue_activity_from_state_fractions(state_frac: dict[str, float]) -> float | None:
    if not state_frac:
        return None
    return float(
        np.clip(
            sum(
                float(state_frac.get(key, 0.0)) * float(WARP_STATE_ACTIVITY[key])
                for key in WARP_STATE_KEYS
            ),
            0.0,
            1.0,
        )
    )


def _weighted_mean(values: list[float], weights: np.ndarray) -> float:
    if not values:
        return 0.0
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def canonicalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw gpusim snapshot into the flat schema expected by gpu_statmech.

    If the snapshot is already in the flat schema, it is returned with only
    minimal normalisation.
    """
    state_frac = _normalize_warp_state_fractions(snapshot)
    issue_activity = _issue_activity_from_state_fractions(state_frac)
    memory_stall = _memory_stall_fraction_from_state_fractions(state_frac)
    state_stall = _stall_fraction_from_state_fractions(state_frac)

    if "sm_active_warps" not in snapshot and "sm_instr_mix" not in snapshot:
        active_warps = float(snapshot.get("active_warps", 0.0))
        stall_fraction = float(snapshot.get("stall_fraction", 0.0))
        return {
            "cycle": float(snapshot.get("cycle", 1.0)),
            "active_warps": active_warps,
            "stall_fraction": stall_fraction,
            "issue_activity": float(
                snapshot.get(
                    "issue_activity",
                    issue_activity
                    if issue_activity is not None
                    else np.clip(active_warps * (1.0 - stall_fraction), 0.0, 1.0),
                )
            ),
            "memory_stall_fraction": float(snapshot.get("memory_stall_fraction", memory_stall)),
            "warp_state_frac": state_frac,
            "instr_mix": dict(snapshot.get("instr_mix", {})),
            "l2_hit_rate": float(snapshot.get("l2_hit_rate", 0.0)),
            "hbm_bw_util": float(snapshot.get("hbm_bw_util", snapshot.get("hbm_bw_utilization", 0.0))),
            "smem_util": float(snapshot.get("smem_util", snapshot.get("smem_utilization", 0.0))),
            "reg_util": float(snapshot.get("reg_util", snapshot.get("reg_utilization", 0.0))),
            "bw_nvlink": float(snapshot.get("bw_nvlink", 0.0)),
            "blocks_executed": int(snapshot.get("blocks_executed", 1)),
            "threads_per_block": int(snapshot.get("threads_per_block", 128)),
        }

    sm_max_warps = max(float(snapshot.get("sm_max_warps", 1.0)), 1.0)
    mean_active_warps = _mean_or(0.0, snapshot.get("sm_active_warps", [])) / sm_max_warps
    mean_stall_frac = (
        state_stall
        if state_stall is not None
        else _mean_or(0.0, snapshot.get("sm_stall_frac", []))
    )
    cycle = float(snapshot.get("total_virtual_cycles", 0.0))
    if cycle <= 0.0:
        # Old gpusim traces used `cycle` as a block index, not a duration.
        cycle = 1.0

    return {
        "cycle": cycle,
        "active_warps": float(np.clip(mean_active_warps, 0.0, 1.0)),
        "stall_fraction": float(np.clip(mean_stall_frac, 0.0, 1.0)),
        "issue_activity": float(
            issue_activity
            if issue_activity is not None
            else np.clip(mean_active_warps * (1.0 - mean_stall_frac), 0.0, 1.0)
        ),
        "memory_stall_fraction": memory_stall,
        "warp_state_frac": state_frac,
        "instr_mix": _average_instr_mix(snapshot.get("sm_instr_mix", [])),
        "l2_hit_rate": float(snapshot.get("l2_hit_rate", 0.0)),
        "hbm_bw_util": float(snapshot.get("hbm_bw_utilization", 0.0)),
        "smem_util": float(snapshot.get("smem_utilization", 0.0)),
        "reg_util": float(snapshot.get("reg_utilization", 0.0)),
        "bw_nvlink": float(snapshot.get("bw_nvlink", 0.0)),
        "blocks_executed": 1,
        "threads_per_block": int(snapshot.get("threads_per_block", 128)),
    }


@dataclass(frozen=True)
class TraceObservables:
    """
    Aggregate observables extracted from a snapshot trace.
    """
    mean_active_warp_fraction: float
    mean_stall_fraction: float
    mean_issue_activity: float
    mean_memory_stall_fraction: float
    mean_reg_utilization: float
    mean_smem_utilization: float
    mean_l2_hit_rate: float
    mean_hbm_bw_utilization: float
    mean_nvlink_bw_utilization: float
    mean_warp_state_fractions: dict[str, float]
    n_snapshots: int

    @property
    def memory_feed_efficiency_proxy(self) -> float:
        """
        A cold-to-hot feedability proxy derived from simulator observables.

        Higher L2 hit rate and lower HBM pressure imply easier feeding of useful
        work from colder levels.
        """
        return float(
            np.clip(
                0.4 * self.mean_l2_hit_rate
                + 0.4 * (1.0 - self.mean_hbm_bw_utilization)
                + 0.2 * (1.0 - self.mean_memory_stall_fraction),
                0.0,
                1.0,
            )
        )


def aggregate_trace_observables(snapshots: list[dict[str, Any]]) -> TraceObservables:
    """
    Aggregate simulator observables from either flat or raw gpusim snapshots.
    """
    if not snapshots:
        return TraceObservables(
            mean_active_warp_fraction=0.0,
            mean_stall_fraction=0.0,
            mean_issue_activity=0.0,
            mean_memory_stall_fraction=0.0,
            mean_reg_utilization=0.0,
            mean_smem_utilization=0.0,
            mean_l2_hit_rate=0.0,
            mean_hbm_bw_utilization=0.0,
            mean_nvlink_bw_utilization=0.0,
            mean_warp_state_fractions={key: 0.0 for key in WARP_STATE_KEYS},
            n_snapshots=0,
        )

    snaps = [canonicalize_snapshot(s) for s in snapshots]
    weights = np.asarray([max(float(s.get("cycle", 1.0)), 0.0) for s in snaps], dtype=float)
    if not np.any(weights > 0.0):
        weights = np.ones(len(snaps), dtype=float)

    active = _weighted_mean([s["active_warps"] for s in snaps], weights)
    stall = _weighted_mean([s["stall_fraction"] for s in snaps], weights)
    issue = _weighted_mean([s["issue_activity"] for s in snaps], weights)
    memory_stall = _weighted_mean([s["memory_stall_fraction"] for s in snaps], weights)
    reg = _weighted_mean([s["reg_util"] for s in snaps], weights)
    smem = _weighted_mean([s["smem_util"] for s in snaps], weights)
    l2 = _weighted_mean([s["l2_hit_rate"] for s in snaps], weights)
    hbm = _weighted_mean([s["hbm_bw_util"] for s in snaps], weights)
    nvlink = _weighted_mean([s["bw_nvlink"] for s in snaps], weights)
    state_fracs = {
        key: _weighted_mean(
            [float(s.get("warp_state_frac", {}).get(key, 0.0)) for s in snaps],
            weights,
        )
        for key in WARP_STATE_KEYS
    }

    return TraceObservables(
        mean_active_warp_fraction=active,
        mean_stall_fraction=stall,
        mean_issue_activity=issue,
        mean_memory_stall_fraction=memory_stall,
        mean_reg_utilization=reg,
        mean_smem_utilization=smem,
        mean_l2_hit_rate=l2,
        mean_hbm_bw_utilization=hbm,
        mean_nvlink_bw_utilization=nvlink,
        mean_warp_state_fractions=state_fracs,
        n_snapshots=len(snaps),
    )

"""
Simulator-trace observable extraction and normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def canonicalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw gpusim snapshot into the flat schema expected by gpu_statmech.

    If the snapshot is already in the flat schema, it is returned with only
    minimal normalisation.
    """
    if "sm_active_warps" not in snapshot and "sm_instr_mix" not in snapshot:
        return {
            "cycle": float(snapshot.get("cycle", 1.0)),
            "active_warps": float(snapshot.get("active_warps", 0.0)),
            "stall_fraction": float(snapshot.get("stall_fraction", 0.0)),
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
    mean_stall_frac = _mean_or(0.0, snapshot.get("sm_stall_frac", []))

    return {
        # gpusim's `cycle` field is a block index, not a duration in cycles.
        # Use a unit-duration interval for normalized per-snapshot analysis.
        "cycle": 1.0,
        "active_warps": float(np.clip(mean_active_warps, 0.0, 1.0)),
        "stall_fraction": float(np.clip(mean_stall_frac, 0.0, 1.0)),
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
    mean_reg_utilization: float
    mean_smem_utilization: float
    mean_l2_hit_rate: float
    mean_hbm_bw_utilization: float
    mean_nvlink_bw_utilization: float
    n_snapshots: int

    @property
    def memory_feed_efficiency_proxy(self) -> float:
        """
        A cold-to-hot feedability proxy derived from simulator observables.

        Higher L2 hit rate and lower HBM pressure imply easier feeding of useful
        work from colder levels.
        """
        return float(np.clip(
            0.5 * self.mean_l2_hit_rate
            + 0.5 * (1.0 - self.mean_hbm_bw_utilization),
            0.0,
            1.0,
        ))


def aggregate_trace_observables(snapshots: list[dict[str, Any]]) -> TraceObservables:
    """
    Aggregate simulator observables from either flat or raw gpusim snapshots.
    """
    if not snapshots:
        return TraceObservables(
            mean_active_warp_fraction=0.0,
            mean_stall_fraction=0.0,
            mean_issue_activity=0.0,
            mean_reg_utilization=0.0,
            mean_smem_utilization=0.0,
            mean_l2_hit_rate=0.0,
            mean_hbm_bw_utilization=0.0,
            mean_nvlink_bw_utilization=0.0,
            n_snapshots=0,
        )

    snaps = [canonicalize_snapshot(s) for s in snapshots]

    active = float(np.mean([s["active_warps"] for s in snaps]))
    stall = float(np.mean([s["stall_fraction"] for s in snaps]))
    reg = float(np.mean([s["reg_util"] for s in snaps]))
    smem = float(np.mean([s["smem_util"] for s in snaps]))
    l2 = float(np.mean([s["l2_hit_rate"] for s in snaps]))
    hbm = float(np.mean([s["hbm_bw_util"] for s in snaps]))
    nvlink = float(np.mean([s["bw_nvlink"] for s in snaps]))
    issue = float(np.clip(active * (1.0 - stall), 0.0, 1.0))

    return TraceObservables(
        mean_active_warp_fraction=active,
        mean_stall_fraction=stall,
        mean_issue_activity=issue,
        mean_reg_utilization=reg,
        mean_smem_utilization=smem,
        mean_l2_hit_rate=l2,
        mean_hbm_bw_utilization=hbm,
        mean_nvlink_bw_utilization=nvlink,
        n_snapshots=len(snaps),
    )

"""
Power and energy model for GPU execution.

Maps simulator MicrostateSnapshot fields → physical energy decomposition:

    E_total  = E_compute + E_sram + E_dram + E_leakage
    W_hw     = E_total - Q_waste
    η_hw     = W_hw / E_total

All energies are in nano-Joules (nJ) per kernel execution.

The model is parameterised by the H100 power envelope and per-component
energy costs derived from published characterisation data.  It is designed
to accept the dict/dataclass output of the gpusim Python bindings directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observables import canonicalize_snapshot
from .partition_function import (
    INSTRUCTION_CLASS_INPUT_ENERGY,
    INSTRUCTION_CLASS_USEFUL_WORK,
    WARP_STATE_BASE_INPUT_ENERGY,
)


# ---------------------------------------------------------------------------
# Energy cost parameters (H100 SXM5)
# ---------------------------------------------------------------------------

@dataclass
class EnergyParams:
    """Per-operation energy costs in pico-Joules."""

    # Compute
    fp16_mac_pj: float = 0.08    # FP16 multiply-accumulate (tensor core)
    fp32_mac_pj: float = 0.30    # FP32 multiply-accumulate (CUDA core)
    int_op_pj:   float = 0.05    # integer operation
    sfu_op_pj:   float = 0.40    # special function unit (sin/cos/exp/sqrt)

    # Memory (per byte transferred)
    reg_access_pj_per_byte:  float = 0.10
    smem_access_pj_per_byte: float = 0.50
    l2_access_pj_per_byte:   float = 2.00
    hbm_access_pj_per_byte:  float = 20.0

    # Leakage (per SM per clock cycle, in pJ)
    leakage_per_sm_per_cycle_pj: float = 5.0

    # Clock frequency (GHz) — used to convert cycle counts to time
    clock_ghz: float = 1.98      # H100 SXM5 boost clock


H100_ENERGY_PARAMS = EnergyParams()

_STALL_STATE_KEYS = (
    "long_scoreboard",
    "short_scoreboard",
    "barrier",
    "exec_dep",
    "mem_throttle",
    "fetch",
)


def _normalized_instr_mix(instr_mix: dict[str, Any]) -> dict[str, float]:
    keys = tuple(INSTRUCTION_CLASS_INPUT_ENERGY)
    if not instr_mix:
        return {"fp32": 1.0, **{key: 0.0 for key in keys if key != "fp32"}}

    norm = {key: max(float(instr_mix.get(key, 0.0)), 0.0) for key in keys}
    total = sum(norm.values())
    if total <= 0.0:
        return {"fp32": 1.0, **{key: 0.0 for key in keys if key != "fp32"}}
    return {key: value / total for key, value in norm.items()}


def _state_fractions(snapshot: dict[str, Any], active_warps: float, stall_frac: float) -> dict[str, float]:
    state_frac = snapshot.get("warp_state_frac", {})
    if isinstance(state_frac, dict) and state_frac:
        total = sum(max(float(state_frac.get(key, 0.0)), 0.0) for key in WARP_STATE_BASE_INPUT_ENERGY)
        if total > 0.0:
            return {
                key: max(float(state_frac.get(key, 0.0)), 0.0) / total
                for key in WARP_STATE_BASE_INPUT_ENERGY
            }

    active = float(max(min(active_warps, 1.0), 0.0))
    stalled = active * float(max(min(stall_frac, 1.0), 0.0))
    eligible = max(active - stalled, 0.0)
    idle = max(1.0 - active, 0.0)
    long_scoreboard = 0.6 * stalled
    short_scoreboard = 0.25 * stalled
    exec_dep = 0.15 * stalled
    return {
        "eligible": eligible,
        "long_scoreboard": long_scoreboard,
        "short_scoreboard": short_scoreboard,
        "barrier": 0.0,
        "exec_dep": exec_dep,
        "mem_throttle": 0.0,
        "fetch": 0.0,
        "idle": idle,
    }


# ---------------------------------------------------------------------------
# Energy decomposition result
# ---------------------------------------------------------------------------

@dataclass
class EnergyDecomposition:
    """
    Full energy decomposition for a single kernel execution.

    E_total = E_compute + E_sram + E_dram + E_leakage
    Q_waste = E_stall + E_idle + E_unnecessary_movement
    W_hw    = E_total - Q_waste
    η_hw    = W_hw / E_total
    """
    # Input energy (nJ)
    E_total_nj: float

    # Useful hardware work (nJ)
    W_hw_nj: float

    # Waste decomposition (nJ)
    Q_stall_nj: float            # cycles lost to pipeline stalls
    Q_idle_nj: float             # cycles with no active warps
    Q_unnecessary_movement_nj: float  # redundant data movement

    # Component breakdown (nJ)
    E_compute_nj: float          # arithmetic operations
    E_sram_nj: float             # register + shared memory accesses
    E_dram_nj: float             # L2 + HBM accesses
    E_leakage_nj: float          # static leakage

    # Derived efficiency
    @property
    def eta_hw(self) -> float:
        """η_hw = W_hw / E_total"""
        return self.W_hw_nj / max(self.E_total_nj, 1e-12)

    @property
    def Q_waste_nj(self) -> float:
        return self.Q_stall_nj + self.Q_idle_nj + self.Q_unnecessary_movement_nj

    @property
    def waste_fraction(self) -> float:
        return self.Q_waste_nj / max(self.E_total_nj, 1e-12)

    def waste_breakdown(self) -> dict[str, float]:
        """Fractional waste by source (each as fraction of E_total)."""
        total = max(self.E_total_nj, 1e-12)
        return {
            "stall":                self.Q_stall_nj / total,
            "idle":                 self.Q_idle_nj / total,
            "unnecessary_movement": self.Q_unnecessary_movement_nj / total,
            "useful":               self.W_hw_nj / total,
        }


# ---------------------------------------------------------------------------
# Core energy model
# ---------------------------------------------------------------------------

def compute_energy(
    snapshot: dict[str, Any],
    n_sm: int = 132,
    params: EnergyParams | None = None,
) -> EnergyDecomposition:
    """
    Compute the energy decomposition from a single gpusim MicrostateSnapshot.

    The snapshot dict is expected to have the following fields (all optional
    with sensible defaults so partial snapshots work):

        cycle           : int   — clock cycle of this snapshot
        active_warps    : float — mean active warps across SMs (fraction of max)
        stall_fraction  : float — mean fraction of cycles lost to stalls
        instr_mix       : dict  — {fp16, fp32, int, sfu, mem, tensor_core} fractions
        l2_hit_rate     : float — L2 cache hit rate ∈ [0, 1]
        hbm_bw_util     : float — HBM bandwidth utilisation ∈ [0, 1]
        smem_util       : float — shared memory utilisation ∈ [0, 1]
        blocks_executed : int   — number of thread blocks in this snapshot
        threads_per_block: int  — threads per block
    """
    if params is None:
        params = H100_ENERGY_PARAMS

    snap = canonicalize_snapshot(snapshot)

    cycle = float(snap.get("cycle", 1.0))
    active_warps = float(snap.get("active_warps", 0.5))
    stall_frac = float(snap.get("stall_fraction", 0.2))
    memory_stall_frac = float(snap.get("memory_stall_fraction", 0.0))
    instr_mix = _normalized_instr_mix(snap.get("instr_mix", {}))
    l2_hit_rate = float(snap.get("l2_hit_rate", 0.8))
    hbm_bw_util = float(snap.get("hbm_bw_util", 0.3))
    smem_util = float(snap.get("smem_util", 0.5))
    reg_util = float(snap.get("reg_util", 0.5))
    issue_activity = float(snap.get("issue_activity", active_warps * (1.0 - stall_frac)))

    blocks = int(snap.get("blocks_executed", 1))
    threads_per_blk = int(snap.get("threads_per_block", 128))
    active_sm_count = int(snap.get("active_sm_count", n_sm))
    total_warp_cycles = float(
        snap.get("total_warp_cycles", max(blocks * threads_per_blk / 32.0 * cycle, 0.0))
    )

    state_frac = _state_fractions(snap, active_warps, stall_frac)
    mean_state_input = sum(
        float(state_frac.get(state, 0.0)) * WARP_STATE_BASE_INPUT_ENERGY[state]
        for state in WARP_STATE_BASE_INPUT_ENERGY
    )
    effective_issue_activity = float(np.clip(issue_activity, 0.0, 1.0))
    issue_warp_cycles = total_warp_cycles * effective_issue_activity

    instr_input_pj = sum(
        instr_mix[key] * INSTRUCTION_CLASS_INPUT_ENERGY[key]
        for key in INSTRUCTION_CLASS_INPUT_ENERGY
    )
    useful_work_pj = sum(
        instr_mix[key] * INSTRUCTION_CLASS_USEFUL_WORK[key]
        for key in INSTRUCTION_CLASS_USEFUL_WORK
    )
    mem_share = float(instr_mix.get("mem", 0.0))

    control_input_pj = total_warp_cycles * mean_state_input + issue_warp_cycles * instr_input_pj

    reg_bytes = issue_warp_cycles * 8.0 * max(reg_util, 0.25)
    smem_bytes = issue_warp_cycles * 16.0 * smem_util
    l2_bytes = issue_warp_cycles * 16.0 * mem_share * l2_hit_rate
    hbm_bytes = issue_warp_cycles * 32.0 * mem_share * max(
        hbm_bw_util,
        (1.0 - l2_hit_rate) * (0.5 + 0.5 * memory_stall_frac),
    )

    E_sram_pj = (
        reg_bytes * params.reg_access_pj_per_byte
        + smem_bytes * params.smem_access_pj_per_byte
    )
    E_dram_pj = (
        l2_bytes * params.l2_access_pj_per_byte
        + hbm_bytes * params.hbm_access_pj_per_byte
    )
    E_compute_pj = control_input_pj
    E_leakage_pj = params.leakage_per_sm_per_cycle_pj * max(active_sm_count, 1) * cycle

    E_total_pj = control_input_pj + E_sram_pj + E_dram_pj + E_leakage_pj
    E_total_nj = E_total_pj * 1e-3

    stall_proxy = (
        sum(float(state_frac.get(state, 0.0)) for state in _STALL_STATE_KEYS)
        * (control_input_pj + memory_stall_frac * E_dram_pj)
    )
    idle_proxy = float(state_frac.get("idle", 0.0)) * (control_input_pj + E_leakage_pj)
    movement_proxy = (
        0.5 * mem_share * (1.0 - l2_hit_rate) + 0.5 * hbm_bw_util
    ) * E_dram_pj
    feed_efficiency = float(
        max(
            0.0,
            min(
                1.0,
                0.5 * l2_hit_rate
                + 0.3 * (1.0 - hbm_bw_util)
                + 0.2 * (1.0 - memory_stall_frac),
            ),
        )
    )
    useful_proxy = issue_warp_cycles * useful_work_pj * feed_efficiency

    proxy_total = useful_proxy + stall_proxy + idle_proxy + movement_proxy
    if proxy_total <= 1e-18:
        proxy_total = 1.0
    scale = E_total_pj / proxy_total

    W_hw_pj = useful_proxy * scale
    Q_stall_pj = stall_proxy * scale
    Q_idle_pj = idle_proxy * scale
    Q_unnecessary_pj = movement_proxy * scale

    return EnergyDecomposition(
        E_total_nj=E_total_nj,
        W_hw_nj=W_hw_pj * 1e-3,
        Q_stall_nj=Q_stall_pj * 1e-3,
        Q_idle_nj=Q_idle_pj * 1e-3,
        Q_unnecessary_movement_nj=Q_unnecessary_pj * 1e-3,
        E_compute_nj=E_compute_pj * 1e-3,
        E_sram_nj=E_sram_pj * 1e-3,
        E_dram_nj=E_dram_pj * 1e-3,
        E_leakage_nj=E_leakage_pj * 1e-3,
    )


def aggregate_energy(snapshots: list[dict[str, Any]], **kwargs) -> EnergyDecomposition:
    """
    Aggregate energy decompositions across a list of MicrostateSnapshots.

    Returns a single EnergyDecomposition summed over all snapshots.
    """
    if not snapshots:
        return EnergyDecomposition(
            E_total_nj=0.0, W_hw_nj=0.0,
            Q_stall_nj=0.0, Q_idle_nj=0.0, Q_unnecessary_movement_nj=0.0,
            E_compute_nj=0.0, E_sram_nj=0.0, E_dram_nj=0.0, E_leakage_nj=0.0,
        )

    parts = [compute_energy(s, **kwargs) for s in snapshots]

    def _sum(attr: str) -> float:
        return sum(getattr(p, attr) for p in parts)

    return EnergyDecomposition(
        E_total_nj=_sum("E_total_nj"),
        W_hw_nj=_sum("W_hw_nj"),
        Q_stall_nj=_sum("Q_stall_nj"),
        Q_idle_nj=_sum("Q_idle_nj"),
        Q_unnecessary_movement_nj=_sum("Q_unnecessary_movement_nj"),
        E_compute_nj=_sum("E_compute_nj"),
        E_sram_nj=_sum("E_sram_nj"),
        E_dram_nj=_sum("E_dram_nj"),
        E_leakage_nj=_sum("E_leakage_nj"),
    )

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

from dataclasses import dataclass, field
from typing import Any

from .observables import canonicalize_snapshot


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

    # Extract snapshot fields with defaults
    cycle           = float(snap.get("cycle", 1))
    active_warps    = float(snap.get("active_warps", 0.5))    # fraction [0,1]
    stall_frac      = float(snap.get("stall_fraction", 0.2))
    instr_mix       = snap.get("instr_mix", {})
    l2_hit_rate     = float(snap.get("l2_hit_rate", 0.8))
    hbm_bw_util     = float(snap.get("hbm_bw_util", 0.3))
    smem_util       = float(snap.get("smem_util", 0.5))
    blocks          = int(snap.get("blocks_executed", 1))
    threads_per_blk = int(snap.get("threads_per_block", 128))

    # Instruction mix fractions (default to an all-fp32 mix)
    f_fp16 = float(instr_mix.get("fp16", 0.0))
    f_fp32 = float(instr_mix.get("fp32", 1.0 - f_fp16))
    f_int  = float(instr_mix.get("int", 0.0))
    f_sfu  = float(instr_mix.get("sfu", 0.0))
    f_tc   = float(instr_mix.get("tensor_core", 0.0))
    # normalise
    total_f = max(f_fp16 + f_fp32 + f_int + f_sfu + f_tc, 1e-9)
    f_fp16 /= total_f; f_fp32 /= total_f; f_int /= total_f
    f_sfu  /= total_f; f_tc   /= total_f

    # Total thread-cycles for this snapshot
    total_threads   = blocks * threads_per_blk
    thread_cycles   = total_threads * cycle

    # --- Compute energy (nJ) ---
    # Number of active arithmetic thread-cycles
    active_thread_cycles = thread_cycles * active_warps * (1.0 - stall_frac)

    # Ops per thread-cycle (assuming 1 op/thread/cycle when active)
    n_fp16_ops = active_thread_cycles * f_fp16
    n_fp32_ops = active_thread_cycles * f_fp32
    n_int_ops  = active_thread_cycles * f_int
    n_sfu_ops  = active_thread_cycles * f_sfu
    n_tc_ops   = active_thread_cycles * f_tc

    E_compute_pj = (
        n_fp16_ops * params.fp16_mac_pj +
        n_fp32_ops * params.fp32_mac_pj +
        n_int_ops  * params.int_op_pj   +
        n_sfu_ops  * params.sfu_op_pj   +
        n_tc_ops   * params.fp16_mac_pj   # tensor core same cost as FP16
    )

    # --- SRAM energy (nJ) ---
    # Shared memory bytes touched per active thread-cycle (rough proxy: 16B/thread)
    smem_bytes = active_thread_cycles * 16.0 * smem_util
    reg_bytes  = active_thread_cycles * 8.0          # ~8B register traffic/thread/cycle

    E_sram_pj = (
        smem_bytes * params.smem_access_pj_per_byte +
        reg_bytes  * params.reg_access_pj_per_byte
    )

    # --- DRAM energy (nJ) ---
    # HBM bandwidth: H100 has ~3.35 TB/s ≈ 2094 bytes/cycle at 1.6 GHz
    hbm_bytes_per_cycle = 2094.0 * n_sm
    hbm_bytes_total = hbm_bytes_per_cycle * cycle * hbm_bw_util

    # L2 bytes = HBM bytes / (1 - l2_hit_rate)  (cache amplification)
    l2_bytes = hbm_bytes_total / max(1.0 - l2_hit_rate, 0.01)

    E_dram_pj = (
        hbm_bytes_total * params.hbm_access_pj_per_byte +
        l2_bytes        * params.l2_access_pj_per_byte
    )

    # --- Leakage (nJ) ---
    E_leakage_pj = params.leakage_per_sm_per_cycle_pj * n_sm * cycle

    # --- Total ---
    E_total_pj = E_compute_pj + E_sram_pj + E_dram_pj + E_leakage_pj
    E_total_nj = E_total_pj * 1e-3

    # --- Waste decomposition ---
    # Stall waste: energy burned during stall cycles (mostly leakage + incomplete pipes)
    stall_cycles = thread_cycles * stall_frac
    Q_stall_pj = (
        stall_cycles * params.leakage_per_sm_per_cycle_pj * n_sm * 0.5  # half-power stall
    )

    # Idle waste: energy from SMs with no active warps
    idle_frac = max(0.0, 1.0 - active_warps)
    Q_idle_pj = idle_frac * E_leakage_pj

    # Unnecessary data movement (proxy: L2 miss traffic beyond what arithmetic needs)
    # A perfectly reusing kernel would load each byte exactly once from HBM.
    # Excess = (actual HBM traffic) - (minimum HBM traffic for ops performed)
    min_hbm_bytes = active_thread_cycles * (f_fp32 * 8.0 + f_fp16 * 4.0)  # operand bytes
    excess_hbm_bytes = max(0.0, hbm_bytes_total - min_hbm_bytes)
    Q_unnecessary_pj = excess_hbm_bytes * params.hbm_access_pj_per_byte

    Q_waste_pj = Q_stall_pj + Q_idle_pj + Q_unnecessary_pj
    W_hw_pj    = max(0.0, E_total_pj - Q_waste_pj)

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

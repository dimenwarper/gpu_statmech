"""
Experiment 02: Memory Hierarchy Thermal Fingerprint
=====================================================
The GPU memory hierarchy (reg → smem → L2 → HBM) is a 1-D chain of
thermal reservoirs at increasing "temperatures" (latency-cycles).

This experiment:
  1. Shows the effective temperature of each level (T_eff = latency_ratio).
  2. Plots how Z_memory changes as a function of β — each level's
     contribution to the transfer-matrix product.
  3. Sweeps the working-set size across levels and shows how the
     minimum reuse requirement changes.
  4. Shows the roofline ridge point as a function of HBM bandwidth.

All purely from the hardware spec numbers in H100_MEMORY_LEVELS.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    MemoryLevel,
    _transfer_matrix,
    z_memory,
    beta_sweep,
)
from gpu_statmech.carnot import derive_carnot_limit

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

print("=" * 60)
print("Experiment 02: Memory Hierarchy Thermal Fingerprint")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Print memory level properties
# ---------------------------------------------------------------------------

print()
print("  H100 Memory Hierarchy")
print(f"  {'Level':<12} {'Capacity':>12} {'Bandwidth':>14} {'Latency':>10} {'Energy/byte':>13} {'T_eff':>8}")
print("  " + "-" * 73)
ref_lat = H100_MEMORY_LEVELS[0].latency_cycles
for lvl in H100_MEMORY_LEVELS:
    T_eff = lvl.latency_cycles / ref_lat
    cap_str = (f"{lvl.capacity_bytes // 1024} KB"
               if lvl.capacity_bytes < 1024**3
               else f"{lvl.capacity_bytes // (1024**3)} GB")
    print(f"  {lvl.name:<12} {cap_str:>12} "
          f"{lvl.bandwidth_bytes_per_cycle:>10.0f} B/cyc "
          f"{lvl.latency_cycles:>8.0f} cyc "
          f"{lvl.energy_per_byte_pj:>9.1f} pJ/B "
          f"{T_eff:>8.1f}×")

# ---------------------------------------------------------------------------
# 2. Z_memory β-sweep and per-level transfer matrix norms
# ---------------------------------------------------------------------------

betas = np.linspace(0.05, 8.0, 200).tolist()
z_mem_vals = [z_memory(b, H100_MEMORY_LEVELS, n_bins=64) for b in betas]

# Per-level transfer matrix spectral norm (largest singular value) at each β
# This captures how much each level "squeezes" the state distribution
norms = {lvl.name: [] for lvl in H100_MEMORY_LEVELS[:-1]}
for b in betas:
    for i in range(len(H100_MEMORY_LEVELS) - 1):
        T = _transfer_matrix(H100_MEMORY_LEVELS[i], H100_MEMORY_LEVELS[i + 1], b, n_bins=32)
        norms[H100_MEMORY_LEVELS[i].name].append(float(np.linalg.norm(T, ord=2)))

# ---------------------------------------------------------------------------
# 3. Minimum reuse factors from the Carnot limit
# ---------------------------------------------------------------------------

limit = derive_carnot_limit()
print()
print("  Minimum reuse factors (from Carnot-optimal conditions)")
print(f"  {'Level':<12} {'min_reuse':>12}  {'Interpretation'}")
print("  " + "-" * 60)
for lvl_name, reuse in limit.min_reuse_factors.items():
    print(f"  {lvl_name:<12} {reuse:>10.1f}×  "
          f"  each byte loaded must be used ≥{reuse:.0f}× before eviction")

# ---------------------------------------------------------------------------
# 4. Roofline ridge as function of HBM bandwidth
# ---------------------------------------------------------------------------

bw_range_gb_s = np.linspace(100, 4000, 50)  # GB/s
bw_range_bytes_per_cycle = bw_range_gb_s * 1e9 / 1.98e9  # at 1.98 GHz
ridge_points = (H100_SM_CONFIG.peak_flops_per_cycle
                / bw_range_bytes_per_cycle)
h100_ridge = (H100_SM_CONFIG.peak_flops_per_cycle
              / H100_MEMORY_LEVELS[-1].bandwidth_bytes_per_cycle)
print()
print("  Roofline Ridge Point (AI_min = peak_FLOPS / HBM_bandwidth)")
print(f"    H100 peak FLOPS/cycle  = {H100_SM_CONFIG.peak_flops_per_cycle:.0f}")
print(f"    H100 HBM bandwidth     = {H100_MEMORY_LEVELS[-1].bandwidth_bytes_per_cycle:.0f} B/cycle")
print(f"    Roofline ridge (H100)  = {h100_ridge:.2f} FLOP/byte")
print(f"    (Operations above this are compute-bound; below are memory-bound)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("H100 Memory Hierarchy Thermal Fingerprint", fontsize=14, fontweight="bold")

# --- Z_memory(β) -----------------------------------------------------------
ax = axes[0, 0]
ax.semilogy(betas, z_mem_vals, color="#2563eb", lw=2)
ax.axvline(limit.beta_optimal, color="#dc2626", ls="--", lw=1.5, label=f"β_opt = {limit.beta_optimal:.2f}")
ax.set_xlabel("β", fontsize=11)
ax.set_ylabel("Z_memory  (log scale)", fontsize=11)
ax.set_title("Memory Partition Function  Z_memory(β)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# --- Transfer matrix norms per level ---------------------------------------
ax = axes[0, 1]
colors = ["#2563eb", "#16a34a", "#d97706"]
for (lvl_name, norm_vals), color in zip(norms.items(), colors):
    ax.plot(betas, norm_vals, lw=2, color=color,
            label=f"{lvl_name} → next")
ax.axvline(limit.beta_optimal, color="#dc2626", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("β", fontsize=11)
ax.set_ylabel("Spectral norm  ‖T‖₂", fontsize=11)
ax.set_title("Transfer Matrix Norms per Level  ‖T_{l→l+1}‖₂(β)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Effective temperature ladder ------------------------------------------
ax = axes[1, 0]
levels = list(limit.T_eff.keys())
T_effs = list(limit.T_eff.values())
colors_bar = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
bars = ax.bar(levels, T_effs, color=colors_bar, edgecolor="white", linewidth=1.5)
ax.set_ylabel("T_eff  (latency ratio vs registers)", fontsize=11)
ax.set_title("Effective Temperatures  T_eff = latency(l) / latency(reg)", fontsize=12)
for bar, T in zip(bars, T_effs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{T:.0f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(T_effs) * 1.15)
ax.grid(axis="y", alpha=0.3)
ax.set_yscale("log")

# --- Roofline ridge vs HBM bandwidth ---------------------------------------
ax = axes[1, 1]
ax.plot(bw_range_gb_s, ridge_points, color="#7c3aed", lw=2)
ax.axvline(3350, color="#dc2626", ls="--", lw=1.5, label="H100 HBM (3.35 TB/s)")
ax.axhline(h100_ridge, color="#16a34a", ls=":", lw=1.5,
           label=f"H100 ridge = {h100_ridge:.1f} FLOP/byte")
ax.set_xlabel("HBM Bandwidth  (GB/s)", fontsize=11)
ax.set_ylabel("Roofline Ridge  (FLOP/byte)", fontsize=11)
ax.set_title("Roofline Ridge vs HBM Bandwidth", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out = FIGURES / "02_memory_hierarchy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved → {out}")
plt.close()

print("\nDone.\n")

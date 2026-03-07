"""
Experiment 01: H100 Carnot Curve
=================================
Sweeps β from low (hot/loaded) to high (cold/idle) and plots:
  - η_hw(β) = 1 - <E(β)>          hardware efficiency
  - S(β)                           entropy (execution-state degeneracy)
  - Cv(β)                          specific heat (sensitivity to load)
  - Decomposed log-Z contributions  (compute / memory / comm)

The peak of η_hw is η_hw,max — the H100 Carnot limit.

All from hardware spec numbers only. No simulator, no GPU needed.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make gpu_statmech importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    beta_sweep,
)
from gpu_statmech.carnot import derive_carnot_limit, verify_roofline_recovery

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Run the β sweep
# ---------------------------------------------------------------------------

print("=" * 60)
print("Experiment 01: H100 Carnot Curve")
print("=" * 60)

betas = np.linspace(0.05, 8.0, 300).tolist()
print(f"  β sweep: {betas[0]:.2f} → {betas[-1]:.2f}  ({len(betas)} points)")

states = beta_sweep(betas)
etas      = [1.0 - max(0.0, min(1.0, s.mean_waste)) for s in states]
entropies = [s.entropy         for s in states]
cv        = [s.specific_heat   for s in states]
lz_c      = [s.log_Z_compute   for s in states]
lz_m      = [s.log_Z_memory    for s in states]
lz_k      = [s.log_Z_comm      for s in states]

# ---------------------------------------------------------------------------
# Carnot limit
# ---------------------------------------------------------------------------

limit = derive_carnot_limit()
beta_opt = limit.beta_optimal
eta_max  = limit.eta_hw_max

roofline = verify_roofline_recovery()

print()
print("  H100 Carnot Limit")
print(f"    η_hw,max          = {eta_max:.4f}  ({eta_max*100:.2f}%)")
print(f"    β_optimal         = {beta_opt:.4f}")
print(f"    naive Carnot η    = {roofline['naive_carnot_efficiency']:.4f}"
      f"  (1 - T_reg/T_HBM)")
print()
print("  Roofline Recovery")
print(f"    roofline ridge    = {roofline['roofline_ridge_flop_per_byte']:.2f} FLOP/byte")
print(f"    Carnot AI_min     = {roofline['carnot_ai_min_flop_per_byte']:.2f} FLOP/byte")
print(f"    ratio             = {roofline['ratio']:.4f}  (should be ≈ 1.0)")
print()
print("  Effective temperatures (T_eff = latency_cycles / latency_reg)")
for name, T in limit.T_eff.items():
    print(f"    {name:<12s}  T_eff = {T:.1f}")
print()
print("  Carnot-optimal conditions")
print(f"    AI_min            = {limit.roofline_intensity:.2f} FLOP/byte")
print(f"    min warp occ.     = {limit.min_warp_occupancy:.3f}")
for lvl, r in limit.min_reuse_factors.items():
    print(f"    min reuse {lvl:<6s}  = {r:.1f}×")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# --- η_hw(β) ---------------------------------------------------------------
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(betas, [e * 100 for e in etas], color="#2563eb", lw=2, label=r"$\eta_{hw}(\beta)$")
ax0.axvline(beta_opt, color="#dc2626", ls="--", lw=1.5, label=f"β_optimal = {beta_opt:.2f}")
ax0.axhline(eta_max * 100, color="#16a34a", ls=":", lw=1.5,
            label=f"η_hw,max = {eta_max*100:.2f}%")
ax0.set_xlabel("β  (inverse resource-pressure)", fontsize=11)
ax0.set_ylabel("η_hw  (%)", fontsize=11)
ax0.set_title("H100 Hardware Efficiency vs Inverse Temperature", fontsize=13, fontweight="bold")
ax0.legend(fontsize=10)
ax0.set_xlim(betas[0], betas[-1])
ax0.set_ylim(0, 105)
ax0.grid(alpha=0.3)

# --- S(β) ------------------------------------------------------------------
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(betas, entropies, color="#7c3aed", lw=2)
ax1.axvline(beta_opt, color="#dc2626", ls="--", lw=1.5, alpha=0.7)
ax1.set_xlabel("β", fontsize=11)
ax1.set_ylabel("S  (nats per warp)", fontsize=11)
ax1.set_title("Entropy  S(β)", fontsize=12, fontweight="bold")
ax1.set_xlim(betas[0], betas[-1])
ax1.grid(alpha=0.3)

# --- Cv(β) -----------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(betas, cv, color="#d97706", lw=2)
ax2.axvline(beta_opt, color="#dc2626", ls="--", lw=1.5, alpha=0.7)
ax2.set_xlabel("β", fontsize=11)
ax2.set_ylabel("Cv  (per DOF)", fontsize=11)
ax2.set_title("Specific Heat  Cv(β)", fontsize=12, fontweight="bold")
ax2.set_xlim(betas[0], betas[-1])
ax2.set_ylim(bottom=0)
ax2.grid(alpha=0.3)

fig.suptitle("H100 Thermodynamic Carnot Curve  —  theoretical calculations only",
             fontsize=12, style="italic", y=1.01)

out = FIGURES / "01_carnot_curve.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved → {out}")
plt.close()


# ---------------------------------------------------------------------------
# log-Z decomposition
# ---------------------------------------------------------------------------

fig2, ax = plt.subplots(figsize=(10, 5))

# Normalise each component by its value at β=1 for visual comparison
b1_idx = np.argmin(np.abs(np.array(betas) - 1.0))
ax.plot(betas, [v - lz_c[b1_idx] for v in lz_c], color="#2563eb", lw=2, label="ln Z_compute")
ax.plot(betas, [v - lz_m[b1_idx] for v in lz_m], color="#16a34a", lw=2, label="ln Z_memory")
ax.plot(betas, [v - lz_k[b1_idx] for v in lz_k], color="#d97706", lw=2, label="ln Z_comm")
ax.axvline(beta_opt, color="#dc2626", ls="--", lw=1.5, label=f"β_optimal")
ax.set_xlabel("β", fontsize=11)
ax.set_ylabel("Δ ln Z  (relative to β=1)", fontsize=11)
ax.set_title("Partition Function Decomposition:  ln Z = ln Z_compute + ln Z_memory + ln Z_comm",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(betas[0], betas[-1])
ax.grid(alpha=0.3)

out2 = FIGURES / "01_logz_decomposition.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Figure saved → {out2}")
plt.close()

print("\nDone.\n")

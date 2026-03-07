"""
Experiment 04: Resonance Condition
====================================
The resonance condition measures compute–communication balance:

    η_overlap = T_overlapped / max(T_compute, T_comm)

This experiment:
  1. Plots η_overlap as a 2D heatmap over (T_compute, T_comm) space —
     the resonance ridge runs along T_compute = T_comm.
  2. Shows η_overlap as a function of the ratio r = T_comm / T_compute —
     peaked at r = 1 (resonance), falling off symmetrically.
  3. Shows how the optimal resonance point shifts with overlap_fraction
     (0 = sequential, 1 = perfectly pipelined).
  4. For each parallelism strategy on LLaMA-7B / 8 GPUs, plots where
     the (T_compute, T_comm) operating point lands relative to the
     resonance ridge.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.multi_gpu import resonance_condition
from gpu_statmech.parallelism import (
    GPT2_SMALL, LLAMA_7B,
    ParallelismConfig,
    build_parallelism_topology,
    estimate_comm_volumes,
    estimate_compute_time_s,
    estimate_comm_time_s,
    enumerate_configs,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

print("=" * 60)
print("Experiment 04: Resonance Condition")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. η_overlap 2D heatmap
# ---------------------------------------------------------------------------

t_vals = np.logspace(-3, 3, 200)   # seconds, log-spaced
T_c, T_k = np.meshgrid(t_vals, t_vals)
ETA = np.vectorize(resonance_condition)(T_c, T_k, 1.0)

# ---------------------------------------------------------------------------
# 2. η_overlap vs ratio r = T_comm / T_compute
# ---------------------------------------------------------------------------

ratios = np.logspace(-3, 3, 500)
eta_by_ratio = {
    "overlap=1.0 (perfect)": [resonance_condition(1.0, r, 1.0) for r in ratios],
    "overlap=0.7":           [resonance_condition(1.0, r, 0.7) for r in ratios],
    "overlap=0.3":           [resonance_condition(1.0, r, 0.3) for r in ratios],
    "overlap=0.0 (none)":    [resonance_condition(1.0, r, 0.0) for r in ratios],
}

print()
print("  η_overlap at resonance (r = T_comm / T_compute = 1.0):")
for label, vals in eta_by_ratio.items():
    idx = np.argmin(np.abs(ratios - 1.0))
    print(f"    {label:<28s}  η = {vals[idx]:.3f}")

# ---------------------------------------------------------------------------
# 3. Operating points for real parallelism configs (LLaMA-7B / 8 GPUs)
# ---------------------------------------------------------------------------

model = LLAMA_7B
n_gpu = 8
configs = enumerate_configs(n_gpu, model, max_tp=8, max_pp=8)

print()
print(f"  LLaMA-7B operating points on 8 GPUs:")
print(f"  {'Config':<22} {'T_compute (s)':>14} {'T_comm (s)':>12} "
      f"{'r=Tk/Tc':>10} {'η_overlap':>10}")
print("  " + "-" * 72)

op_points = []
for cfg in configs:
    topo  = build_parallelism_topology(cfg)
    cvols = estimate_comm_volumes(cfg, model)
    t_c   = estimate_compute_time_s(cfg, model, eta_hw=0.5)
    t_k   = estimate_comm_time_s(cvols, topo)
    eta   = resonance_condition(t_c, t_k)
    ratio = t_k / max(t_c, 1e-15)
    op_points.append((cfg, t_c, t_k, ratio, eta))
    print(f"  {cfg.label:<22} {t_c:>14.4f}s  {t_k if t_k < 1e10 else float('inf'):>10.4f}s  "
          f"{min(ratio, 999.0):>10.2f}   {eta:>10.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Resonance Condition  η_overlap = T_overlapped / max(T_compute, T_comm)",
             fontsize=13, fontweight="bold")

# --- 2D heatmap -------------------------------------------------------------
ax = axes[0]
im = ax.contourf(np.log10(T_c), np.log10(T_k), ETA,
                 levels=50, cmap="RdYlGn", vmin=0, vmax=1)
ax.contour(np.log10(T_c), np.log10(T_k), ETA,
           levels=[0.5, 0.7, 0.9], colors="white", linewidths=0.8, alpha=0.6)
# Resonance ridge: T_c = T_k → diagonal
diag = np.linspace(-3, 3, 100)
ax.plot(diag, diag, "w--", lw=2, label="T_compute = T_comm (resonance)")
fig.colorbar(im, ax=ax, label="η_overlap")
ax.set_xlabel("log₁₀(T_compute)  [s]", fontsize=10)
ax.set_ylabel("log₁₀(T_comm)  [s]", fontsize=10)
ax.set_title("η_overlap Heatmap", fontsize=12)
ax.legend(fontsize=8)

# --- η_overlap vs ratio r ---------------------------------------------------
ax = axes[1]
colors = ["#2563eb", "#16a34a", "#d97706", "#9ca3af"]
for (label, vals), color in zip(eta_by_ratio.items(), colors):
    ax.semilogx(ratios, vals, lw=2, color=color, label=label)
ax.axvline(1.0, color="black", ls=":", lw=1.5, label="r = 1 (resonance)")
ax.set_xlabel("r  =  T_comm / T_compute", fontsize=11)
ax.set_ylabel("η_overlap", fontsize=11)
ax.set_title("η_overlap vs Compute/Comm Ratio", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)

# --- Operating points on the heatmap ----------------------------------------
ax = axes[2]
im2 = ax.contourf(np.log10(T_c), np.log10(T_k), ETA,
                  levels=50, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.8)
diag2 = np.linspace(-4, 4, 100)
ax.plot(diag2, diag2, "w--", lw=1.5, alpha=0.7)
fig.colorbar(im2, ax=ax, label="η_overlap")

phase_colors = {
    "ferromagnetic":          "#2563eb",
    "antiferromagnetic":      "#dc2626",
    "domain_wall":            "#16a34a",
    "spin_glass":             "#d97706",
    "quasi_antiferromagnetic": "#7c3aed",
}
plotted_phases: set[str] = set()
for cfg, t_c, t_k, ratio, eta in op_points:
    if t_k > 1e8:   # inf comm (no links)
        continue
    phase = cfg.dominant_phase
    color = phase_colors.get(phase, "gray")
    label = phase if phase not in plotted_phases else None
    plotted_phases.add(phase)
    ax.scatter(np.log10(t_c), np.log10(t_k),
               color=color, s=80, zorder=5, label=label,
               edgecolors="white", linewidths=0.8)
    ax.annotate(cfg.label, (np.log10(t_c), np.log10(t_k)),
                textcoords="offset points", xytext=(5, 5),
                fontsize=6, color="white", fontweight="bold")

ax.set_xlabel("log₁₀(T_compute)  [s]", fontsize=10)
ax.set_ylabel("log₁₀(T_comm)  [s]", fontsize=10)
ax.set_title(f"LLaMA-7B Operating Points  ({n_gpu} GPUs)", fontsize=12)
if plotted_phases:
    ax.legend(fontsize=8, framealpha=0.7)

plt.tight_layout()
out = FIGURES / "04_resonance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved → {out}")
plt.close()

print("\nDone.\n")

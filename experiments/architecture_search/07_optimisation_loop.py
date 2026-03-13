"""
Experiment 07 — Thermodynamic Architecture Search Loop

Runs the full optimisation loop for up to 15 iterations (≤ 450 proposals)
and tracks:
  • Pareto hypervolume convergence
  • Best Carnot-condition satisfaction rate and expressiveness per iteration
  • Evolution of the Pareto frontier across iterations
  • How the oracle's AI prior shifts toward the roofline ridge with feedback

Thermodynamic quality metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``thermo_score`` = fraction of Carnot conditions satisfied ∈ {0, 0.2, …, 1}.
A kernel satisfying all five conditions (AI ≥ ridge, working sets ≤ capacity,
reuse ≥ minimum, occupancy ≥ minimum, zero unnecessary movement) scores 1.0.

Key result: Hypervolume grows over the first several iterations before
converging. The oracle's AI distribution narrows toward the roofline ridge
as feedback drives proposals into the Carnot-optimal region.

Figures: figures/07_loop_convergence.png
         figures/07_pareto_evolution.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.loop import LoopConfig, OptimisationLoop
from gpu_statmech.pareto import ParetoPoint, hypervolume_2d, pareto_frontier

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

N_ITER     = 15
BATCH_SIZE = 30

print("=" * 60)
print("Experiment 07 — Thermodynamic Architecture Search Loop")
print("=" * 60)

carnot_limit = derive_carnot_limit()

cfg = LoopConfig(
    n_proposals_per_iter=BATCH_SIZE,
    eta_threshold=0.0,          # include all kernels as Pareto candidates
    max_iterations=N_ITER,
    convergence_tolerance=1e-5,
    patience=5,
    seed=42,
    verbose=True,
)

# ── run the loop, capturing per-iteration snapshots ───────────────────────────
snapshots: list = []

def capture(state):
    snapshots.append(state)

loop  = OptimisationLoop(carnot_limit, cfg, on_iteration=capture)
final = loop.run()
n_completed = len(snapshots)

# ── tabular summary ───────────────────────────────────────────────────────────
print()
print(f"{'Iter':>4}  {'Pool':>5}  {'Front':>5}  {'HV':>10}  "
      f"{'Best η':>8}  {'Best expr':>9}  {'Carnot✓':>7}")
print("-" * 60)
for s in snapshots:
    print(
        f"{s.iteration:4d}  {len(s.all_kernels):5d}  {s.frontier_size:5d}  "
        f"{s.hypervolume:10.6f}  {s.best_eta:8.4f}  "
        f"{s.best_expressiveness:9.4f}  {s.n_carnot_optimal:7d}"
    )

print()
print(loop.convergence_report())
print()

# top-5 overall
top5 = loop.best_kernels(n=5)
print("Top-5 kernels by combined score:")
print(f"  {'Name':<30}  {'η':>6}  {'expr':>6}  {'combined':>8}  {'Carnot?':>7}")
for ck in top5:
    print(
        f"  {ck.proposal.name:<30}  {ck.thermo_score:6.4f}  "
        f"{ck.expressiveness_score:6.4f}  {ck.combined_score:8.4f}  "
        f"{'✓' if ck.is_carnot_optimal else '✗':>7}"
    )

# ── oracle prior drift ────────────────────────────────────────────────────────
# Re-run a fresh loop for N_ITER steps, capturing AI prior mean/std each step
loop2 = OptimisationLoop(carnot_limit, cfg)
ai_means: list[float] = []
ai_stds:  list[float] = []

for _ in range(N_ITER):
    loop2.step()
    ai_means.append(loop2._oracle.prior.log_ai_mean)
    ai_stds.append(loop2._oracle.prior.log_ai_std)

ridge_log    = float(np.log(carnot_limit.roofline_intensity))
ridge_log10  = ridge_log / np.log(10)
ai_means_10  = [m / np.log(10) for m in ai_means]
ai_stds_10   = [s / np.log(10) for s in ai_stds]

# ── Figure 1: convergence plots ───────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
fig1.suptitle(
    f"Architecture Search Loop Convergence — H100\n"
    f"η_hw,max = {carnot_limit.eta_hw_max * 100:.2f}%  |  "
    f"{n_completed} iterations × {BATCH_SIZE} proposals",
    fontsize=12, fontweight="bold",
)

iters = [s.iteration for s in snapshots]
hvs   = [s.hypervolume          for s in snapshots]
betas = [s.best_eta             for s in snapshots]
bexps = [s.best_expressiveness  for s in snapshots]
bcomb = [s.best_combined        for s in snapshots]

# 1a. Hypervolume
ax = axes[0, 0]
ax.plot(iters, hvs, "o-", color="#4c72b0", lw=2, ms=5)
ax.fill_between(iters, hvs, alpha=0.15, color="#4c72b0")
ax.set_xlabel("Iteration")
ax.set_ylabel("Pareto Hypervolume")
ax.set_title("Hypervolume Convergence")
ax.grid(True, alpha=0.3)
if hvs:
    ax.annotate(
        f"Final HV = {hvs[-1]:.6f}",
        xy=(iters[-1], hvs[-1]),
        xytext=(max(0, iters[-1] - 4), hvs[-1] * 0.7 + 0.001),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=8,
    )

# 1b. Best scores per iteration
ax = axes[0, 1]
ax.plot(iters, betas, "s--", color="#dd8452", lw=1.8, ms=5,
        label="Best Carnot-cond. fraction")
ax.plot(iters, bexps, "^--", color="#55a868", lw=1.8, ms=5,
        label="Best expressiveness")
ax.plot(iters, bcomb, "o-",  color="#c44e52", lw=2,   ms=5,
        label="Best combined")
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")
ax.set_title("Best Scores per Iteration")
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# 1c. AI prior drift (log10 scale) — from second loop run
xs = list(range(N_ITER))
ax = axes[1, 0]
ax.plot(xs, ai_means_10, "o-", color="#8172b2", lw=2, ms=5,
        label="Prior mean log₁₀(AI)")
ax.fill_between(
    xs,
    [m - s for m, s in zip(ai_means_10, ai_stds_10)],
    [m + s for m, s in zip(ai_means_10, ai_stds_10)],
    alpha=0.2, color="#8172b2", label="±1 std",
)
ax.axhline(ridge_log10, color="red", lw=1.5, ls="--",
           label=f"Roofline ridge ({carnot_limit.roofline_intensity:.2f} FLOP/B)")
ax.set_xlabel("Iteration")
ax.set_ylabel("log₁₀(Arithmetic Intensity)")
ax.set_title("Oracle AI Prior Drift\n(converging toward roofline ridge)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1d. Frontier size per iteration
front_sizes = [s.frontier_size for s in snapshots]
carnot_cnts = [s.n_carnot_optimal for s in snapshots]
ax = axes[1, 1]
ax.bar(iters, front_sizes, color="#64b5cd", label="Frontier size", alpha=0.7)
ax.bar(iters, carnot_cnts, color="#c44e52", label="Carnot-optimal ✓", alpha=0.9)
ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_title("Pareto Frontier Size per Iteration")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

fig1.tight_layout()
out1 = FIGURES / "07_loop_convergence.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out1}")
plt.close(fig1)

# ── Figure 2: Pareto frontier evolution ───────────────────────────────────────
# Pick up to 4 evenly-spaced snapshot indices to show progression
n_panels = min(4, n_completed)
if n_panels > 0:
    panel_indices = sorted(set(
        int(round(i * (n_completed - 1) / max(n_panels - 1, 1)))
        for i in range(n_panels)
    ))
else:
    panel_indices = []

if panel_indices:
    fig2, axes2 = plt.subplots(1, len(panel_indices), figsize=(14, 4), sharey=True)
    if len(panel_indices) == 1:
        axes2 = [axes2]
    fig2.suptitle(
        "Pareto Frontier Evolution  (Carnot-cond. fraction  vs  Expressiveness)",
        fontsize=12, fontweight="bold",
    )

    for ax, snap_idx in zip(axes2, panel_indices):
        s = snapshots[snap_idx]
        candidates = [ParetoPoint.from_compiled(ck) for ck in s.all_kernels]
        front = pareto_frontier(candidates)

        all_etas  = [p.eta_fraction   for p in candidates]
        all_exprs = [p.expressiveness for p in candidates]
        ax.scatter(all_etas, all_exprs, c="#cccccc", s=12, alpha=0.5, zorder=1)

        if front:
            f_etas  = [p.eta_fraction   for p in front]
            f_exprs = [p.expressiveness for p in front]
            ax.scatter(f_etas, f_exprs, c="#c44e52", s=40, zorder=3, label="Frontier")
            pts_sorted = sorted(front, key=lambda p: p.eta_fraction)
            xs2 = [p.eta_fraction   for p in pts_sorted]
            ys2 = [p.expressiveness for p in pts_sorted]
            ax.step(xs2, ys2, where="post", color="#c44e52", lw=1.5, alpha=0.7)

        hv = hypervolume_2d(front)
        ax.set_title(
            f"Iter {s.iteration}\n"
            f"pool={len(candidates)}  front={len(front)}\n"
            f"HV={hv:.5f}",
            fontsize=9,
        )
        ax.set_xlabel("Carnot-cond. fraction")
        if ax is axes2[0]:
            ax.set_ylabel("Expressiveness")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    out2 = FIGURES / "07_pareto_evolution.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out2}")
    plt.close(fig2)
else:
    print("No snapshots captured — skipping Pareto evolution figure.")

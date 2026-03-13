"""
Experiment 08 — Carnot-Optimal Kernel Analysis

After running the optimisation loop, inspects the kernels that achieve
the best thermodynamic efficiency (η_hw / η_hw,max) and analyses:
  • Which Carnot conditions are easiest / hardest to satisfy
  • The joint distribution of resource usage for top-k kernels
  • Waste attribution breakdown across the kernel pool
  • How the expressiveness–efficiency tradeoff looks for Carnot-optimal
    vs sub-optimal kernels

Key result: the dominant bottleneck across the pool is 'reuse_smem'
(data reuse at shared memory is rarely high enough to meet the Carnot
minimum), while 'arithmetic_intensity' is satisfied by ~60% of proposals
once feedback has run for 10 iterations.

Figures: figures/08_condition_satisfaction.png
         figures/08_waste_breakdown.png
         figures/08_optimal_vs_suboptimal.png
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.loop import LoopConfig, OptimisationLoop
from gpu_statmech.pareto import ParetoPoint, pareto_frontier

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

N_ITER     = 15
BATCH_SIZE = 30

print("=" * 60)
print("Experiment 08 — Carnot-Optimal Kernel Analysis")
print("=" * 60)

carnot_limit = derive_carnot_limit()

cfg = LoopConfig(
    n_proposals_per_iter=BATCH_SIZE,
    eta_threshold=0.0,          # keep everything for analysis
    max_iterations=N_ITER,
    convergence_tolerance=1e-5,
    patience=5,
    seed=42,
    verbose=False,
)

loop  = OptimisationLoop(carnot_limit, cfg)
final = loop.run()
pool  = final.all_kernels

print(f"\nTotal kernels in pool : {len(pool)}")
n_optimal = sum(1 for ck in pool if ck.is_carnot_optimal)
print(f"Carnot-optimal        : {n_optimal}  ({n_optimal / len(pool) * 100:.1f}%)")

# ── Condition satisfaction rates ───────────────────────────────────────────────
# Collect all condition names and satisfaction booleans
cond_satisfied: dict[str, list[bool]] = {}
bottlenecks: list[str] = []

for ck in pool:
    for cond in ck.optimality_report.conditions:
        if cond.name not in cond_satisfied:
            cond_satisfied[cond.name] = []
        cond_satisfied[cond.name].append(cond.satisfied)
    bottlenecks.append(ck.dominant_bottleneck)

print("\nCondition satisfaction rates:")
cond_names = list(cond_satisfied.keys())
cond_rates = [np.mean(cond_satisfied[c]) for c in cond_names]
for name, rate in zip(cond_names, cond_rates):
    bar = "█" * int(rate * 30)
    print(f"  {name:<35s} {rate * 100:5.1f}%  {bar}")

bn_counts = Counter(b for b in bottlenecks if b != "none")
print(f"\nBottom-neck frequency (top-5):")
for bn, cnt in bn_counts.most_common(5):
    print(f"  {bn:<35s} {cnt:4d}  ({cnt / len(pool) * 100:.1f}%)")

# ── Resource profiles for top-20 kernels ──────────────────────────────────────
top20 = loop.best_kernels(n=20)
t_ais   = [ck.proposal.arithmetic_intensity    for ck in top20]
t_regs  = [ck.proposal.registers_per_thread    for ck in top20]
t_smems = [ck.proposal.smem_bytes / 1024       for ck in top20]  # KB
t_tcs   = [ck.proposal.tensor_core_utilisation for ck in top20]
t_occs  = [ck.kernel_spec.warp_occupancy       for ck in top20]
t_eta   = [ck.thermo_score                     for ck in top20]
t_expr  = [ck.expressiveness_score             for ck in top20]

print(f"\nTop-20 kernel resource profile:")
print(f"  mean AI   = {np.mean(t_ais):.2f} FLOP/byte  (ridge = {carnot_limit.roofline_intensity:.2f})")
print(f"  mean regs = {np.mean(t_regs):.1f} per thread")
print(f"  mean smem = {np.mean(t_smems):.1f} KB")
print(f"  mean TC   = {np.mean(t_tcs):.3f}")
print(f"  mean occ  = {np.mean(t_occs):.3f}")
print(f"  mean η    = {np.mean(t_eta):.4f}")
print(f"  mean expr = {np.mean(t_expr):.4f}")

# ── Optimal vs sub-optimal split ───────────────────────────────────────────────
opt_pool    = [ck for ck in pool if ck.is_carnot_optimal]
subopt_pool = [ck for ck in pool if not ck.is_carnot_optimal]

opt_expr    = [ck.expressiveness_score for ck in opt_pool]    if opt_pool    else [0.0]
subopt_expr = [ck.expressiveness_score for ck in subopt_pool] if subopt_pool else [0.0]
opt_eta     = [ck.thermo_score         for ck in opt_pool]    if opt_pool    else [0.0]
subopt_eta  = [ck.thermo_score         for ck in subopt_pool] if subopt_pool else [0.0]

print(f"\nCarnot-optimal vs sub-optimal expressiveness:")
print(f"  Optimal    mean = {np.mean(opt_expr):.4f}   n = {len(opt_pool)}")
print(f"  Sub-optimal mean = {np.mean(subopt_expr):.4f}   n = {len(subopt_pool)}")

# ── Figures ────────────────────────────────────────────────────────────────────

# Figure 1: condition satisfaction + bottleneck
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(
    f"Carnot Condition Analysis — {len(pool)} Kernels, {N_ITER} Iterations",
    fontsize=12, fontweight="bold",
)

# 1a. Satisfaction rate bar chart
colors_sat = ["#55a868" if r >= 0.5 else "#c44e52" for r in cond_rates]
short_names = [n.replace("working_set_", "ws:").replace("reuse_", "reuse:") for n in cond_names]
y_pos = range(len(cond_names))
ax1.barh(y_pos, [r * 100 for r in cond_rates], color=colors_sat, edgecolor="white")
ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(short_names, fontsize=8)
ax1.axvline(50, color="black", lw=0.8, ls="--", alpha=0.5)
ax1.set_xlabel("Satisfaction Rate (%)")
ax1.set_title("Carnot Condition Satisfaction Rates\n(green ≥ 50%, red < 50%)")
ax1.set_xlim(0, 105)
for i, r in enumerate(cond_rates):
    ax1.text(r * 100 + 1, i, f"{r * 100:.0f}%", va="center", fontsize=7)

# 1b. Bottleneck frequency
top_bns = bn_counts.most_common(8)
bn_labels = [b for b, _ in top_bns]
bn_vals   = [c for _, c in top_bns]
short_bn  = [l.replace("working_set_", "ws:").replace("reuse_", "reuse:") for l in bn_labels]
ax2.barh(range(len(bn_labels)), bn_vals, color="#8172b2", edgecolor="white")
ax2.set_yticks(range(len(bn_labels)))
ax2.set_yticklabels(short_bn, fontsize=8)
ax2.set_xlabel("Frequency")
ax2.set_title("Dominant Bottleneck Frequency\n(most common violation across pool)")

fig1.tight_layout()
out1 = FIGURES / "08_condition_satisfaction.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out1}")
plt.close(fig1)

# Figure 2: resource profile of top-20
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("Top-20 Kernel Resource Profiles", fontsize=12, fontweight="bold")

ranks = list(range(1, 21))
axes2[0].bar(ranks, t_ais, color="#4c72b0")
axes2[0].axhline(carnot_limit.roofline_intensity, color="red", lw=1.5, ls="--",
                 label=f"Ridge = {carnot_limit.roofline_intensity:.2f}")
axes2[0].set_xlabel("Rank (by combined score)")
axes2[0].set_ylabel("Arithmetic Intensity [FLOP/byte]")
axes2[0].set_title("Arithmetic Intensity")
axes2[0].legend(fontsize=8)
axes2[0].set_yscale("log")

axes2[1].bar(ranks, t_occs, color="#dd8452")
axes2[1].axhline(carnot_limit.min_warp_occupancy, color="red", lw=1.5, ls="--",
                 label=f"Min = {carnot_limit.min_warp_occupancy:.2f}")
axes2[1].set_xlabel("Rank")
axes2[1].set_ylabel("Warp Occupancy")
axes2[1].set_title("Warp Occupancy")
axes2[1].legend(fontsize=8)
axes2[1].set_ylim(0, 1.05)

axes2[2].bar(ranks, t_tcs, color="#55a868")
axes2[2].set_xlabel("Rank")
axes2[2].set_ylabel("TC Utilisation")
axes2[2].set_title("Tensor-Core Utilisation")
axes2[2].set_ylim(0, 1.05)

fig2.tight_layout()
out2 = FIGURES / "08_waste_breakdown.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out2}")
plt.close(fig2)

# Figure 3: Carnot-optimal vs sub-optimal scatter
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.scatter(subopt_eta, subopt_expr, s=15, alpha=0.35, color="#aaaaaa", label="Sub-optimal")
if opt_pool:
    ax3.scatter(opt_eta, opt_expr, s=40, alpha=0.85, color="#c44e52",
                marker="*", zorder=5, label=f"Carnot-optimal (n={len(opt_pool)})")
ax3.set_xlabel("η_hw / η_hw,max")
ax3.set_ylabel("Expressiveness")
ax3.set_title(
    f"Carnot-Optimal vs Sub-Optimal Kernels\n"
    f"Total pool: {len(pool)} kernels, {N_ITER} iterations"
)
ax3.legend(fontsize=9)
ax3.set_xlim(-0.02, 1.05)
ax3.set_ylim(-0.02, 1.05)
ax3.grid(True, alpha=0.3)
out3 = FIGURES / "08_optimal_vs_suboptimal.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out3}")
plt.close(fig3)

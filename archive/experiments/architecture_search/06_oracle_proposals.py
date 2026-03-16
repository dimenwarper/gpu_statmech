"""
Experiment 06 — Oracle Proposal Distribution

Visualises the initial proposal distribution of the PhysicsOracle:
  • Arithmetic intensity vs tensor-core utilisation scatter
  • Block-size and access-pattern frequency histograms
  • Warp-occupancy distribution from the occupancy model
  • Expressiveness score distribution

All drawn from the prior (no feedback), so this characterises the
baseline diversity of proposals before any learning.

Key result: the prior covers ~4 decades of AI, all block sizes with
equal probability, and produces a wide spread of expressiveness scores
centred around 0.5 — giving the loop room to discover good regions.

Figures: figures/06_oracle_proposals.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── project import ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.compiler import KernelCompiler, expressiveness_score, warp_occupancy
from gpu_statmech.oracle import ACCESS_PATTERNS, VALID_BLOCK_SIZES, PhysicsOracle

# ── setup ─────────────────────────────────────────────────────────────────────
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

N_PROPOSALS = 500
RNG_SEED = 0

print("=" * 60)
print("Experiment 06 — Oracle Proposal Distribution")
print("=" * 60)

carnot_limit = derive_carnot_limit()
oracle   = PhysicsOracle(carnot_limit, seed=RNG_SEED)
compiler = KernelCompiler(carnot_limit)

rng       = np.random.default_rng(RNG_SEED)
proposals = oracle.propose(n=N_PROPOSALS, rng=rng)

# ── compute derived quantities ────────────────────────────────────────────────
ais      = np.array([p.arithmetic_intensity     for p in proposals])
tc_utils = np.array([p.tensor_core_utilisation  for p in proposals])
occs     = np.array([warp_occupancy(p)           for p in proposals])
exprs    = np.array([expressiveness_score(p, carnot_limit) for p in proposals])
blocks   = [p.block_size               for p in proposals]
patterns = [p.memory_access_pattern    for p in proposals]

ridge = carnot_limit.roofline_intensity

print(f"\nProposals generated : {N_PROPOSALS}")
print(f"Roofline ridge AI   : {ridge:.3f} FLOP/byte")
print(f"\nArithmetic intensity summary:")
print(f"  min  = {ais.min():.3f}")
print(f"  mean = {ais.mean():.3f}")
print(f"  max  = {ais.max():.1f}")
print(f"  fraction above ridge = {(ais >= ridge).mean() * 100:.1f}%")
print(f"\nWarp occupancy summary:")
print(f"  mean = {occs.mean():.3f}")
print(f"  min  = {occs.min():.3f}")
print(f"  max  = {occs.max():.3f}")
print(f"\nExpressiveness summary:")
print(f"  mean = {exprs.mean():.3f}")
print(f"  std  = {exprs.std():.3f}")

block_counts   = {b: blocks.count(b) for b in VALID_BLOCK_SIZES}
pattern_counts = {p: patterns.count(p) for p in ACCESS_PATTERNS}
print(f"\nBlock size distribution:")
for b, c in block_counts.items():
    print(f"  {b:5d}: {c:4d}  ({c / N_PROPOSALS * 100:.1f}%)")
print(f"\nAccess pattern distribution:")
for p, c in pattern_counts.items():
    print(f"  {p:12s}: {c:4d}  ({c / N_PROPOSALS * 100:.1f}%)")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax_scatter = fig.add_subplot(gs[0, :2])
ax_occ     = fig.add_subplot(gs[0, 2])
ax_blocks  = fig.add_subplot(gs[1, 0])
ax_pattern = fig.add_subplot(gs[1, 1])
ax_expr    = fig.add_subplot(gs[1, 2])

# 1. AI vs TC scatter (log AI)
sc = ax_scatter.scatter(
    np.log10(ais), tc_utils,
    c=exprs, cmap="plasma", alpha=0.55, s=18, vmin=0, vmax=1,
)
ax_scatter.axvline(np.log10(ridge), color="red", lw=1.5, ls="--",
                   label=f"Roofline ridge ({ridge:.2f} FLOP/B)")
ax_scatter.set_xlabel("log₁₀(Arithmetic Intensity)  [FLOP/byte]")
ax_scatter.set_ylabel("Tensor-Core Utilisation")
ax_scatter.set_title("Proposal Distribution: AI vs TC utilisation\n(colour = expressiveness)")
ax_scatter.legend(fontsize=8)
plt.colorbar(sc, ax=ax_scatter, label="Expressiveness")

# 2. Warp occupancy histogram
ax_occ.hist(occs, bins=20, color="#4c72b0", edgecolor="white", linewidth=0.5)
ax_occ.axvline(carnot_limit.min_warp_occupancy, color="red", lw=1.5, ls="--",
               label=f"Min for latency hiding ({carnot_limit.min_warp_occupancy:.2f})")
ax_occ.set_xlabel("Warp Occupancy")
ax_occ.set_ylabel("Count")
ax_occ.set_title("Warp Occupancy Distribution")
ax_occ.legend(fontsize=7)

# 3. Block size bar chart
bs_labels = [str(b) for b in VALID_BLOCK_SIZES]
bs_vals   = [block_counts[b] for b in VALID_BLOCK_SIZES]
ax_blocks.bar(bs_labels, bs_vals, color="#dd8452", edgecolor="white")
ax_blocks.set_xlabel("Block Size (threads)")
ax_blocks.set_ylabel("Count")
ax_blocks.set_title("Block Size Frequency\n(uniform prior)")

# 4. Access pattern bar chart
ap_labels = list(ACCESS_PATTERNS.keys())
ap_vals   = [pattern_counts[p] for p in ap_labels]
ap_colors = ["#55a868", "#c44e52", "#8172b2"]
ax_pattern.bar(ap_labels, ap_vals, color=ap_colors, edgecolor="white")
ax_pattern.set_xlabel("Access Pattern")
ax_pattern.set_ylabel("Count")
ax_pattern.set_title("Memory Access Pattern\nFrequency (uniform prior)")

# 5. Expressiveness histogram
ax_expr.hist(exprs, bins=25, color="#64b5cd", edgecolor="white", linewidth=0.5)
ax_expr.axvline(exprs.mean(), color="navy", lw=1.5, ls="--",
                label=f"Mean = {exprs.mean():.3f}")
ax_expr.set_xlabel("Expressiveness Score")
ax_expr.set_ylabel("Count")
ax_expr.set_title("Expressiveness Score Distribution\n(prior sample)")
ax_expr.legend(fontsize=8)

fig.suptitle(
    f"Oracle Prior: {N_PROPOSALS} Proposals — H100 Carnot Limit η_hw,max = {carnot_limit.eta_hw_max * 100:.2f}%",
    fontsize=13, fontweight="bold",
)

out = FIGURES / "06_oracle_proposals.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.close(fig)

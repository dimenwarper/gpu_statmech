"""
Experiment 03: Multi-GPU Scaling Efficiency
============================================
Runs the current multi-GPU topology proxy for 1 → 64 GPUs across four
interconnect topologies:
  - NVLink-4 clique        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - NVSwitch fabric        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - PCIe Gen 5 ring        (J = 1.00, BW =  64 GB/s)   weakly coupled
  - InfiniBand HDR mesh    (J = 5.00, BW =  50 GB/s)   spin-glass

Key quantities:
  - η_multi,max(n_gpu) — current multi-GPU ceiling proxy
  - scaling_efficiency = η_multi,max / η_hw,max,single
  - resonance_eta — communication log-share complement from the current model
  - comm_overhead_fraction

Important caveat: the current multi_gpu.py path is still a legacy J-only
communication model. The reported "resonance_eta" here is not a timing-based
overlap metric and does not directly use link bandwidth or latency.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG
from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.multi_gpu import TopologyGraph, derive_multi_gpu_carnot_limit

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

print("=" * 60)
print("Experiment 03: Multi-GPU Scaling Efficiency")
print("=" * 60)

# ---------------------------------------------------------------------------
# Single-GPU reference limit (computed once)
# ---------------------------------------------------------------------------

single_limit = derive_carnot_limit(n_beta=100)
eta_single   = single_limit.eta_hw_max
print(f"\n  Single-GPU η_hw,max = {eta_single:.4f}  ({eta_single*100:.2f}%)")
print("  Note: multi-GPU values below are legacy topology proxies, not timing-accurate")
print("        overlap predictions.")

# ---------------------------------------------------------------------------
# GPU counts and topology builders
# ---------------------------------------------------------------------------

n_gpu_values = [1, 2, 4, 8, 16, 32, 64]

TOPOLOGIES = {
    "NVLink-4 clique":     TopologyGraph.nvlink_clique,
    "NVSwitch fabric":     TopologyGraph.nvswitch_fabric,
    "PCIe Gen5 ring":      TopologyGraph.pcie_ring,
    "InfiniBand fat-tree": TopologyGraph.infiniband_fat_tree,
}

COLORS = {
    "NVLink-4 clique":     "#2563eb",
    "NVSwitch fabric":     "#16a34a",
    "PCIe Gen5 ring":      "#d97706",
    "InfiniBand fat-tree": "#dc2626",
}

MARKERS = {
    "NVLink-4 clique":     "o",
    "NVSwitch fabric":     "s",
    "PCIe Gen5 ring":      "^",
    "InfiniBand fat-tree": "D",
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

results: dict[str, dict] = {name: {
    "eta_multi_max": [],
    "scaling_eff":   [],
    "resonance_eta": [],
    "comm_overhead": [],
} for name in TOPOLOGIES}

for n in n_gpu_values:
    print(f"\n  n_gpu = {n}")
    for topo_name, builder in TOPOLOGIES.items():
        topo = TopologyGraph(n_gpu=1, links=[], name="single_gpu") if n == 1 else builder(n)
        limit = derive_multi_gpu_carnot_limit(
            topo,
            eta_hw_max_single=eta_single,
            n_beta=80, n_bins=32,
        )
        r = results[topo_name]
        r["eta_multi_max"].append(limit.eta_multi_max)
        r["scaling_eff"].append(limit.scaling_efficiency())
        r["resonance_eta"].append(limit.resonance_eta)
        r["comm_overhead"].append(limit.comm_overhead_fraction)
        print(f"    {topo_name:<22s}  η_multi,max={limit.eta_multi_max:.4f}  "
              f"scaling={limit.scaling_efficiency():.4f}  "
              f"log-share proxy={limit.resonance_eta:.4f}  "
              f"comm={limit.comm_overhead_fraction*100:.1f}%")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("  Summary at n_gpu = 64  (legacy topology proxy)")
print(f"  {'Topology':<24} {'η_multi,max':>12} {'scaling_eff':>12} "
      f"{'log-share':>11} {'comm_overhead':>14}")
print("  " + "-" * 77)
for name, r in results.items():
    print(f"  {name:<24} {r['eta_multi_max'][-1]:>12.4f} "
          f"{r['scaling_eff'][-1]:>12.4f} "
          f"{r['resonance_eta'][-1]:>11.4f} "
          f"{r['comm_overhead'][-1]*100:>12.1f}%")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-GPU Topology Proxy  (legacy J-only communication model)",
             fontsize=14, fontweight="bold")

# --- η_multi,max(n_gpu) -------------------------------------------------------
ax = axes[0, 0]
ax.axhline(eta_single * 100, color="black", ls=":", lw=1.5, label=f"1-GPU limit  {eta_single*100:.1f}%")
for name, r in results.items():
    ax.plot(n_gpu_values, [v * 100 for v in r["eta_multi_max"]],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("η_multi,max  (%)", fontsize=11)
ax.set_title("Multi-GPU Ceiling Proxy  η_multi,max(n)", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- scaling_efficiency -------------------------------------------------------
ax = axes[0, 1]
ax.axhline(1.0, color="black", ls=":", lw=1.5, label="perfect scaling")
for name, r in results.items():
    ax.plot(n_gpu_values, r["scaling_eff"],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("η_multi,max / η_hw,max,single", fontsize=11)
ax.set_title("Scaling Proxy  η_multi,max / η_single", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- resonance_eta(n_gpu) -----------------------------------------------------
ax = axes[1, 0]
for name, r in results.items():
    ax.plot(n_gpu_values, r["resonance_eta"],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("legacy proxy", fontsize=11)
ax.set_title("Communication Log-Share Complement  (legacy proxy)", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- comm_overhead(n_gpu) -----------------------------------------------------
ax = axes[1, 1]
for name, r in results.items():
    ax.plot(n_gpu_values, [v * 100 for v in r["comm_overhead"]],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("Comm overhead  (%)", fontsize=11)
ax.set_title("Communication Overhead Fraction  (DOFs in comm subsystem)", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out = FIGURES / "03_scaling_efficiency.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved → {out}")
plt.close()

print("\nDone.\n")

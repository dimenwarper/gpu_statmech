"""
Experiment 03: Multi-GPU Scaling Efficiency
============================================
Runs the energy-based multi-GPU limit for a canonical distributed-training
communication load across 1 → 64 GPUs and four interconnect topologies:
  - NVLink-4 clique        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - NVSwitch fabric        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - PCIe Gen 5 ring        (J = 1.00, BW =  64 GB/s)   weakly coupled
  - InfiniBand HDR mesh    (J = 5.00, BW =  50 GB/s)   spin-glass

Key quantities:
  - η_multi,max(n_gpu) — energy-based multi-GPU ceiling
  - scaling_efficiency = η_multi,max / η_hw,max,single
  - balance_proxy — useful-work / communication-energy balance proxy
  - comm_energy_share — fraction of total input energy spent in communication

Communication demand closure:
  - Workload: LLaMA-7B
  - Pattern:  pure data parallelism (DP-n all-reduce style demand)
  - target_comm_load = total_bytes / (BW_ref * T_compute_ref)

The balance proxy is still not a timing-overlap metric.
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
from gpu_statmech.multi_gpu import (
    TopologyGraph,
    derive_multi_gpu_carnot_limit,
    normalise_comm_demand,
)
from gpu_statmech.parallelism import (
    LLAMA_7B,
    ParallelismConfig,
    estimate_comm_volumes,
    estimate_compute_time_s,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)
N_BETA = 20
N_BINS = 16

print("=" * 60)
print("Experiment 03: Multi-GPU Scaling Efficiency")
print("=" * 60)

# ---------------------------------------------------------------------------
# Single-GPU reference limit (computed once)
# ---------------------------------------------------------------------------

single_limit = derive_carnot_limit(n_beta=N_BETA, n_bins=N_BINS)
eta_single   = single_limit.eta_hw_max
print(f"\n  Single-GPU η_hw,max = {eta_single:.4f}  ({eta_single*100:.2f}%)")
print("  Canonical workload: LLaMA-7B, pure data parallelism (DP-n).")
print("  target_comm_load is derived from per-step communication bytes divided by")
print("  the reference-link capacity over the estimated compute window.")
print("  The balance proxy is not a timing-overlap prediction.")

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
    "balance_proxy": [],
    "comm_energy": [],
} for name in TOPOLOGIES}

for n in n_gpu_values:
    print(f"\n  n_gpu = {n}")
    dp_config = ParallelismConfig(dp=n)
    comm_volumes = estimate_comm_volumes(dp_config, LLAMA_7B)
    t_compute_ref = estimate_compute_time_s(dp_config, LLAMA_7B, eta_hw=eta_single)
    target_comm_load = normalise_comm_demand(comm_volumes.total_bytes, t_compute_ref)
    print(f"    target_comm_load={target_comm_load:.4f}  bytes/step={comm_volumes.total_bytes/1e9:.2f} GB")
    for topo_name, builder in TOPOLOGIES.items():
        topo = TopologyGraph(n_gpu=1, links=[], name="single_gpu") if n == 1 else builder(n)
        limit = derive_multi_gpu_carnot_limit(
            topo,
            eta_hw_max_single=eta_single,
            target_activity=single_limit.target_activity,
            target_comm_load=target_comm_load,
            n_beta=N_BETA, n_bins=N_BINS,
        )
        r = results[topo_name]
        r["eta_multi_max"].append(limit.eta_multi_max)
        r["scaling_eff"].append(limit.scaling_efficiency())
        r["balance_proxy"].append(limit.resonance_eta)
        r["comm_energy"].append(limit.comm_overhead_fraction)
        print(f"    {topo_name:<22s}  η_multi,max={limit.eta_multi_max:.4f}  "
              f"scaling={limit.scaling_efficiency():.4f}  "
              f"balance={limit.resonance_eta:.4f}  "
              f"comm energy={limit.comm_overhead_fraction*100:.1f}%")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("  Summary at n_gpu = 64")
print(f"  {'Topology':<24} {'η_multi,max':>12} {'scaling_eff':>12} "
      f"{'balance':>11} {'comm_energy':>14}")
print("  " + "-" * 77)
for name, r in results.items():
    print(f"  {name:<24} {r['eta_multi_max'][-1]:>12.4f} "
          f"{r['scaling_eff'][-1]:>12.4f} "
          f"{r['balance_proxy'][-1]:>11.4f} "
          f"{r['comm_energy'][-1]*100:>12.1f}%")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-GPU Efficiency Limit  (<W_hw>/<E_in> with topology penalties)",
             fontsize=14, fontweight="bold")

# --- η_multi,max(n_gpu) -------------------------------------------------------
ax = axes[0, 0]
ax.axhline(eta_single * 100, color="black", ls=":", lw=1.5, label=f"1-GPU limit  {eta_single*100:.1f}%")
for name, r in results.items():
    ax.plot(n_gpu_values, [v * 100 for v in r["eta_multi_max"]],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("η_multi,max  (%)", fontsize=11)
ax.set_title("Multi-GPU Ceiling  η_multi,max(n)", fontsize=12)
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
ax.set_title("Scaling Efficiency  η_multi,max / η_single", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- balance proxy ------------------------------------------------------------
ax = axes[1, 0]
for name, r in results.items():
    ax.plot(n_gpu_values, r["balance_proxy"],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("energy-balance proxy", fontsize=11)
ax.set_title("Useful-Work / Communication Balance Proxy", fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(n_gpu_values)
ax.set_xticklabels(n_gpu_values)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- comm energy share --------------------------------------------------------
ax = axes[1, 1]
for name, r in results.items():
    ax.plot(n_gpu_values, [v * 100 for v in r["comm_energy"]],
            color=COLORS[name], marker=MARKERS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Number of GPUs", fontsize=11)
ax.set_ylabel("Comm energy share  (%)", fontsize=11)
ax.set_title("Communication Energy Fraction", fontsize=12)
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

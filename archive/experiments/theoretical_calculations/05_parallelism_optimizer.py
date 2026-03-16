"""
Experiment 05: Parallelism Optimizer — Pareto Frontiers
=========================================================
Runs the thermodynamic parallelism optimizer for two real models:
  - GPT-2 Small   (117M params,  8 GPUs)
  - LLaMA-7B      (7B params,   64 GPUs)

For each, produces:
  1. A scatter plot of all configs on (comm_overhead, η_multi) axes,
     coloured by thermodynamic phase, with the Pareto frontier highlighted.
  2. A stacked bar chart of communication volume breakdown by strategy
     (AllReduce / AllGather / ReduceScatter / P2P / AllToAll / CP).
  3. A printed table of all Pareto-efficient configs.

All purely theoretical — hardware spec numbers + model parameters.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG
from gpu_statmech.parallelism import (
    GPT2_SMALL, LLAMA_7B, ModelParams,
    ParallelismConfig,
    optimise_parallelism,
    estimate_comm_volumes,
    enumerate_configs,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

PHASE_COLORS = {
    "ferromagnetic":           "#2563eb",
    "antiferromagnetic":       "#dc2626",
    "domain_wall":             "#16a34a",
    "spin_glass":              "#d97706",
    "quasi_antiferromagnetic": "#7c3aed",
    "unknown":                 "#9ca3af",
}

PHASE_LABELS = {
    "ferromagnetic":           "DP (ferromagnetic)",
    "antiferromagnetic":       "TP (antiferromagnetic)",
    "domain_wall":             "PP (domain wall)",
    "spin_glass":              "EP (spin-glass)",
    "quasi_antiferromagnetic": "CP (quasi-AF)",
}

COMM_COLORS = {
    "dp_allreduce":     "#2563eb",
    "tp_allgather":     "#16a34a",
    "tp_reducescatter": "#22c55e",
    "pp_p2p":           "#d97706",
    "ep_alltoall":      "#dc2626",
    "cp_allgather":     "#7c3aed",
}

COMM_LABELS = {
    "dp_allreduce":     "DP AllReduce",
    "tp_allgather":     "TP AllGather",
    "tp_reducescatter": "TP ReduceScatter",
    "pp_p2p":           "PP P2P",
    "ep_alltoall":      "EP AllToAll",
    "cp_allgather":     "CP AllGather",
}


def run_experiment(model: ModelParams, n_gpu: int, model_name: str) -> None:
    print(f"\n  {'='*50}")
    print(f"  {model_name}  /  {n_gpu} GPUs")
    print(f"  {'='*50}")
    print(f"  Parameters: {model.n_params/1e9:.2f}B  |  "
          f"L={model.n_layers}  H={model.hidden_dim}  "
          f"heads={model.n_heads}  seq={model.seq_len}  batch={model.batch_size}")

    result = optimise_parallelism(
        n_gpu, model,
        max_tp=8, max_pp=8,
        eta_hw_single=0.5,
        n_beta=60, n_bins=32,
    )

    print(f"\n  Theoretical η_multi,max : {result.multi_gpu_limit.eta_multi_max:.4f}")
    print(f"  Configs evaluated       : {len(result.scores)}")
    print(f"  Pareto-efficient        : {len(result.pareto_configs)}")
    print(f"\n  Recommended: {result.recommended.config.label}")
    print(f"    η_multi      = {result.recommended.eta_multi:.4f}")
    print(f"    comm_overhead= {result.recommended.comm_overhead*100:.1f}%")
    print(f"    η_overlap    = {result.recommended.resonance_eta:.4f}")
    print(f"    thermo_phase = {result.recommended.thermo_phase}")

    print(f"\n  Pareto Frontier:")
    print(f"  {'Config':<22} {'η_multi':>8} {'comm%':>7} "
          f"{'η_overlap':>10} {'bottleneck':>14} {'phase'}")
    print("  " + "-" * 72)
    for s in result.pareto_configs:
        print(f"  {s.config.label:<22} {s.eta_multi:>8.4f} "
              f"{s.comm_overhead*100:>6.1f}% "
              f"{s.resonance_eta:>10.4f} "
              f"{s.dominant_bottleneck:>14}  {s.thermo_phase}")

    # --- Fig 1: Pareto scatter -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{model_name}  /  {n_gpu} GPUs  —  Parallelism Optimizer",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    # All configs
    seen_phases: set[str] = set()
    for s in result.scores:
        phase = s.thermo_phase
        color = PHASE_COLORS.get(phase, "gray")
        label = PHASE_LABELS.get(phase, phase) if phase not in seen_phases else None
        seen_phases.add(phase)
        ax.scatter(s.comm_overhead * 100, s.eta_multi,
                   color=color, s=60, alpha=0.6, zorder=3, label=label)

    # Pareto frontier (highlighted)
    pareto_x = [s.comm_overhead * 100 for s in result.pareto_configs]
    pareto_y = [s.eta_multi           for s in result.pareto_configs]
    sorted_pareto = sorted(zip(pareto_x, pareto_y))
    px, py = zip(*sorted_pareto) if sorted_pareto else ([], [])
    ax.plot(px, py, "k--", lw=1.5, zorder=4, label="Pareto frontier")
    ax.scatter(pareto_x, pareto_y, color="black", s=100, zorder=5, marker="*")

    # Recommended
    ax.scatter(result.recommended.comm_overhead * 100, result.recommended.eta_multi,
               color="gold", s=200, zorder=6, marker="*",
               edgecolors="black", linewidths=1.5, label="Recommended")

    # Carnot ceiling
    ax.axhline(result.multi_gpu_limit.eta_multi_max,
               color="#6b7280", ls=":", lw=1.5,
               label=f"η_multi,max = {result.multi_gpu_limit.eta_multi_max:.4f}")

    # Config labels for Pareto configs
    for s in result.pareto_configs:
        ax.annotate(s.config.label,
                    (s.comm_overhead * 100, s.eta_multi),
                    textcoords="offset points", xytext=(6, 3), fontsize=7)

    ax.set_xlabel("Communication Overhead  (%)", fontsize=11)
    ax.set_ylabel("η_multi  (efficiency)", fontsize=11)
    ax.set_title("All Configs + Pareto Frontier", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # --- Fig 2: comm volume breakdown ----------------------------------------
    ax = axes[1]
    configs_to_show = result.pareto_configs[:8]   # top-8 Pareto configs
    labels = [s.config.label for s in configs_to_show]
    keys   = ["dp_allreduce", "tp_allgather", "tp_reducescatter",
              "pp_p2p", "ep_alltoall", "cp_allgather"]

    bottoms = np.zeros(len(configs_to_show))
    for key in keys:
        vals = []
        for s in configs_to_show:
            bd = s.comm_volumes.breakdown()
            vals.append(bd.get(key, 0.0) * 100)
        ax.bar(range(len(configs_to_show)), vals, bottom=bottoms,
               color=COMM_COLORS[key], label=COMM_LABELS[key], edgecolor="white")
        bottoms += np.array(vals)

    ax.set_xticks(range(len(configs_to_show)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of total comm volume  (%)", fontsize=11)
    ax.set_title("Communication Volume Breakdown  (Pareto configs)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    out = FIGURES / f"05_parallelism_{safe_name}_{n_gpu}gpu.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Run both experiments
# ---------------------------------------------------------------------------

print("=" * 60)
print("Experiment 05: Parallelism Optimizer — Pareto Frontiers")
print("=" * 60)

run_experiment(GPT2_SMALL, n_gpu=8,  model_name="GPT-2 Small")
run_experiment(LLAMA_7B,   n_gpu=64, model_name="LLaMA-7B")

print("\nDone.\n")

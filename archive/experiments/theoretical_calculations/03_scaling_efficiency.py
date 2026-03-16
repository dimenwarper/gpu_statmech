"""
Experiment 03: Multi-GPU Scaling Efficiency
============================================
Runs the energy-based multi-GPU limit for several canonical distributed-
training communication scenarios across 1 -> 64 GPUs and four interconnect
topologies:
  - NVLink-4 clique        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - NVSwitch fabric        (J = 0.10, BW = 900 GB/s)   ferromagnetic
  - PCIe Gen 5 ring        (J = 1.00, BW =  64 GB/s)   weakly coupled
  - InfiniBand fat-tree    (J = 5.00, BW =  50 GB/s)   spin-glass

Canonical workload scenarios:
  - LLaMA-7B pure data parallelism      DP-n
  - LLaMA-7B pure pipeline parallelism  PP-n
  - LLaMA-7B pure context parallelism   CP-n
  - LLaMA-7B pure tensor parallelism    TP-n

Outputs:
  - 03_scaling_efficiency.png   : scaling efficiency across scenarios
  - 03_comm_energy_share.png    : communication energy share across scenarios
  - 03_comm_load_headroom.png   : target communication load vs topology capacity

The communication-demand closure is:
  target_comm_load = total_bytes / (BW_ref * T_compute_ref)

Blank points in the scenario plots indicate infeasible operating points where a
slow topology cannot satisfy the required communication load under the current
closure.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpu_statmech.carnot import derive_carnot_limit
from gpu_statmech.multi_gpu import (
    TopologyGraph,
    derive_multi_gpu_carnot_limit,
    mean_topology_comm_load,
    normalise_comm_demand,
)
from gpu_statmech.parallelism import (
    LLAMA_7B,
    ModelParams,
    ParallelismConfig,
    estimate_comm_volumes,
    estimate_compute_time_s,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)
N_BETA = 20
N_BINS = 16
COMM_CAPACITY_FIELD = 1e6
COMM_CAPACITY_BETA = 1.0


@dataclass(frozen=True)
class ScenarioSpec:
    slug: str
    title: str
    config_builder: Callable[[int], ParallelismConfig]
    n_gpu_values: list[int]
    model: ModelParams = field(default_factory=lambda: LLAMA_7B)


def _single_gpu_topology() -> TopologyGraph:
    return TopologyGraph(n_gpu=1, links=[], name="single_gpu")


SCENARIOS = [
    ScenarioSpec(
        slug="dp",
        title="LLaMA-7B pure data parallelism (DP-n)",
        config_builder=lambda n: ParallelismConfig(dp=n),
        n_gpu_values=[1, 2, 4, 8, 16, 32, 64],
    ),
    ScenarioSpec(
        slug="pp",
        title="LLaMA-7B pure pipeline parallelism (PP-n)",
        config_builder=lambda n: ParallelismConfig(pp=n),
        n_gpu_values=[1, 2, 4, 8, 16, 32],
    ),
    ScenarioSpec(
        slug="cp",
        title="LLaMA-7B pure context parallelism (CP-n)",
        config_builder=lambda n: ParallelismConfig(cp=n),
        n_gpu_values=[1, 2, 4, 8, 16, 32, 64],
    ),
    ScenarioSpec(
        slug="tp",
        title="LLaMA-7B pure tensor parallelism (TP-n)",
        config_builder=lambda n: ParallelismConfig(tp=n),
        n_gpu_values=[1, 2, 4, 8, 16, 32],
    ),
]

TOPOLOGIES = {
    "NVLink-4 clique": TopologyGraph.nvlink_clique,
    "NVSwitch fabric": TopologyGraph.nvswitch_fabric,
    "PCIe Gen5 ring": TopologyGraph.pcie_ring,
    "InfiniBand fat-tree": TopologyGraph.infiniband_fat_tree,
}

COLORS = {
    "NVLink-4 clique": "#2563eb",
    "NVSwitch fabric": "#16a34a",
    "PCIe Gen5 ring": "#d97706",
    "InfiniBand fat-tree": "#dc2626",
}

MARKERS = {
    "NVLink-4 clique": "o",
    "NVSwitch fabric": "s",
    "PCIe Gen5 ring": "^",
    "InfiniBand fat-tree": "D",
}


def _initialise_topology_results() -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "eta_multi_max": [],
            "scaling_eff": [],
            "balance_proxy": [],
            "comm_energy": [],
            "capacity": [],
            "feasible": [],
        }
        for name in TOPOLOGIES
    }


def _evaluate_scenario(
    scenario: ScenarioSpec,
    eta_single: float,
    target_activity: float | None,
) -> dict[str, object]:
    print(f"\nScenario: {scenario.title}")
    print("  " + "-" * (len(scenario.title) + 10))

    scenario_results = _initialise_topology_results()
    bytes_gb: list[float] = []
    target_comm_loads: list[float] = []

    for n in scenario.n_gpu_values:
        config = scenario.config_builder(n)
        comm_volumes = estimate_comm_volumes(config, scenario.model)
        t_compute_ref = estimate_compute_time_s(config, scenario.model, eta_hw=eta_single)
        target_comm_load = normalise_comm_demand(comm_volumes.total_bytes, t_compute_ref)

        bytes_gb.append(comm_volumes.total_bytes / 1e9)
        target_comm_loads.append(target_comm_load)

        print(
            f"  n_gpu={n:<2d}  bytes/step={comm_volumes.total_bytes / 1e9:>8.2f} GB  "
            f"target_comm_load={target_comm_load:>8.4f}"
        )

        for topo_name, builder in TOPOLOGIES.items():
            topo = _single_gpu_topology() if n == 1 else builder(n)
            capacity = mean_topology_comm_load(
                COMM_CAPACITY_BETA,
                topo,
                comm_field=COMM_CAPACITY_FIELD,
            )
            topo_results = scenario_results[topo_name]
            topo_results["capacity"].append(capacity)

            try:
                limit = derive_multi_gpu_carnot_limit(
                    topo,
                    eta_hw_max_single=eta_single,
                    target_activity=target_activity,
                    target_comm_load=target_comm_load,
                    n_beta=N_BETA,
                    n_bins=N_BINS,
                )
            except ValueError as exc:
                topo_results["eta_multi_max"].append(float("nan"))
                topo_results["scaling_eff"].append(float("nan"))
                topo_results["balance_proxy"].append(float("nan"))
                topo_results["comm_energy"].append(float("nan"))
                topo_results["feasible"].append(False)
                print(
                    f"    {topo_name:<22s}  infeasible  "
                    f"(capacity={capacity:.4f}, reason={exc})"
                )
                continue

            topo_results["eta_multi_max"].append(limit.eta_multi_max)
            topo_results["scaling_eff"].append(limit.scaling_efficiency())
            topo_results["balance_proxy"].append(limit.resonance_eta)
            topo_results["comm_energy"].append(limit.comm_overhead_fraction)
            topo_results["feasible"].append(True)
            print(
                f"    {topo_name:<22s}  eta_multi,max={limit.eta_multi_max:.4f}  "
                f"scaling={limit.scaling_efficiency():.4f}  "
                f"comm energy={limit.comm_overhead_fraction * 100:.2f}%"
            )

    return {
        "spec": scenario,
        "results": scenario_results,
        "bytes_gb": bytes_gb,
        "target_comm_load": target_comm_loads,
    }


def _format_metric(value: float, as_percent: bool = False) -> str:
    if not math.isfinite(value):
        return "infeasible"
    if as_percent:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def _print_summary(all_results: dict[str, dict[str, object]]) -> None:
    print("\nSummary at largest simulated GPU count per scenario")
    print("=" * 60)
    for scenario in SCENARIOS:
        data = all_results[scenario.slug]
        n_values = data["spec"].n_gpu_values
        max_n = n_values[-1]
        bytes_gb = data["bytes_gb"][-1]
        target_load = data["target_comm_load"][-1]
        print(
            f"\n{scenario.title}\n"
            f"  max n_gpu={max_n}  bytes/step={bytes_gb:.2f} GB  "
            f"target_comm_load={target_load:.4f}"
        )
        print(
            f"  {'Topology':<24} {'scaling_eff':>12} {'comm energy':>14} {'capacity':>12}"
        )
        print("  " + "-" * 68)
        for topo_name in TOPOLOGIES:
            topo_results = data["results"][topo_name]
            scaling = topo_results["scaling_eff"][-1]
            comm_energy = topo_results["comm_energy"][-1]
            capacity = topo_results["capacity"][-1]
            print(
                f"  {topo_name:<24} {_format_metric(scaling):>12} "
                f"{_format_metric(comm_energy, as_percent=True):>14} {capacity:>12.4f}"
            )


def _plot_metric_grid(
    all_results: dict[str, dict[str, object]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_name: str,
    *,
    percent: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
    reference_line: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, scenario in zip(axes.flat, SCENARIOS):
        data = all_results[scenario.slug]
        n_values = data["spec"].n_gpu_values
        for topo_name in TOPOLOGIES:
            series = np.asarray(data["results"][topo_name][metric_key], dtype=float)
            if percent:
                series = series * 100.0
            ax.plot(
                n_values,
                series,
                color=COLORS[topo_name],
                marker=MARKERS[topo_name],
                lw=2,
                ms=6,
                label=topo_name,
            )

        if reference_line is not None:
            ax.axhline(reference_line, color="black", ls=":", lw=1.2)

        ax.set_title(scenario.title, fontsize=11)
        ax.set_xlabel("Number of GPUs", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels(n_values)
        if y_min is not None or y_max is not None:
            ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved -> {out}")
    plt.close(fig)


def _plot_capacity_grid(all_results: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Communication-Demand Headroom  (target load vs topology capacity)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, scenario in zip(axes.flat, SCENARIOS):
        data = all_results[scenario.slug]
        n_values = data["spec"].n_gpu_values
        ax.plot(
            n_values,
            data["target_comm_load"],
            color="black",
            marker="o",
            lw=2,
            ms=6,
            ls="--",
            label="workload target",
        )
        for topo_name in TOPOLOGIES:
            capacity = data["results"][topo_name]["capacity"]
            ax.plot(
                n_values,
                capacity,
                color=COLORS[topo_name],
                marker=MARKERS[topo_name],
                lw=2,
                ms=6,
                label=topo_name,
            )

        ax.set_title(scenario.title, fontsize=11)
        ax.set_xlabel("Number of GPUs", fontsize=10)
        ax.set_ylabel("normalized comm load", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels(n_values)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "03_comm_load_headroom.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved -> {out}")
    plt.close(fig)


def _plot_workload_pressure(all_results: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Scenario Communication Pressure", fontsize=14, fontweight="bold")

    for scenario in SCENARIOS:
        data = all_results[scenario.slug]
        n_values = data["spec"].n_gpu_values
        axes[0].plot(n_values, data["bytes_gb"], marker="o", lw=2, ms=6, label=scenario.slug.upper())
        axes[1].plot(
            n_values,
            data["target_comm_load"],
            marker="o",
            lw=2,
            ms=6,
            label=scenario.slug.upper(),
        )

    axes[0].set_title("Communication volume per step", fontsize=11)
    axes[0].set_xlabel("Number of GPUs", fontsize=10)
    axes[0].set_ylabel("bytes/step  (GB)", fontsize=10)
    axes[0].set_xscale("log", base=2)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Normalized communication demand", fontsize=11)
    axes[1].set_xlabel("Number of GPUs", fontsize=10)
    axes[1].set_ylabel("target_comm_load", fontsize=10)
    axes[1].set_xscale("log", base=2)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES / "03_comm_workload_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved -> {out}")
    plt.close(fig)


print("=" * 60)
print("Experiment 03: Multi-GPU Scaling Efficiency")
print("=" * 60)

single_limit = derive_carnot_limit(n_beta=N_BETA, n_bins=N_BINS)
eta_single = single_limit.eta_hw_max
print(f"\n  Single-GPU eta_hw,max = {eta_single:.4f}  ({eta_single * 100:.2f}%)")
print("  Canonical workload family: LLaMA-7B under pure DP / PP / CP / TP scenarios.")
print("  Blank points indicate that the communication-demand closure is infeasible")
print("  for that topology at the requested workload pressure.")

scenario_results = {
    scenario.slug: _evaluate_scenario(
        scenario,
        eta_single,
        single_limit.target_activity,
    )
    for scenario in SCENARIOS
}

_print_summary(scenario_results)

_plot_metric_grid(
    scenario_results,
    metric_key="scaling_eff",
    ylabel="eta_multi,max / eta_hw,max,single",
    title="Scaling Efficiency Across Communication Scenarios",
    out_name="03_scaling_efficiency.png",
    y_min=0.0,
    y_max=1.05,
    reference_line=1.0,
)
_plot_metric_grid(
    scenario_results,
    metric_key="comm_energy",
    ylabel="communication energy share (%)",
    title="Communication Energy Share Across Communication Scenarios",
    out_name="03_comm_energy_share.png",
    percent=True,
    y_min=0.0,
)
_plot_capacity_grid(scenario_results)
_plot_workload_pressure(scenario_results)

print("\nDone.\n")

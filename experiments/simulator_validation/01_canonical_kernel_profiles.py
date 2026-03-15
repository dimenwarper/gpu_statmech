"""
Experiment 01 — Canonical gpusim Kernel Profiles

Runs the canonical simulator-backed kernel suite used by
`scripts/run_gpusim_analysis.py`, then visualizes:
  - measured hardware efficiency vs inferred single-kernel ceiling
  - observed issue/stall behavior
  - inferred operating point (beta and memory feed efficiency)
  - observed vs model-predicted thermodynamic state families
  - exact warp-state residuals as a secondary diagnostic

This is the first simulator-backed experiment category. It is not based on
hardware spec numbers alone; it uses synthetic gpusim traces produced from the
canonical kernel profiles in `gpu_statmech.gpusim_driver`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_statmech.gpusim_driver import (  # noqa: E402
    canonical_kernel_profiles,
    load_gpusim_module,
    run_kernel_suite,
)
from gpu_statmech.observables import (  # noqa: E402
    WARP_STATE_FAMILY_KEYS,
    warp_state_family_fractions,
)
from gpu_statmech.thermo import analyse_protocol  # noqa: E402


FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

PHASE_COLORS = {
    "compute_bound": "#55a868",
    "memory_bound": "#dd8452",
    "latency_bound": "#c44e52",
    "mixed": "#4c72b0",
}

STATE_ORDER = [
    "eligible",
    "exec_dep",
    "short_scoreboard",
    "long_scoreboard",
    "mem_throttle",
    "barrier",
    "fetch",
    "idle",
]

FAMILY_ORDER = list(WARP_STATE_FAMILY_KEYS)
FAMILY_COLORS = {
    "productive": "#4c72b0",
    "dependency": "#8172b2",
    "memory": "#dd8452",
    "sync_frontend": "#937860",
    "idle": "#b5bdc5",
}
FAMILY_LABELS = {
    "productive": "productive",
    "dependency": "dependency",
    "memory": "memory",
    "sync_frontend": "sync/fetch",
    "idle": "idle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical gpusim kernel profiles and generate summary plots."
    )
    parser.add_argument(
        "--gpu",
        choices=("h100", "a100"),
        default="h100",
        help="gpusim GPU preset to run.",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=None,
        help="Canonical kernel profile to run. Repeat to select multiple kernels.",
    )
    return parser.parse_args()


def _paired_family_bars(
    ax: plt.Axes,
    names: list[str],
    observed: dict[str, list[float]],
    predicted: dict[str, list[float]],
) -> None:
    y = np.arange(len(names), dtype=float)
    height = 0.34
    left_obs = np.zeros(len(names), dtype=float)
    left_pred = np.zeros(len(names), dtype=float)
    for family in FAMILY_ORDER:
        obs_vals = np.asarray(observed[family], dtype=float)
        pred_vals = np.asarray(predicted[family], dtype=float)
        ax.barh(
            y + height / 2,
            obs_vals * 100.0,
            height=height,
            left=left_obs * 100.0,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.barh(
            y - height / 2,
            pred_vals * 100.0,
            height=height,
            left=left_pred * 100.0,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.6,
        )
        left_obs += obs_vals
        left_pred += pred_vals

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("State-family occupancy (%)")
    ax.set_title("Observed (solid) vs predicted (transparent) families")
    ax.grid(True, axis="x", alpha=0.25)


def main() -> int:
    args = parse_args()
    profiles = canonical_kernel_profiles(args.kernel)
    profile_desc = {profile.name: profile.description for profile in profiles}

    try:
        gpusim_module = load_gpusim_module()
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    traces = run_kernel_suite(gpusim_module, profiles=profiles, gpu=args.gpu)
    protocol = analyse_protocol(traces)
    analyses = sorted(protocol.kernel_analyses, key=lambda a: [p.name for p in profiles].index(a.kernel_name))

    names = [analysis.kernel_name for analysis in analyses]
    x = np.arange(len(names))
    eta = np.asarray([analysis.eta_hw * 100.0 for analysis in analyses], dtype=float)
    eta_max = np.asarray([analysis.eta_hw_max * 100.0 for analysis in analyses], dtype=float)
    eta_frac = np.asarray([analysis.eta_hw_fraction * 100.0 for analysis in analyses], dtype=float)
    issue = np.asarray([analysis.observables.mean_issue_activity for analysis in analyses], dtype=float)
    stall = np.asarray([analysis.observables.mean_stall_fraction for analysis in analyses], dtype=float)
    mem_stall = np.asarray(
        [analysis.observables.mean_memory_stall_fraction for analysis in analyses],
        dtype=float,
    )
    beta = np.asarray([analysis.thermo_state.beta for analysis in analyses], dtype=float)
    feed = np.asarray(
        [analysis.thermo_state.memory_feed_efficiency for analysis in analyses],
        dtype=float,
    )
    phases = [analysis.dominant_phase for analysis in analyses]
    phase_colors = [PHASE_COLORS.get(phase, "#4c72b0") for phase in phases]

    observed_state = {
        state: [analysis.observables.mean_warp_state_fractions.get(state, 0.0) for analysis in analyses]
        for state in STATE_ORDER
    }
    predicted_state = {
        state: [analysis.thermo_state.warp_state_fractions.get(state, 0.0) for analysis in analyses]
        for state in STATE_ORDER
    }
    observed_families = {
        family: [analysis.observables.mean_warp_state_family_fractions.get(family, 0.0) for analysis in analyses]
        for family in FAMILY_ORDER
    }
    predicted_families = {
        family: [
            warp_state_family_fractions(analysis.thermo_state.warp_state_fractions).get(family, 0.0)
            for analysis in analyses
        ]
        for family in FAMILY_ORDER
    }

    print("=" * 72)
    print("Experiment 01 — Canonical gpusim Kernel Profiles")
    print("=" * 72)
    print(f"GPU preset          : {args.gpu}")
    print(f"Kernels             : {', '.join(names)}")
    print(
        "Protocol summary    : "
        f"eta_hw={protocol.eta_hw * 100:.2f}%  "
        f"eta_hw/eta_hw,max={protocol.eta_hw_fraction * 100:.2f}%  "
        f"dominant_bottleneck={protocol.dominant_bottleneck()}"
    )
    print("\nPer-kernel summary:")
    for analysis in analyses:
        print(
            f"  {analysis.kernel_name:<16s} "
            f"phase={analysis.dominant_phase:<13s} "
            f"eta={analysis.eta_hw * 100:6.2f}%  "
            f"eta/eta_max={analysis.eta_hw_fraction * 100:6.2f}%  "
            f"beta={analysis.thermo_state.beta:6.3f}  "
            f"issue={analysis.observables.mean_issue_activity:5.3f}  "
            f"stall={analysis.observables.mean_stall_fraction:5.3f}  "
            f"mem_stall={analysis.observables.mean_memory_stall_fraction:5.3f}"
        )

    # Figure 1: overview
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(
        f"Canonical gpusim Kernel Profiles — {args.gpu.upper()}",
        fontsize=13,
        fontweight="bold",
    )

    width = 0.36
    ax = axes[0, 0]
    ax.bar(x - width / 2, eta, width=width, color=phase_colors, edgecolor="white", label="eta_hw")
    ax.bar(
        x + width / 2,
        eta_max,
        width=width,
        color="#d9d9d9",
        edgecolor="#666666",
        linewidth=0.8,
        label="eta_hw,max",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Measured Efficiency vs Inferred Ceiling")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    for idx, frac in enumerate(eta_frac):
        ax.text(x[idx] - width / 2, eta[idx] + 0.3, f"{frac:.1f}%", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    series = [issue, stall, mem_stall]
    labels = ["issue", "stall", "mem_stall"]
    colors = ["#4c72b0", "#c44e52", "#dd8452"]
    for offset, values, label, color in zip((-width, 0.0, width), series, labels, colors):
        ax.bar(x + offset, values, width=width, label=label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title("Observed Issue and Stall Mix")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    for idx, analysis in enumerate(analyses):
        ax.scatter(
            analysis.observables.mean_issue_activity,
            analysis.observables.mean_memory_stall_fraction,
            s=80 + 4.0 * eta_frac[idx],
            color=phase_colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.annotate(analysis.kernel_name, (issue[idx], mem_stall[idx]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlim(0.0, max(issue.max() * 1.15, 0.5))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Issue activity")
    ax.set_ylabel("Memory stall fraction")
    ax.set_title("Observed Operating Regimes")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, beta, color="#8172b2", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Inferred beta (log scale)")
    ax.set_title("Inferred Operating Point")
    ax.grid(True, axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, feed, color="#55a868", marker="o", linewidth=2.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Memory feed efficiency")

    fig1.tight_layout()
    out1 = FIGURES / "01_canonical_overview.png"
    fig1.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved figure         : {out1}")

    # Figure 2: family-level fit + exact-state residuals
    fig2, (ax_family, ax_resid) = plt.subplots(
        1,
        2,
        figsize=(15.5, 5.8),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig2.suptitle(
        "Family-Level Fit with Exact-State Residual Diagnostic",
        fontsize=13,
        fontweight="bold",
    )
    _paired_family_bars(ax_family, names, observed_families, predicted_families)

    residual = np.asarray(
        [
            [
                predicted_state[state][idx] - observed_state[state][idx]
                for state in STATE_ORDER
            ]
            for idx in range(len(names))
        ],
        dtype=float,
    )
    vmax = float(max(np.max(np.abs(residual)), 1e-6))
    im = ax_resid.imshow(residual, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_resid.set_xticks(np.arange(len(STATE_ORDER)))
    ax_resid.set_xticklabels(STATE_ORDER, rotation=35, ha="right")
    ax_resid.set_yticks(np.arange(len(names)))
    ax_resid.set_yticklabels(names)
    ax_resid.set_title("Exact-state residuals\n(predicted - observed)")
    for row in range(residual.shape[0]):
        for col in range(residual.shape[1]):
            ax_resid.text(
                col,
                row,
                f"{residual[row, col] * 100:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    cbar = fig2.colorbar(im, ax=ax_resid, fraction=0.046, pad=0.04)
    cbar.set_label("Residual occupancy fraction")

    family_handles = [Patch(facecolor=FAMILY_COLORS[family], label=FAMILY_LABELS[family]) for family in FAMILY_ORDER]
    fig2.legend(handles=family_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.36, -0.02))
    fig2.tight_layout(rect=(0, 0.05, 1, 1))
    out2 = FIGURES / "01_warp_state_match.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figure         : {out2}")

    print("\nKernel profile descriptions:")
    for name in names:
        print(f"  {name:<16s} {profile_desc[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Experiment 02 — Simulator Intervention Recommendation

This experiment tests the actual thesis of the simulator path:

  given only a baseline trace, can the thermodynamic model recommend the
  intervention class that most improves efficiency?

We build a small counterfactual neighborhood around the canonical `gpusim`
kernel families, run every baseline plus every intervention, and compare:

  - stat-mech recommendation
  - raw counter family heuristic
  - roofline-like heuristic
  - occupancy-only heuristic
  - random choice

The main metric is oracle attainment ratio:

  realized_gain(chosen lever) / realized_gain(oracle-best lever)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_statmech.carnot import derive_carnot_limit  # noqa: E402
from gpu_statmech.energy import aggregate_energy  # noqa: E402
from gpu_statmech.gpusim_driver import (  # noqa: E402
    canonical_kernel_profiles,
    load_gpusim_module,
    run_kernel_suite,
)
from gpu_statmech.gpusim_recommendation import (  # noqa: E402
    BASELINE_STRESS_KEYS,
    INTERVENTION_KEYS,
    RecommendationBaseline,
    apply_intervention,
    generate_recommendation_baselines,
    oracle_attainment_ratio,
    recommend_intervention_occupancy_only,
    recommend_intervention_raw_counter,
    recommend_intervention_roofline,
    recommend_intervention_statmech,
    statmech_intervention_scores,
)
from gpu_statmech.partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG  # noqa: E402
from gpu_statmech.thermo import analyse_kernel  # noqa: E402


FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

METHOD_ORDER = (
    "stat_mech",
    "raw_counter",
    "roofline",
    "occupancy_only",
    "random",
)
METHOD_LABELS = {
    "stat_mech": "stat-mech",
    "raw_counter": "raw counters",
    "roofline": "roofline",
    "occupancy_only": "occupancy only",
    "random": "random",
}
METHOD_COLORS = {
    "stat_mech": "#4c72b0",
    "raw_counter": "#dd8452",
    "roofline": "#55a868",
    "occupancy_only": "#c44e52",
    "random": "#8172b2",
}
LEVER_LABELS = {
    "locality": "locality",
    "occupancy": "occupancy",
    "tensorize": "tensorize",
}
STRESS_LABELS = {
    "base": "base",
    "memory_stressed": "memory stressed",
    "footprint_stressed": "footprint stressed",
    "compute_unoptimized": "compute unoptimized",
}
EXPERIMENT_GRID_CAP_X = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simulator recommendation experiment and plot oracle-attainment metrics."
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
        help="Canonical kernel family to include. Repeat to select multiple families.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for the random baseline and bootstrap intervals.",
    )
    return parser.parse_args()


def _bootstrap_ci(values: list[float], *, seed: int, n_samples: int = 2000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = sample.mean()
    lo, hi = np.quantile(means, [0.05, 0.95])
    return (float(lo), float(hi))


def _experiment_scale_profiles(profiles: list) -> list:
    """
    Downscale grid sizes for the recommendation study.

    The experiment is about relative intervention choice, not wall-clock
    fidelity, so we cap the grid to keep the simulator runtime tractable.
    """

    scaled = []
    for profile in profiles:
        gx, gy, gz = profile.grid
        scaled.append(replace(profile, grid=(min(gx, EXPERIMENT_GRID_CAP_X), gy, gz)))
    return scaled


def _all_profiles_for_experiment(
    baselines: list[RecommendationBaseline],
) -> tuple[list, dict[str, dict[str, str]]]:
    profiles = []
    intervention_names: dict[str, dict[str, str]] = {}
    for baseline in baselines:
        profiles.append(baseline.profile)
        intervention_names[baseline.key] = {}
        for lever in INTERVENTION_KEYS:
            variant = apply_intervention(baseline.profile, lever)
            profiles.append(variant)
            intervention_names[baseline.key][lever] = variant.name
    return profiles, intervention_names


def _method_recommendations(analysis: object, *, rng: np.random.Generator) -> dict[str, str]:
    return {
        "stat_mech": recommend_intervention_statmech(analysis),
        "raw_counter": recommend_intervention_raw_counter(analysis),
        "roofline": recommend_intervention_roofline(analysis),
        "occupancy_only": recommend_intervention_occupancy_only(analysis),
        "random": str(rng.choice(INTERVENTION_KEYS)),
    }


def _plot_oracle_attainment(records: list[dict[str, object]], *, seed: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Simulator Recommendation Study — Oracle Attainment", fontsize=13, fontweight="bold")

    means = []
    errors_low = []
    errors_high = []
    accuracy = []
    for idx, method in enumerate(METHOD_ORDER):
        vals = [float(record["attainment"][method]) for record in records]
        mean = float(np.mean(vals)) if vals else 0.0
        lo, hi = _bootstrap_ci(vals, seed=seed + idx)
        means.append(mean)
        errors_low.append(max(mean - lo, 0.0))
        errors_high.append(max(hi - mean, 0.0))
        acc = np.mean([record["recommended"][method] == record["oracle_lever"] for record in records]) if records else 0.0
        accuracy.append(float(acc))

    x = np.arange(len(METHOD_ORDER))
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    ax = axes[0]
    ax.bar(x, means, color=colors, edgecolor="white", yerr=np.vstack([errors_low, errors_high]), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean oracle attainment ratio")
    ax.set_title("How much of the oracle-best gain does each method recover?")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, mean in enumerate(means):
        ax.text(idx, mean + 0.03, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(x, accuracy, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Top-1 lever accuracy")
    ax.set_title("How often does each method pick the oracle-best lever?")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, acc in enumerate(accuracy):
        ax.text(idx, acc + 0.03, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGURES / "02_oracle_attainment.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_statmech_supporting(records: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("Stat-mech Recommendation Diagnostics", fontsize=13, fontweight="bold")

    lever_to_idx = {lever: idx for idx, lever in enumerate(INTERVENTION_KEYS)}
    confusion = np.zeros((len(INTERVENTION_KEYS), len(INTERVENTION_KEYS)), dtype=float)
    for record in records:
        oracle_idx = lever_to_idx[str(record["oracle_lever"])]
        pred_idx = lever_to_idx[str(record["recommended"]["stat_mech"])]
        confusion[oracle_idx, pred_idx] += 1.0

    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(confusion)
    np.divide(confusion, np.maximum(row_totals, 1.0), out=normalized, where=row_totals > 0)
    ax = axes[0]
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(INTERVENTION_KEYS)))
    ax.set_xticklabels([LEVER_LABELS[k] for k in INTERVENTION_KEYS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(INTERVENTION_KEYS)))
    ax.set_yticklabels([LEVER_LABELS[k] for k in INTERVENTION_KEYS])
    ax.set_xlabel("Recommended by stat-mech")
    ax.set_ylabel("Oracle-best lever")
    ax.set_title("Normalized confusion matrix")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized frequency")

    ax = axes[1]
    stress_groups = []
    stress_means = []
    for stress in BASELINE_STRESS_KEYS:
        vals = [float(record["attainment"]["stat_mech"]) for record in records if record["stress"] == stress]
        if not vals:
            continue
        stress_groups.append(STRESS_LABELS[stress])
        stress_means.append(float(np.mean(vals)))
    xpos = np.arange(len(stress_groups))
    ax.bar(xpos, stress_means, color=METHOD_COLORS["stat_mech"], edgecolor="white")
    ax.set_xticks(xpos)
    ax.set_xticklabels(stress_groups, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean oracle attainment ratio")
    ax.set_title("Stat-mech attainment by baseline stress")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, mean in enumerate(stress_means):
        ax.text(idx, mean + 0.03, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGURES / "02_statmech_confusion.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    families = _experiment_scale_profiles(canonical_kernel_profiles(args.kernel))
    baselines = generate_recommendation_baselines(families)
    all_profiles, intervention_names = _all_profiles_for_experiment(baselines)

    try:
        gpusim_module = load_gpusim_module()
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    traces = run_kernel_suite(gpusim_module, profiles=all_profiles, gpu=args.gpu)
    carnot_limit = derive_carnot_limit(H100_SM_CONFIG, H100_MEMORY_LEVELS)
    baseline_names = {baseline.profile.name for baseline in baselines}
    analyses = {
        name: analyse_kernel(name, snaps, carnot_limit=carnot_limit, n_beta=80)
        for name, snaps in traces.items()
        if name in baseline_names
    }
    eta_by_name = {
        name: aggregate_energy(snaps, n_sm=H100_SM_CONFIG.n_sm).eta_hw
        for name, snaps in traces.items()
    }

    records: list[dict[str, object]] = []
    skipped = 0
    for baseline in baselines:
        baseline_analysis = analyses[baseline.profile.name]
        gains = {
            lever: eta_by_name[intervention_names[baseline.key][lever]] - baseline_analysis.eta_hw
            for lever in INTERVENTION_KEYS
        }
        oracle_lever = max(INTERVENTION_KEYS, key=lambda lever: gains[lever])
        oracle_gain = max(0.0, float(gains[oracle_lever]))
        if oracle_gain <= 1e-9:
            skipped += 1
            continue

        recommended = _method_recommendations(baseline_analysis, rng=rng)
        stat_scores = statmech_intervention_scores(baseline_analysis)
        attainment = {
            method: oracle_attainment_ratio(float(gains[lever]), oracle_gain)
            for method, lever in recommended.items()
        }
        records.append(
            {
                "family": baseline.family,
                "stress": baseline.stress,
                "baseline_name": baseline.profile.name,
                "baseline_eta": baseline_analysis.eta_hw,
                "oracle_lever": oracle_lever,
                "oracle_gain": oracle_gain,
                "gains": gains,
                "recommended": recommended,
                "attainment": attainment,
                "dominant_phase": baseline_analysis.dominant_phase,
                "dominant_bottleneck": baseline_analysis.bottleneck.dominant_source,
                "stat_scores": stat_scores,
            }
        )

    if not records:
        print("No actionable baselines found; all interventions had zero or negative gain.", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Experiment 02 — Simulator Intervention Recommendation")
    print("=" * 72)
    print(f"GPU preset          : {args.gpu}")
    print(f"Kernel families     : {', '.join(profile.name for profile in families)}")
    print(f"Actionable baselines: {len(records)} / {len(baselines)} (skipped {skipped})")

    oracle_by_stress: dict[str, list[str]] = defaultdict(list)
    for record in records:
        oracle_by_stress[str(record["stress"])].append(str(record["oracle_lever"]))

    print("\nOracle lever by stress:")
    for stress in BASELINE_STRESS_KEYS:
        levers = oracle_by_stress.get(stress, [])
        if not levers:
            continue
        counts = {lever: levers.count(lever) for lever in INTERVENTION_KEYS}
        pretty = ", ".join(f"{lever}={counts[lever]}" for lever in INTERVENTION_KEYS)
        print(f"  {STRESS_LABELS[stress]:<18s} {pretty}")

    print("\nMethod summary:")
    for method in METHOD_ORDER:
        vals = [float(record["attainment"][method]) for record in records]
        accuracy = np.mean([record["recommended"][method] == record["oracle_lever"] for record in records])
        print(
            f"  {METHOD_LABELS[method]:<15s} "
            f"attainment={np.mean(vals):.3f}  "
            f"top1={accuracy:.3f}"
        )

    sample_records = sorted(records, key=lambda record: (-float(record["oracle_gain"]), str(record["baseline_name"])))[:6]
    print("\nTop actionable baselines:")
    for record in sample_records:
        gains_text = ", ".join(f"{lever}={100.0 * float(record['gains'][lever]):5.2f} pp" for lever in INTERVENTION_KEYS)
        print(
            f"  {record['baseline_name']:<34s} "
            f"oracle={record['oracle_lever']:<11s} "
            f"phase={record['dominant_phase']:<13s} "
            f"bottleneck={record['dominant_bottleneck']:<24s} "
            f"{gains_text}"
        )

    out1 = _plot_oracle_attainment(records, seed=args.seed)
    out2 = _plot_statmech_supporting(records)
    print(f"\nSaved figure         : {out1}")
    print(f"Saved figure         : {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run a canonical kernel suite through gpusim and analyse the resulting traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gpu_statmech.gpusim_driver import (
    build_protocol_report,
    canonical_kernel_profiles,
    load_gpusim_module,
    run_kernel_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical workloads through gpusim and analyse them with gpu_statmech."
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
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="List the available canonical kernel profiles and exit.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = canonical_kernel_profiles(args.kernel)

    if args.list_kernels:
        for profile in profiles:
            print(f"{profile.name}: {profile.description}")
        return 0

    try:
        gpusim_module = load_gpusim_module()
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    traces = run_kernel_suite(gpusim_module, profiles=profiles, gpu=args.gpu)
    report = build_protocol_report(traces)

    print(f"Ran {len(report['kernels'])} canonical kernel(s) on gpusim {args.gpu}.")
    print(
        "Protocol: "
        f"eta_hw={report['protocol']['eta_hw']*100:.2f}%  "
        f"eta_hw/eta_hw,max={report['protocol']['eta_hw_fraction']*100:.2f}%  "
        f"dominant_bottleneck={report['protocol']['dominant_bottleneck']}"
    )
    for kernel in report["kernels"]:
        print(
            f"[{kernel['kernel_name']}] "
            f"phase={kernel['dominant_phase']}  "
            f"bottleneck={kernel['dominant_bottleneck']}  "
            f"eta={kernel['eta_hw']*100:.2f}%  "
            f"beta={kernel['beta']:.3f}  "
            f"issue={kernel['mean_issue_activity']:.3f}  "
            f"stall={kernel['mean_stall_fraction']:.3f}  "
            f"mem_stall={kernel['mean_memory_stall_fraction']:.3f}"
        )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote JSON report to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

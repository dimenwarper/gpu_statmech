"""
End-to-end driver for running canonical workloads through gpusim and
analysing the resulting traces with gpu_statmech.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping, Sequence

from .thermo import analyse_protocol


@dataclass(frozen=True)
class GpuSimKernelProfile:
    """Resource-profile description for a canonical simulator workload."""

    name: str
    description: str
    threads_per_block: int
    regs_per_thread: int
    smem_per_block: int
    grid: tuple[int, int, int]
    instr_mix: dict[str, float]


CANONICAL_KERNEL_PROFILES: tuple[GpuSimKernelProfile, ...] = (
    GpuSimKernelProfile(
        name="gemm_tc",
        description="Tensor-core-dominant GEMM tile",
        threads_per_block=256,
        regs_per_thread=96,
        smem_per_block=64 * 1024,
        grid=(4096, 1, 1),
        instr_mix={
            "fp16": 0.20,
            "fp32": 0.05,
            "int": 0.00,
            "sfu": 0.00,
            "mem": 0.15,
            "tensor_core": 0.60,
        },
    ),
    GpuSimKernelProfile(
        name="flash_attention",
        description="Attention-style kernel with SRAM reuse pressure",
        threads_per_block=128,
        regs_per_thread=128,
        smem_per_block=96 * 1024,
        grid=(3072, 1, 1),
        instr_mix={
            "fp16": 0.30,
            "fp32": 0.05,
            "int": 0.00,
            "sfu": 0.05,
            "mem": 0.30,
            "tensor_core": 0.30,
        },
    ),
    GpuSimKernelProfile(
        name="layernorm",
        description="Low-intensity normalization kernel",
        threads_per_block=256,
        regs_per_thread=48,
        smem_per_block=16 * 1024,
        grid=(8192, 1, 1),
        instr_mix={
            "fp16": 0.00,
            "fp32": 0.45,
            "int": 0.05,
            "sfu": 0.10,
            "mem": 0.40,
            "tensor_core": 0.00,
        },
    ),
    GpuSimKernelProfile(
        name="softmax_reduce",
        description="Reduction-heavy softmax stage",
        threads_per_block=256,
        regs_per_thread=64,
        smem_per_block=32 * 1024,
        grid=(4096, 1, 1),
        instr_mix={
            "fp16": 0.00,
            "fp32": 0.40,
            "int": 0.00,
            "sfu": 0.20,
            "mem": 0.40,
            "tensor_core": 0.00,
        },
    ),
    GpuSimKernelProfile(
        name="transpose_bw",
        description="Bandwidth-dominated transpose/copy kernel",
        threads_per_block=256,
        regs_per_thread=40,
        smem_per_block=48 * 1024,
        grid=(8192, 1, 1),
        instr_mix={
            "fp16": 0.00,
            "fp32": 0.10,
            "int": 0.00,
            "sfu": 0.00,
            "mem": 0.90,
            "tensor_core": 0.00,
        },
    ),
)


def canonical_kernel_profiles(
    names: Sequence[str] | None = None,
) -> list[GpuSimKernelProfile]:
    """Return the canonical kernel suite, optionally filtered by name."""

    if names is None:
        return list(CANONICAL_KERNEL_PROFILES)

    selected = []
    by_name = {profile.name: profile for profile in CANONICAL_KERNEL_PROFILES}
    for name in names:
        if name not in by_name:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"unknown kernel profile '{name}' (available: {available})")
        selected.append(by_name[name])
    return selected


def load_gpusim_module() -> Any:
    """
    Import the `gpusim` Python extension module with a concrete remediation hint.
    """

    try:
        return importlib.import_module("gpusim")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `gpusim`. Build/install the sibling repo's Python "
            "extension first, for example: `cd ../gpusim && maturin develop "
            "--features python`."
        ) from exc


def _build_instr_mix(gpusim_module: Any, instr_mix: Mapping[str, float]) -> Any:
    return gpusim_module.InstrMix(
        fp16=float(instr_mix.get("fp16", 0.0)),
        fp32=float(instr_mix.get("fp32", 0.0)),
        int_frac=float(instr_mix.get("int", 0.0)),
        sfu=float(instr_mix.get("sfu", 0.0)),
        mem=float(instr_mix.get("mem", 0.0)),
        tensor_core=float(instr_mix.get("tensor_core", 0.0)),
    )


def build_kernel_spec(gpusim_module: Any, profile: GpuSimKernelProfile) -> Any:
    """Construct a `gpusim.KernelSpec` from a canonical profile."""

    return gpusim_module.KernelSpec(
        name=profile.name,
        threads_per_block=profile.threads_per_block,
        regs_per_thread=profile.regs_per_thread,
        smem_per_block=profile.smem_per_block,
        grid=profile.grid,
        instr_mix=_build_instr_mix(gpusim_module, profile.instr_mix),
    )


def run_kernel_suite(
    gpusim_module: Any,
    profiles: Sequence[GpuSimKernelProfile] | None = None,
    gpu: str = "h100",
) -> dict[str, list[dict[str, Any]]]:
    """Run a canonical workload suite through `gpusim` and return raw traces."""

    if profiles is None:
        profiles = CANONICAL_KERNEL_PROFILES

    if not hasattr(gpusim_module.GpuSim, gpu):
        raise ValueError(f"unknown GPU preset '{gpu}'")

    traces: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        simulator = getattr(gpusim_module.GpuSim, gpu)()
        spec = build_kernel_spec(gpusim_module, profile)
        traces[profile.name] = simulator.run(spec)
    return traces


def build_protocol_report(
    traces: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build a compact JSON-serializable report from a set of kernel traces."""

    protocol = analyse_protocol(dict(traces))
    kernels = []
    for analysis in protocol.kernel_analyses:
        kernels.append(
            {
                "kernel_name": analysis.kernel_name,
                "eta_hw": analysis.eta_hw,
                "eta_hw_max": analysis.eta_hw_max,
                "eta_hw_fraction": analysis.eta_hw_fraction,
                "dominant_phase": analysis.dominant_phase,
                "dominant_bottleneck": analysis.bottleneck.dominant_source,
                "beta": analysis.thermo_state.beta,
                "work_field": analysis.thermo_state.work_field,
                "memory_feed_efficiency": analysis.thermo_state.memory_feed_efficiency,
                "mean_issue_activity": analysis.observables.mean_issue_activity,
                "mean_stall_fraction": analysis.observables.mean_stall_fraction,
                "mean_memory_stall_fraction": (
                    analysis.observables.mean_memory_stall_fraction
                ),
            }
        )

    return {
        "protocol": {
            "eta_hw": protocol.eta_hw,
            "eta_hw_max": protocol.eta_hw_max,
            "eta_hw_fraction": protocol.eta_hw_fraction,
            "dominant_bottleneck": protocol.dominant_bottleneck(),
        },
        "kernels": kernels,
    }

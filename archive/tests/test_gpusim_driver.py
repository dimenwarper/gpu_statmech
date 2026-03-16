import importlib

import pytest

from gpu_statmech.gpusim_driver import (
    GpuSimKernelProfile,
    build_kernel_spec,
    build_protocol_report,
    canonical_kernel_profiles,
    load_gpusim_module,
    run_kernel_suite,
)


class _FakeInstrMix:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeKernelSpec:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSimulator:
    def run(self, spec):
        _ = spec
        return [
            {
                "active_warps": 0.5,
                "stall_fraction": 0.2,
                "issue_activity": 0.4,
                "memory_stall_fraction": 0.1,
                "instr_mix": {"fp32": 1.0},
                "l2_hit_rate": 0.7,
                "hbm_bw_util": 0.3,
                "smem_util": 0.2,
                "reg_util": 0.2,
            }
        ]


class _FakeGpuSimFactory:
    @staticmethod
    def h100():
        return _FakeSimulator()

    @staticmethod
    def a100():
        return _FakeSimulator()


class _FakeGpuSimModule:
    InstrMix = _FakeInstrMix
    KernelSpec = _FakeKernelSpec
    GpuSim = _FakeGpuSimFactory


def test_canonical_kernel_profile_selection():
    profiles = canonical_kernel_profiles(["gemm_tc", "transpose_bw"])
    assert [profile.name for profile in profiles] == ["gemm_tc", "transpose_bw"]


def test_unknown_kernel_profile_raises():
    with pytest.raises(ValueError, match="unknown kernel profile"):
        canonical_kernel_profiles(["does_not_exist"])


def test_build_kernel_spec_uses_gpusim_api():
    profile = GpuSimKernelProfile(
        name="toy",
        description="toy",
        threads_per_block=128,
        regs_per_thread=32,
        smem_per_block=4096,
        grid=(4, 1, 1),
        instr_mix={"fp16": 0.2, "fp32": 0.8},
    )
    spec = build_kernel_spec(_FakeGpuSimModule, profile)
    assert spec.kwargs["name"] == "toy"
    assert spec.kwargs["instr_mix"].kwargs["fp32"] == pytest.approx(0.8)


def test_run_kernel_suite_returns_named_traces():
    traces = run_kernel_suite(
        _FakeGpuSimModule,
        profiles=canonical_kernel_profiles(["gemm_tc", "layernorm"]),
        gpu="h100",
    )
    assert set(traces) == {"gemm_tc", "layernorm"}
    assert all(isinstance(trace, list) and trace for trace in traces.values())


def test_build_protocol_report_is_json_ready():
    traces = {
        "gemm_tc": [{"active_warps": 0.6, "stall_fraction": 0.1, "instr_mix": {"fp32": 1.0}}],
        "transpose_bw": [{"active_warps": 0.4, "stall_fraction": 0.4, "instr_mix": {"fp32": 1.0}}],
    }
    report = build_protocol_report(traces)
    assert "protocol" in report
    assert len(report["kernels"]) == 2
    assert report["protocol"]["eta_hw"] >= 0.0


def test_load_gpusim_module_error_message(monkeypatch):
    real_import = importlib.import_module

    def _fake_import(name: str):
        if name == "gpusim":
            raise ModuleNotFoundError("missing")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ModuleNotFoundError, match="maturin develop --features python"):
        load_gpusim_module()

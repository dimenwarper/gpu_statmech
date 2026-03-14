# Ideas: GPU Simulation-Driven Architecture Search

**Date**: 2026-03-14
**Context**: Integrate `/root/projects/gpusim` (Rust, cycle-accurate) with `gpu_statmech` (Python, statistical mechanics theory) to replace heuristic proxy scores with simulation-validated signals in the architecture search loop.

---

## Core Thesis

The architecture search loop (Phase 3) currently scores kernel proposals using a heuristic `arch_score` in `compiler.py`. By routing proposals through gpusim, we get a **ground-truth simulated η_hw** as the Pareto score — making the search loop physics-validated end-to-end.

---

## Direction 1: `simulator.py` Bridge Module

Create `src/gpu_statmech/simulator.py` as a clean integration layer between the two projects.

**Responsibilities:**
- Translate `KernelSpec` (gpu_statmech) → `KernelSpec` / `InstrMix` (gpusim Python bindings)
- Run simulations via gpusim's PyO3 bindings (`GpuSim.h100().run(spec)`)
- Translate `MicrostateSnapshot` trace → `ThermodynamicState`
- Expose a simple API usable by `compiler.py` and `loop.py`

**Sketch:**
```python
simulate(kernel_spec: KernelSpec) -> ThermodynamicState
simulate_multi_gpu(kernel_spec, parallelism_config) -> MultiGPUThermodynamicState
```

**Notes:**
- gpusim field names already match gpu_statmech (designed this way)
- gpusim Python bindings exist via PyO3 (`maturin develop --features python`)
- Keep gpusim as a separate Rust project; call via Python bindings only

---

## Direction 2: β Back-Solver

Use simulated warp state distributions to **back-solve β** for any given workload, grounding the abstract inverse-resource-pressure parameter in observable simulation data.

**Method:**
1. Run gpusim with a given `InstrMix`, collect warp state fractions from `MicrostateSnapshot` (sm_stall_frac, active warps, etc.)
2. Fit β by minimizing KL divergence between:
   - **Empirical**: `p_sim(warp_state)` from the trace
   - **Theoretical**: `p(σ) ∝ exp[-β(E_in(σ) - h·W_hw(σ))]`
3. Return fitted β as the operating point for that kernel

**Why this matters:**
- Currently β is a free sweep parameter; this pins it to a real workload
- Enables per-kernel β estimates → more accurate η_hw predictions
- Could feed back into the Carnot curve to show where real workloads sit

**New function:** `fit_beta(trace: list[MicrostateSnapshot]) -> float`

---

## Direction 3: Simulation-Driven Pareto Scoring (Replace `arch_score`)

Upgrade the search loop to use simulated η_hw instead of the heuristic proxy.

**Current flow:**
```
Oracle proposes KernelSpec
  → compiler.py computes heuristic arch_score
  → Pareto frontier over (arch_score, expressiveness)
```

**New flow:**
```
Oracle proposes KernelSpec
  → simulator.py runs gpusim
  → extract simulated η_hw from trace
  → Pareto frontier over (simulated η_hw / η_hw_max, expressiveness)
```

**Changes needed:**
- `compiler.py`: add `simulate=True` flag to `compile_kernel()`, routes to simulator if available
- `loop.py`: pass simulated scores to Pareto instead of proxy scores
- Graceful fallback to `arch_score` if gpusim not installed

---

## Direction 4: Carnot Curve Validation Experiment

A new experiment (`experiments/theoretical_calculations/04_carnot_validation.py`) that:
1. Sweeps `InstrMix` from 100% memory-bound → 100% compute-bound (thermodynamic phase sweep)
2. Runs each mix through gpusim, extracts empirical η_hw
3. Overlays on the theoretical Carnot curve from `carnot.py`

**Expected output:** `figures/04_carnot_validation.png` — the **killer paper figure** showing theory vs simulation agreement across the full efficiency spectrum.

---

## Direction 5: Fit Multi-GPU Coupling Constants J from Simulation

Currently J values are hand-assigned heuristics:
- `J_DP = 1.0`, `J_TP = 1.5`, `J_PP = 0.8`, `J_EP = 5.0`, `J_CP = 1.4`

**Method:**
1. Run AllReduce / AllGather / AllToAll simulations at 1→64 GPU scale in gpusim
2. Measure communication energy fraction from `CollectiveStats`
3. Back-fit J per parallelism strategy by matching `η_multi` predictions to simulated results
4. Replace hard-coded J constants with fitted values + uncertainty estimates

**New function:** `fit_coupling_constants(cluster_trace) -> dict[str, float]`

---

## Suggested Implementation Order

1. **`simulator.py` bridge** (Direction 1) — unlocks everything else
2. **β back-solver** (Direction 2) — validates theory, needed for paper
3. **Simulation-driven Pareto** (Direction 3) — upgrades the search loop
4. **Carnot validation experiment** (Direction 4) — paper figure
5. **Fit J constants** (Direction 5) — strengthens multi-GPU story

---

## Open Questions

- Are gpusim Python bindings already compiled? (`maturin develop --features python` needed)
- Simulation cost per kernel: is gpusim fast enough to run 20–50 proposals per loop iteration?
- Should `simulate=True` be the default in loop.py, or opt-in?
- For the paper: validate on A100 config as well (gpusim supports both H100 and A100)?

---

## Gap Analysis: What Needs to Happen Before the Loop is Simulation-Driven

The loop infrastructure (`oracle.py`, `loop.py`, `compiler.py`, `pareto.py`) is fully scaffolded and runs end-to-end today. The gaps are in the **scoring signal quality** and the **simulator bridge**.

### 🔴 Critical (blocking)

**1. `thermo_score` is not real η_hw**
Currently in `compiler.py`:
```python
thermo = sum(1 for c in report.conditions if c.satisfied) / n_cond
```
This counts Carnot conditions satisfied — not actual η_hw. The comment in the code even acknowledges the real `eta_hw_fraction` collapses to ~0 for all practical kernels under the current theory. This is the core signal that needs to come from gpusim.

**2. `simulator.py` doesn't exist**
The bridge between the two projects. Everything else depends on it.

**3. gpusim Python bindings need to be compiled**
`maturin develop --features python` hasn't been run. Hard prerequisite.

### 🟡 Important (needed for correctness)

**4. `KernelProposal` → `gpusim.InstrMix` field mapping is undefined**
How `tensor_core_utilisation`, `arithmetic_intensity`, and `memory_access_pattern` map to
`InstrMix(fp16, mem, tensor_core, ...)` hasn't been written down or implemented anywhere.

**5. `MicrostateSnapshot` → η_hw aggregation is undefined**
How to reduce a per-block trace into a single η_hw scalar for the Pareto score needs to be designed.

### 🟢 Nice-to-have (non-blocking for the loop, needed for paper)

**6. β back-solver** — pins β to real workloads, needed for theory validation figures

**7. End-to-end integration test** — smoke test for the full loop with simulator in the stack

### Critical Path

```
Build gpusim Python bindings  (maturin develop --features python)
          ↓
Write simulator.py            (KernelProposal → InstrMix → run → η_hw)
          ↓
Wire into compiler.py         (thermo_score = simulated η_hw, fallback to condition count)
          ↓
Loop runs end-to-end with real scores  ✓
          ↓
β back-solver + Carnot validation experiment  (paper story)
```

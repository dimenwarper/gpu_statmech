# gpu_statmech

**Deriving optimal neural network architectures from the thermodynamics of GPU hardware.**

Full project brief: [`docs/project_brief.md`](docs/project_brief.md)

---

## Core Idea

A GPU is a thermodynamic engine. Like Carnot's heat engine, it has a theoretical maximum efficiency derivable from first principles. This project derives that limit and uses it to design neural architectures that operate near it.

Overall efficiency decomposes into two layers:

```
η = η_hw × η_task
```

- **η_hw** — hardware efficiency: fraction of input energy that becomes *any* computation
- **η_task** — task efficiency: fraction of that computation that is *useful* for the downstream task
- **η_max = η_hw,max × η_task,max** — the GPU Carnot limit

The simulator lives in the sister repo [`gpusim`](https://github.com/dimenwarper/gpusim) (Rust + PyO3). This repo is the analysis and search layer on top of it.

---

## Plan of Attack

### Phase 0 — Theory & Core Thermodynamics
*Goal: a computable η_hw,max for the H100, and the Carnot-optimal computation class.*

**Partition function approximation (three-component factorization):**

```
Z ≈ Z_compute × Z_memory × Z_comm
```

| Component | Method | Notes |
|---|---|---|
| Z_compute | Mean-field over SMs, warp-level factorization | Accurate to O(1/N_SM); N=132 makes this tight |
| Z_memory | **Transfer matrix over the 4-level hierarchy** | Exact — 1D chain reg→smem→L2→HBM is analytically solvable |
| Z_comm | Mean-field on cluster graph with topology-dependent J_gh | NVLink J≈0.1, IB J≈5.0 |
| Validation | Empirical density-of-states histogram from simulator traces | Ground truth for calibration of the above |

From Z we derive F, S, ⟨E⟩, C_V and — crucially — η_hw,max as a function of hardware parameters (SM count, memory capacities, bandwidths).

**Carnot-optimal conditions (necessary for η → η_max):**
- Arithmetic intensity ≥ B_HBM / peak_FLOPS (roofline, but *derived* not assumed)
- Working set at each level ≤ capacity at that level
- Data reuse factor at level l ≥ C_{l−1} / B_l × peak_FLOPS
- Warp occupancy sufficient to hide memory latency
- Zero unnecessary data movement

Validation: roofline model must fall out as a special case of the two-level limit.

**Deliverables:** `src/gpu_statmech/partition_function.py`, `src/gpu_statmech/carnot.py`

---

### Phase 1 — Thermodynamic Analysis Module
*Goal: consume gpusim MicrostateSnapshots and output η_hw, waste decomposition, distance from η_max.*

Wraps the gpusim Python API and adds:
- **Energy model:** map instruction mix + memory access pattern → Joules (compute, SRAM, DRAM, leakage)
- **η_hw computation:** W_hw / E_in per kernel and per full protocol
- **Waste decomposition:** idle compute, unnecessary data movement, pipeline stalls, synchronisation — each as a fraction of E_in
- **Bottleneck attribution:** for each % point below η_max, identify the specific constraint (e.g. "12% lost: working set 312KB > SMEM capacity 228KB → L2 spill")
- **Phase identification:** compute-bound / memory-bound / latency-bound regime + boundary locations
- **Entropy estimation:** perturbation analysis over scheduling variants

**Deliverables:** `src/gpu_statmech/thermo.py`, `src/gpu_statmech/energy.py`

---

### Phase 1.5 — Multi-GPU Extension
*Goal: coupled-engine Carnot limit, topology-dependent η_multi,max.*

- Z_comm fully implemented with topology graph (NVLink, NVSwitch, PCIe, IB, RoCE)
- Coupling constant J_gh per link type
- η_multi,max derivation from coupled Hamiltonian H = Σ H_local + Σ J_gh H_comm
- Parallelism strategies as thermodynamic phases:
  - DP → ferromagnetic, TP → antiferromagnetic, PP → domain wall, EP → spin-glass
- Resonance condition: η_overlap = T_overlapped / max(T_compute, T_comm)
- Parallelism optimizer: enumerate (dp, tp, pp, ep, cp) configs, score each by η_multi

**Deliverables:** `src/gpu_statmech/multi_gpu.py`, `src/gpu_statmech/parallelism.py`

---

### Phase 2 — LLM Architecture Oracle
*Goal: generate candidate CUDA kernels constrained to the Carnot-optimal class.*

- Prompt template encodes the derived Carnot-optimal conditions for the target GPU
- Each proposal: forward kernel + backward kernel + communication kernels (multi-GPU)
- Output format: kernel spec with thread block dims, SMEM layout, register budget, memory access pattern, tensor core utilisation
- Compilation pipeline: CUDA → PTX → gpusim KernelSpec → thermodynamic score
- Expressiveness scoring: proxy task performance (small synthetic benchmarks) before full training

**Deliverables:** `src/gpu_statmech/oracle.py`, `src/gpu_statmech/compiler.py`

---

### Phase 3 — Optimisation Loop
*Goal: Pareto frontier over (η_hw / η_hw,max, expressiveness).*

Closed loop:
1. Oracle proposes batch of N ≈ 20–50 kernel architectures
2. Simulator scores each: η_hw, waste decomposition, distance from η_max
3. Candidates above η threshold evaluated on proxy expressiveness tasks
4. Pareto frontier computed over (η_hw, expressiveness)
5. Physics-grounded feedback to oracle: waste attribution, bottleneck location, restructuring suggestions
6. Repeat until frontier stabilises

**Deliverables:** `src/gpu_statmech/loop.py`, `src/gpu_statmech/pareto.py`

---

### Phase 4 — Training & Validation
*Goal: demonstrate on CIFAR-10 and TinyStories.*

| Task | Scale | Baseline | Target |
|---|---|---|---|
| CIFAR-10 | 0.3–10M params, single H100 | ResNet-18 | ≥93% acc, η_hw/η_hw,max ≥ 0.3 |
| TinyStories | 10M params, single H100 | GPT-2 small | Match perplexity, η_hw/η_hw,max ≥ 0.35 |
| TinyStories | 85–125M params, 8×H100 | GPT-1 | Match perplexity, validate parallelism optimizer |

Key metric beyond accuracy/perplexity: **≥15% closer to η_hw,max than baselines at equivalent task performance.**

---

### Phase 5 — Paper
*Target: ICML / NeurIPS / ICLR*

Narrative: derive the GPU Carnot limit → characterise Carnot-optimal computations → show neural architectures can live in this class → demonstrate on real tasks.

---

## Repository Layout

```
gpu_statmech/
├── docs/
│   └── project_brief.md       # Full project brief
├── src/gpu_statmech/
│   ├── partition_function.py  # Z_compute, Z_memory (transfer matrix), Z_comm
│   ├── carnot.py              # η_hw,max derivation, Carnot-optimal conditions
│   ├── thermo.py              # Thermodynamic analysis from simulator traces
│   ├── energy.py              # Energy/power model (compute, SRAM, DRAM, leakage)
│   ├── multi_gpu.py           # Coupled-engine Carnot limit, topology ranking
│   ├── parallelism.py         # Parallelism optimizer (dp/tp/pp/ep/cp)
│   ├── oracle.py              # LLM architecture oracle
│   ├── compiler.py            # CUDA → PTX → KernelSpec pipeline
│   ├── loop.py                # Optimisation loop
│   └── pareto.py              # Pareto frontier utilities
└── tests/
```

---

## Key Design Decisions

**Why the transfer matrix for Z_memory?**
The memory hierarchy (reg → smem → L2 → HBM) is a 1D chain, and the partition function of a 1D chain is exactly computable via transfer matrices — no approximation. This means the memory thermodynamics (where most of the interesting physics lives) has zero approximation error. Z_compute and Z_comm use mean-field, which is accurate to O(1/N) for N=132 SMs and N_GPU GPUs respectively.

**Why start from the simulator traces for validation?**
The simulator already produces MicrostateSnapshots with SM occupancy, instruction mix, stall fractions, L2 hit rate, and HBM bandwidth utilisation. We build an empirical density-of-states histogram g(E) from varied kernel runs, compute Z_empirical(β) = Σ g(E) exp(−βE), and compare against the analytical approximation. Agreement within ~10% validates the factorisation; discrepancy tells us which subsystem has stronger-than-expected correlations.

**Why kernel-native architectures?**
The Carnot conditions are constraints on execution — tiling, reuse, occupancy — that cannot be expressed purely at the mathematical (PyTorch) level. An architecture described as a CUDA kernel *is* the execution protocol; there is no compilation gap where efficiency is lost.

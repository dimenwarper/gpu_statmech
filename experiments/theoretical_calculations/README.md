# Theoretical Calculations

First-principles thermodynamic predictions derived entirely from H100
hardware spec numbers. No simulator, no GPU, no kernel execution required —
only `numpy` and `matplotlib`.

The full computation chain is:

```
H100 datasheet numbers
    → partition function Z(β) = Z_compute × Z_memory × Z_comm
        → free energy F, entropy S, mean effective energy <E_eff>,
          mean input energy <E_in>, useful work <W_hw>, specific heat Cv
            → η_hw,max  (single-GPU Carnot limit)
                → η_multi,max  (multi-GPU coupled limit)
                    → parallelism Pareto frontier
```

Current caveats:
- Experiment 01 now uses the fixed-load single-GPU model `eta_hw = <W_hw>/<E_in>` with `target_activity = 0.20`.
- Experiment 01 now also includes a first-order memory-feed closure, so the
  reported single-GPU efficiency is lower than the earlier compute-only value.
- Experiment 02 removes the old "min reuse" claim because that formula is not yet calibrated as a literal reuse count.
- Experiment 03 now uses the energy-based multi-GPU limit with a fixed
  communication-demand closure derived from a canonical workload family
  (pure DP, PP, CP, and TP). The current caveat is no longer missing demand,
  but still-coarse topology modeling.

## Experiments

### 01 — H100 Carnot Curve [`01_carnot_curve.py`](01_carnot_curve.py)

Sweeps β (inverse resource-pressure) from hot/loaded to cold/idle at fixed
target activity and plots `eta_hw(beta) = <W_hw>/<E_in>`, `S(beta)`,
`Cv(beta)`, the decomposed `ln Z` components, and the solved work field `h(beta)`.

**Key result:** At `target_activity = 0.20`, the current fixed-load model
with the first-order memory-feed closure reaches **16.43%** at the upper end
of the default sweep (`beta = 10.0`). The load closure is now well-defined,
and the compute-memory coupling is no longer missing from the single-GPU
path, but the present normalized energy tables still do not produce an
interior beta optimum. The plotted entropy and specific-heat diagnostics now
use a higher-resolution memory grid (`n_bins = 256`) and a wider derivative
stencil (`d_beta = 5e-3`) so the curves are numerically smoother. The roofline
ridge point (0.51 FLOP/byte) is still recovered exactly from the Carnot
arithmetic-intensity condition — ratio = 1.0000.

| Quantity | Value |
|---|---|
| η_hw,max | 16.43% |
| β_optimal | 10.0 |
| h*(β_opt) | 3.6909 |
| target activity | 0.20 |
| Roofline ridge | 0.51 FLOP/byte |
| Naive Carnot η (T_reg/T_HBM) | 99.83% |
| T_eff (HBM vs registers) | 600× |

**Figures:** `figures/01_carnot_curve.png`, `figures/01_logz_decomposition.png`

---

### 02 — Memory Hierarchy Thermal Fingerprint [`02_memory_hierarchy.py`](02_memory_hierarchy.py)

Treats the 4-level memory hierarchy (reg → smem → L2 → HBM) as a chain of
thermal reservoirs at increasing effective temperatures and computes the
exact transfer matrix partition function at each level.

**Key result:** The HBM level runs 600× hotter than registers. The experiment
now reports per-level arithmetic-intensity thresholds directly from the
bandwidth model and keeps the roofline plot in bytes/cycle so its units match
the partition-function code.

| Level | T_eff | AI_min (FLOP/byte) |
|---|---|---|
| registers | 1× | 4.000 |
| smem | 23× | 4.000 |
| L2 | 200× | 0.320 |
| HBM | 600× | 0.512 |

The old `min_reuse_factors` printout is intentionally omitted here until the
underlying formula is recalibrated as a true dimensionless reuse count.

**Figure:** `figures/02_memory_hierarchy.png`

---

### 03 — Multi-GPU Scaling Efficiency [`03_scaling_efficiency.py`](03_scaling_efficiency.py)

Runs the energy-based multi-GPU limit for several canonical distributed-training
communication scenarios across four interconnect topologies:

- LLaMA-7B pure data parallelism      `DP-n`
- LLaMA-7B pure pipeline parallelism  `PP-n`
- LLaMA-7B pure context parallelism   `CP-n`
- LLaMA-7B pure tensor parallelism    `TP-n`

The communication-demand closure is

`target_comm_load = total_bytes / (BW_ref × T_compute_ref)`

so every topology is evaluated at the same workload-implied communication
pressure, not just as a bare graph. Blank points in the plots mark infeasible
operating points where the requested communication pressure exceeds the
topology's attainable load under the current closure.

**Key result:** The multi-GPU path now uses the same `<W_hw>/<E_in>` framing
as the single-GPU model, folds link bandwidth/latency into the effective
communication cost, and solves a communication field to match the required
load. Across the scenario family, the results now read the way you would
expect:

- `DP` and `PP` are light-demand workloads, so all fabrics remain close to the
  single-GPU ceiling.
- `CP` is heavier and shows a modest but visible topology split.
- `TP` is the real stress case: NVLink/NVSwitch stay feasible through `TP32`,
  while PCIe Gen5 ring and InfiniBand fat-tree become infeasible by `TP16`
  because the required normalized communication load exceeds their capacity.

| Scenario | Max degree | NVLink/NVSwitch | PCIe Gen5 ring | InfiniBand fat-tree |
|---|---|---|---|---|
| DP | 64 | 1.0000 | 0.9995 | 0.9974 |
| PP | 32 | 1.0000 | 0.9999 | 0.9992 |
| CP | 64 | 1.0000 | 0.9994 | 0.9968 |
| TP | 32 | 1.0000 | infeasible | infeasible |

The next missing physics is not demand closure anymore; it is richer
collective-aware routing and congestion. A heavier TP/EP-style workload should
also separate the topologies much more strongly than this DP example.

**Figures:** `figures/03_scaling_efficiency.png`, `figures/03_comm_energy_share.png`, `figures/03_comm_load_headroom.png`, `figures/03_comm_workload_profiles.png`

---

### 04 — Resonance Condition [`04_resonance.py`](04_resonance.py)

Plots η_overlap = T_overlapped / max(T_compute, T_comm) as a 2D heatmap
over (T_compute, T_comm) space, and locates where real parallelism
configurations land relative to the resonance ridge.

**Key result:** LLaMA-7B on 8 GPUs is severely compute-dominated
(T_compute ≈ 22s vs T_comm ≤ 2.5s). All configs sit far below the resonance
ridge with η_overlap < 0.11 — meaning communication is never the bottleneck
at this scale.

| Config | T_compute | T_comm | η_overlap |
|---|---|---|---|
| DP8 | 22.3s | 0.031s | 0.001 |
| TP8 | 22.3s | 2.44s | 0.110 |
| DP4+TP2 | 22.3s | 0.64s | 0.029 |

**Figure:** `figures/04_resonance.png`

---

### 05 — Parallelism Optimizer [`05_parallelism_optimizer.py`](05_parallelism_optimizer.py)

Runs the full thermodynamic parallelism optimizer — enumerates all valid
(dp, tp, pp, ep, cp) configs, scores each by η_multi = η_multi,max × η_overlap,
computes the Pareto frontier on (η_multi, −comm_overhead), and identifies
the recommended strategy.

**Key result:**

| Model | GPUs | Recommended | η_multi | η_overlap | Phase |
|---|---|---|---|---|---|
| GPT-2 Small | 8 | TP2+PP4 | 0.451 | 0.469 | domain_wall |
| LLaMA-7B | 64 | DP16+TP4 | 0.810 | 0.841 | ferromagnetic |

GPT-2: all 9 valid configs land on the Pareto frontier (no one dominates).
LLaMA-7B: 6 of 16 configs are Pareto-efficient; DP16+TP4 wins on η_multi.

**Figures:** `figures/05_parallelism_gpt_2_small_8gpu.png`, `figures/05_parallelism_llama_7b_64gpu.png`

---

## Running all experiments

```bash
uv run --with matplotlib python experiments/theoretical_calculations/01_carnot_curve.py
uv run --with matplotlib python experiments/theoretical_calculations/02_memory_hierarchy.py
uv run --with matplotlib python experiments/theoretical_calculations/03_scaling_efficiency.py
uv run --with matplotlib python experiments/theoretical_calculations/04_resonance.py
uv run --with matplotlib python experiments/theoretical_calculations/05_parallelism_optimizer.py
```

Figures are saved to [`figures/`](figures/).

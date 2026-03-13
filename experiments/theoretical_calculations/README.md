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
- Experiment 02 removes the old "min reuse" claim because that formula is not yet calibrated as a literal reuse count.
- Experiment 03 is still a legacy J-only topology proxy; it does not directly use link bandwidth or latency in the thermodynamic limit.

## Experiments

### 01 — H100 Carnot Curve [`01_carnot_curve.py`](01_carnot_curve.py)

Sweeps β (inverse resource-pressure) from hot/loaded to cold/idle at fixed
target activity and plots `eta_hw(beta) = <W_hw>/<E_in>`, `S(beta)`,
`Cv(beta)`, the decomposed `ln Z` components, and the solved work field `h(beta)`.

**Key result:** At `target_activity = 0.20`, the current fixed-load model
reaches **30.95%** at the upper end of the default sweep (`beta = 10.0`).
That means the load closure is now well-defined, but the present normalized
energy tables still do not produce an interior beta optimum. The roofline
ridge point (0.51 FLOP/byte) is still recovered exactly from the Carnot
arithmetic-intensity condition — ratio = 1.0000.

| Quantity | Value |
|---|---|
| η_hw,max | 30.95% |
| β_optimal | 10.0 |
| h*(β_opt) | 1.9597 |
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

Runs the current multi-GPU topology proxy for 1→64 GPUs across four
interconnect topologies using the coupled partition function
`ln Z_multi = N × ln Z_single + ln Z_comm_topology`.

**Key result:** After the single-GPU refactor, the current multi-GPU path is
still ceiling-clipped and J-only, so `eta_multi,max` stays flat at the
single-GPU value (**0.3095**) across topologies. The only topology-dependent
signal left in this experiment is the legacy communication log-share proxy,
which is weakest for InfiniBand at 64 GPUs.

| Topology | η_multi,max @ 64 GPU | scaling_eff | legacy proxy |
|---|---|---|---|
| NVLink-4 clique | 0.3095 | 1.0000 | 0.9976 |
| NVSwitch fabric | 0.3095 | 1.0000 | 0.9976 |
| PCIe Gen5 ring | 0.3095 | 1.0000 | 0.9996 |
| InfiniBand fat-tree | 0.3095 | 1.0000 | 0.9788 |

Treat this as a topology-coupling proxy only. It is not yet a physically
grounded overlap or scaling-efficiency prediction.

**Figure:** `figures/03_scaling_efficiency.png`

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
uv run python experiments/theoretical_calculations/01_carnot_curve.py
uv run python experiments/theoretical_calculations/02_memory_hierarchy.py
uv run python experiments/theoretical_calculations/03_scaling_efficiency.py
uv run python experiments/theoretical_calculations/04_resonance.py
uv run python experiments/theoretical_calculations/05_parallelism_optimizer.py
```

Figures are saved to [`figures/`](figures/).

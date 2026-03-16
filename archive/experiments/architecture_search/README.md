# Architecture Search

Physics-guided kernel architecture search (Phases 2–3).

The oracle proposes kernel configurations sampled from a prior over the
Carnot-optimal design space.  The compiler scores each proposal on two
axes — thermodynamic quality and expressiveness — and the optimisation
loop evolves the prior toward the Pareto frontier over both objectives.

## Computation chain

```
CarnotLimit (from partition_function + carnot)
    → PhysicsOracle.propose()          sample kernel configurations
        → KernelCompiler.compile()     score (thermo, expressiveness)
            → Pareto frontier          non-dominated set
                → Oracle.feedback()   update prior toward top-k
                    → repeat
```

## Thermodynamic quality metric

`thermo_score` = fraction of Carnot conditions satisfied ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
(five conditions: AI ≥ ridge, working sets ≤ capacity, reuse ≥ minimum,
occupancy ≥ minimum, no unnecessary movement).

The full multiplicative η_hw / η_hw,max from `check_carnot_optimality` collapses
to ~0 for any practical kernel because the theoretical Carnot reuse minimums
are in the millions-fold regime.  The condition-satisfaction fraction is the
right signal for guiding search.

## Experiments

### 06 — Oracle Proposal Distribution [`06_oracle_proposals.py`](06_oracle_proposals.py)

Draws 500 proposals from the initial (uninformative) prior and plots
the joint distribution of arithmetic intensity vs tensor-core utilisation,
block-size and access-pattern frequencies, warp-occupancy distribution,
and expressiveness score distribution.

**Key results:**

| Quantity | Value |
|---|---|
| Proposals generated | 500 |
| Fraction above roofline ridge | 48.4% |
| Expressiveness mean ± std | 0.573 ± 0.160 |
| Block-size distribution | ~20% each (uniform prior) |
| Access-pattern distribution | ~33% each (uniform prior) |

**Figure:** `figures/06_oracle_proposals.png`

---

### 07 — Optimisation Loop [`07_optimisation_loop.py`](07_optimisation_loop.py)

Runs 15 iterations × 30 proposals = 450 total kernel evaluations.
Tracks Pareto hypervolume convergence, best per-iteration scores,
oracle AI prior drift toward the roofline ridge, and frontier evolution.

**Key results:**

| Quantity | Value |
|---|---|
| Final hypervolume | 0.875000 |
| Best Carnot-cond. fraction | 0.875 (7/8 conditions) |
| Best expressiveness | 1.000 |
| Best combined score | 1.875 |
| HV jump at iteration 6 | +0.222 (oracle learns high-AI region) |

The hypervolume grows from 0.61 (iteration 0) to 0.875 (iteration 13–14),
driven by the oracle's AI prior converging toward and above the roofline
ridge as feedback identifies high-AI proposals as thermodynamically
superior.

**Figures:** `figures/07_loop_convergence.png`, `figures/07_pareto_evolution.png`

---

### 08 — Carnot-Optimal Kernel Analysis [`08_carnot_optimal_kernels.py`](08_carnot_optimal_kernels.py)

Analyses the full 450-kernel pool for condition satisfaction rates,
dominant bottlenecks, resource profiles of the top-20 kernels, and
the expressiveness split between Carnot-optimal and sub-optimal kernels.

**Key results:**

| Condition | Satisfaction Rate |
|---|---|
| arithmetic_intensity | 88.4% |
| working_set_smem | 100.0% |
| working_set_registers | 33.6% |
| reuse_smem | 78.7% |
| reuse_L2 | 51.6% |
| reuse_HBM | 0.0% |
| warp_occupancy | 23.8% |
| unnecessary_data_movement | 8.4% |

The dominant bottleneck is `unnecessary_data_movement` (91.6% of proposals),
followed by `reuse_HBM` (5.6%).  The HBM reuse condition is never satisfied
because the Carnot minimum (≈26M ops/byte) is far above any practical
arithmetic intensity.  This identifies the key gap between theory and
practice: the Carnot-optimal HBM reuse requires a working set that fits
entirely in L2 cache, which demands a model size ≤ 50 MB.

Top-20 kernels by combined score show: mean AI = 1.20 FLOP/byte (2.3× ridge),
near-full tensor-core utilisation (TC = 0.955), full warp occupancy (1.000),
and 32 registers/thread (minimum register pressure).

**Figures:** `figures/08_condition_satisfaction.png`,
`figures/08_waste_breakdown.png`,
`figures/08_optimal_vs_suboptimal.png`

---

## Running all experiments

```bash
uv run python experiments/architecture_search/06_oracle_proposals.py
uv run python experiments/architecture_search/07_optimisation_loop.py
uv run python experiments/architecture_search/08_carnot_optimal_kernels.py
```

Figures are saved to [`figures/`](figures/).

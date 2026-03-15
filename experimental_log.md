# Experimental Log

## 2026-03-13

### Theory audit

- `experiments/theoretical_calculations/01_carnot_curve.py` used `eta = 1 - mean_waste`, so the reported optimum was a sweep-boundary artifact rather than a true interior maximum.
- `experiments/theoretical_calculations/02_memory_hierarchy.py` mixed bandwidth units in the roofline plot and printed a dimensionful reuse threshold as if it were a reuse count.
- `experiments/theoretical_calculations/03_scaling_efficiency.py` ranked topologies mostly through the communication coupling `J`; its reported overlap metric did not use link bandwidth or latency.

### Model changes

- Replaced the waste-only compute ensemble with a first-principles form:

  `p(sigma) proportional to exp[-beta (E_in(sigma) - h W_hw(sigma))]`

- Added explicit compute-side observables for:
  - mean input energy `<E_in>`
  - mean useful hardware work `<W_hw>`
  - mean activity `<A>`
  - hardware efficiency `eta_hw = <W_hw> / <E_in>`
- Added `solve_work_field(beta, target_activity, ...)` so the useful-work field `h` is solved from the load closure `<A> = target_activity`.
- Updated `derive_carnot_limit()` to default to a fixed-load sweep with `target_activity = 0.20`.

### Current observations

- The fixed-load closure removes the arbitrary free-field sweep and makes `h` an interpretable operating-point variable.
- With the current normalized energy tables, the fixed-load Carnot sweep still peaks at the upper beta boundary (`beta = 10.0` for the default sweep). The closure fixes the interpretation of `h`, but it does not by itself create an interior beta optimum.
- The single-GPU and multi-GPU paths are now both expressed in `<W_hw> / <E_in>` terms.
- The multi-GPU path now also has a fixed communication-demand closure.
- The remaining weak spot in the multi-GPU theory is now the coarseness of the communication model: routing, congestion, and collective schedule details are still collapsed into a simple topology-level field.

### Validation

- `uv run pytest -q` -> `366 passed`

### Experiment clean-up

- Updated `experiments/theoretical_calculations/01_carnot_curve.py` to use the fixed-load single-GPU model `eta_hw = <W_hw>/<E_in>` with `target_activity = 0.20`, and to report the current boundary-optimum caveat explicitly.
- Updated `experiments/theoretical_calculations/02_memory_hierarchy.py` to remove the misleading literal reuse-count claim and to keep the roofline plot in bytes/cycle, matching the partition-function units.
- Updated `experiments/theoretical_calculations/03_scaling_efficiency.py` to use the energy-based multi-GPU ceiling and to label the remaining balance metric as a non-timing proxy.
- Re-ran experiments 01-03 with `uv run --with matplotlib python ...` to refresh their figures and outputs.

### Compute-memory closure and search objective

- Added a first-order compute-memory closure in `partition_function.py`:
  - `memory_level_occupancies(beta, ...)` computes exact occupancy marginals from the transfer-matrix chain.
  - `memory_feed_efficiency(beta, ...)` converts warm-level occupancy into a feedability factor in `[0, 1]`.
  - the single-GPU path now scales delivered useful work by this feedability factor and stores it on `ThermodynamicState`.
- Added an architecture-facing search objective in `compiler.py` by upgrading the old expressiveness heuristic to reward:
  - tensor-core use
  - roofline-saturating arithmetic intensity
  - locality/feedability (coalescing, reuse, low redundant movement)
  - occupancy
- Re-ran experiment 01 after the closure; at `target_activity = 0.20` the reported single-GPU limit dropped from `30.95%` to `16.46%`, which is directionally consistent with the missing memory coupling being restored.

### Multi-GPU energy refactor

- Refactored `multi_gpu.py` onto the same `<W_hw> / <E_in>` formulation as the single-GPU path:
  - multi-GPU states now carry `mean_input_energy`, `mean_useful_work`, `mean_comm_input_energy`, and `eta_multi`
  - the communication subsystem now folds link bandwidth and latency into the effective link cost instead of using only `J`
  - `derive_multi_gpu_carnot_limit()` now maximizes `eta_multi = <W_hw>/<E_in>` instead of `1 - mean_waste`
- Updated `experiments/theoretical_calculations/03_scaling_efficiency.py` to use the energy-based multi-GPU ceiling and to report communication energy share instead of the old log-share language.
- Re-ran experiment 03. The result is now theoretically consistent with the refactor, but still nearly flat across topologies because the model did not yet enforce a communication-demand closure.

### Communication-demand closure

- Added a first communication-demand closure to `multi_gpu.py`:
  - `normalise_comm_demand(total_bytes, reference_window_s)` converts workload bytes into a dimensionless communication pressure.
  - `solve_comm_field(beta, target_comm_load, topology)` solves the communication field `g` so the topology actually carries the required load.
  - the communication partition now uses `exp[-beta (J_eff - g alpha) u]` per link, where `alpha` is the normalized delivered service of the link.
- Updated `parallelism.py` so topology Carnot limits are now evaluated at workload-derived communication pressure rather than zero-demand topology-only sweeps.
- Updated experiment 03 to use a canonical LLaMA-7B pure-DP communication load. Topology differences now move in the right direction (NVLink/NVSwitch best, InfiniBand worst), but the gap is still modest because pure DP is a relatively light communication workload and the model still lacks collective-aware routing/congestion.

### Experiment 03 scenario sweep

- Expanded `experiments/theoretical_calculations/03_scaling_efficiency.py` from a single pure-DP sweep into a four-scenario family:
  - `DP-n`
  - `PP-n`
  - `CP-n`
  - `TP-n`
- The experiment now emits four figures instead of one:
  - `03_scaling_efficiency.png`
  - `03_comm_energy_share.png`
  - `03_comm_load_headroom.png`
  - `03_comm_workload_profiles.png`
- The new outputs are materially more informative:
  - `DP` and `PP` stay near the single-GPU ceiling on every topology.
  - `CP` shows a modest but visible split between NVLink/NVSwitch and slower fabrics.
  - `TP` finally behaves like a real stress case: PCIe Gen5 ring and InfiniBand fat-tree become infeasible by `TP16`, while NVLink/NVSwitch stay feasible through `TP32`.
- This is the first experiment-03 version that makes the communication-demand closure visibly useful on the plots instead of only in the equations.

### Experiment 01 numerical smoothing

- Increased the numerical resolution used by `experiments/theoretical_calculations/01_carnot_curve.py` for the plotted diagnostics:
  - `n_bins = 256`
  - `d_beta = 5e-3`
- Also stopped clipping the plotted `Cv(β)` to non-negative values; the panel now shows the signed quantity around a zero reference line.
- The `Cv(β)` plot now uses a local quartic fit to `ln Z(β)` over a 21-point window before taking curvature, which suppresses the cold-end second-derivative noise without changing the underlying efficiency sweep.
- This does not change the qualitative theory. It only reduces derivative noise in the displayed `S(β)` and `Cv(β)` curves.
- With the smoother settings, the reported fixed-load single-GPU ceiling is `16.43%` at `beta = 10.0`.

## 2026-03-14

### Simulator-observable inference

- Inspected the sibling `gpusim` repo and verified that the Python boundary already exports usable per-block microstate snapshots:
  - per-SM active warps
  - per-SM stall fractions
  - per-SM instruction mix
  - register / shared-memory utilization
  - L2 hit rate
  - HBM bandwidth utilization
- Added `src/gpu_statmech/observables.py` so `gpu_statmech` can ingest raw `gpusim` snapshots directly instead of assuming the older flat mock-snapshot schema.
- `energy.py` now normalizes raw simulator snapshots before computing the existing energy decomposition, which keeps the old interface working while making the simulator path usable.
- `thermo.py` now has two beta-inference modes:
  - `observable_match` (new default): infer the operating point by matching simulator-observed issue activity, stall fraction, and a memory-feed proxy against the partition-function model
  - `crude_waste_logit` (retained): the older waste-fraction logit estimate
- Added `mean_compute_mem_stall_fraction(...)` to `partition_function.py` so the observable-matching path can compare simulator stall observables to an internally consistent model-side stall prediction.
- Added tests for:
  - raw `gpusim` snapshot normalization
  - observable aggregation
  - observable-matching thermo inference
  - backwards-compatible crude inference

### Validation

- `uv run pytest -q` -> `381 passed`

### Canonical gpusim driver

- Added `src/gpu_statmech/gpusim_driver.py` plus `scripts/run_gpusim_analysis.py`.
- The new driver runs a canonical suite of simulator workloads:
  - `gemm_tc`
  - `flash_attention`
  - `layernorm`
  - `softmax_reduce`
  - `transpose_bw`
- The report path is intentionally simple:
  - run the kernel suite in `gpusim`
  - ingest the raw block traces here
  - analyse the resulting protocol with the existing thermodynamic stack
  - print a compact operating-point summary per kernel and for the full protocol
- This is the first reusable end-to-end demo path for simulator-driven inference in the repo.

### Validation

- `uv run pytest -q` -> `382 passed`

## 2026-03-15

### Simulator-backed plotting experiment

- Added `experiments/simulator_validation/01_canonical_kernel_profiles.py`.
- The new experiment reuses the same canonical kernel profiles and `gpusim` driver path as `scripts/run_gpusim_analysis.py`, but saves figures instead of only printing a text summary.
- Current outputs:
  - `experiments/simulator_validation/figures/01_canonical_overview.png`
  - `experiments/simulator_validation/figures/01_warp_state_match.png`
- The overview figure shows:
  - measured `eta_hw` vs inferred `eta_hw,max`
  - observed issue / stall / memory-stall fractions
  - issue-vs-memory-stall operating regimes
  - inferred `beta` and memory-feed efficiency
- The second figure now treats exact warp-state occupancy as a secondary diagnostic:
  - the primary comparison is observed vs model-predicted coarse state families
    (`productive`, `dependency`, `memory`, `sync/fetch`, `idle`)
  - the exact 8-state occupancy mismatch is shown only as a residual heatmap
    so it is clearly a calibration diagnostic rather than the main fit target

### Simulator intervention recommendation

- Added `src/gpu_statmech/gpusim_recommendation.py` plus
  `experiments/simulator_validation/02_intervention_recommendation.py`.
- The new experiment turns the simulator path into a recommendation study:
  - generate controlled stressed baselines from the canonical kernel families
  - generate one counterfactual variant per intervention lever
  - recommend a lever from the baseline trace alone
  - compare against the oracle-best lever found by actually running all
    counterfactuals in `gpusim`
- Current intervention levers are:
  - `locality`
  - `occupancy`
  - `tensorize`
- Current stressed baseline modes are:
  - `base`
  - `memory_stressed`
  - `footprint_stressed`
  - `compute_unoptimized`
- Current outputs:
  - `experiments/simulator_validation/figures/02_oracle_attainment.png`
  - `experiments/simulator_validation/figures/02_statmech_confusion.png`
- On the current H100 simulator sweep:
  - actionable baselines: `18 / 20`
  - mean oracle attainment:
    - `stat-mech`: `0.865`
    - `raw counters`: `0.870`
    - `roofline`: `0.759`
    - `occupancy only`: `0.001`
    - `random`: `0.728`
- Interpretation:
  - the thermodynamic recommender is already substantially better than the
    weak baselines (`roofline`, `occupancy only`, `random`)
  - the strongest simple baseline is still the raw counter-family heuristic,
    which slightly outperforms the current stat-mech mapping on this synthetic
    suite
  - that is a useful gap, not a contradiction: the simulator interventions are
    still tightly aligned to the same coarse families used in the raw-counter
    policy, so the next improvement is to make the thermodynamic mapping more
    discriminative than simple family-majority rules

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

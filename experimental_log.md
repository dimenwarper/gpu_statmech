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
- The single-GPU path is now expressed in `<W_hw> / <E_in>` terms. The multi-GPU path still uses a waste-based proxy and a temporary ceiling clamp to stay consistent with the single-GPU limit.

### Validation

- `uv run pytest -q` -> `358 passed`

### Experiment clean-up

- Updated `experiments/theoretical_calculations/01_carnot_curve.py` to use the fixed-load single-GPU model `eta_hw = <W_hw>/<E_in>` with `target_activity = 0.20`, and to report the current boundary-optimum caveat explicitly.
- Updated `experiments/theoretical_calculations/02_memory_hierarchy.py` to remove the misleading literal reuse-count claim and to keep the roofline plot in bytes/cycle, matching the partition-function units.
- Updated `experiments/theoretical_calculations/03_scaling_efficiency.py` to label the current multi-GPU path honestly as a legacy topology proxy rather than a timing-accurate overlap model.
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

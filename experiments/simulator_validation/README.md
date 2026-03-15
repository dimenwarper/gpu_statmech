# Simulator Validation

Simulator-backed experiments built from `gpusim` traces rather than hardware
spec numbers alone.

These scripts use the same canonical kernel profiles and ingestion path as
`scripts/run_gpusim_analysis.py`, then save figures to `figures/`.

## Experiments

### 01 — Canonical gpusim Kernel Profiles [`01_canonical_kernel_profiles.py`](01_canonical_kernel_profiles.py)

Runs the canonical `gpusim` kernel suite:

- `gemm_tc`
- `flash_attention`
- `layernorm`
- `softmax_reduce`
- `transpose_bw`

and visualizes:

- measured `eta_hw` vs inferred `eta_hw,max`
- observed issue / stall / memory-stall fractions
- inferred operating point (`beta`, memory-feed efficiency)
- observed vs model-predicted thermodynamic state families
- exact warp-state residuals as a secondary diagnostic

**Figures:** `figures/01_canonical_overview.png`, `figures/01_warp_state_match.png`

### 02 — Simulator Intervention Recommendation [`02_intervention_recommendation.py`](02_intervention_recommendation.py)

Builds a counterfactual neighborhood around the canonical `gpusim` kernel
families, then asks a decision question:

- from the baseline trace alone, which intervention class should we try next?

The script compares:

- stat-mech recommendation
- raw counter-family heuristic
- roofline-style heuristic
- occupancy-only heuristic
- random choice

against an oracle that evaluates every candidate intervention on the simulator.

Primary metric:

- oracle attainment ratio = realized gain from the chosen lever / realized gain
  from the oracle-best lever

**Figures:** `figures/02_oracle_attainment.png`, `figures/02_statmech_confusion.png`

## Running

The `gpusim` Python extension must be installed into the same environment used
to run the script. One working path is:

```bash
uv run --with matplotlib --with maturin bash -lc '
  cd ../gpusim
  maturin develop --features python
  cd ../gpu_statmech
  python experiments/simulator_validation/01_canonical_kernel_profiles.py --gpu h100
  python experiments/simulator_validation/02_intervention_recommendation.py --gpu h100
'
```

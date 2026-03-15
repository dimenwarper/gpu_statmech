# Experiments

Each subdirectory is a self-contained experiment category. Scripts print
results to stdout and save figures to a `figures/` subfolder.

## Subdirectories

| Directory | What it covers | Dependencies |
|---|---|---|
| [`theoretical_calculations/`](theoretical_calculations/) | First-principles thermodynamic predictions from hardware spec numbers alone. No simulator, no GPU. | `numpy`, `matplotlib` |
| [`simulator_validation/`](simulator_validation/) | Simulator-backed validation and visualization using `gpusim` traces and the canonical kernel profiles. | `numpy`, `matplotlib`, `gpusim` |

## Running

```bash
# install experiment deps once
uv pip install -e ".[experiments]"

# run any script directly
uv run python experiments/theoretical_calculations/01_carnot_curve.py
uv run python experiments/simulator_validation/01_canonical_kernel_profiles.py
```

## Planned

| Directory | Phase | What it will cover |
|---|---|---|
| `kernel_analysis/` | Phase 1 | Waste decomposition and bottleneck attribution for real CUDA kernels |
| `architecture_search/` | Phase 2–3 | LLM oracle proposals scored by η_hw; Pareto frontier over (η, expressiveness) |
| `training_runs/` | Phase 4 | CIFAR-10 and TinyStories results vs ResNet-18 / GPT-2 baselines |

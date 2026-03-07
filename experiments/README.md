# Experiments

Each subdirectory is a self-contained experiment category. Scripts print
results to stdout and save figures to a `figures/` subfolder.

## Subdirectories

| Directory | What it covers | Dependencies |
|---|---|---|
| [`theoretical_calculations/`](theoretical_calculations/) | First-principles thermodynamic predictions from hardware spec numbers alone. No simulator, no GPU. | `numpy`, `matplotlib` |

## Running

```bash
# install experiment deps once
uv pip install -e ".[experiments]"

# run any script directly
uv run python experiments/theoretical_calculations/01_carnot_curve.py
```

## Planned

| Directory | Phase | What it will cover |
|---|---|---|
| `simulator_validation/` | Phase 0 validation | Compare analytical Z(β) against empirical density-of-states from gpusim traces |
| `kernel_analysis/` | Phase 1 | Waste decomposition and bottleneck attribution for real CUDA kernels |
| `architecture_search/` | Phase 2–3 | LLM oracle proposals scored by η_hw; Pareto frontier over (η, expressiveness) |
| `training_runs/` | Phase 4 | CIFAR-10 and TinyStories results vs ResNet-18 / GPT-2 baselines |

"""
gpu_statmech — thermodynamic analysis of GPU computation.

Layer structure:
  partition_function.py  — Z_compute (mean-field), Z_memory (transfer matrix), Z_comm
  carnot.py              — η_hw,max derivation and Carnot-optimal conditions
  energy.py              — power/energy model from simulator traces
  thermo.py              — thermodynamic analysis module (Phase 1)
  observables.py         — simulator snapshot normalisation + observable extraction
  multi_gpu.py           — coupled-engine Carnot limit (Phase 1.5)
  parallelism.py         — parallelism optimizer (Phase 1.5)
  gpusim_driver.py       — canonical gpusim workload driver + reporting
  gpusim_recommendation.py — simulator intervention study helpers
  oracle.py              — physics-based kernel proposal oracle (Phase 2)
  compiler.py            — KernelProposal → KernelSpec compiler + scorer (Phase 2)
  pareto.py              — Pareto frontier utilities (Phase 3)
  loop.py                — thermodynamic architecture search loop (Phase 3)
  utils.py               — shared analysis helpers for diagnostic curves
"""

"""
gpu_statmech — thermodynamic analysis of GPU computation.

Layer structure:
  partition_function.py  — Z_compute (mean-field), Z_memory (transfer matrix), Z_comm
  carnot.py              — η_hw,max derivation and Carnot-optimal conditions
  energy.py              — power/energy model from simulator traces
  thermo.py              — thermodynamic analysis module (Phase 1)
  multi_gpu.py           — coupled-engine Carnot limit (Phase 1.5)
  parallelism.py         — parallelism optimizer (Phase 1.5)
"""

# gpusim Integration Gap Analysis

What needs to be added to `/root/projects/gpusim` so that `gpu_statmech` can
consume it as a simulation backend.

The target API from the Python side is roughly:

```python
sim   = GpuSim.h100()
trace = sim.run(kernel_spec, launch_config)   # → list[Microstate]
Ē     = EnergyFunctional().time_average(trace)
```

Gaps are ordered by priority.

---

## Gap 1 — Python Bindings (PyO3)  `[CRITICAL]`

**Current state:** None. gpusim is Rust-only; there is no way to call it from
Python.

**What's needed:** A `pyo3`-based `gpusim` Python extension module that exposes:

| Python symbol | Wraps |
|---|---|
| `GpuSim(sm_config)` | `GPU` + `KernelExecutor` |
| `GpuSim.h100()` / `GpuSim.a100()` | `GPU::h100()` / `GPU::a100()` |
| `GpuSim.run(spec, config) → list[dict]` | `KernelExecutor::launch()` + trace |
| `Cluster(nodes, gpus_per_node, ...)` | `Cluster` |
| `Cluster.h100_dgx(n)` | `Cluster::h100_dgx()` |
| `Cluster.run_kernel_on(device, spec, config)` | `Cluster::launch_kernel_on()` |
| `Cluster.all_reduce(bytes, algo)` | `Cluster::all_reduce()` |
| `Cluster.all_gather(bytes)` | `Cluster::all_gather()` |
| `Cluster.transfer(src, dst, bytes)` | `Cluster::transfer()` |

The trace returned by `run()` should be a list of per-step snapshot dicts that
`gpu_statmech` can deserialize into `Microstate` objects (see Gap 2).

**Files to change:** `Cargo.toml` (add `pyo3` dep + `[lib]` crate-type),
new `src/python.rs` module.

---

## Gap 2 — Microstate Trace Emission  `[CRITICAL]`

**Current state:** `executor.rs` writes one `LiveMetrics` JSON snapshot
**per block** containing only `sm_active_blocks: Vec<u32>`. There is no
per-cycle output and no warp-level or memory-level detail.

**What's needed:** After each block (or at configurable granularity), emit a
snapshot that maps directly onto `gpu_statmech.Microstate`:

```
MicrostateSnapshot {
    cycle:           u64,
    gpu_id:          usize,

    // per SM
    sm_active_warps: Vec<u32>,       // active_warps per SM
    sm_max_warps:    u32,            // from SmConfig
    sm_instr_mix:    Vec<InstrMix>,  // FP16/FP32/INT/SFU/MEM/TC fractions per SM
    sm_stall_frac:   Vec<f32>,       // fraction of warps stalled per SM

    // memory hierarchy
    reg_utilization:    f32,
    smem_utilization:   f32,
    l1_hit_rate:        f32,
    l2_hit_rate:        f32,
    l2_utilization:     f32,
    hbm_bw_utilization: f32,

    // bandwidth channels
    bw_sm_to_smem: f32,   // fraction of theoretical peak
    bw_sm_to_l2:   f32,
    bw_l2_to_hbm:  f32,
    bw_nvlink:     f32,   // only non-zero in multi-GPU mode
}
```

`gpu_statmech` will deserialize this into a `Microstate` via a thin adapter
(no format changes needed on the Python side as long as the field names match).

**Files to change:** `src/metrics.rs` (add `MicrostateSnapshot`),
`src/executor.rs` (emit snapshot at configurable granularity).

---

## Gap 3 — Warp State Transitions  `[HIGH]`

**Current state:** `scheduler.rs` defines `WarpState` with 8 variants
(`Eligible`, `LongScoreboard`, `ShortScoreboard`, `Barrier`, `ExecDep`,
`MemThrottle`, `Fetch`, `Idle`). Every `WarpSlot` is created with
`WarpState::Eligible` and **never transitions**. The schedulers (LRR, GTO,
TwoLevel) treat all warps as unconditionally eligible.

**What's needed:** Wire state transitions driven by the memory latency model
(Gap 4) and instruction dependencies:

| Transition | Trigger |
|---|---|
| `Eligible → LongScoreboard` | Warp issues a global memory (HBM) load |
| `LongScoreboard → Eligible` | Simulated HBM latency (~600 cycles) elapsed |
| `Eligible → ShortScoreboard` | Warp issues SMEM load or short-latency arith |
| `ShortScoreboard → Eligible` | ~25 cycles elapsed |
| `Eligible → Barrier` | Warp reaches `__syncthreads()` |
| `Barrier → Eligible` | All other warps in block reach barrier |
| `Eligible → ExecDep` | RAW hazard: result of prior instruction not ready |
| `ExecDep → Eligible` | Producing instruction writes-back |
| `Eligible → MemThrottle` | Memory request queue saturated |

The schedulers should then filter to only issue `Eligible` warps and use stall
state to compute `stall_frac` for the microstate snapshot.

**Files to change:** `src/scheduler.rs` (state machine transitions),
`src/executor.rs` (call transition logic per warp-issue cycle).

---

## Gap 4 — Memory Latency and Bandwidth Counters  `[HIGH]`

**Current state:** `memory.rs` implements `L2Cache` and `HBM` as sparse
`HashMap<usize, u8>` with instant reads/writes and no latency, no bandwidth
tracking, and no cache eviction.

**What's needed:**

### 4a. Per-access latency simulation
Attach a simulated latency (in cycles) to each memory operation so the
executor can model scoreboard stalls:

```rust
impl HBM {
    pub fn read_latency(&self) -> u32 { 600 }   // cycles
}
impl L2Cache {
    pub fn access_latency(&self) -> u32 { 200 }
}
// SMEM latency lives in sm.rs: ~25 cycles
```

### 4b. Bandwidth utilization counters
Track bytes transferred per simulated time unit on each channel so
`bw_*_utilization` fields in the microstate snapshot can be populated:

```rust
pub struct HBM {
    ...
    pub bytes_read_this_step:    u64,
    pub bytes_written_this_step: u64,
    pub peak_bandwidth_bps:      u64,   // already exists (3.4 TB/s)
}
```

Reset counters at each snapshot interval; compute utilization as
`bytes_transferred / (peak_bw * snapshot_duration)`.

### 4c. Cache hit/miss tracking
`L2Cache` currently returns zeroes for unwritten addresses and always
"hits" (in the sense that it never models miss penalty). Need:

```rust
pub struct L2Cache {
    ...
    pub hits:   u64,
    pub misses: u64,
}
// hit_rate = hits / (hits + misses)
```

True set-associative eviction is out of scope; a simpler working-set model
(mark address as present/absent, evict LRU when over capacity) is sufficient.

**Files to change:** `src/memory.rs`.

---

## Gap 5 — Instruction Mix Tracking  `[MEDIUM]`

**Current state:** `executor.rs` calls `(kernel.func)(&mut ctx)` for each
thread sequentially. There is no classification of what instruction type each
kernel "issues" — the concept does not exist in the current model.

**What's needed:** Since the kernels are Rust closures (not real machine code),
true instruction counting is not feasible. Instead, add **kernel metadata** that
the caller provides at launch time, specifying the expected instruction mix as
fractions:

```rust
pub struct InstrMix {
    pub fp16:        f32,
    pub fp32:        f32,
    pub int:         f32,
    pub sfu:         f32,
    pub mem:         f32,
    pub tensor_core: f32,
}

// Added to LaunchConfig:
pub instr_mix: Option<InstrMix>,
```

If `instr_mix` is `None`, the executor uses a default (100% FP32). If provided,
the snapshot records the declared mix. This is intentionally approximate —
accurate instruction profiling requires PTX-level execution which is out of
scope for Phase 1.

**Files to change:** `src/kernel.rs` (add `InstrMix`, extend `LaunchConfig`),
`src/executor.rs` (propagate to snapshot).

---

## Gap 6 — Missing Collective Operations  `[MEDIUM]`

**Current state:** `cluster.rs` implements `all_reduce` (Ring/Tree/Direct),
`all_gather` (Ring), and `broadcast` (Tree). Missing:

| Missing operation | Used by parallelism strategy |
|---|---|
| `reduce_scatter` | ZeRO / FSDP |
| `all_to_all` | Expert Parallelism (MoE token dispatch) |

Also, the `bottleneck_link()` helper assumes a two-tier topology (NVLink
intra-node, IB inter-node). For topologies with only PCIe or with mixed link
types, this gives wrong results.

**What's needed:**

```rust
impl Cluster {
    pub fn reduce_scatter(&self, bytes_per_gpu: u64) -> CollectiveStats { ... }
    pub fn all_to_all(&self, bytes_per_gpu: u64) -> CollectiveStats { ... }
}
```

Formulae (ring algorithm):
- **ReduceScatter:** same as AllGather — `(N-1)/N · B / bw + (N-1)·latency`
- **AllToAll:** `(N-1) · B / bw + (N-1)·latency` (each GPU sends to every other)

**Files to change:** `src/cluster.rs`.

---

## Gap 7 — Expanded Interconnect Types  `[MEDIUM]`

**Current state:** `interconnect.rs` models only `NVLink` and `InfiniBand`.
`cluster.rs` hardcodes the two-tier (NVLink + IB) topology.

**What's needed** to match `gpu_statmech`'s `Topology` graph (which supports
NVLink, NVSwitch, PCIe Gen5, IB NDR/HDR, and Ethernet/RoCE):

```rust
pub struct PCIeConfig {
    pub bandwidth_gb_s: f64,  // 64 GB/s for Gen5
    pub latency_us:     f64,  // ~3.5 µs
}

pub struct RoCEConfig {
    pub bandwidth_gb_s: f64,
    pub latency_us:     f64,
}
```

And an enum to unify them:

```rust
pub enum LinkConfig {
    NVLink(NVLinkConfig),
    InfiniBand(InfiniBandConfig),
    PCIe(PCIeConfig),
    RoCE(RoCEConfig),
}
```

A more flexible `Cluster` constructor should accept an adjacency map
`HashMap<(DeviceId, DeviceId), LinkConfig>` rather than fixed
NVLink-within / IB-across topology.

**Files to change:** `src/interconnect.rs`, `src/cluster.rs`.

---

## Gap 8 — Kernel Representation from Python  `[HIGH]`

**Current state:** Kernels are Rust closures (`Box<dyn Fn(&mut ThreadCtx)>`).
There is no way to define or run a kernel from Python.

For Phase 1 (simulator instrumentation), `gpu_statmech` does not need to
execute real CUDA kernels — it needs to **simulate the hardware behaviour** of
a kernel described by its resource profile. The Python side should be able to
pass:

```python
KernelSpec(
    name          = "flash_attn_fwd",
    threads_per_block = 128,
    regs_per_thread   = 64,
    smem_per_block    = 49152,   # bytes
    instr_mix = InstrMix(fp16=0.6, mem=0.3, tensor_core=0.1),
    grid = (1024, 1, 1),
)
```

and get back a `list[Microstate]` trace. No actual kernel code is executed;
the simulator infers utilization from the resource profile and the latency
model (Gap 4).

For Phase 3 (LLM oracle), real CUDA / PTX execution will be needed. That is
out of scope here — but the `KernelSpec` struct should be forward-compatible
by accepting an optional `ptx_source: Option<String>` field for later use.

**Files to change:** `src/kernel.rs` (add `KernelSpec`), `src/python.rs`
(expose it to Python via PyO3).

---

## Summary

| # | Gap | Priority | Files |
|---|---|---|---|
| 1 | Python bindings (PyO3) | CRITICAL | `Cargo.toml`, new `src/python.rs` |
| 2 | Microstate trace emission | CRITICAL | `src/metrics.rs`, `src/executor.rs` |
| 3 | Warp state transitions | HIGH | `src/scheduler.rs`, `src/executor.rs` |
| 4 | Memory latency + bandwidth counters | HIGH | `src/memory.rs` |
| 5 | Instruction mix tracking | MEDIUM | `src/kernel.rs`, `src/executor.rs` |
| 6 | Missing collectives (ReduceScatter, AllToAll) | MEDIUM | `src/cluster.rs` |
| 7 | Expanded interconnect types (PCIe, RoCE) | MEDIUM | `src/interconnect.rs`, `src/cluster.rs` |
| 8 | Kernel representation from Python | HIGH | `src/kernel.rs`, `src/python.rs` |

Gaps 1, 2, and 8 together constitute the **minimum viable integration**:
a Python caller that can describe a kernel by its resource profile, run it
through the simulator, and get back a `list[Microstate]` trace that
`gpu_statmech` can feed into `EnergyFunctional`. Gaps 3 and 4 are needed
to make the stall and bandwidth fields in `Microstate` meaningful rather
than zeroed out. Gaps 5–7 are quality-of-life improvements that improve
fidelity and coverage.

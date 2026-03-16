# Phase 0 Theory: The Statistical Mechanics of GPU Computation

## Starting Point: What Are We Actually Modelling?

A GPU executing a kernel is a physical system with a huge number of possible internal states. At any given clock cycle, you could ask: which warps are active on which SMs? What instruction is each warp executing? What data is sitting in registers vs. shared memory vs. L2 vs. HBM? We call one complete answer to all those questions a **microstate** σ. The set of all possible microstates is Ω.

The key physical insight is this: not all microstates are equally "useful." A microstate where every warp is executing a tensor-core multiply is extremely useful. A microstate where every warp is stalled waiting on HBM is pure waste. The GPU's job — and the job of the programmer — is to steer the system toward useful microstates and away from wasteful ones.

This is precisely the structure that statistical mechanics was invented to analyse.

---

## The Boltzmann Framework

In classical statistical mechanics, a system in thermal equilibrium at temperature T occupies microstate σ with probability:

$$p(\sigma) = \frac{e^{-E(\sigma)/k_B T}}{Z}$$

where E(σ) is the energy of that microstate and Z = Σ_σ exp(−E(σ)/k_BT) is the **partition function** — a normalisation constant that encodes *everything* thermodynamic about the system.

We transplant this framework directly onto the GPU, with one crucial redefinition: **E(σ) is not physical heat-energy, it is waste energy** — the fraction of input power consumed by microstate σ that does not produce useful computation. A warp in the `eligible` state (executing useful work) contributes E = 0. A warp in the `long_scoreboard` state (stalled on HBM latency) contributes E = 1. All other warp states fall between these extremes according to how much of their cycle budget they waste:

| Warp state | Waste E(s) | Interpretation |
|---|---|---|
| `eligible` | 0.0 | Executing useful work |
| `exec_dep` | 0.3 | Short data dependency stall |
| `fetch` | 0.2 | Instruction cache miss |
| `short_scoreboard` | 0.6 | Waiting on L2 latency |
| `mem_throttle` | 0.8 | Too many outstanding memory requests |
| `barrier` | 0.9 | Synchronisation stall |
| `long_scoreboard` | 1.0 | Waiting on HBM latency |
| `idle` | 1.0 | No active warp |

The inverse temperature β is reinterpreted as **resource pressure**: high β (low T) means the system is lightly loaded, low β (high T) means it is heavily loaded and approaching saturation. This is not just an analogy — it is the right mathematical object because the GPU's warp scheduler naturally samples from something close to a Boltzmann distribution over warp states, and β controls how tightly the system is constrained to low-waste states.

---

## Why Z Factorises (and Why That's Crucial)

The full microstate space of an H100 is astronomically large. 132 SMs × 64 warps × 8 warp states × 4 memory levels × continuous bandwidth utilisation — the exact Z is intractable. But the GPU has a beautiful physical structure we can exploit: **SMs are nearly independent**.

Two warps on different SMs don't share registers or shared memory. They interact only through two shared global resources: L2 cache bandwidth and HBM bandwidth. If we ignore those interactions for a moment, the system factorises exactly:

$$Z_{\text{total}} \approx Z_{\text{compute}} \times Z_{\text{memory}} \times Z_{\text{comm}}$$

Each factor captures a different subsystem.

---

## Z_compute: Mean-Field Over SMs

Since warps on different SMs are independent, the SM partition function factorises down to the single-warp level:

$$Z_{\text{warp}}(\beta) = \sum_{s \in \text{warp states}} e^{-\beta \cdot \text{waste}(s)}$$

$$Z_{\text{SM}}(\beta) = Z_{\text{warp}}(\beta)^{\text{warps/SM}}$$

$$Z_{\text{compute}}(\beta) = Z_{\text{SM}}(\beta)^{N_{\text{SM}}} \times \text{correction}(\beta)$$

Z_warp is a sum over just 8 states — trivially computable. However, `Z_SM^132` would overflow float64 (it reaches ~10^7500), so we immediately move to **log space**:

$$\ln Z_{\text{compute}} = N_{\text{SM}} \times \text{warps/SM} \times \ln Z_{\text{warp}} + \ln(\text{correction})$$

We never materialise Z_compute itself — all thermodynamic calculations use ln Z directly.

The bandwidth interaction between SMs is handled as a **mean-field correction**: we solve a self-consistency equation for the Lagrange multiplier λ that enforces the constraint that total expected HBM demand equals HBM bandwidth supply. Concretely, we iterate:

1. Given λ, compute the mean fraction of warps in memory-stalled states under the modified Boltzmann weights `exp(-β(waste + λ × mem_stall_mask))`
2. Check whether `N_SM × warps/SM × mean_stall_frac ≤ BW_HBM`
3. Adjust λ until the constraint is satisfied

This correction is accurate to O(1/N_SM) — excellent for N_SM = 132.

---

## Z_memory: Exact Transfer Matrix Over the 1D Chain

Here we get something genuinely special. The memory hierarchy is a **1D chain**:

```
registers → shared memory → L2 → HBM
  (cold)                          (hot)
```

A 1D chain has an exact solution via **transfer matrices** — a method from condensed matter physics used to solve the 1D Ising model and similar systems. We need no approximation here at all.

Discretise the occupancy fraction at each level into n_bins bins: u ∈ {0, 1/n, 2/n, ..., 1}. The transfer matrix between adjacent levels encodes the Boltzmann weight of transitioning from occupancy u_i at the colder level to u_j at the warmer level:

$$T_l[u_i, u_j] = \exp\!\big(-\beta \cdot \text{cost}(u_i, u_j, l)\big)$$

where the cost combines:
- **Data-movement waste**: energy proportional to |u_j − u_i| × capacity × energy_per_byte at the warmer level, normalised by the maximum possible movement cost
- **Latency waste**: fraction of cycles spent waiting, proportional to u_j × (latency_l / latency_HBM)

The full memory partition function is then:

$$Z_{\text{memory}} = \mathbf{1}^T \cdot T_{\text{reg→smem}} \cdot T_{\text{smem→L2}} \cdot T_{\text{L2→HBM}} \cdot \mathbf{1}$$

This is a sequence of matrix-vector products. For n_bins = 64 it costs O(64³) = trivial. The result is **exact** — the 1D topology eliminates all the combinatorial complexity that makes higher-dimensional systems hard.

This is also where the memory-hierarchy temperature gradient lives. The effective temperature of each level is proportional to its access latency:

| Level | Latency (cycles) | T_eff (relative) |
|---|---|---|
| Registers | ~1 | 1 |
| Shared memory | ~23 | 23 |
| L2 cache | ~200 | 200 |
| HBM | ~600 | 600 |

The thermodynamic force driving useful work is the gradient T_HBM >> T_reg. The GPU "engine" extracts work by moving data downhill through this gradient and computing on it at the coldest accessible level.

---

## Z_comm: Mean-Field on the Cluster Graph

For multi-GPU systems, each inter-GPU link is approximated as an independent harmonic oscillator over link utilisation u ∈ [0, 1]:

$$Z_{\text{link}} = \int_0^1 e^{-\beta J_{gh} u} \, du = \frac{1 - e^{-\beta J_{gh}}}{\beta J_{gh}}$$

The coupling constant J_gh encodes the thermodynamic cost of using that link:

| Link type | Bandwidth | J_gh |
|---|---|---|
| NVLink 4.0 / NVSwitch | 900 GB/s | 0.1 (cheap) |
| PCIe Gen 5 | 64 GB/s | 1.0 |
| InfiniBand NDR | 50 GB/s | 5.0 |
| RoCE | 50 GB/s | 10.0 (expensive) |

The two limiting cases are instructive:
- β J_gh → 0 (free communication): Z_link → 1, no thermodynamic cost
- β J_gh >> 1 (expensive communication): Z_link → 1/(β J_gh), large thermodynamic penalty

The full Z_comm is the product over all edges in the cluster graph.

---

## From Z to η_hw,max

Once we have Z(β) = Z_compute × Z_memory × Z_comm, all thermodynamic quantities follow from derivatives of ln Z. Since ln Z is extensive (proportional to the number of degrees of freedom n_dof = N_SM × warps_per_SM), we normalise to get intensive (per-DOF) quantities:

**Mean waste fraction** (∈ [0, 1]):
$$\langle E \rangle(\beta) = -\frac{1}{n_{\text{dof}}} \frac{\partial \ln Z}{\partial \beta}$$

**Free energy per DOF:**
$$F(\beta) = -\frac{\ln Z}{\beta \cdot n_{\text{dof}}}$$

**Entropy per DOF** (nats per warp):
$$S(\beta) = \beta \big(\langle E \rangle - F\big)$$

**Specific heat per DOF** (≥ 0):
$$C_V(\beta) = \frac{\beta^2}{n_{\text{dof}}} \frac{\partial^2 \ln Z}{\partial \beta^2} = \beta^2 \cdot \text{var}(E_{\text{per-DOF}}) \geq 0$$

Note the sign: $\partial^2 \ln Z / \partial \beta^2 = \text{var}(E) \geq 0$ (variance is non-negative), so C_V ≥ 0 always. A negative specific heat would indicate a thermodynamically unstable system.

**Hardware efficiency:**
$$\eta_{\text{hw}}(\beta) = 1 - \langle E \rangle(\beta)$$

**The Carnot limit:**
$$\eta_{\text{hw,max}} = \max_\beta \, \eta_{\text{hw}}(\beta)$$

### Why Is There an Interior Maximum?

The efficiency η_hw(β) peaks at an interior β because the two limiting regimes are both inefficient:

- **β → ∞ (cold, lightly loaded):** Each warp that runs does so efficiently, but most warps are idle — no work to give them. η_hw is limited by idle capacity.
- **β → 0 (hot, heavily loaded):** Every warp is active, but they're fighting for HBM bandwidth and stalling on each other. Waste from stalls, bank conflicts, and memory pressure dominates. η_hw falls.

The Carnot limit η_hw,max is where these two waste mechanisms are exactly balanced. It is determined entirely by the hardware parameters (SM count, memory capacities, bandwidths, latencies) — independent of any task. This is the GPU's Carnot limit in the exact sense of Carnot's original result.

---

## The Carnot-Optimal Conditions

The Carnot engine achieves its efficiency limit through a specific protocol: reversible **isothermal** steps (exchanging heat at constant temperature) alternating with **adiabatic** transitions (changing temperature without exchanging heat). The GPU analog is:

- **Isothermal compute step:** Load a tile from the warmer level l into level l−1, compute on it exhaustively at level l−1 (extracting all useful work before eviction), write results back. The SM is fully occupied and the bandwidth between levels is saturated.
- **Adiabatic transition:** Move between compute phases without wasting bandwidth — no unnecessary data movement, no pipeline flushes, no synchronisation stalls.

Formalising this maximum-work protocol yields five necessary conditions for a kernel to achieve η_hw ≈ η_hw,max:

### Condition 1: Arithmetic Intensity ≥ Roofline Ridge Point

$$\text{AI} \geq \text{AI}_{\min} = \frac{\text{peak FLOPS/cycle}}{\text{BW}_{\text{HBM}} \text{ bytes/cycle}}$$

This is the standard roofline model — but in our framework it is *derived* from the partition function rather than assumed as a separate model. It emerges because being below the ridge means the kernel is HBM-bandwidth-limited (spending cycles waiting on memory rather than computing), which is thermodynamically equivalent to the engine operating below its Carnot efficiency.

### Condition 2: Working Set ≤ Capacity at Each Level

If the working set at level l exceeds the capacity of that level, data must spill to level l+1 (warmer), incurring its higher latency and energy cost. This is the hardware analog of the engine being forced to exchange heat with a warmer reservoir than necessary — direct thermodynamic loss. This condition enforces **tiling**: tile sizes must be chosen so that the active data fits within the fast levels.

### Condition 3: Data Reuse Factor ≥ Minimum at Each Level

$$\text{reuse}(l) \geq \frac{C_{l-1}}{B_l} \times \text{peak FLOPS}$$

Each byte loaded from level l must be used at least this many times before eviction, to amortise the load cost. C_{l-1} is the capacity of the colder level (where computation happens), B_l is the bandwidth from the warmer level, and peak FLOPS/cycle sets the rate at which bytes can be processed. Below this threshold, the kernel cannot stay compute-bound and HBM traffic becomes a thermodynamic bottleneck.

### Condition 4: Warp Occupancy ≥ Latency-Hiding Minimum

$$\text{occupancy} \geq \frac{\text{latency}_{\text{HBM}}}{\text{warps/SM}} \approx \frac{600}{64} \approx 10 \text{ warps/SM}$$

With HBM latency of ~600 cycles and one instruction issued per warp per cycle, you need ~600 warps in flight to keep the SM busy while a memory request is outstanding. With 64 warps per SM, you need at minimum ~10/64 ≈ 15% occupancy just to hide HBM latency. Below this threshold, the SM idles waiting for memory — thermodynamic waste from idle capacity.

### Condition 5: Zero Unnecessary Data Movement

Any byte loaded that was already resident at a colder level, or any byte written that will be re-read before eviction, is pure waste with no thermodynamic justification. This is the strongest and most practically demanding condition. It rules out naive implementations that reload the same data multiple times, materialise intermediate tensors unnecessarily, or fail to exploit cache residency.

A kernel satisfying all five conditions is **Carnot-optimal** and operates at η_hw ≈ η_hw,max.

---

## The Specific Heat and What It Tells You

The other thermodynamic quantities have physical interpretations worth noting.

**Entropy** S = β(⟨E⟩ − F) measures the degeneracy of execution states at a given utilisation level: how many distinct microstate trajectories achieve the same macroscopic efficiency. High entropy means the computation is robust and can be scheduled many ways without losing efficiency. Low entropy means efficiency is fragile and depends critically on exact scheduling decisions. This is observable in practice as the difference between a kernel that performs consistently across different GPU generations and one that is highly tuned to a specific microarchitecture.

**Specific heat** C_V = β² var(E) measures how sharply the GPU's efficiency changes under resource pressure. A high C_V means a broad, flat η(β) curve — the GPU degrades gracefully as load increases. A low C_V means a sharp phase transition between efficient and inefficient regimes. This is observable as the difference between a kernel that scales smoothly with batch size and one that falls off a cliff at a specific occupancy threshold.

---

## Validation: Roofline Recovery

A necessary sanity check is that the Carnot-optimal arithmetic intensity condition reduces to the standard roofline ridge point in the limit of a two-level memory hierarchy with infinite compute. In this limit:
- Only registers and HBM exist (Z_memory becomes a single 2-level transfer matrix)
- The capacity constraint at the register level is the only binding constraint

The roofline ridge point is AI* = peak_FLOPS / BW_HBM. Our `verify_roofline_recovery()` function checks that `limit.roofline_intensity ≈ AI*` — both are computed from the same hardware parameters and should agree to within numerical error. This provides a ground-truth validation of the entire derivation chain.

---

## Implementation Notes

### Always Work in Log Space

Z_compute overflows float64 for any realistic GPU configuration. For the H100:

```
Z_SM^132 = (Z_warp^64)^132 ≈ 8^(64×132) ≈ 10^7500
```

We therefore never materialise Z itself. All thermodynamic calculations operate on ln Z directly. The factorisation Z = Z_compute × Z_memory × Z_comm becomes ln Z = ln Z_compute + ln Z_memory + ln Z_comm, which is numerically stable.

### Numerical Derivatives

The mean waste ⟨E⟩ and specific heat C_V are computed from finite differences of ln Z:

```
⟨E⟩ ≈ -(ln Z(β + δ) - ln Z(β - δ)) / (2δ n_dof)   [central difference]

C_V ≈ β² × (-ln Z(β+2δ) + 16 ln Z(β+δ) - 30 ln Z(β) + 16 ln Z(β-δ) - ln Z(β-2δ)) / (12δ² n_dof)
                                                          [4-point stencil]
```

The 4-point stencil for C_V gives O(δ⁴) accuracy, sufficient for the β step sizes we use (δ = 10⁻⁴).

### The n_dof Normalisation

ln Z is an **extensive** quantity: ln Z ≈ n_dof × ln Z_warp + (memory and comm terms). Without normalisation, ⟨E⟩ ≈ n_dof × ⟨e_warp⟩ ≈ 8448 × 0.46 ≈ 3900, which is not a fraction. Dividing by n_dof = N_SM × warps_per_SM = 132 × 64 = 8448 gives ⟨E⟩ ∈ [0, 1], and consequently η_hw = 1 − ⟨E⟩ ∈ [0, 1].

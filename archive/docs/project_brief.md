# The Statistical Mechanics of GPU

**Deriving Optimal Computation Protocols from the Thermodynamics of GPU Hardware**

**Status:** Pre-Kickoff · **Classification:** Research — Exploratory · **Date:** March 2026

---

## 1. Executive Summary

This project treats a GPU as a thermodynamic engine and asks a question directly analogous to the one Carnot asked of heat engines: given the physical constraints of the machine, what is the theoretical maximum rate at which it can perform useful work, and what protocol achieves that maximum?

A heat engine operates between a hot reservoir and a cold reservoir; its maximum efficiency η = 1 − T_C/T_H is derived from first principles and is independent of the engine's internal mechanism. We propose an analogous derivation for GPUs—but with a critical twist that has no classical analog. A GPU takes in free energy (electrical power), performs operations, and dissipates waste. But unlike a steam engine, where "work" (force × distance) is unambiguous, a GPU's "useful work" can only be defined relative to a downstream task. A multiply-accumulate that contributes to a correct prediction is work; the same multiply-accumulate inside a dead computation path is waste. This means the GPU's efficiency decomposes into two layers: *hardware efficiency* η_hw (what fraction of input energy becomes any computation at all) and *task efficiency* η_task (what fraction of that computation is actually useful for the task). The overall Carnot limit is η_max = η_hw,max × η_task,max.

The core research program is:

1. **Derive the hardware Carnot limit η_hw,max from first principles.** Starting from the microstate space of the GPU, define thermodynamic quantities (entropy, temperature, free energy, work) rigorously. Derive the maximum operation rate as a function of hardware parameters. This is a purely theoretical result—a number (or function) that characterizes the GPU as a physical system, independent of any task.

2. **Characterize the Carnot-optimal computation class.** Derive the necessary and sufficient conditions on a computation's structure (memory access pattern, arithmetic intensity, parallelism, data reuse) for it to achieve η_hw,max. This defines a class of computations that maximally exploit the hardware.

3. **Within the Carnot-optimal class, maximize η_task for specific downstream tasks.** Use an LLM-based oracle to search for neural architectures—expressed as complete CUDA kernels—whose computations satisfy the hardware Carnot conditions (high η_hw) while also being task-efficient learners (high η_task). The key question is whether the intersection of "hardware-optimal" and "task-optimal" is non-empty—and our hypothesis is that it is, because the constraints that make a computation hardware-efficient (locality, hierarchy, reuse) are themselves useful inductive biases for real-world tasks.

The framework extends naturally to multi-GPU systems, where the interconnect topology introduces additional thermodynamic constraints and the engine analogy generalizes to a coupled engine network.

---

## 2. Problem Statement

Derive, from first principles, the maximum rate at which a GPU can perform useful computation, and use this derivation to design neural network architectures that operate at or near this theoretical limit.

This is structured as three nested problems:

1. **The physics problem (η_hw).** Define the GPU as a thermodynamic system. Derive its maximum hardware efficiency η_hw,max—the maximum fraction of electrical energy convertible to *any* computation—as a function of architectural parameters (SM count, memory hierarchy, bandwidth ratios). This is a pure theory result, independent of any task.

2. **The characterization problem (Carnot-optimal class).** Given η_hw,max, determine what properties a computation must have to saturate it. Derive the necessary structure of Carnot-optimal kernels: their memory access patterns, arithmetic intensity, data reuse factors, and parallelism. This defines a class of computations, not a specific computation.

3. **The architecture search problem (η_task).** Within the Carnot-optimal computation class, find parameterized function approximators (neural networks) that maximize η_task—the fraction of hardware work that is useful for a specific downstream task (CIFAR-10, TinyStories). This is where the LLM oracle and expressiveness scoring enter. They operate within a search space that is *derived from the physics*, not defined by human intuition, and they optimize for task relevance within that space.

The deliverable for each candidate architecture is a complete set of CUDA kernels implementing forward and backward passes. The architecture is not described abstractly and then compiled to a GPU—it is conceived, from the start, as a GPU execution protocol.

---

## 3. Theoretical Framework

### 3.1 The GPU as a Thermodynamic Engine — and the Problem of Defining Work

A classical heat engine converts thermal energy into mechanical work, operating between two thermal reservoirs. A GPU converts electrical energy into computational work, operating between the power supply (energy source) and the thermal environment (heat sink). We formalize this analogy—but immediately confront a subtlety that has no analog in classical thermodynamics.

**Energy input.** The GPU consumes power P_in (measured in Watts). Over a computation lasting time T, the total energy input is E_in = P_in · T. This is the analog of the heat Q_H absorbed from the hot reservoir.

**The problem of defining useful work.** For a steam engine, work is unambiguous: force times distance. For a GPU, the definition of "useful work" splits into two layers, because the GPU itself has no concept of whether its operations are meaningful:

- **Layer 1: Hardware work (W_hw).** The total number of operations the GPU actually executes, weighted by their energy cost. This includes every multiply-accumulate, every memory load, every instruction—regardless of whether it contributes to a correct answer. W_hw is entirely determined by the hardware execution and is task-independent.

- **Layer 2: Task work (W_task).** The subset of W_hw that actually reduces the loss on the downstream task. A 4096×4096 attention GEMM where only 10% of the entries influence the prediction contributes W_hw proportional to the full GEMM but W_task proportional to only the relevant 10%. A normalization layer that is necessary for training stability contributes to W_task; a normalization layer that could be removed without affecting convergence does not.

The true useful work is W = W_task, and the overall efficiency decomposes as:

*η = W_task / E_in = (W_hw / E_in) × (W_task / W_hw) = η_hw × η_task*

**Two Carnot limits.** This decomposition reveals two distinct theoretical maxima:

- **η_hw,max (hardware Carnot limit):** The maximum fraction of input energy that can be converted to *any* computation, regardless of whether that computation is useful for a task. This is derivable purely from the GPU's physical architecture—memory hierarchy, bandwidth ratios, SM count, pipeline structure. It is the focus of Section 3.3–3.5 and is the direct analog of the classical Carnot efficiency.

- **η_task,max (task Carnot limit):** The maximum fraction of hardware work that can be made task-relevant. This depends on the task: for some tasks (dense matrix multiplication for a well-conditioned linear system), nearly all compute is useful—η_task ≈ 1. For others (sparse attention in a language model where most tokens are irrelevant to the prediction), the theoretical minimum computation may be far less than what any known architecture performs—η_task << 1.

The overall Carnot limit is their product: **η_max = η_hw,max × η_task,max**.

**Why this decomposition matters.** It separates two fundamentally different optimization problems:

- Maximizing η_hw is a *physics and systems* problem: keep the GPU's pipeline full, data in the right memory level, bandwidth saturated. The solution is hardware-specific and task-independent.

- Maximizing η_task is a *learning theory and architecture* problem: find the computation that extracts maximum task-relevant information per operation. The solution is task-specific and (in principle) hardware-independent.

Current ML practice overwhelmingly focuses on η_task (designing architectures for accuracy) and treats η_hw as an afterthought (hoping the compiler does a good job). Performance engineering focuses on η_hw (writing fast kernels) for fixed architectures, without questioning whether the architecture's operations are task-efficient. This project is the first to *jointly derive* bounds on both and to search for architectures that simultaneously approach both limits.

**The link between the layers.** The two layers are not independent. The constraint that an architecture must achieve high η_hw (its computation must be Carnot-optimal at the hardware level) restricts the space of possible architectures. Within that restricted space, we search for high η_task (architectures that are also powerful learners). The deep question is: does the intersection of "hardware-efficient" and "task-efficient" contain anything useful? Our hypothesis is yes—and that the intersection may contain architectures that are *better learners* than existing ones, because the hardware constraints force a structure that is itself a useful inductive bias (locality, hierarchy, data reuse) that happens to align with properties of real-world data.

**Waste heat.** Everything that is not useful work is waste: Q_waste = E_in − W_task. Waste decomposes into:

- **Hardware waste (E_in − W_hw):** Idle transistors, unnecessary data movement, pipeline stalls, synchronization overhead. Addressed by maximizing η_hw.
- **Task waste (W_hw − W_task):** Operations that the GPU executes efficiently but that don't contribute to reducing the loss. Addressed by maximizing η_task.

### 3.2 Microstate Space and Entropy

A microstate σ is a complete specification of the GPU at a given clock cycle:

- **SM occupancy vector:** For each of N streaming multiprocessors, the number and identity of active warps, their instruction type (FP16/FP32/INT/SFU/MEM), and pipeline stage.
- **Memory hierarchy state:** For each level (registers, shared memory, L1/L2 cache, HBM), which data elements are resident. Model as occupation numbers n_i for data element i at memory level l.
- **Bandwidth channel utilization:** Fraction of theoretical peak bandwidth consumed on each interconnect (SM↔shared mem, SM↔L2, L2↔HBM).

The microstate space Ω is discrete and finite. A computation protocol induces a trajectory σ(t) through Ω. The thermodynamic entropy of the GPU at a given macroscopic utilization level is:

*S(U) = k_B · ln |{σ ∈ Ω : utilization(σ) = U}|*

where U is a coarse-grained utilization measure (e.g., the fraction of SMs performing useful arithmetic). High entropy at a given utilization level means many distinct microstate trajectories achieve that utilization—the system has many "ways" to be at that efficiency. Low entropy means the utilization level is highly constrained and fragile.

### 3.3 Temperature and the Memory Hierarchy

In the engine analogy, temperature maps to the *resource pressure* on the GPU. Define a hierarchy of temperatures corresponding to the memory levels:

- **T_reg (registers):** The "coldest" level. Data here is immediately available; accessing it dissipates minimal waste energy. Capacity is severely limited (~256KB per SM).
- **T_smem (shared memory):** Slightly "warmer." Access costs ~20–30 cycles of latency (during which the SM could be doing useful work). Capacity: 228KB per SM.
- **T_L2 (L2 cache):** "Warm." Access costs ~200 cycles. Capacity: 50MB total.
- **T_HBM (HBM):** The "hottest" level. Access costs ~400–800 cycles and consumes significant energy (~20× per bit compared to registers). Capacity: 80GB.

The temperature gradient T_HBM > T_L2 > T_smem > T_reg defines the thermodynamic force that drives data "downhill" from HBM toward registers. The GPU performs useful work by moving data through this gradient and computing on it at the coldest accessible level. The maximum work is extracted when data follows the steepest thermodynamic gradient—loaded once from HBM, maximally reused in registers/shared memory, and written back once.

The Carnot limit is determined by the ratio of temperatures at the extremes:

*η_max = 1 − T_reg / T_HBM*

In practice, T_reg / T_HBM is the ratio of register access cost to HBM access cost (~0 / ~600 cycles ≈ 0), so the naive Carnot limit is η_max → 1. But this is misleading—it ignores capacity constraints. The real limit is set by the *finite capacity* at each temperature level, which forces data to spend time at warm levels. The corrected Carnot limit is:

*η_max = f(C_reg, C_smem, C_L2, C_HBM, B_reg, B_smem, B_L2, B_HBM, N_SM, ...)*

where C_l is the capacity at level l, B_l is the bandwidth, and f is derived from the partition function of the GPU's memory system. Deriving f explicitly is a primary research objective of this project.

### 3.4 The Partition Function of a GPU

The partition function Z encodes the statistical mechanics of the GPU:

*Z = Σ_{σ ∈ Ω} exp(−E(σ) / k_B T)*

where E(σ) is the energy of microstate σ (the waste energy: power consumed minus useful work performed in that state). From Z, all thermodynamic quantities follow:

- **Free energy:** F = −k_B T · ln Z. This is the maximum useful work extractable from the GPU at temperature T.
- **Average energy:** ⟨E⟩ = −∂(ln Z)/∂β, where β = 1/k_B T. This is the expected waste at thermal equilibrium.
- **Entropy:** S = −∂F/∂T. This measures the degeneracy of execution states at a given efficiency level.
- **Specific heat:** C_V = ∂⟨E⟩/∂T. This measures how sensitive the GPU's efficiency is to changes in resource pressure—a high specific heat means the GPU gracefully degrades under load; a low specific heat means it has sharp phase transitions between efficient and inefficient regimes.

The key theoretical result we seek is the *equation of state* of the GPU: a relationship between the macroscopic observables (utilization, throughput, power) derived from Z. This equation of state, once derived, tells us everything about the GPU's thermodynamic limits—just as the ideal gas law PV = nRT tells us everything about the thermodynamic limits of a gas.

### 3.5 Deriving the Maximum Work Protocol

The Carnot engine achieves maximum efficiency by executing a specific protocol: two isothermal steps (where the engine exchanges heat reversibly with the reservoirs) and two adiabatic steps (where the engine changes temperature without exchanging heat). The analog for a GPU is a computation protocol that cycles through the memory hierarchy reversibly.

**Isothermal compute step:** The GPU loads a tile of data from level l into level l−1 (e.g., HBM → shared memory), computes on it exhaustively at level l−1, and writes results back to level l. "Exhaustively" means the data is reused the maximum number of times permitted by the capacity at level l−1 before being evicted. During this step, the SM is fully occupied and the bandwidth between levels l and l−1 is saturated.

**Adiabatic transition:** The GPU transitions between compute phases without wasting bandwidth—no unnecessary data movement, no pipeline flushes, no synchronization stalls. Intermediate results that are needed by the next compute phase are retained in the fastest accessible level.

The maximum work protocol is the one that maximizes the ratio W/E_in by:

1. Maximizing data reuse at each level of the hierarchy (extracting all useful compute from data before evicting it)
2. Minimizing the number of transitions between levels (each transition dissipates waste proportional to the temperature difference)
3. Overlapping data movement with computation (hiding latency so that transitions happen "for free" while the SM is busy computing on previously loaded data)

This gives us a set of *necessary conditions* on any Carnot-optimal kernel:

- **Arithmetic intensity ≥ B_HBM / peak_FLOPS** (the roofline condition—but now derived, not assumed)
- **Working set at each level ≤ capacity at that level** (tiling must respect hierarchy capacities)
- **Data reuse factor at level l ≥ C_{l-1} / B_l × peak_FLOPS** (each byte loaded must be used enough times to amortize the load cost)
- **Warp occupancy sufficient to hide memory latency** (enough active warps to keep the SM busy during loads)
- **Zero unnecessary data movement** (no reads of data that are already resident; no writes of data that will be reread before eviction)

These conditions define the Carnot-optimal computation class. Any kernel satisfying all of them operates at or near η_max.

### 3.6 Multi-GPU: Coupled Engine Networks

For a system of G GPUs, each GPU is an engine, and the interconnect topology defines the coupling between engines. The global microstate is:

*Σ = (σ₁, σ₂, ..., σ_G, C)*

where σ_g is the microstate of GPU g and C is the communication state vector. The total system Hamiltonian is:

*H(Σ) = Σ_g H_local(σ_g) + Σ_{(g,h) ∈ edges} J_{gh} · H_comm(σ_g, σ_h)*

where H_local is the single-GPU Hamiltonian from Section 3.4, and H_comm captures the communication cost. The coupling constant J_{gh} encodes the physical link properties:

- **NVLink (intra-node):** 900 GB/s bidirectional (H100), latency ~1μs. J_{gh} ≈ 0.1 (cheap communication).
- **NVSwitch (full bisection):** All-to-all within a node at full bandwidth. J_{gh} ≈ 0.1 for all pairs within the switch domain.
- **PCIe Gen5:** 64 GB/s per direction, latency ~2–5μs. J_{gh} ≈ 1.0 (moderate cost).
- **InfiniBand NDR (inter-node):** 400 Gb/s per port, latency ~1–2μs network + software overhead. J_{gh} ≈ 5.0 (expensive).
- **Ethernet/RoCE (inter-node):** 100–400 Gb/s, higher latency and protocol overhead. J_{gh} ≈ 10.0 (most expensive).

The multi-GPU Carnot limit is *lower* than the single-GPU limit because inter-GPU communication is an additional source of waste. The multi-GPU efficiency is:

*η_multi = W_total / (Σ_g E_in,g)*

and the Carnot limit η_multi,max depends on the topology. A fully connected topology (NVSwitch) has a higher limit than a ring or tree topology because it has lower communication waste for collective operations.

### 3.7 Parallelism Strategies as Thermodynamic Phases

Different parallelism strategies correspond to different ordered phases of the coupled engine network. Each phase has a characteristic symmetry, communication pattern, and efficiency:

- **Data parallelism (DP):** Ferromagnetic phase. All engines in the same state (same model). Coupling only through global order parameter (gradient all-reduce). Efficiency approaches single-GPU η_max when communication cost is small relative to compute.

- **Tensor parallelism (TP):** Antiferromagnetic phase. Neighboring engines hold complementary fragments and couple at every step. High communication frequency. Only efficient when J_{gh} is small (fast interconnect).

- **Pipeline parallelism (PP):** Domain wall phase. Sharp boundaries between engine states. Communication flows directionally. Efficiency limited by bubble fraction (a thermodynamic defect).

- **Expert parallelism (EP):** Spin-glass phase. Disordered, input-dependent communication with frustration (routing imbalance). Load balancing is the analog of annealing.

- **Context/Sequence parallelism (CP/SP):** Helical phase. Ordered along the sequence axis with periodic communication.

Phase transitions between parallelism strategies are sharp functions of model size, sequence length, and interconnect bandwidth. The formalism should predict the optimal transition points—the multi-GPU analog of phase diagrams.

### 3.8 Compute-Communication Overlap as a Coupled-Engine Resonance

Perfect compute-communication overlap corresponds to a resonance condition in the coupled engine network: each engine produces data at exactly the rate its neighbors consume it. Define the overlap ratio:

*η_overlap = T_overlapped / max(T_compute, T_comm)*

At η_overlap = 1, the coupled system operates at its Carnot limit—no engine ever idles waiting for data from another engine. This resonance condition constrains the architecture: the computation between communication points must produce exactly the right volume of data at exactly the right rate. This is a derived constraint, not an optimization target.

---

## 4. System Architecture

The system has four major components that interact in a closed loop:

### 4.1 GPU Simulator (Existing)

We have an existing GPU simulator. It operates at the kernel level: it takes actual CUDA kernels (or PTX/SASS representations) as input. Required outputs for single-GPU:

- Per-SM warp occupancy and instruction mix at each cycle
- Memory hierarchy residency map (which tensors/tiles at which level)
- Bandwidth utilization per channel per cycle
- Pipeline stall counts and causes (data dependency, resource conflict, memory latency)
- **Useful work accounting:** classification of each operation as useful (contributes to output) or waste (data movement, synchronization, idle)
- **Power model:** estimated energy consumption decomposed by component (compute, SRAM access, DRAM access, leakage)

**Multi-GPU extension requirements:**

- Configurable interconnect topology: arbitrary graph of GPU nodes with per-edge bandwidth, latency, and protocol (NVLink, NVSwitch, PCIe, InfiniBand, RoCE)
- Per-link utilization traces: bytes in flight, direction, congestion/backpressure events per cycle
- Collective operation modeling: ring all-reduce, tree all-reduce, all-to-all, all-gather, reduce-scatter with accurate bandwidth and latency modeling for each topology
- Synchronization event tracking: barrier waits, bubble time, per-GPU idle cycles attributed to communication dependencies
- Compute-communication overlap detection
- Preset topology configurations: DGX H100 (8×H100, NVSwitch), DGX SuperPOD (32 nodes, InfiniBand fat tree), custom topologies via adjacency matrix

**Deliverable:** a Python API that takes CUDA kernels (or PTX), a parallelism configuration, and a topology specification, and returns per-GPU microstate traces, energy decomposition (W, Q_waste), and the efficiency η = W / E_in.

### 4.2 Thermodynamic Analysis Module

This module computes the thermodynamic quantities from the simulator traces:

- **Efficiency η** for each kernel and for the full computation protocol
- **Distance from Carnot limit:** η_max − η, decomposed by waste source (idle compute, unnecessary data movement, pipeline stalls, synchronization)
- **Entropy of the execution:** how many distinct microstate trajectories could achieve the same macroscopic utilization (estimated via perturbation analysis in the simulator)
- **Phase identification:** whether the computation is in the compute-bound, memory-bound, or latency-bound regime (and where the phase boundaries are)
- **Bottleneck attribution:** for each percentage point of efficiency lost below η_max, identify the specific hardware constraint responsible (e.g., "12% lost to shared memory capacity—working set exceeds 228KB, forcing spill to L2")

### 4.3 LLM Architecture Oracle

An LLM prompted to generate neural architectures as complete CUDA kernel specifications, constrained to the Carnot-optimal computation class.

The prompt template now includes the *derived* Carnot-optimal conditions (from Section 3.5):

- The minimum arithmetic intensity for this GPU
- The maximum working set size at each memory level
- The minimum data reuse factor at each level
- The minimum warp occupancy for latency hiding
- Explicit prohibition of unnecessary data movement patterns

Each architecture proposal consists of:

- **Forward kernels:** CUDA kernels with thread block dimensions, shared memory layout, register budget, memory access patterns, tensor core utilization—all designed to satisfy the Carnot-optimal conditions.
- **Backward kernels:** Corresponding kernels for the backward pass.
- **Communication kernels (multi-GPU):** Specifying inter-GPU data flow and collective patterns.
- **Memory management specification:** Tensor allocation, materialization vs. recomputation, and flow through the hierarchy.

The key difference from the previous framing: the LLM is not searching freely over all possible kernels. It is searching *within a derived, principled search space*—the set of kernels that satisfy the Carnot-optimal conditions. This dramatically constrains the search and gives the LLM structured guidance on *why* certain kernel designs are better than others.

### 4.4 Optimization Loop

1. LLM proposes a batch of N candidate architectures as kernel specifications (N ≈ 20–50), constrained to the Carnot-optimal class
2. Each candidate's kernels are compiled and fed to the simulator to compute η and the distance from η_max
3. Candidates with η above a threshold are evaluated on proxy tasks (expressiveness scoring)
4. Compute Pareto frontier over (η, expressiveness)
5. Feed Pareto-optimal candidates + thermodynamic analysis back to LLM. The feedback is now physically grounded: "this kernel achieves 72% of η_max; 18% is lost to shared memory bank conflicts causing stalls at cycle 14000–15000; 10% is lost to warp divergence in the conditional branch at line 47. Restructure the shared memory access pattern to eliminate bank conflicts and replace the conditional with a predicated instruction."
6. Repeat until Pareto frontier stabilizes

### 4.5 Multi-GPU Parallelism Optimizer

For each candidate architecture, determines the optimal parallelism configuration by minimizing waste in the coupled engine network:

*max_{P} η_multi(architecture, P, topology)*

where P = (dp, tp, pp, ep, cp). The optimizer:

1. Enumerates feasible parallelism configurations
2. For each, computes η_multi using the simulator
3. Determines the optimal mapping of parallelism dimensions to physical topology
4. Evaluates whether the resonance condition (η_overlap ≈ 1) is achievable
5. Returns the Pareto frontier over parallelism configs, reporting η_multi decomposed into single-GPU efficiency and communication overhead

### 4.6 Topology-Aware Architecture Co-Design

Joint optimization of architecture and parallelism for a specific topology. The LLM receives the topology and the derived Carnot limit for that topology, then proposes architectures natively suited to the coupled engine structure:

- Asymmetric layer structures matching the topology's bandwidth hierarchy
- Built-in "communication slots" timed to achieve resonance
- Expert routing patterns matching the physical connectivity
- Activation checkpointing at pipeline stage boundaries

---

## 5. Concrete Use Cases

### 5.1 Use Case A: Image Classification (CIFAR-10)

**Objective:** Design a Carnot-optimal image classifier on CIFAR-10.

**Setup:** The LLM oracle proposes kernel-level architectures for 32×32×3 images with 10 output classes. Single H100 base case, 8×H100 for multi-GPU variant. Parameter count range: 0.3–10M.

**Why CIFAR-10 is a good test of the theory:** At 32×32 spatial resolution with small models, the workload is far too small to saturate an H100. Standard PyTorch implementations achieve <10% MFU—meaning they operate at <10% of even the naive Carnot limit. This is the hardest utilization problem: the GPU is thermodynamically "cold" (low compute pressure), and most of its capacity is wasted. If the formalism can derive kernels that operate significantly closer to η_max for this small workload, it demonstrates that the theoretical framework adds genuine value beyond existing optimization techniques.

**Kernel-level design space:** The 32×32 image fits entirely in shared memory. The Carnot-optimal conditions demand: load the image once from HBM, perform all computation in registers/shared memory with maximum reuse, write only the 10-dimensional output back. This may favor a single fused kernel implementing the entire forward pass (or large chunks of it) without returning to HBM for intermediate results. The formalism should predict whether a convolution-like or attention-like or entirely novel compute primitive achieves higher η for these dimensions.

**Evaluation protocol:**

- Train on CIFAR-10 (50K images) with standard augmentation
- Report test accuracy, MFU, throughput (images/sec), energy per image, and η_hw / η_hw,max
- Profile with Nsight Compute to validate η predictions

**Success criteria:** ≥93% test accuracy with η_hw / η_hw,max ≥ 0.3 (vs. <0.1 for standard PyTorch). Stretch: ≥96% accuracy with a structurally novel architecture.

### 5.2 Use Case B: Autoregressive Language Modeling (TinyStories)

**Objective:** Design a Carnot-optimal autoregressive language model on TinyStories.

**Setup:** Two scale points:

- **Tiny (≈10M params):** Single H100. Baseline: small GPT-2-style transformer (4–6 layers, hidden dim 256–384). Training data: TinyStories (~500M tokens).
- **Small (≈85–125M params):** Single H100 or 8×H100. Baseline: GPT-1 (~117M params). Training data: TinyStories + optional OpenWebText subset (~2B tokens).

**Why TinyStories tests the theory differently than CIFAR-10:** Autoregressive language modeling introduces causal structure (triangular attention mask), sequential token generation (inherently serial), and a KV cache that grows with sequence length. These create additional thermodynamic constraints beyond the static memory hierarchy. The causal mask wastes ~50% of attention compute in a naive implementation—this is a thermodynamic loss that the formalism should quantify and that Carnot-optimal kernels should eliminate. At 10M parameters, the KV cache for 512 tokens fits in L2, creating an interesting regime where the "temperature" of the KV cache is T_L2 rather than T_HBM.

**Kernel-level design space:**

- **Causal attention as a thermodynamic constraint:** The triangular mask creates asymmetric data reuse. The formalism should derive the optimal tiling for causal attention directly from the Carnot conditions—or determine that an attention-free architecture achieves higher η by avoiding the triangular waste entirely.
- **KV cache at different temperatures:** At 10M params, KV cache fits in L2 (ε₂ ≈ 200 cycles). At 125M params, it may spill to HBM (ε₃ ≈ 600 cycles). The phase transition between these regimes should be predicted by the formalism.
- **Decode-phase efficiency:** Single-token decode is latency-bound. The Carnot limit for decode is much lower than for prefill—the theory should quantify this gap and predict what architectural features minimize it.

**Evaluation protocol:**

- Train on TinyStories with cosine LR schedule
- Report validation perplexity, MFU (prefill and decode), tokens/sec/GPU, Joules/token, and η_hw / η_hw,max
- For 125M on 8-GPU: report scaling efficiency and validate parallelism optimizer

**Success criteria:** Match baseline perplexity at η_hw / η_hw,max ≥ 0.35 for training (vs. <0.15 for standard small transformers). More importantly: the formalism should correctly predict the η gap between prefill and decode phases, and between different sequence lengths.

**Scaling roadmap:** TinyStories → OpenWebText (GPT-2 scale) → larger corpora (LLaMA scale). Each transition exercises different aspects of the formalism.

---

## 6. Concrete Deliverables

1. **Derivation of the hardware Carnot limit** η_hw,max as a function of hardware parameters, with the full partition function calculation and equation of state. Formal definition of the two-layer efficiency decomposition η = η_hw × η_task and characterization of each layer's theoretical maximum.

2. **Characterization of the Carnot-optimal computation class:** necessary and sufficient conditions on kernel structure for η → η_max. Proof that these conditions reduce to the roofline model in the appropriate limit.

3. **Instrumented single-GPU simulator** with Python API accepting CUDA kernels, producing microstate traces, energy decomposition, and η_hw / η_hw,max.

4. **Multi-GPU simulator extension** with configurable topologies, coupled-engine Carnot limit computation, and communication overhead decomposition.

5. **Parallelism optimizer** deriving optimal (dp, tp, pp, ep, cp) from the coupled-engine thermodynamics.

6. **LLM oracle pipeline** with Carnot-optimal constraints built into the prompt, kernel-level output, and expressiveness scoring.

7. **Optimization loop** with physics-grounded feedback (waste decomposition by source, distance from η_max, bottleneck attribution).

8. **Validation experiments** on CIFAR-10 and TinyStories at specified scales, reporting η_hw / η_hw,max alongside traditional metrics.

9. **Paper draft** targeting ICML/NeurIPS/ICLR. The paper's narrative: derive the GPU Carnot limit → characterize Carnot-optimal computations → show neural architectures can live in this class → demonstrate on real tasks.

---

## 7. Validation Strategy

### 7.1 Theoretical Validation

Validate the derivation itself:

- **Roofline recovery:** Show that the Carnot-optimal conditions reduce to the roofline bound (arithmetic intensity ≥ bytes/FLOP) in the limit of a two-level memory hierarchy with infinite compute. This should fall out of the partition function calculation.
- **Known kernel ranking:** Compute η_hw / η_hw,max for a set of well-understood kernels (optimized GEMM, naive GEMM, FlashAttention, naive attention, fused softmax, unfused softmax). The ordering should match known performance rankings. If the formalism assigns η_FlashAttention > η_naive_attention, that's a sanity check. If it also quantifies *how much* efficiency FlashAttention recovers and attributes it to specific waste sources, that's a stronger validation.
- **Phase boundary prediction:** Predict the compute-bound → memory-bound transition for parameterized kernel families. Compare with empirical profiling.

### 7.2 Architecture Validation

- **Classification baseline:** ResNet-18 on CIFAR-10, matched parameter count and compute.
- **Language modeling baseline:** Small GPT-2-style transformer (10M) and GPT-1 (125M) on TinyStories, matched parameters and tokens.
- **Metrics:** Accuracy/perplexity, η_hw / η_hw,max (hardware efficiency relative to Carnot limit), η_task (estimated task efficiency — see Open Question 2), MFU, throughput, energy per sample/token.
- **Target:** ≥15% closer to η_hw,max than baselines at equivalent accuracy/perplexity. Additionally, characterize η_task for both the discovered architecture and baselines to determine whether hardware-optimal architectures sacrifice, match, or improve task efficiency.

### 7.3 Multi-GPU Validation

- **Topology sensitivity:** Same architecture on DGX H100 (NVSwitch), A100 ×8 (NVLink pairwise), 8× PCIe. The derived η_multi,max should correctly rank topologies.
- **Scaling prediction:** Predict η_multi at 8 GPUs from single-GPU η and topology parameters. Compare with measurements.
- **Phase transition detection:** Sweep model size on fixed topology. Predict parallelism phase transitions.
- **Parallelism sanity check:** Verify optimizer recommendations for standard transformers are consistent with established heuristics.

---

## 8. Timeline

| Phase | Deliverables | Success Criteria |
|-------|-------------|-----------------|
| Phase 0: Theory | Derivation of GPU Carnot limit; partition function calculation; Carnot-optimal computation class characterization; literature review | Closed-form (or computable) η_hw,max for H100; Carnot-optimal conditions derived; roofline model recovered as special case |
| Phase 1: Single-GPU Simulator | Instrumented simulator accepting CUDA kernels; thermodynamic analysis module computing η, waste decomposition, distance from η_max | η_hw / η_hw,max computed for reference kernels; ordering matches known performance; waste decomposition identifies correct bottlenecks |
| Phase 1.5: Multi-GPU Extension | Coupled-engine Carnot limit computation; topology-dependent η_multi,max; collective operation modeling | Correctly ranks topologies by η_multi,max; collectives within 10% of NCCL benchmarks |
| Phase 2: Constrained Oracle | LLM kernel proposal pipeline with Carnot-optimal constraints; CUDA compilation; expressiveness scoring | Generates valid kernels satisfying Carnot-optimal conditions; expressiveness scores correlate with downstream performance |
| Phase 3: Architecture Search | Pareto frontier (η_hw / η_hw,max, expressiveness) at single-GPU and 8-GPU scales | ≥3 Pareto-optimal architectures per scale that differ structurally from known designs |
| Phase 4: Training & Validation | Train on CIFAR-10 and TinyStories; compare against baselines | ≥1 architecture closer to η_hw,max by ≥15% at equivalent accuracy/perplexity |
| Phase 5: Paper | Full paper: theory, derivation, experiments, predictions | Submission-ready for ICML/NeurIPS/ICLR |

---

## 9. Key Hypotheses to Test

1. **The GPU Carnot limit is computable.** The partition function (or a tractable approximation) can be evaluated for realistic GPU architectures, yielding a concrete η_hw,max. Testable in Phase 0.

2. **The Carnot-optimal class is non-trivially constraining.** The derived conditions exclude most existing kernel implementations while admitting a rich space of computations that includes viable neural network building blocks. Testable in Phase 0/1.

3. **η_max is significantly higher than current practice.** The gap between η_hw,max and the best existing kernels (e.g., cuBLAS GEMM, FlashAttention) is ≥10%, indicating there is headroom the formalism can exploit. Testable in Phase 1.

4. **LLM proposals within the Carnot-optimal class are compilable and trainable.** The constrained search space still admits ≥50 structurally diverse, valid architectures. Testable in Phase 2.

5. **The Carnot-optimal class admits task-efficient architectures (η_hw × η_task is jointly maximizable).** Operating near η_hw,max does not require sacrificing η_task—there exist architectures in the Carnot-optimal class that match baseline accuracy/perplexity. More speculatively: the hardware constraints (locality, hierarchical data reuse, tiling) may themselves be useful inductive biases, so that Carnot-optimal architectures are *better* learners than unconstrained ones for structured data like images and language. Testable in Phase 4.

6. **The multi-GPU Carnot limit correctly ranks topologies.** η_multi,max derived from the coupled-engine formalism predicts the relative throughput of different cluster configurations.

7. **Parallelism phase transitions are predicted by the theory.** The coupled-engine model predicts sharp transitions between parallelism strategies as model size varies on a fixed topology.

8. **Kernel-native Carnot-optimal architectures outperform framework-compiled architectures.** The LLM-crafted kernels achieve higher η than equivalent mathematical architectures implemented through PyTorch + torch.compile, validating that the Carnot-optimal design space contains gains inaccessible to automated compilers.

---

## 10. Anticipated Results

- **A concrete number for η_hw,max.** For the H100, we expect η_hw,max to be in the range 0.7–0.85 for compute-bound workloads (limited by memory hierarchy overhead, pipeline structure, and warp scheduling constraints). For memory-bound workloads, η_hw,max will be lower and will depend on the bandwidth ratios. The fact that even the best existing kernels achieve ~0.5–0.6 suggests significant headroom.

- **η_task will vary dramatically by architecture.** For a dense transformer on CIFAR-10, η_task may be quite low—full attention over all 32×32 patches computes many pairwise relationships that are irrelevant to classification. A Carnot-optimal architecture forced into local, hierarchical computation by the hardware constraints may paradoxically achieve *higher* η_task because its inductive bias (locality, data reuse) aligns with the spatial structure of images.

- **The η_hw × η_task trade-off curve will be non-convex.** We expect to find architectures that are Pareto-dominated by neither pure hardware optimization nor pure task optimization—architectures that achieve a better product η_hw × η_task than either extreme. These are the genuinely novel architectures the framework is designed to discover.

- **Carnot-optimal kernels will be heavily fused.** The "zero unnecessary data movement" condition strongly favors monolithic kernels where multiple mathematical operations execute within a single kernel launch, with all intermediate state in registers/shared memory.

- **Tile dimensions will be derived, not chosen.** The optimal tile sizes at each memory level will follow from the Carnot conditions and the capacity/bandwidth constraints—they won't be powers of 2 for aesthetic reasons but specific values that maximize data reuse at each level.

- **Attention may not be Carnot-optimal.** Quadratic attention has a data reuse pattern that may violate the Carnot conditions for long sequences (the working set exceeds shared memory capacity, forcing spills to L2/HBM). The formalism may determine that linear attention, state-space models, or a novel operation achieves higher η.

- **Different architectures at different scales.** The Carnot limit is scale-dependent (different workload sizes hit different constraints). The formalism may reveal that the optimal architecture at 10M params is fundamentally different from the optimal architecture at 125M params—not just wider/deeper but structurally different.

- **Multi-GPU Carnot limit will predict when to change parallelism strategies.** As model size grows on a fixed 8-GPU node, the coupled-engine model will predict sharp transitions (e.g., DP-only → DP+TP at a specific parameter count determined by HBM capacity and NVLink bandwidth).

- **Backward-pass-aware design.** The Carnot conditions apply separately to forward and backward passes (which have different memory access patterns). The optimal architecture may have forward-pass structures chosen because they produce backward kernels that are closer to η_max.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| The partition function is intractable for realistic GPUs | Critical | Use mean-field approximations, coarse-graining, or Monte Carlo sampling of the microstate space. Even an approximate η_max that ranks kernels correctly is valuable. |
| η_max is so close to current best practice that there's no headroom | High | This would itself be an interesting result (existing kernels are near-optimal). Pivot to the multi-GPU case where headroom is likely larger due to communication overhead. |
| The Carnot-optimal class excludes all useful neural network architectures | Critical | Relax the conditions to ε-Carnot-optimal (within ε of η_hw,max) and show that the relaxed class admits neural architectures. Characterize the trade-off between η_hw and η_task—even if the strict Carnot-optimal class is too restrictive, the η_hw × η_task Pareto frontier is still valuable. |
| LLM cannot generate valid CUDA kernels satisfying the Carnot constraints | High | Start with a constrained kernel DSL; gradually expand to raw CUDA. Provide Carnot-optimal templates. Fall back to human engineer translating LLM specifications. |
| Novel architectures don't train stably | Medium | muP or spectral initialization; hyperparameter search budget; fallback to showing the theoretical results + Pareto analysis without full training. |
| Multi-GPU simulator fidelity diverges from real clusters | High | Validate against NCCL benchmarks on actual DGX H100; calibrate empirically. |
| The engine analogy breaks down for GPUs | High | Identify precisely where and why. The breakdown itself is informative—it characterizes the limits of the thermodynamic framework. Document as a negative result if necessary. |

---

## 12. Required Resources

### 12.1 Compute

- GPU simulator runs (single-GPU): CPU-only, ~1000 core-hours
- GPU simulator runs (multi-GPU): ~5000–10000 core-hours
- Partition function estimation (Monte Carlo sampling of microstate space): ~2000 core-hours
- LLM oracle calls: ~10000–20000 API calls
- CUDA compilation and kernel validation: ~500 core-hours
- Proxy task training: ~200 GPU-hours
- CIFAR-10 training (Use Case A): ~50 GPU-hours
- TinyStories training (Use Case B): ~100 GPU-hours at 10M scale, ~300 GPU-hours at 125M scale
- NCCL/profiling benchmarks: ~200 GPU-hours

### 12.2 People

- **Research engineer (primary):** Systems-level GPU programming (CUDA, PTX, warp-level intrinsics), experience with GPU simulators. Owns simulator instrumentation, thermodynamic analysis module, kernel validation pipeline, and optimization loop.
- **PI/lead researcher:** Statistical mechanics + ML architecture expertise. Owns the Carnot limit derivation, partition function calculation, Carnot-optimal class characterization, paper, and high-level direction. This person needs to be comfortable with actual stat mech calculations—partition functions, mean-field approximations, phase diagrams.
- **ML engineer (part-time, Phase 4 onward):** Training infrastructure, hyperparameter optimization, benchmark evaluation.

---

## 13. Open Questions for Discussion

1. **Is the partition function tractable?** The microstate space of a real GPU is enormous. Can we compute Z (or a useful approximation) analytically for simplified GPU models, then correct numerically for realistic hardware? Mean-field theory, replica methods, and renormalization group techniques from condensed matter physics may be applicable.

2. **Measuring η_task in practice.** The two-layer decomposition η = η_hw × η_task is clean in theory, but measuring η_task empirically is hard. How do you determine which operations in a trained network "contribute to the output"? One approach: ablation-based attribution (remove operations and measure accuracy drop). Another: information-theoretic (measure mutual information between intermediate activations and the label). A pragmatic proxy: compare the compute cost of the discovered architecture against the information-theoretic minimum for the task (e.g., Bayes-optimal classification of CIFAR-10 requires processing at most ~10 bits of information per image × 50K images). The gap between actual compute and this minimum is an upper bound on 1 − η_task.

3. **Ergodicity and equilibrium.** The GPU's execution is deterministic, not stochastic. Can we define a meaningful "equilibrium" for a deterministic system? One approach: define the ensemble over all valid schedules for a given computation, and show that the warp scheduler's complexity makes the system effectively ergodic from the application's perspective.

4. **Kernel representation for the LLM.** Raw CUDA, kernel-level DSL, or PTX? The Carnot-optimal constraints may make this easier—the search space is narrower, so a more constrained DSL may suffice.

5. **Backward pass kernels.** Options: (a) LLM generates both, (b) auto-differentiate at kernel level (Enzyme), (c) finite-difference verification. The Carnot conditions apply separately to backward kernels, which is an additional constraint but also additional guidance for the LLM.

6. **Landauer's limit.** In principle, the thermodynamic cost of irreversible computation (k_B T ln 2 per bit erased) sets an absolute floor on waste. At room temperature this is ~3×10⁻²¹ J/bit—negligible compared to current GPU energy scales (~10⁻¹² J per operation). But does it become relevant in any limit? Probably not for this project, but worth noting for theoretical completeness.

7. **Is η_max the same for all computations?** Intuitively, different computations (GEMM vs. reduction vs. scatter) have different Carnot limits because they have different memory access patterns and arithmetic intensities. The derivation should yield η_max as a function of the computation's structural properties, not just the hardware. Characterizing this function is part of the theory deliverable.

8. **Coupling constant calibration.** Should J_{gh} be measured empirically for each cluster, or derived from specs? What calibration error is tolerable?

9. **Dynamic parallelism.** Can the framework model time-varying parallelism as a non-equilibrium thermodynamic process?

10. **Numerical correctness of LLM kernels.** Need a robust validation pipeline checking outputs against reference implementations at float32 precision, with FP16/BF16 tolerances.

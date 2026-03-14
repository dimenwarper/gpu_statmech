"""
Parallelism optimizer for multi-GPU neural network training.

Scores parallelism configurations (dp, tp, pp, ep, cp) by their
multi-GPU thermodynamic efficiency η_multi and finds the Pareto
frontier over (η_multi, −communication_overhead).

Each parallelism strategy maps to a thermodynamic phase of the
inter-GPU communication, characterised by its coupling constant J:

    DP  → ferromagnetic           (AllReduce,          J_DP = 1.0 × J_link)
    TP  → antiferromagnetic       (AllGather + RS,      J_TP = 1.5 × J_link)
    PP  → domain_wall             (P2P at boundary,     J_PP = 0.8 × J_link)
    EP  → spin_glass              (AllToAll,            J_EP = 5.0 × J_link)
    CP  → quasi_antiferromagnetic (AllGather on KV,     J_CP = 1.4 × J_link)

Communication volumes (bytes per training iteration):

    DP AllReduce        : 2 × N_params × D_dtype
    TP AllGather        : 2L × (B/dp) × S × H × D_dtype
    TP ReduceScatter    : 2L × (B/dp) × S × H × D_dtype
    PP P2P              : (B/(dp×pp)) × S × H × D_dtype  (per stage boundary)
    EP AllToAll (×2)    : 2 × (B/dp) × S × H × D_dtype  (per expert layer)
    CP AllGather (KV)   : 4L × (B/dp) × (S/cp) × H × D_dtype

Compute time per training iteration:

    T_compute = 6 × N_params × B × S / (n_gpu × peak_TFLOPS × 1e12 × η_hw)

Factor of 6: forward (2N ops) + backward (4N ops) per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .partition_function import (
    H100_MEMORY_LEVELS,
    H100_SM_CONFIG,
    LINK_PRESETS,
    LinkConfig,
    MemoryLevel,
    SMConfig,
    TopologyEdge,
)
from .multi_gpu import (
    THERMO_PHASE,
    MultiGPUCarnotLimit,
    TopologyGraph,
    derive_multi_gpu_carnot_limit,
    normalise_comm_demand,
    resonance_condition,
)


# H100 peak FP16 tensor-core throughput (TFLOPS, from NVIDIA datasheet)
H100_PEAK_TFLOPS: float = 989.0


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    Key transformer model parameters for communication volume estimation.

    Attributes:
        n_params:    Total parameter count (e.g. 7_000_000_000 for LLaMA-7B).
        n_layers:    Transformer depth (stacked attention + FFN blocks).
        hidden_dim:  Hidden dimension H (residual stream width).
        n_heads:     Number of attention heads.
        n_experts:   Number of MoE experts (1 = dense model).
        seq_len:     Sequence length S (tokens).
        batch_size:  Global batch size B (tokens across all GPUs per iteration).
        dtype_bytes: Bytes per parameter/activation (2 = fp16/bf16, 4 = fp32).
    """
    n_params: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    n_experts: int = 1
    seq_len: int = 2048
    batch_size: int = 1024
    dtype_bytes: int = 2


# Model presets
GPT2_SMALL  = ModelParams(117_000_000,    12,  768,  12, seq_len=1024,  batch_size=512)
GPT2_MEDIUM = ModelParams(345_000_000,    24, 1024,  16, seq_len=1024,  batch_size=512)
LLAMA_7B    = ModelParams(7_000_000_000,  32, 4096,  32, seq_len=2048,  batch_size=1024)
LLAMA_70B   = ModelParams(70_000_000_000, 80, 8192,  64, seq_len=2048,  batch_size=2048)


# ---------------------------------------------------------------------------
# Parallelism configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelismConfig:
    """
    A parallelism strategy expressed as degree counts.

    The total GPU count is: n_gpu = dp × tp × pp × ep × cp.

    Attributes:
        dp: Data-parallel degree.    Replicates model; reduces gradients via AllReduce.
        tp: Tensor-parallel degree.  Shards weight matrices within a layer.
        pp: Pipeline-parallel degree. Assigns layer blocks to GPU stages.
        ep: Expert-parallel degree.  Distributes MoE experts across GPUs.
        cp: Context-parallel degree. Shards the sequence dimension.
    """
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1

    @property
    def n_gpu(self) -> int:
        return self.dp * self.tp * self.pp * self.ep * self.cp

    @property
    def label(self) -> str:
        """Human-readable strategy label, e.g. 'DP4+TP2'."""
        parts: list[str] = []
        if self.dp > 1: parts.append(f"DP{self.dp}")
        if self.tp > 1: parts.append(f"TP{self.tp}")
        if self.pp > 1: parts.append(f"PP{self.pp}")
        if self.ep > 1: parts.append(f"EP{self.ep}")
        if self.cp > 1: parts.append(f"CP{self.cp}")
        return "+".join(parts) if parts else "single-GPU"

    @property
    def dominant_strategy(self) -> str:
        """Name of the parallelism dimension with the highest degree."""
        dims = {"dp": self.dp, "tp": self.tp, "pp": self.pp,
                "ep": self.ep, "cp": self.cp}
        return max(dims, key=dims.get)

    @property
    def dominant_phase(self) -> str:
        """Thermodynamic phase of the dominant parallelism strategy."""
        return THERMO_PHASE.get(self.dominant_strategy, "unknown")


# ---------------------------------------------------------------------------
# Communication volume estimation
# ---------------------------------------------------------------------------

@dataclass
class CommVolumes:
    """Communication volume estimates for one training iteration (bytes)."""
    dp_allreduce_bytes:     float = 0.0
    tp_allgather_bytes:     float = 0.0
    tp_reducescatter_bytes: float = 0.0
    pp_p2p_bytes:           float = 0.0
    ep_alltoall_bytes:      float = 0.0
    cp_allgather_bytes:     float = 0.0

    @property
    def total_bytes(self) -> float:
        return (
            self.dp_allreduce_bytes
            + self.tp_allgather_bytes
            + self.tp_reducescatter_bytes
            + self.pp_p2p_bytes
            + self.ep_alltoall_bytes
            + self.cp_allgather_bytes
        )

    def breakdown(self) -> dict[str, float]:
        """Fractional breakdown of communication volume by strategy."""
        total = max(self.total_bytes, 1.0)
        return {
            "dp_allreduce":     self.dp_allreduce_bytes     / total,
            "tp_allgather":     self.tp_allgather_bytes     / total,
            "tp_reducescatter": self.tp_reducescatter_bytes / total,
            "pp_p2p":           self.pp_p2p_bytes           / total,
            "ep_alltoall":      self.ep_alltoall_bytes      / total,
            "cp_allgather":     self.cp_allgather_bytes     / total,
        }


def estimate_comm_volumes(
    config: ParallelismConfig,
    model: ModelParams,
) -> CommVolumes:
    """
    Estimate per-iteration communication volumes for each parallelism strategy.

    Derivations:
        DP AllReduce   : Ring allreduce = 2 × N × D  (two passes of model size).
        TP AllGather   : 2 ops/layer × L layers × local_tokens × H × D.
        TP ReduceScatter: same volume as AllGather (column-then-row matmul).
        PP P2P         : activations per microbatch × (pp−1) stage boundaries.
        EP AllToAll    : 2 × local_tokens × H × D × expert_layers
                         (forward dispatch + backward combine).
        CP AllGather   : 4 × L × local_tokens/cp × H × D
                         (K and V gathered for full-sequence attention).

    Args:
        config: Parallelism configuration.
        model:  Model hyperparameters.

    Returns:
        CommVolumes with per-strategy byte counts.
    """
    L  = model.n_layers
    H  = model.hidden_dim
    B  = model.batch_size
    S  = model.seq_len
    D  = model.dtype_bytes
    N  = model.n_params

    dp = max(config.dp, 1)
    tp = max(config.tp, 1)
    pp = max(config.pp, 1)
    ep = max(config.ep, 1)
    cp = max(config.cp, 1)

    # DP: ring AllReduce (2 passes over model parameters)
    dp_allreduce = 2.0 * N * D if config.dp > 1 else 0.0

    # TP: 2 AllGather + 2 ReduceScatter per layer on activations
    # (Megatron-style column-parallel then row-parallel linear)
    local_tokens_tp = (B / dp) * S
    tp_allgather     = 2.0 * L * local_tokens_tp * H * D if config.tp > 1 else 0.0
    tp_reducescatter = 2.0 * L * local_tokens_tp * H * D if config.tp > 1 else 0.0

    # PP: P2P send of activation tensor at each stage boundary per microbatch
    # Microbatch = global batch / (dp × pp_accumulation_steps)
    micro_tokens = B / (dp * pp)
    pp_p2p = micro_tokens * S * H * D * (pp - 1) if config.pp > 1 else 0.0

    # EP: AllToAll routing of tokens to experts (forward + backward)
    # Assume top-1 routing; expert layers = every other layer for MoE
    local_tokens_ep = (B / dp) * S
    n_expert_layers = max(L // 2, 1)
    ep_alltoall = (2.0 * local_tokens_ep * H * D * n_expert_layers
                   if config.ep > 1 else 0.0)

    # CP: AllGather K and V for full-sequence attention each layer
    # Local tokens after CP split: (B/dp) × (S/cp)
    local_tokens_cp = (B / dp) * (S / cp)
    cp_allgather = 4.0 * L * local_tokens_cp * H * D if config.cp > 1 else 0.0

    return CommVolumes(
        dp_allreduce_bytes=dp_allreduce,
        tp_allgather_bytes=tp_allgather,
        tp_reducescatter_bytes=tp_reducescatter,
        pp_p2p_bytes=pp_p2p,
        ep_alltoall_bytes=ep_alltoall,
        cp_allgather_bytes=cp_allgather,
    )


# ---------------------------------------------------------------------------
# Timing estimates
# ---------------------------------------------------------------------------

def estimate_compute_time_s(
    config: ParallelismConfig,
    model: ModelParams,
    sm_config: SMConfig | None = None,
    eta_hw: float = 0.5,
    peak_tflops: float = H100_PEAK_TFLOPS,
) -> float:
    """
    Estimate compute time per training iteration (seconds).

        T_compute = 6 × N_params × B × S / (n_gpu × peak_TFLOPS × 1e12 × η_hw)

    Factor of 6: forward pass ≈ 2N FLOP/token, backward ≈ 4N FLOP/token.
    η_hw reduces effective TFLOPS from the peak.

    Args:
        config:      Parallelism configuration.
        model:       Model hyperparameters.
        sm_config:   SM configuration (unused, kept for API consistency).
        eta_hw:      Hardware efficiency fraction (default 0.5 = 50% of peak).
        peak_tflops: Peak FP16 throughput in TFLOPS (default: H100 = 989).

    Returns:
        Compute time in seconds.
    """
    flops_total = 6.0 * model.n_params * model.batch_size * model.seq_len
    flops_per_gpu = flops_total / max(config.n_gpu, 1)
    effective_tflops = max(peak_tflops * eta_hw, 1.0)
    return flops_per_gpu / (effective_tflops * 1e12)


def estimate_comm_time_s(
    comm_volumes: CommVolumes,
    topology: TopologyGraph,
) -> float:
    """
    Estimate communication time per training iteration (seconds).

        T_comm ≈ total_bytes / bottleneck_bandwidth

    Uses the bottleneck (minimum per-link) bandwidth.  In practice
    collective algorithms exploit all links, but the bottleneck still
    limits throughput for the most congested collective.

    Returns 0.0 if there is nothing to communicate (zero volume).
    Returns infinity if there is volume but the topology has no links.

    Args:
        comm_volumes: Communication volume estimates.
        topology:     Multi-GPU topology graph.

    Returns:
        Communication time in seconds.
    """
    if comm_volumes.total_bytes == 0.0:
        return 0.0
    bw_bytes_s = topology.bottleneck_bandwidth_gb_s() * 1e9
    if bw_bytes_s < 1.0:
        return float("inf")
    return comm_volumes.total_bytes / bw_bytes_s


# ---------------------------------------------------------------------------
# Topology builder for a parallelism config
# ---------------------------------------------------------------------------

# Coupling constant scale factors per parallelism strategy.
# These encode the thermodynamic phase of the communication pattern:
#   TP (antiferromagnetic): tight coupling, AllGather alternates state → 1.5 ×
#   PP (domain wall):       one-way P2P, weaker coupling              → 0.8 ×
#   DP (ferromagnetic):     AllReduce, replicas align                 → 1.0 ×
#   EP (spin-glass):        AllToAll, random routing disorder         → 5.0 ×
#   CP (quasi-AF):          like TP but on sequence dimension         → 1.4 ×
_J_SCALE: dict[str, float] = {
    "tp": 1.5,
    "pp": 0.8,
    "dp": 1.0,
    "ep": 5.0,
    "cp": 1.4,
}


def build_parallelism_topology(
    config: ParallelismConfig,
    intra_node_link: LinkConfig | None = None,
    inter_node_link: LinkConfig | None = None,
    gpus_per_node: int = 8,
) -> TopologyGraph:
    """
    Build the effective topology graph implied by a parallelism configuration.

    Each parallelism dimension introduces directed edges between its GPU group,
    scaled by a coupling modifier J_scale that encodes the thermodynamic phase:

        J_ij = J_scale[strategy] × J_link

    Link selection:
      - TP always uses intra-node (requires NVLink bandwidth).
      - PP may span nodes.
      - DP uses intra-node for small dp, inter-node for large dp.
      - EP and CP follow the same rule as DP.

    Args:
        config:           Parallelism configuration.
        intra_node_link:  Link type within a node (default: NVSwitch).
        inter_node_link:  Link type across nodes (default: InfiniBand).
        gpus_per_node:    Number of GPUs per physical node (default: 8).

    Returns:
        TopologyGraph representing the full multi-GPU communication graph.
    """
    if intra_node_link is None:
        intra_node_link = LINK_PRESETS["nvswitch"]
    if inter_node_link is None:
        inter_node_link = LINK_PRESETS["infiniband"]

    dp = config.dp; tp = config.tp; pp = config.pp
    ep = config.ep; cp = config.cp
    n_gpu = config.n_gpu
    links: list[TopologyEdge] = []

    def _scaled_link(strategy: str, span_nodes: bool) -> LinkConfig:
        base = inter_node_link if span_nodes else intra_node_link
        scale = _J_SCALE.get(strategy, 1.0)
        return LinkConfig(
            name=f"{base.name}_{strategy}",
            bandwidth_gb_s=base.bandwidth_gb_s,
            latency_us=base.latency_us,
            coupling_J=base.coupling_J * scale,
        )

    def gpu_idx(dp_i: int, tp_i: int, pp_i: int, ep_i: int, cp_i: int) -> int:
        return (dp_i * (tp * pp * ep * cp)
                + tp_i * (pp * ep * cp)
                + pp_i * (ep * cp)
                + ep_i * cp
                + cp_i)

    def intra_node(g1: int, g2: int) -> bool:
        return (g1 // gpus_per_node) == (g2 // gpus_per_node)

    for dp_i in range(dp):
        for tp_i in range(tp):
            for pp_i in range(pp):
                for ep_i in range(ep):
                    for cp_i in range(cp):
                        g = gpu_idx(dp_i, tp_i, pp_i, ep_i, cp_i)

                        # TP: cyclic ring within TP group
                        if config.tp > 1:
                            g2 = gpu_idx(dp_i, (tp_i + 1) % tp, pp_i, ep_i, cp_i)
                            lc = _scaled_link("tp", not intra_node(g, g2))
                            links.append(TopologyEdge(g, g2, lc))

                        # PP: linear chain to next stage
                        if config.pp > 1 and pp_i + 1 < pp:
                            g2 = gpu_idx(dp_i, tp_i, pp_i + 1, ep_i, cp_i)
                            lc = _scaled_link("pp", not intra_node(g, g2))
                            links.append(TopologyEdge(g, g2, lc))

                        # DP: all-to-all within DP group (AllReduce)
                        if config.dp > 1:
                            for dp_j in range(dp):
                                if dp_j != dp_i:
                                    g2 = gpu_idx(dp_j, tp_i, pp_i, ep_i, cp_i)
                                    lc = _scaled_link("dp", not intra_node(g, g2))
                                    links.append(TopologyEdge(g, g2, lc))

                        # EP: all-to-all within EP group (AllToAll)
                        if config.ep > 1:
                            for ep_j in range(ep):
                                if ep_j != ep_i:
                                    g2 = gpu_idx(dp_i, tp_i, pp_i, ep_j, cp_i)
                                    lc = _scaled_link("ep", not intra_node(g, g2))
                                    links.append(TopologyEdge(g, g2, lc))

                        # CP: cyclic ring within CP group (like TP)
                        if config.cp > 1:
                            g2 = gpu_idx(dp_i, tp_i, pp_i, ep_i, (cp_i + 1) % cp)
                            lc = _scaled_link("cp", not intra_node(g, g2))
                            links.append(TopologyEdge(g, g2, lc))

    return TopologyGraph(n_gpu=n_gpu, links=links, name=config.label)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ParallelismScore:
    """Thermodynamic score for one parallelism configuration."""
    config: ParallelismConfig
    eta_multi: float           # achieved η_multi ∈ [0, 1]
    eta_multi_max: float       # topology Carnot limit ∈ [0, 1]
    eta_hw_fraction: float     # η_multi / η_multi_max ∈ [0, 1]
    comm_overhead: float       # T_comm / max(T_compute, T_comm) ∈ [0, 1]
    resonance_eta: float       # η_overlap ∈ [0, 1]
    dominant_bottleneck: str   # "compute" | "communication" | "balanced"
    thermo_phase: str          # thermodynamic phase name
    comm_volumes: CommVolumes  # per-strategy byte counts

    def summary(self) -> str:
        return (
            f"{self.config.label:<22s}  "
            f"η_multi={self.eta_multi:.4f}  "
            f"({self.eta_hw_fraction * 100:.1f}% of max)  "
            f"comm={self.comm_overhead * 100:.1f}%  "
            f"η_overlap={self.resonance_eta:.3f}  "
            f"phase={self.thermo_phase}"
        )


def score_config(
    config: ParallelismConfig,
    model: ModelParams,
    topology: TopologyGraph | None = None,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    eta_hw_single: float = 0.5,
    peak_tflops: float = H100_PEAK_TFLOPS,
    n_beta: int = 50,
    n_bins: int = 32,
) -> ParallelismScore:
    """
    Score a parallelism configuration by its multi-GPU thermodynamic efficiency.

    Pipeline:
      1. Build topology from config (if not provided).
      2. Estimate communication volumes and timings.
      3. Derive η_multi,max from the topology partition function.
      4. Compute resonance η_overlap.
      5. Compute effective η_multi = η_multi,max × η_overlap.
      6. Identify dominant bottleneck.

    Args:
        config:        Parallelism configuration to score.
        model:         Model hyperparameters.
        topology:      Override topology (default: built from config).
        sm_config:     SM configuration (default: H100).
        memory_levels: Memory hierarchy (default: H100).
        eta_hw_single: Assumed single-GPU η_hw for timing and normalisation.
        peak_tflops:   GPU peak TFLOPS.
        n_beta:        Beta sweep points for partition function.
        n_bins:        Transfer matrix resolution.

    Returns:
        ParallelismScore with all thermodynamic quantities.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS
    if topology is None:
        topology = build_parallelism_topology(config)

    comm_volumes = estimate_comm_volumes(config, model)
    t_compute = estimate_compute_time_s(
        config, model, sm_config, eta_hw_single, peak_tflops,
    )
    t_comm = estimate_comm_time_s(comm_volumes, topology)
    target_comm_load = normalise_comm_demand(comm_volumes.total_bytes, t_compute)

    # η_overlap measures compute–communication balance
    eta_overlap = resonance_condition(t_compute, t_comm, overlap_fraction=1.0)

    # Communication overhead: fraction of wall time in comm
    t_wall = max(t_compute, t_comm, 1e-15)
    comm_overhead = min(t_comm / t_wall, 1.0)

    # Multi-GPU Carnot limit for this topology
    multi_limit = derive_multi_gpu_carnot_limit(
        topology, sm_config, memory_levels,
        eta_hw_max_single=eta_hw_single,
        target_comm_load=target_comm_load,
        n_beta=n_beta, n_bins=n_bins,
    )

    # Effective η_multi = Carnot limit × resonance penalty
    eta_multi = multi_limit.eta_multi_max * eta_overlap
    eta_hw_fraction = eta_multi / max(multi_limit.eta_multi_max, 1e-12)

    # Dominant bottleneck
    if t_comm > t_compute * 1.2:
        bottleneck = "communication"
    elif t_compute > t_comm * 1.2:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    return ParallelismScore(
        config=config,
        eta_multi=eta_multi,
        eta_multi_max=multi_limit.eta_multi_max,
        eta_hw_fraction=eta_hw_fraction,
        comm_overhead=comm_overhead,
        resonance_eta=eta_overlap,
        dominant_bottleneck=bottleneck,
        thermo_phase=config.dominant_phase,
        comm_volumes=comm_volumes,
    )


# ---------------------------------------------------------------------------
# Config enumeration
# ---------------------------------------------------------------------------

def enumerate_configs(
    n_gpu: int,
    model: ModelParams,
    max_tp: int = 8,
    max_pp: int = 8,
    include_ep: bool = False,
    include_cp: bool = False,
) -> list[ParallelismConfig]:
    """
    Enumerate valid parallelism configurations for n_gpu GPUs.

    Validity constraints:
      - dp × tp × pp × ep × cp == n_gpu  (exact GPU count)
      - tp ≤ min(max_tp, n_heads) and n_heads % tp == 0  (head-divisible TP)
      - hidden_dim % tp == 0                              (dim-divisible TP)
      - tp is a power of 2 (hardware alignment)
      - pp is a power of 2 (pipeline scheduling)
      - ep ≤ n_experts and n_experts % ep == 0 (only if include_ep)
      - cp is a power of 2 (only if include_cp)

    Args:
        n_gpu:      Total GPU count.
        model:      Model hyperparameters (used for head/dim constraints).
        max_tp:     Maximum tensor-parallel degree (default 8).
        max_pp:     Maximum pipeline-parallel degree (default 8).
        include_ep: Include expert-parallel configs (requires n_experts > 1).
        include_cp: Include context-parallel configs.

    Returns:
        Deduplicated list of valid configs, sorted by (tp, pp, dp, ep, cp).
    """
    def powers_of_2_up_to(limit: int) -> list[int]:
        vals: list[int] = []
        v = 1
        while v <= limit:
            vals.append(v)
            v *= 2
        return vals

    tp_cap = min(max_tp, n_gpu, model.n_heads, model.hidden_dim)
    tp_values = [
        t for t in powers_of_2_up_to(tp_cap)
        if model.n_heads % t == 0 and model.hidden_dim % t == 0
    ]
    if not tp_values:
        tp_values = [1]

    pp_values = powers_of_2_up_to(min(max_pp, n_gpu))
    if not pp_values:
        pp_values = [1]

    ep_values = [1]
    if include_ep and model.n_experts > 1:
        ep_values = [
            e for e in powers_of_2_up_to(min(model.n_experts, n_gpu))
            if model.n_experts % e == 0
        ]
        if not ep_values:
            ep_values = [1]

    cp_values = [1]
    if include_cp:
        cp_values = powers_of_2_up_to(min(8, n_gpu))
        if not cp_values:
            cp_values = [1]

    seen: set[tuple[int, int, int, int, int]] = set()
    configs: list[ParallelismConfig] = []

    for tp in tp_values:
        for pp in pp_values:
            for ep in ep_values:
                for cp in cp_values:
                    denom = tp * pp * ep * cp
                    if n_gpu % denom != 0:
                        continue
                    dp = n_gpu // denom
                    key = (dp, tp, pp, ep, cp)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append(ParallelismConfig(
                        dp=dp, tp=tp, pp=pp, ep=ep, cp=cp,
                    ))

    configs.sort(key=lambda c: (c.tp, c.pp, c.dp, c.ep, c.cp))
    return configs


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(
    scores: list[ParallelismScore],
) -> list[ParallelismScore]:
    """
    Find Pareto-efficient configurations on (η_multi, 1 − comm_overhead).

    A config is dominated if another config has *both*:
      - η_multi ≥ candidate's  AND
      - comm_overhead ≤ candidate's
    with at least one strict inequality.

    Returns the Pareto frontier sorted by η_multi descending.
    """
    if not scores:
        return []

    frontier: list[ParallelismScore] = []
    for candidate in scores:
        dominated = any(
            other is not candidate
            and other.eta_multi >= candidate.eta_multi
            and other.comm_overhead <= candidate.comm_overhead
            and (other.eta_multi > candidate.eta_multi
                 or other.comm_overhead < candidate.comm_overhead)
            for other in scores
        )
        if not dominated:
            frontier.append(candidate)

    frontier.sort(key=lambda s: s.eta_multi, reverse=True)
    return frontier


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass
class ParallelismOptimResult:
    """Full result of the parallelism optimizer."""
    scores: list[ParallelismScore]          # all evaluated configurations
    pareto_configs: list[ParallelismScore]  # Pareto-efficient subset
    recommended: ParallelismScore           # highest η_multi overall
    multi_gpu_limit: MultiGPUCarnotLimit    # theoretical ceiling (DGX topology)

    def summary(self) -> str:
        lines = [
            f"Parallelism Optimizer  [{self.recommended.config.n_gpu} GPUs]",
            "",
            f"  Theoretical η_multi,max : {self.multi_gpu_limit.eta_multi_max:.4f}",
            f"  Recommended config      : {self.recommended.config.label}",
            f"  Recommended η_multi     : {self.recommended.eta_multi:.4f}",
            f"  Comm overhead           : {self.recommended.comm_overhead * 100:.1f}%",
            f"  η_overlap               : {self.recommended.resonance_eta:.3f}",
            f"  Thermo phase            : {self.recommended.thermo_phase}",
            "",
            f"  Pareto frontier ({len(self.pareto_configs)} configs):",
        ]
        for s in self.pareto_configs:
            lines.append(f"    {s.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main optimizer entry point
# ---------------------------------------------------------------------------

def optimise_parallelism(
    n_gpu: int,
    model: ModelParams,
    topology: TopologyGraph | None = None,
    sm_config: SMConfig | None = None,
    memory_levels: list[MemoryLevel] | None = None,
    max_tp: int = 8,
    max_pp: int = 8,
    include_ep: bool = False,
    include_cp: bool = False,
    eta_hw_single: float = 0.5,
    peak_tflops: float = H100_PEAK_TFLOPS,
    n_beta: int = 50,
    n_bins: int = 32,
) -> ParallelismOptimResult:
    """
    Find the Pareto-optimal parallelism strategy for the given model and GPU count.

    Steps:
      1. Enumerate all valid (dp, tp, pp, ep, cp) configurations.
      2. Score each by η_multi using the topology partition function.
      3. Compute the Pareto frontier on (η_multi, −comm_overhead).
      4. Return all scores, frontier, recommended config, and theoretical limit.

    Args:
        n_gpu:         Number of GPUs.
        model:         Model hyperparameters.
        topology:      Topology for computing the theoretical ceiling.
                       Each config's own topology is built from its strategy.
                       Default: DGX H100 with n_gpu GPUs.
        sm_config:     SM configuration (default: H100).
        memory_levels: Memory hierarchy (default: H100).
        max_tp:        Maximum tensor-parallel degree.
        max_pp:        Maximum pipeline-parallel degree.
        include_ep:    Include expert-parallel configs.
        include_cp:    Include context-parallel configs.
        eta_hw_single: Assumed single-GPU hardware efficiency for timing.
        peak_tflops:   GPU peak TFLOPS (default: H100 989 TFLOPS).
        n_beta:        Beta sweep points for partition function.
        n_bins:        Transfer matrix resolution.

    Returns:
        ParallelismOptimResult with scores, Pareto frontier, and recommendation.
    """
    if sm_config is None:
        sm_config = H100_SM_CONFIG
    if memory_levels is None:
        memory_levels = H100_MEMORY_LEVELS

    # Default ceiling topology: DGX H100 with n_nodes nodes
    if topology is None:
        n_nodes = max(1, (n_gpu + 7) // 8)
        topology = TopologyGraph.dgx_h100(n_nodes)

    configs = enumerate_configs(n_gpu, model, max_tp, max_pp, include_ep, include_cp)
    if not configs:
        configs = [ParallelismConfig(dp=n_gpu)]  # fallback: pure DP

    scores: list[ParallelismScore] = [
        score_config(
            c, model,
            topology=build_parallelism_topology(c),
            sm_config=sm_config,
            memory_levels=memory_levels,
            eta_hw_single=eta_hw_single,
            peak_tflops=peak_tflops,
            n_beta=n_beta,
            n_bins=n_bins,
        )
        for c in configs
    ]

    # Theoretical ceiling: best-case topology (DGX NVLink/NVSwitch)
    multi_limit = derive_multi_gpu_carnot_limit(
        topology, sm_config, memory_levels,
        eta_hw_max_single=eta_hw_single,
        n_beta=n_beta, n_bins=n_bins,
    )

    frontier = pareto_frontier(scores)
    recommended = max(scores, key=lambda s: s.eta_multi)

    return ParallelismOptimResult(
        scores=scores,
        pareto_configs=frontier,
        recommended=recommended,
        multi_gpu_limit=multi_limit,
    )

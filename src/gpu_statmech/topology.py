"""
topology.py — Interconnect topology as an interaction Hamiltonian.

Models the coupling graph between GPU "sites."  Each edge (g, h) carries a
LinkConfig specifying physical bandwidth, latency, and the coupling constant
J_{gh} that appears in the multi-GPU energy functional:

    E(Σ) = Σ_g E_local(σ_g) + Σ_{(g,h)∈edges} J_{gh} · E_comm(σ_g, σ_h)

Coupling constants encode relative communication cost (higher = more expensive):
    NVLink / NVSwitch   J ≈ 0.1   (cheap, ~900 GB/s, ~1 µs)
    PCIe Gen5           J ≈ 1.0   (moderate, ~64 GB/s, ~3.5 µs)
    InfiniBand NDR      J ≈ 5.0   (expensive, ~50 GB/s, ~2 µs + SW overhead)
    Ethernet/RoCE       J ≈ 10.0  (most expensive, high latency)

Per Section 3.6 of the project brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# Interconnect types and physical specs
# ---------------------------------------------------------------------------


class InterconnectType(Enum):
    """Physical interconnect technology between GPUs."""
    NVLINK    = auto()   # Intra-node direct NVLink (peer-to-peer)
    NVSWITCH  = auto()   # Intra-node via NVSwitch fabric (all-to-all)
    PCIE_GEN5 = auto()   # PCIe Gen5 (host-mediated)
    IB_NDR    = auto()   # InfiniBand NDR (inter-node)
    IB_HDR    = auto()   # InfiniBand HDR (inter-node, older)
    ETH_ROCE  = auto()   # Ethernet / RoCE (inter-node)


# Per-link coupling constants J_{gh} (dimensionless, higher = costlier comm).
# Values from Section 3.6 of the project brief.
COUPLING_CONSTANTS: dict[InterconnectType, float] = {
    InterconnectType.NVLINK:    0.1,
    InterconnectType.NVSWITCH:  0.1,
    InterconnectType.PCIE_GEN5: 1.0,
    InterconnectType.IB_NDR:    5.0,
    InterconnectType.IB_HDR:    5.0,
    InterconnectType.ETH_ROCE:  10.0,
}

# Physical specifications: bandwidth (GB/s) and one-way latency (µs).
LINK_SPECS: dict[InterconnectType, dict[str, float]] = {
    InterconnectType.NVLINK:    {"bandwidth_gbps": 900.0, "latency_us": 1.0},
    InterconnectType.NVSWITCH:  {"bandwidth_gbps": 900.0, "latency_us": 1.0},
    InterconnectType.PCIE_GEN5: {"bandwidth_gbps":  64.0, "latency_us": 3.5},
    InterconnectType.IB_NDR:    {"bandwidth_gbps":  50.0, "latency_us": 2.0},
    InterconnectType.IB_HDR:    {"bandwidth_gbps":  25.0, "latency_us": 2.0},
    InterconnectType.ETH_ROCE:  {"bandwidth_gbps":  50.0, "latency_us": 5.0},
}


# ---------------------------------------------------------------------------
# Link configuration
# ---------------------------------------------------------------------------


@dataclass
class LinkConfig:
    """
    Configuration for a single directed GPU-to-GPU link.

    Attributes:
        interconnect_type   Physical technology
        bandwidth_gbps      Peak one-directional bandwidth in GB/s
        latency_us          One-way latency in microseconds
        coupling_constant   J_{gh}: relative communication cost weight
    """
    interconnect_type: InterconnectType
    bandwidth_gbps: float
    latency_us: float
    coupling_constant: float

    @classmethod
    def from_type(cls, itype: InterconnectType) -> LinkConfig:
        """Construct a LinkConfig using the standard specs for a given type."""
        specs = LINK_SPECS[itype]
        return cls(
            interconnect_type=itype,
            bandwidth_gbps=specs["bandwidth_gbps"],
            latency_us=specs["latency_us"],
            coupling_constant=COUPLING_CONSTANTS[itype],
        )

    @property
    def bandwidth_bytes_per_us(self) -> float:
        """Convenience: bandwidth in bytes/µs."""
        return self.bandwidth_gbps * 1e3  # GB/s → MB/µs → bytes/µs needs ×1e9/1e6 = ×1e3

    def transfer_time_us(self, bytes_: int) -> float:
        """Estimated transfer time in µs for a given message size."""
        return self.latency_us + bytes_ / (self.bandwidth_gbps * 1e9 / 1e6)


# ---------------------------------------------------------------------------
# Topology (interaction Hamiltonian graph)
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    """
    Interconnect topology as an interaction Hamiltonian on a GPU graph.

    Nodes are GPU indices 0 … num_gpus-1.
    Edges are undirected (bidirectional) LinkConfig objects.

    The coupling constants J_{gh} define the interaction strength in:
        E_interaction = Σ_{(g,h)∈edges} J_{gh} · E_comm(σ_g, σ_h)

    This is formally analogous to a spin system on an irregular lattice
    with position-dependent coupling constants (Section 3.6).
    """
    num_gpus: int
    # Directed adjacency map: (src, dst) → LinkConfig.
    # add_link() always populates both directions.
    _links: dict[tuple[int, int], LinkConfig] = field(
        default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_link(self, gpu_a: int, gpu_b: int, link: LinkConfig) -> None:
        """Add a bidirectional link between gpu_a and gpu_b."""
        self._check_gpu_id(gpu_a)
        self._check_gpu_id(gpu_b)
        if gpu_a == gpu_b:
            raise ValueError("Cannot add a self-link")
        self._links[(gpu_a, gpu_b)] = link
        self._links[(gpu_b, gpu_a)] = link

    def _check_gpu_id(self, gid: int) -> None:
        if not (0 <= gid < self.num_gpus):
            raise ValueError(f"GPU id {gid} out of range [0, {self.num_gpus})")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def link(self, gpu_a: int, gpu_b: int) -> LinkConfig | None:
        """Return the LinkConfig for edge (gpu_a → gpu_b), or None."""
        return self._links.get((gpu_a, gpu_b))

    def coupling_constant(self, gpu_a: int, gpu_b: int) -> float:
        """
        J_{gh}: coupling constant between GPU g and GPU h.

        Returns inf if the two GPUs are not directly connected
        (no direct link → communication is infinitely expensive in the
        Hamiltonian, i.e., it should not be used for direct transfers).
        """
        lnk = self._links.get((gpu_a, gpu_b))
        return lnk.coupling_constant if lnk is not None else float("inf")

    def neighbors(self, gpu_id: int) -> list[int]:
        """Return sorted list of GPUs directly connected to gpu_id."""
        return sorted(b for (a, b) in self._links if a == gpu_id)

    def edges(self) -> Iterator[tuple[int, int, LinkConfig]]:
        """
        Iterate over undirected edges as (gpu_a, gpu_b, link) with gpu_a < gpu_b.
        """
        seen: set[tuple[int, int]] = set()
        for (a, b), lnk in self._links.items():
            if (b, a) not in seen:
                seen.add((a, b))
                yield (a, b, lnk)

    @property
    def num_edges(self) -> int:
        return len(self._links) // 2

    def mean_coupling(self) -> float:
        """Mean J_{gh} over all edges — overall communication cost pressure."""
        js = [lnk.coupling_constant for _, _, lnk in self.edges()]
        return sum(js) / len(js) if js else 0.0

    def is_fully_connected(self) -> bool:
        expected = self.num_gpus * (self.num_gpus - 1) // 2
        return self.num_edges == expected

    # ------------------------------------------------------------------
    # Preset topology constructors
    # ------------------------------------------------------------------

    @classmethod
    def dgx_h100(cls) -> Topology:
        """
        DGX H100: 8 GPUs fully connected via NVSwitch fabric.

        All 28 pairs share a 900 GB/s bidirectional NVSwitch link with J ≈ 0.1.
        This is the "ferromagnetic" base topology — cheapest all-to-all comm.
        """
        topo = cls(num_gpus=8)
        nvswitch = LinkConfig.from_type(InterconnectType.NVSWITCH)
        for i in range(8):
            for j in range(i + 1, 8):
                topo.add_link(i, j, nvswitch)
        return topo

    @classmethod
    def dgx_a100(cls) -> Topology:
        """
        DGX A100: 8 GPUs connected via NVLink (600 GB/s) pairwise mesh.
        Less dense than NVSwitch; not all pairs have equal bandwidth.
        Modeled here as full mesh at 600 GB/s for simplicity.
        """
        topo = cls(num_gpus=8)
        nvlink = LinkConfig.from_type(InterconnectType.NVLINK)
        # Override to A100 bandwidth
        nvlink_a100 = LinkConfig(
            interconnect_type=InterconnectType.NVLINK,
            bandwidth_gbps=600.0,
            latency_us=1.0,
            coupling_constant=COUPLING_CONSTANTS[InterconnectType.NVLINK],
        )
        for i in range(8):
            for j in range(i + 1, 8):
                topo.add_link(i, j, nvlink_a100)
        return topo

    @classmethod
    def dgx_superpod(cls, num_nodes: int = 32) -> Topology:
        """
        DGX SuperPOD: multiple DGX H100 nodes connected via InfiniBand fat-tree.

        Within each 8-GPU node: NVSwitch (J ≈ 0.1).
        Across nodes: one representative inter-node IB link (J ≈ 5.0).

        GPU numbering: node n → GPUs [n*8 … n*8+7].
        """
        gpus_per_node = 8
        total = num_nodes * gpus_per_node
        topo = cls(num_gpus=total)

        nvswitch = LinkConfig.from_type(InterconnectType.NVSWITCH)
        ib      = LinkConfig.from_type(InterconnectType.IB_NDR)

        for node in range(num_nodes):
            base = node * gpus_per_node
            # Intra-node: full NVSwitch mesh
            for i in range(gpus_per_node):
                for j in range(i + 1, gpus_per_node):
                    topo.add_link(base + i, base + j, nvswitch)
            # Inter-node: connect GPU 0 of each node to GPU 0 of every other node
            for other in range(node + 1, num_nodes):
                topo.add_link(base, other * gpus_per_node, ib)

        return topo

    @classmethod
    def pcie_cluster(cls, num_gpus: int) -> Topology:
        """
        PCIe-only cluster: all GPU pairs connected via PCIe Gen5 (J ≈ 1.0).
        Representative of commodity cloud instances without NVLink.
        """
        topo = cls(num_gpus=num_gpus)
        pcie = LinkConfig.from_type(InterconnectType.PCIE_GEN5)
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                topo.add_link(i, j, pcie)
        return topo

    @classmethod
    def from_adjacency(
        cls,
        num_gpus: int,
        adj: dict[tuple[int, int], InterconnectType],
    ) -> Topology:
        """
        Build a topology from an explicit adjacency dict.

        adj: {(gpu_a, gpu_b): InterconnectType}
        Only one direction per undirected edge is required.
        """
        topo = cls(num_gpus=num_gpus)
        for (a, b), itype in adj.items():
            if (b, a) not in adj or a < b:
                topo.add_link(a, b, LinkConfig.from_type(itype))
        return topo

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Topology(num_gpus={self.num_gpus}, "
            f"num_edges={self.num_edges}, "
            f"mean_coupling={self.mean_coupling():.2f})"
        )

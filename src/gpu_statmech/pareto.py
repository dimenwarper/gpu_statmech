"""
Pareto frontier utilities for the architecture search optimisation loop (Phase 3).

The optimisation loop maximises two objectives simultaneously:

  1. **η_hw / η_hw,max**  (thermodynamic efficiency, from Carnot checker)
  2. **expressiveness**   (proxy representational power, from compiler)

A kernel A dominates kernel B if A is at least as good as B on *all*
objectives and strictly better on *at least one*.  The Pareto frontier is
the maximal antichain under dominance — the set of proposals that are not
dominated by any other.

The module provides:

  • ``ParetoPoint``        — a wrapper pairing a ``CompiledKernel`` with its
                            two-objective coordinates.
  • ``is_dominated``       — dominance predicate for a pair of points.
  • ``pareto_frontier``    — non-dominated subset of a list of ParetoPoints.
  • ``hypervolume_2d``     — hypervolume indicator relative to a reference
                            point (standard convergence metric for MOO).
  • ``crowding_distance``  — NSGA-II crowding distance for diversity
                            preservation in selection.
  • ``pareto_summary``     — human-readable summary of a frontier.

All functions operate on plain ``ParetoPoint`` objects and are independent
of the oracle/compiler internals, so they can be used in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .compiler import CompiledKernel


# ---------------------------------------------------------------------------
# ParetoPoint
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """
    A point in the two-objective Pareto space.

    Attributes
    ----------
    kernel:
        The compiled kernel this point represents.
    eta_fraction:
        η_hw / η_hw,max  ∈ [0, 1]  (thermodynamic efficiency objective).
    expressiveness:
        Proxy expressiveness score  ∈ [0, 1]  (representational power objective).
    """
    kernel: CompiledKernel
    eta_fraction: float
    expressiveness: float

    @property
    def objectives(self) -> tuple[float, float]:
        """(eta_fraction, expressiveness) as a tuple."""
        return (self.eta_fraction, self.expressiveness)

    @classmethod
    def from_compiled(cls, ck: CompiledKernel) -> "ParetoPoint":
        """Construct from a ``CompiledKernel`` using its built-in scores."""
        return cls(
            kernel=ck,
            eta_fraction=ck.thermo_score,
            expressiveness=ck.expressiveness_score,
        )


# ---------------------------------------------------------------------------
# Dominance predicate
# ---------------------------------------------------------------------------

def is_dominated(point: ParetoPoint, other: ParetoPoint) -> bool:
    """
    Return True if ``point`` is dominated by ``other``.

    ``other`` dominates ``point`` if:
      • other.eta_fraction  ≥ point.eta_fraction, AND
      • other.expressiveness ≥ point.expressiveness, AND
      • at least one inequality is strict.

    Parameters
    ----------
    point:
        The candidate point to test.
    other:
        The potential dominator.

    Returns
    -------
    bool
    """
    a_eta, a_expr = point.objectives
    b_eta, b_expr = other.objectives

    at_least_as_good = (b_eta >= a_eta) and (b_expr >= a_expr)
    strictly_better  = (b_eta > a_eta)  or  (b_expr > a_expr)
    return at_least_as_good and strictly_better


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """
    Compute the non-dominated Pareto frontier.

    Uses a simple O(n²) sweep — adequate for the batch sizes used in the
    optimisation loop (typically n ≤ 500 accumulated proposals).

    Parameters
    ----------
    points:
        All candidate points from the current and past iterations.

    Returns
    -------
    list[ParetoPoint]
        The non-dominated subset, sorted by increasing ``eta_fraction``.
    """
    if not points:
        return []

    pts = list(points)
    frontier: list[ParetoPoint] = []

    for candidate in pts:
        dominated = any(is_dominated(candidate, other) for other in pts if other is not candidate)
        if not dominated:
            frontier.append(candidate)

    # Sort by eta_fraction ascending for consistent iteration
    frontier.sort(key=lambda p: p.eta_fraction)
    return frontier


# ---------------------------------------------------------------------------
# Hypervolume indicator (2-D)
# ---------------------------------------------------------------------------

def hypervolume_2d(
    frontier: Sequence[ParetoPoint],
    reference: tuple[float, float] = (0.0, 0.0),
) -> float:
    """
    Compute the 2-D hypervolume indicator for a Pareto frontier.

    The hypervolume is the area of the objective space dominated by the
    frontier and bounded by ``reference`` from below.  It is the standard
    scalar convergence metric for multi-objective optimisation:

      • Higher → frontier covers more of the objective space.
      • Increases monotonically as the frontier improves.
      • Converges when successive iterations add < ε to the hypervolume.

    Parameters
    ----------
    frontier:
        The non-dominated Pareto frontier (need not be sorted).
    reference:
        Lower-bound reference point in objective space.  Typically (0, 0).

    Returns
    -------
    float
        Hypervolume indicator ≥ 0.
    """
    if not frontier:
        return 0.0

    ref_eta, ref_expr = reference

    # Sort by eta_fraction ascending
    pts = sorted(frontier, key=lambda p: p.eta_fraction)

    # Compute area as sum of axis-aligned rectangles between consecutive
    # frontier points and the reference level.
    hv = 0.0
    # Track the highest expressiveness seen so far (monotone front)
    max_expr_seen = ref_expr

    for i, pt in enumerate(pts):
        if pt.eta_fraction <= ref_eta:
            continue
        # Width: horizontal span of this point's dominance region
        if i + 1 < len(pts):
            width = pts[i + 1].eta_fraction - pt.eta_fraction
        else:
            # Last point: no right neighbour on the frontier;
            # its contribution ends at the frontier boundary (eta = 1.0)
            width = 1.0 - pt.eta_fraction

        # Height: how much this point's expressiveness exceeds the reference
        height = max(pt.expressiveness - ref_expr, 0.0)

        hv += width * height

    # Also add the leftmost rectangle from ref_eta to first frontier point
    # with height = first point's expressiveness
    if pts:
        first = pts[0]
        if first.eta_fraction > ref_eta:
            # Covered by first point sweeping left to ref_eta
            left_width  = first.eta_fraction - ref_eta
            left_height = max(first.expressiveness - ref_expr, 0.0)
            hv += left_width * left_height

    return float(hv)


def _hypervolume_2d_exact(
    frontier: Sequence[ParetoPoint],
    reference: tuple[float, float] = (0.0, 0.0),
) -> float:
    """
    Exact 2-D hypervolume using the standard sweep algorithm.

    Replaces the approximate version above with the correct staircase
    integral over sorted non-dominated points.

    Parameters
    ----------
    frontier:
        Non-dominated Pareto frontier.
    reference:
        Reference (nadir) point, dominated by all frontier points.

    Returns
    -------
    float
        Hypervolume indicator.
    """
    if not frontier:
        return 0.0

    ref_eta, ref_expr = reference
    # Sort by eta descending
    pts = sorted(frontier, key=lambda p: p.eta_fraction, reverse=True)

    hv = 0.0
    prev_expr = ref_expr

    for pt in pts:
        if pt.eta_fraction <= ref_eta:
            continue
        height = max(pt.expressiveness - prev_expr, 0.0)
        width  = pt.eta_fraction - ref_eta
        hv += width * height
        prev_expr = max(prev_expr, pt.expressiveness)

    return float(hv)


# Use the exact implementation as the public API
hypervolume_2d = _hypervolume_2d_exact  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Crowding distance (NSGA-II)
# ---------------------------------------------------------------------------

def crowding_distance(frontier: Sequence[ParetoPoint]) -> NDArray:
    """
    Compute the NSGA-II crowding distance for each point on the frontier.

    Crowding distance measures how isolated a point is in objective space.
    Points with large crowding distances are in sparse regions of the
    frontier — they are preferred during selection to maintain diversity.

    Boundary points (minimum and maximum in any objective) receive
    infinite distance.

    Parameters
    ----------
    frontier:
        The Pareto frontier (need not be sorted).

    Returns
    -------
    NDArray, shape (n,)
        Crowding distance for each point, in the same order as ``frontier``.
        Infinite for boundary points.
    """
    n = len(frontier)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([np.inf])
    if n == 2:
        return np.array([np.inf, np.inf])

    pts = list(frontier)
    distances = np.zeros(n, dtype=float)

    for obj_idx in range(2):  # eta_fraction and expressiveness
        values = np.array([p.objectives[obj_idx] for p in pts])
        sorted_idx = np.argsort(values)

        # Boundary points get infinite distance
        distances[sorted_idx[0]]  = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = values[sorted_idx[-1]] - values[sorted_idx[0]]
        if obj_range < 1e-12:
            continue  # all values identical; skip

        for rank in range(1, n - 1):
            i_prev = sorted_idx[rank - 1]
            i_next = sorted_idx[rank + 1]
            distances[sorted_idx[rank]] += (
                (values[i_next] - values[i_prev]) / obj_range
            )

    return distances


# ---------------------------------------------------------------------------
# Pareto summary
# ---------------------------------------------------------------------------

def pareto_summary(
    frontier: Sequence[ParetoPoint],
    hypervolume: float | None = None,
) -> str:
    """
    Return a human-readable summary of a Pareto frontier.

    Parameters
    ----------
    frontier:
        The non-dominated frontier.
    hypervolume:
        Pre-computed hypervolume (computed from ``frontier`` if None).

    Returns
    -------
    str
        Multi-line summary string.
    """
    if not frontier:
        return "Pareto frontier: empty."

    if hypervolume is None:
        hypervolume = _hypervolume_2d_exact(frontier)

    etas  = [p.eta_fraction   for p in frontier]
    exprs = [p.expressiveness for p in frontier]

    best_eta  = max(etas)
    best_expr = max(exprs)
    n_carnot  = sum(1 for p in frontier if p.kernel.is_carnot_optimal)

    # Best combined score
    best_combined = max(p.kernel.combined_score for p in frontier)
    best_overall  = max(frontier, key=lambda p: p.kernel.combined_score)

    lines = [
        f"Pareto frontier: {len(frontier)} non-dominated points",
        f"  Hypervolume indicator     : {hypervolume:.6f}",
        f"  Best η_hw / η_hw,max      : {best_eta:.4f}",
        f"  Best expressiveness       : {best_expr:.4f}",
        f"  Best combined score       : {best_combined:.4f}  [{best_overall.kernel.proposal.name}]",
        f"  Carnot-optimal points     : {n_carnot}/{len(frontier)}",
        "",
        "  Frontier points (η, expr, combined):",
    ]

    sorted_front = sorted(frontier, key=lambda p: p.eta_fraction)
    for p in sorted_front:
        lines.append(
            f"    {p.kernel.proposal.name:<30s} "
            f"η={p.eta_fraction:.3f}  "
            f"expr={p.expressiveness:.3f}  "
            f"combined={p.kernel.combined_score:.3f}"
            + (" ✓" if p.kernel.is_carnot_optimal else "")
        )

    return "\n".join(lines)

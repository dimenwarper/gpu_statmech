"""
Thermodynamic architecture search optimisation loop (Phase 3).

Implements the closed-loop search over kernel architectures described in
the project brief:

  1. Oracle proposes a batch of N kernel architectures.
  2. Compiler scores each: η_hw, expressiveness, waste decomposition.
  3. Proposals above the η threshold are retained as Pareto candidates.
  4. The Pareto frontier is updated over all accumulated proposals.
  5. Physics-grounded feedback is computed and fed back to the oracle.
  6. Repeat until the Pareto hypervolume converges.

Key classes
-----------
LoopConfig:
    Hyper-parameters for the loop (batch size, thresholds, convergence).

LoopState:
    Snapshot of the search state at the end of one iteration, including
    the accumulated kernel pool, current Pareto frontier, and convergence
    diagnostics.

OptimisationLoop:
    Orchestrates oracle → compiler → pareto → feedback, exposing both a
    single-step ``step()`` interface and a full ``run()`` interface.

Convergence criterion
---------------------
The loop is considered converged when the Pareto hypervolume has changed
by less than ``convergence_tolerance`` for ``patience`` consecutive
iterations.  The hypervolume is ∈ [0, 1] for the unit objective square.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .carnot import CarnotLimit, derive_carnot_limit
from .compiler import CompiledKernel, KernelCompiler
from .oracle import KernelProposal, PhysicsOracle
from .pareto import ParetoPoint, hypervolume_2d, pareto_frontier, pareto_summary
from .partition_function import H100_MEMORY_LEVELS, H100_SM_CONFIG, MemoryLevel, SMConfig


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    """
    Hyper-parameters for the optimisation loop.

    Attributes
    ----------
    n_proposals_per_iter:
        Number of kernel proposals generated per iteration.
    eta_threshold:
        Minimum η_hw / η_hw,max required for a proposal to be retained
        as a Pareto candidate.  Proposals below this are discarded but
        still used for oracle feedback.
    max_iterations:
        Hard upper bound on the number of iterations.
    convergence_tolerance:
        The loop stops early when the absolute change in Pareto hypervolume
        is below this value for ``patience`` consecutive iterations.
    patience:
        Number of consecutive below-tolerance iterations before declaring
        convergence.
    seed:
        Random seed for the oracle's proposal generator.  ``None`` → non-
        deterministic.
    verbose:
        If True, print a per-iteration summary to stdout.
    """
    n_proposals_per_iter: int   = 30
    eta_threshold: float        = 0.3
    max_iterations: int         = 20
    convergence_tolerance: float = 1e-4
    patience: int               = 3
    seed: int | None            = 42
    verbose: bool               = False


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    """
    Snapshot of the optimisation loop at the end of iteration ``iteration``.

    Attributes
    ----------
    iteration:
        Completed iteration index (0-based).
    all_kernels:
        All compiled kernels accumulated so far (including sub-threshold ones).
    pareto_front:
        Current Pareto frontier over all accumulated kernels.
    hypervolume:
        Pareto hypervolume indicator at this iteration.
    hypervolume_history:
        Hypervolume after each completed iteration (length = iteration + 1).
    best_eta:
        Highest η_hw / η_hw,max seen so far.
    best_expressiveness:
        Highest expressiveness score seen so far.
    best_combined:
        Highest combined score (η + expressiveness) seen so far.
    converged:
        True if the convergence criterion was satisfied at this iteration.
    feedback_message:
        The compiler's feedback message from the most recent batch.
    """
    iteration: int
    all_kernels: list[CompiledKernel]
    pareto_front: list[ParetoPoint]
    hypervolume: float
    hypervolume_history: list[float]
    best_eta: float
    best_expressiveness: float
    best_combined: float
    converged: bool
    feedback_message: str = ""

    @property
    def n_carnot_optimal(self) -> int:
        """Number of Carnot-optimal kernels on the current frontier."""
        return sum(1 for p in self.pareto_front if p.kernel.is_carnot_optimal)

    @property
    def frontier_size(self) -> int:
        """Number of points on the current Pareto frontier."""
        return len(self.pareto_front)

    def summary(self) -> str:
        """Return a concise one-line summary of this loop state."""
        return (
            f"Iter {self.iteration:3d} | "
            f"pool={len(self.all_kernels):4d} | "
            f"frontier={self.frontier_size:3d} | "
            f"HV={self.hypervolume:.6f} | "
            f"best η={self.best_eta:.3f} | "
            f"best expr={self.best_expressiveness:.3f} | "
            f"Carnot✓={self.n_carnot_optimal}"
            + (" [CONVERGED]" if self.converged else "")
        )


# ---------------------------------------------------------------------------
# OptimisationLoop
# ---------------------------------------------------------------------------

class OptimisationLoop:
    """
    Closed-loop thermodynamic architecture search.

    Orchestrates the oracle → compile → Pareto → feedback cycle.

    Parameters
    ----------
    carnot_limit:
        The target hardware's derived Carnot limit.  Shared by the oracle
        and compiler.
    config:
        Loop hyper-parameters.
    memory_levels:
        Memory hierarchy (defaults to H100).
    sm_config:
        SM configuration (defaults to H100).
    on_iteration:
        Optional callback invoked at the end of each iteration with the
        current ``LoopState``.  Useful for live plotting or logging.

    Examples
    --------
    >>> from gpu_statmech.carnot import derive_carnot_limit
    >>> from gpu_statmech.loop import OptimisationLoop, LoopConfig
    >>> limit = derive_carnot_limit()
    >>> loop = OptimisationLoop(limit, LoopConfig(n_proposals_per_iter=20, max_iterations=5))
    >>> state = loop.run()
    >>> print(state.summary())
    """

    def __init__(
        self,
        carnot_limit: CarnotLimit,
        config: LoopConfig | None = None,
        memory_levels: list[MemoryLevel] | None = None,
        sm_config: SMConfig | None = None,
        on_iteration: Callable[[LoopState], None] | None = None,
    ) -> None:
        self.carnot_limit  = carnot_limit
        self.config        = config or LoopConfig()
        self.memory_levels = memory_levels or H100_MEMORY_LEVELS
        self.sm_config     = sm_config or H100_SM_CONFIG
        self.on_iteration  = on_iteration

        self._oracle   = PhysicsOracle(
            carnot_limit=carnot_limit,
            memory_levels=self.memory_levels,
            sm_config=self.sm_config,
        )
        self._compiler = KernelCompiler(
            carnot_limit=carnot_limit,
            memory_levels=self.memory_levels,
            sm_config=self.sm_config,
        )

        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)
        self._all_kernels: list[CompiledKernel] = []
        self._hypervolume_history: list[float] = []
        self._iteration: int = 0
        self._below_tol_count: int = 0
        self._last_hv: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> LoopState:
        """
        Execute one iteration of the optimisation loop.

        Returns
        -------
        LoopState
            State snapshot at the end of this iteration.
        """
        cfg = self.config

        # 1. Propose
        proposals = self._oracle.propose(n=cfg.n_proposals_per_iter, rng=self._rng)

        # 2. Compile and score
        compiled_batch = self._compiler.batch_compile(proposals)
        self._all_kernels.extend(compiled_batch)

        # 3. Compute scores for oracle feedback (use combined score)
        scores = [ck.combined_score for ck in compiled_batch]

        # 4. Oracle feedback (uses all proposals, not just above-threshold)
        self._oracle.feedback(proposals, scores)

        # 5. Build Pareto frontier over ALL accumulated kernels
        #    (filter by eta_threshold for Pareto candidacy)
        candidates = [
            ParetoPoint.from_compiled(ck)
            for ck in self._all_kernels
            if ck.thermo_score >= cfg.eta_threshold
        ]
        front = pareto_frontier(candidates)
        hv = hypervolume_2d(front)

        # 6. Convergence check
        delta_hv = abs(hv - self._last_hv)
        if delta_hv < cfg.convergence_tolerance:
            self._below_tol_count += 1
        else:
            self._below_tol_count = 0
        converged = self._below_tol_count >= cfg.patience

        self._last_hv = hv
        self._hypervolume_history.append(hv)
        self._iteration += 1

        # 7. Best-so-far stats
        best_eta   = max((ck.thermo_score        for ck in self._all_kernels), default=0.0)
        best_expr  = max((ck.expressiveness_score for ck in self._all_kernels), default=0.0)
        best_comb  = max((ck.combined_score       for ck in self._all_kernels), default=0.0)

        # 8. Feedback message
        fb_msg = self._compiler.feedback_message(compiled_batch)

        state = LoopState(
            iteration=self._iteration - 1,
            all_kernels=list(self._all_kernels),
            pareto_front=front,
            hypervolume=hv,
            hypervolume_history=list(self._hypervolume_history),
            best_eta=best_eta,
            best_expressiveness=best_expr,
            best_combined=best_comb,
            converged=converged,
            feedback_message=fb_msg,
        )

        if cfg.verbose:
            print(state.summary())

        if self.on_iteration is not None:
            self.on_iteration(state)

        return state

    def run(self, n_iterations: int | None = None) -> LoopState:
        """
        Run the optimisation loop to completion.

        Runs until either ``n_iterations`` is reached, ``max_iterations``
        is reached, or the convergence criterion is satisfied.

        Parameters
        ----------
        n_iterations:
            Override for the number of iterations.  If None, uses
            ``config.max_iterations``.

        Returns
        -------
        LoopState
            Final state after the last iteration.
        """
        max_iter = n_iterations if n_iterations is not None else self.config.max_iterations
        state: LoopState | None = None

        for _ in range(max_iter):
            state = self.step()
            if state.converged:
                break

        if state is None:
            # Zero iterations requested — return empty state
            state = LoopState(
                iteration=-1,
                all_kernels=[],
                pareto_front=[],
                hypervolume=0.0,
                hypervolume_history=[],
                best_eta=0.0,
                best_expressiveness=0.0,
                best_combined=0.0,
                converged=False,
            )

        return state

    def reset(self) -> None:
        """Reset the loop and oracle to their initial state."""
        self._oracle.reset()
        self._rng = np.random.default_rng(self.config.seed)
        self._all_kernels.clear()
        self._hypervolume_history.clear()
        self._iteration = 0
        self._below_tol_count = 0
        self._last_hv = 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def convergence_report(self) -> str:
        """
        Return a formatted convergence report over all completed iterations.

        Shows the hypervolume history and highlights where convergence
        was achieved.
        """
        if not self._hypervolume_history:
            return "No iterations completed."

        lines = [
            "=== Convergence Report ===",
            f"  Iterations completed : {self._iteration}",
            f"  Final hypervolume    : {self._last_hv:.6f}",
            f"  Total kernels scored : {len(self._all_kernels)}",
            "",
            "  Hypervolume per iteration:",
        ]
        prev = 0.0
        for i, hv in enumerate(self._hypervolume_history):
            delta = hv - prev
            mark  = " *" if abs(delta) < self.config.convergence_tolerance else ""
            lines.append(f"    [{i:3d}]  HV = {hv:.6f}  Δ = {delta:+.6f}{mark}")
            prev = hv

        patience_str = f"(patience = {self.config.patience})"
        if self._below_tol_count >= self.config.patience:
            lines.append(f"\n  Converged ✓  {patience_str}")
        else:
            lines.append(f"\n  Not yet converged  {patience_str}")

        return "\n".join(lines)

    def best_kernels(self, n: int = 5) -> list[CompiledKernel]:
        """
        Return the top-n kernels by combined score from the accumulated pool.

        Parameters
        ----------
        n:
            Number of top kernels to return.

        Returns
        -------
        list[CompiledKernel]
            Top-n kernels, sorted by combined score descending.
        """
        return sorted(self._all_kernels, key=lambda ck: ck.combined_score, reverse=True)[:n]

    def pareto_report(self) -> str:
        """
        Return a detailed Pareto frontier report for the current state.
        """
        candidates = [
            ParetoPoint.from_compiled(ck)
            for ck in self._all_kernels
            if ck.thermo_score >= self.config.eta_threshold
        ]
        front = pareto_frontier(candidates)
        hv    = hypervolume_2d(front)
        return pareto_summary(front, hypervolume=hv)

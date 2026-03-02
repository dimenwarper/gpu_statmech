"""
free_energy.py — Free energy F = E_hardware(σ) – T · S_model(θ)

The joint objective trading off hardware efficiency against model expressiveness.
Minimizing F finds architectures on the Pareto frontier of utilization and
expressiveness.

    F = E_hardware – T · S_model

where:
    E_hardware  scalar energy from EnergyFunctional or MultiGPUEnergyFunctional
    T           temperature — controls exploration vs. exploitation
    S_model     expressiveness score ∈ [0, 1] (higher = more capable model)

Temperature annealing schedule:
    High T  → favours expressive but hardware-inefficient architectures
    Low T   → favours hardware-native but potentially less capable architectures

Per Sections 3 and 4.4 of the project brief.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

from .microstate import Microstate
from .multi_gpu import GlobalMicrostate, MultiGPUEnergyFunctional
from .energy import EnergyFunctional


# ---------------------------------------------------------------------------
# Annealing schedules
# ---------------------------------------------------------------------------


class AnnealingSchedule:
    """Base class for temperature annealing schedules."""

    def temperature_at(self, step: int) -> float:
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.temperature_at(step)


@dataclass
class GeometricAnnealing(AnnealingSchedule):
    """
    T(k) = T_initial · factor^k

    The default schedule: multiply temperature by `factor` at each iteration.
    Reaches T_min in log(T_min / T_initial) / log(factor) steps.
    """
    initial_temperature: float = 1.0
    factor: float = 0.9
    min_temperature: float = 1e-3

    def __post_init__(self) -> None:
        if not (0.0 < self.factor < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {self.factor}")
        if self.min_temperature <= 0:
            raise ValueError("min_temperature must be > 0")

    def temperature_at(self, step: int) -> float:
        t = self.initial_temperature * (self.factor ** step)
        return max(t, self.min_temperature)


@dataclass
class CosineAnnealing(AnnealingSchedule):
    """
    T(k) = T_min + 0.5·(T_max – T_min)·(1 + cos(π·k/K))

    Smooth cosine decay from T_max to T_min over K steps.
    """
    max_temperature: float = 1.0
    min_temperature: float = 1e-3
    total_steps: int = 20

    def temperature_at(self, step: int) -> float:
        k = min(step, self.total_steps)
        cosine = math.cos(math.pi * k / self.total_steps)
        return self.min_temperature + 0.5 * (
            self.max_temperature - self.min_temperature
        ) * (1.0 + cosine)


@dataclass
class LinearAnnealing(AnnealingSchedule):
    """T(k) = T_max – (T_max – T_min) · k/K"""
    max_temperature: float = 1.0
    min_temperature: float = 1e-3
    total_steps: int = 20

    def temperature_at(self, step: int) -> float:
        k = min(step, self.total_steps)
        return self.max_temperature - (
            self.max_temperature - self.min_temperature
        ) * k / self.total_steps


# ---------------------------------------------------------------------------
# Free energy
# ---------------------------------------------------------------------------


@dataclass
class FreeEnergy:
    """
    F = E_hardware(σ) – T · S_model(θ)

    Computes the joint hardware-expressiveness objective for a candidate
    architecture.  Used by the optimization loop to drive Pareto-frontier
    search.

    Attributes:
        temperature             Current temperature T (updated by anneal())
        single_gpu_energy_fn    EnergyFunctional for single-GPU states
        multi_gpu_energy_fn     MultiGPUEnergyFunctional for cluster states
        annealing_schedule      Optional schedule for automatic T updates
        _step                   Current optimization iteration (auto-incremented
                                by anneal())
    """
    temperature: float = 1.0
    single_gpu_energy_fn: EnergyFunctional = field(
        default_factory=EnergyFunctional
    )
    multi_gpu_energy_fn: MultiGPUEnergyFunctional = field(
        default_factory=MultiGPUEnergyFunctional
    )
    annealing_schedule: AnnealingSchedule = field(
        default_factory=GeometricAnnealing
    )
    _step: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Hardware energy extraction
    # ------------------------------------------------------------------

    def hardware_energy(
        self,
        hardware_state: Union[Microstate, GlobalMicrostate],
    ) -> float:
        """Extract E_hardware from a single-GPU or multi-GPU state."""
        if isinstance(hardware_state, GlobalMicrostate):
            return self.multi_gpu_energy_fn.compute(hardware_state)["total"]
        return self.single_gpu_energy_fn.compute(hardware_state)

    # ------------------------------------------------------------------
    # Free energy computation
    # ------------------------------------------------------------------

    def compute(
        self,
        hardware_state: Union[Microstate, GlobalMicrostate],
        expressiveness_score: float,
    ) -> float:
        """
        F = E_hardware – T · S_model

        Args:
            hardware_state       Single-GPU Microstate or GlobalMicrostate
            expressiveness_score S_model(θ) ∈ [0, 1]; higher = more expressive

        Returns:
            Scalar free energy.  Lower F → better architecture candidate.
        """
        e_hw = self.hardware_energy(hardware_state)
        return e_hw - self.temperature * expressiveness_score

    def decompose(
        self,
        hardware_state: Union[Microstate, GlobalMicrostate],
        expressiveness_score: float,
    ) -> dict[str, float]:
        """
        Return F broken down into its hardware and expressiveness components.

        {
            "E_hardware":     raw hardware energy,
            "T":              current temperature,
            "S_model":        expressiveness score,
            "T_times_S":      T · S_model (the entropic term),
            "F":              total free energy,
        }
        """
        e_hw = self.hardware_energy(hardware_state)
        t_s  = self.temperature * expressiveness_score
        return {
            "E_hardware": e_hw,
            "T":          self.temperature,
            "S_model":    expressiveness_score,
            "T_times_S":  t_s,
            "F":          e_hw - t_s,
        }

    # ------------------------------------------------------------------
    # Pareto analysis
    # ------------------------------------------------------------------

    def pareto_frontier(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Extract the Pareto frontier from a list of candidate dictionaries.

        Each candidate must have keys:
            "E_hardware"  (float, lower is better — hardware efficiency)
            "S_model"     (float, higher is better — expressiveness)
            + any additional metadata fields (passed through unchanged)

        A candidate c dominates c' iff:
            c["E_hardware"] ≤ c'["E_hardware"]  AND
            c["S_model"]    ≥ c'["S_model"]
        with at least one strict inequality.

        Returns the non-dominated subset, sorted by ascending E_hardware.
        """
        if not candidates:
            return []

        frontier = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is c:
                    continue
                # other dominates c if it is at least as good on both axes
                # and strictly better on at least one
                hw_leq = other["E_hardware"] <= c["E_hardware"]
                ex_geq = other["S_model"]    >= c["S_model"]
                hw_lt  = other["E_hardware"] <  c["E_hardware"]
                ex_gt  = other["S_model"]    >  c["S_model"]
                if hw_leq and ex_geq and (hw_lt or ex_gt):
                    dominated = True
                    break
            if not dominated:
                frontier.append(c)

        return sorted(frontier, key=lambda c: c["E_hardware"])

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def anneal(self) -> float:
        """
        Advance the annealing schedule by one step.

        Updates self.temperature using the configured schedule and increments
        the internal step counter.

        Returns:
            New temperature after annealing.
        """
        self._step += 1
        self.temperature = self.annealing_schedule.temperature_at(self._step)
        return self.temperature

    def reset(self) -> None:
        """Reset temperature and step counter to initial values."""
        self._step = 0
        self.temperature = self.annealing_schedule.temperature_at(0)

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_cold(self) -> bool:
        """True when temperature is effectively zero (pure hardware optimisation)."""
        return self.temperature < 1e-3

    @property
    def is_hot(self) -> bool:
        """True when temperature is high (favour expressiveness / exploration)."""
        return self.temperature >= 0.5

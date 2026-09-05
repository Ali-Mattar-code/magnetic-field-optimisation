"""Power-error Pareto sweeps and knee selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrical import ohmic_power
from .metrics import normalised_rmse
from .optimisation import OptimisationResult, solve_currents


@dataclass(frozen=True)
class ParetoPoint:
    regularisation: float
    power_w: float
    nrmse: float
    peak_current_a: float
    result: OptimisationResult


def sweep(
    regularisations: np.ndarray,
    influence: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    resistance: np.ndarray,
    **solver_kwargs: object,
) -> list[ParetoPoint]:
    points: list[ParetoPoint] = []
    for value in np.asarray(regularisations, dtype=float):
        result = solve_currents(
            influence,
            target,
            weights,
            resistance,
            regularisation=float(value),
            **solver_kwargs,
        )
        realised = influence @ result.currents
        points.append(
            ParetoPoint(
                regularisation=float(value),
                power_w=ohmic_power(result.currents, resistance),
                nrmse=normalised_rmse(realised, target),
                peak_current_a=float(np.max(np.abs(result.currents))),
                result=result,
            )
        )
    return points


def knee_index(points: list[ParetoPoint]) -> int:
    if len(points) < 3:
        return 0
    power = np.asarray([p.power_w for p in points])
    error = np.asarray([p.nrmse for p in points])
    power = (power - power.min()) / max(np.ptp(power), 1e-15)
    error = (error - error.min()) / max(np.ptp(error), 1e-15)
    return int(np.argmin(np.hypot(power, error)))


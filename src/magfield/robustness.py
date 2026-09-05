"""Monte Carlo non-ideality analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Coil
from .physics import influence_matrix


@dataclass(frozen=True)
class RobustnessResult:
    target_errors: np.ndarray
    powers: np.ndarray
    median_error: float
    p95_error: float


def monte_carlo(
    coils: list[Coil],
    target_point: np.ndarray,
    currents: np.ndarray,
    resistances: np.ndarray,
    target_value: float,
    *,
    draws: int,
    current_sigma_fraction: float,
    resistance_sigma_fraction: float,
    position_sigma_m: float,
    tilt_sigma_deg: float,
    seed: int,
) -> RobustnessResult:
    """Perturb control and manufacture variables and re-evaluate target field.

    The quick workflow approximates small pose error as an effective observation-point
    displacement. This captures first-order sensitivity without rebuilding every coil mesh.
    """
    rng = np.random.default_rng(seed)
    target_point = np.asarray(target_point, dtype=float).reshape(1, 3)
    currents = np.asarray(currents, dtype=float)
    diagonal_r = np.diag(resistances)
    errors = np.empty(draws)
    powers = np.empty(draws)
    tilt_scale = np.deg2rad(tilt_sigma_deg) * np.mean([c.radius for c in coils])
    for index in range(draws):
        perturbed_i = currents * (1.0 + rng.normal(0.0, current_sigma_fraction, currents.size))
        perturbed_r = diagonal_r * (
            1.0 + rng.normal(0.0, resistance_sigma_fraction, currents.size)
        )
        point_shift = rng.normal(0.0, position_sigma_m, 3)
        point_shift[:2] += rng.normal(0.0, tilt_scale, 2)
        target_row = influence_matrix(target_point + point_shift, coils, component=2)[0]
        realised = float(target_row @ perturbed_i)
        errors[index] = abs(realised - target_value) / abs(target_value)
        powers[index] = float(np.sum(perturbed_r * perturbed_i**2))
    return RobustnessResult(
        target_errors=errors,
        powers=powers,
        median_error=float(np.median(errors)),
        p95_error=float(np.quantile(errors, 0.95)),
    )


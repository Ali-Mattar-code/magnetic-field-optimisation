"""Reference validations for the numerical field model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import Coil
from .physics import analytic_on_axis, loop_field


@dataclass(frozen=True)
class ValidationSummary:
    single_loop_max_relative_error: float
    helmholtz_centre_relative_error: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def run_validation(segments: int = 256) -> ValidationSummary:
    radius = 0.04
    coil = Coil(centre=(0.0, 0.0, 0.0), radius=radius, segments=segments)
    z = np.linspace(0.005, 0.10, 60)
    points = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    numerical = loop_field(points, coil, 1.0)[:, 2]
    analytical = analytic_on_axis(z, radius, 1.0)
    loop_error = float(np.max(np.abs(numerical - analytical) / np.abs(analytical)))

    separation = radius
    first = Coil(centre=(0.0, 0.0, -separation / 2), radius=radius, segments=segments)
    second = Coil(centre=(0.0, 0.0, separation / 2), radius=radius, segments=segments)
    centre = np.array([[0.0, 0.0, 0.0]])
    numerical_pair = float(loop_field(centre, first)[0, 2] + loop_field(centre, second)[0, 2])
    analytical_pair = float(2.0 * analytic_on_axis(separation / 2, radius, 1.0))
    helmholtz_error = abs(numerical_pair - analytical_pair) / abs(analytical_pair)
    return ValidationSummary(
        single_loop_max_relative_error=loop_error,
        helmholtz_centre_relative_error=float(helmholtz_error),
        passed=loop_error < 0.01 and helmholtz_error < 0.01,
    )


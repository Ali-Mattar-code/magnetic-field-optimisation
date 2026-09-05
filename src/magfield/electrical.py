"""Electrical quantities kept distinct from field-shaping losses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import Coil
from .physics import MU0


def coil_resistance(coil: Coil) -> float:
    area = np.pi * coil.wire_radius**2
    return float(coil.resistivity * coil.conductor_length / area)


def resistance_matrix(coils: Sequence[Coil]) -> np.ndarray:
    return np.diag([coil_resistance(coil) for coil in coils])


def ohmic_power(currents: np.ndarray, resistance: np.ndarray) -> float:
    currents = np.asarray(currents, dtype=float)
    return float(currents @ resistance @ currents)


def approximate_inductance_matrix(coils: Sequence[Coil]) -> np.ndarray:
    """Build a stable coarse inductance matrix for energy accounting.

    The approximation is intentionally conservative; it is not used as DC loss.
    """
    n = len(coils)
    matrix = np.zeros((n, n))
    for i, coil_i in enumerate(coils):
        wire = max(coil_i.wire_radius, coil_i.radius * 1e-4)
        self_l = MU0 * coil_i.radius * (np.log(8 * coil_i.radius / wire) - 2)
        matrix[i, i] = max(self_l * coil_i.turns**2, 1e-12)
        for j in range(i):
            coil_j = coils[j]
            distance = np.linalg.norm(np.asarray(coil_i.centre) - np.asarray(coil_j.centre))
            alignment = abs(np.dot(coil_i.unit_normal, coil_j.unit_normal))
            scale = np.sqrt(matrix[i, i] * matrix[j, j])
            coupling = 0.18 * alignment * np.exp(-distance / max(coil_i.radius, coil_j.radius))
            matrix[i, j] = matrix[j, i] = coupling * scale
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.clip(values, 1e-15, None)) @ vectors.T


def stored_energy(currents: np.ndarray, inductance: np.ndarray) -> float:
    currents = np.asarray(currents, dtype=float)
    return float(0.5 * currents @ inductance @ currents)


"""Field-quality metrics with explicit regions and units."""

from __future__ import annotations

import numpy as np


def relative_error(actual: float, target: float) -> float:
    if target == 0:
        raise ValueError("target must be non-zero")
    return float(abs(actual - target) / abs(target))


def normalised_rmse(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    scale = np.ptp(target)
    if scale == 0:
        scale = max(np.max(np.abs(target)), 1e-15)
    return float(np.sqrt(np.mean((actual - target) ** 2)) / scale)


def fwhm(coordinate: np.ndarray, profile: np.ndarray) -> float:
    coordinate = np.asarray(coordinate, dtype=float)
    profile = np.abs(np.asarray(profile, dtype=float))
    if coordinate.size != profile.size or coordinate.size < 3:
        raise ValueError("coordinate and profile must have the same length >=3")
    half = 0.5 * np.max(profile)
    mask = profile >= half
    if not np.any(mask):
        return 0.0
    indices = np.flatnonzero(mask)
    return float(coordinate[indices[-1]] - coordinate[indices[0]])


def leakage_ratio(coordinate: np.ndarray, profile: np.ndarray, guard_half_width: float) -> float:
    coordinate = np.asarray(coordinate, dtype=float)
    profile = np.abs(np.asarray(profile, dtype=float))
    outside = np.abs(coordinate) > guard_half_width
    if not np.any(outside):
        raise ValueError("guard region covers every sample")
    return float(np.max(profile[outside]) / max(np.max(profile), 1e-15))


def side_lobe_ratio(coordinate: np.ndarray, profile: np.ndarray, main_half_width: float) -> float:
    return leakage_ratio(coordinate, profile, main_half_width)


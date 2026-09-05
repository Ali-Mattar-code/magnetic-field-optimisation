"""Solver-verified exploratory ML acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SurrogateBenchmark:
    model: ExtraTreesRegressor
    nrmse: float
    per_output_nrmse: np.ndarray
    measured_speedup: float
    train_samples: int
    test_samples: int


def fit_surrogate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    solver_seconds: float,
    seed: int = 42,
) -> SurrogateBenchmark:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=seed
    )
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    start = perf_counter()
    prediction = model.predict(x_test)
    prediction_seconds = max(perf_counter() - start, 1e-9)
    output_scale = np.maximum(np.ptp(y_test, axis=0), 1e-15)
    per_output = np.sqrt(np.mean((y_test - prediction) ** 2, axis=0)) / output_scale
    nrmse = float(np.mean(per_output))
    per_prediction = prediction_seconds / max(len(x_test), 1)
    return SurrogateBenchmark(
        model=model,
        nrmse=nrmse,
        per_output_nrmse=per_output,
        measured_speedup=float(solver_seconds / max(per_prediction, 1e-9)),
        train_samples=len(x_train),
        test_samples=len(x_test),
    )


def physics_features(raw_geometry: np.ndarray) -> np.ndarray:
    """Augment radius, spacing, and depth with dimensionless physics features."""
    raw_geometry = np.asarray(raw_geometry, dtype=float)
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 3:
        raise ValueError("raw_geometry must have columns [radius, spacing, depth]")
    if np.any(raw_geometry <= 0):
        raise ValueError("geometry values must be positive")
    radius, spacing, depth = raw_geometry.T
    axial_loop_scale = radius**2 / (radius**2 + depth**2) ** 1.5
    return np.column_stack(
        [
            raw_geometry,
            radius / depth,
            spacing / depth,
            spacing / radius,
            depth / radius,
            axial_loop_scale,
            1.0 / depth**3,
        ]
    )

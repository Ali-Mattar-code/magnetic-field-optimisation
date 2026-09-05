"""Build a small geometry-to-solver-output surrogate benchmark.

The labels are generated from the physics solver. The benchmark records measured local
timings and errors; it does not insert the historical thesis speedup.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from magfield.electrical import ohmic_power, resistance_matrix
from magfield.geometry import planar_array
from magfield.optimisation import solve_currents
from magfield.physics import influence_matrix
from magfield.surrogate import fit_surrogate, physics_features


def solve_geometry(radius: float, spacing: float, depth: float) -> tuple[float, float]:
    coils = planar_array(3, 3, spacing, radius, turns=10, segments=48)
    x = np.linspace(-0.08, 0.08, 101)
    samples = np.column_stack([x, np.zeros_like(x), np.full_like(x, depth)])
    matrix = influence_matrix(samples, coils, component=2)
    target = 50e-6 * np.exp(-0.5 * (x / 0.018) ** 2)
    weights = np.where(np.abs(x) < 0.018, 900.0, 30.0)
    row = influence_matrix(np.array([[0.0, 0.0, depth]]), coils, component=2)[0]
    resistance = resistance_matrix(coils)
    result = solve_currents(
        matrix,
        target,
        weights,
        resistance,
        target_row=row,
        target_value=50e-6,
        max_current=18.0,
        regularisation=3e-8,
    )
    return ohmic_power(result.currents, resistance), float(np.max(np.abs(result.currents)))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    raw_geometry = np.column_stack(
        [
            rng.uniform(0.014, 0.024, 400),
            rng.uniform(0.022, 0.034, 400),
            rng.uniform(0.045, 0.070, 400),
        ]
    )
    start = perf_counter()
    labels = np.asarray([solve_geometry(*row) for row in raw_geometry])
    solver_seconds = (perf_counter() - start) / len(raw_geometry)
    features = physics_features(raw_geometry)
    benchmark = fit_surrogate(features, labels, solver_seconds=solver_seconds)
    payload = {
        "evidence_level": "exploratory-local-benchmark",
        "samples": len(raw_geometry),
        "features": [
            "radius",
            "spacing",
            "depth",
            "radius/depth",
            "spacing/depth",
            "spacing/radius",
            "depth/radius",
            "analytical_axial_loop_scale",
            "inverse_depth_cubed",
        ],
        "train_samples": benchmark.train_samples,
        "test_samples": benchmark.test_samples,
        "multi_output_nrmse": benchmark.nrmse,
        "power_nrmse": float(benchmark.per_output_nrmse[0]),
        "peak_current_nrmse": float(benchmark.per_output_nrmse[1]),
        "solver_seconds_per_case": solver_seconds,
        "measured_prediction_speedup": benchmark.measured_speedup,
        "warning": "Timing depends on CPU, solver, discretisation, and batch size.",
    }
    output = Path("results/ml_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

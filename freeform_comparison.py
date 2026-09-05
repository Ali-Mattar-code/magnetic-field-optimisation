"""Exploratory comparison of a regular lattice and a multi-scale loop basis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from magfield.electrical import ohmic_power, resistance_matrix
from magfield.freeform import multi_scale_basis
from magfield.geometry import planar_array
from magfield.metrics import fwhm, leakage_ratio
from magfield.optimisation import solve_currents
from magfield.physics import influence_matrix


def evaluate(coils: list, label: str) -> dict:
    x = np.linspace(-0.10, 0.10, 241)
    points = np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.060)])
    matrix = influence_matrix(points, coils, component=2)
    target = 50e-6 * np.exp(-0.5 * (x / 0.018) ** 2)
    weights = np.where(np.abs(x) <= 0.018, 1500.0, 30.0)
    target_row = influence_matrix(np.array([[0.0, 0.0, 0.060]]), coils, component=2)[0]
    resistance = resistance_matrix(coils)
    result = solve_currents(
        matrix,
        target,
        weights,
        resistance,
        target_row=target_row,
        target_value=50e-6,
        regularisation=3e-8,
        max_current=12.0,
    )
    profile = matrix @ result.currents
    return {
        "label": label,
        "evidence_level": "exploratory-not-cost-equivalent",
        "coils": len(coils),
        "fwhm_m": fwhm(x, profile),
        "leakage_ratio_beyond_40mm": leakage_ratio(x, profile, 0.040),
        "ohmic_power_w": ohmic_power(result.currents, resistance),
        "peak_current_a": float(np.max(np.abs(result.currents))),
        "solver_success": result.success,
    }


if __name__ == "__main__":
    common = {"turns": 10, "segments": 96, "wire_radius": 0.00075, "resistivity": 1.68e-8}
    regular = planar_array(5, 5, 0.026, 0.018, **common)
    exploratory = multi_scale_basis(0.026, **common)
    payload = {
        "warning": "Different conductor bases; do not interpret as equal-cost hardware.",
        "results": [evaluate(regular, "regular-5x5"), evaluate(exploratory, "multi-scale")],
    }
    output = Path("results/freeform_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

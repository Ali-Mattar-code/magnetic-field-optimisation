"""End-to-end reproducible experiment orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .electrical import (
    approximate_inductance_matrix,
    ohmic_power,
    resistance_matrix,
    stored_energy,
)
from .geometry import planar_array
from .metrics import fwhm, leakage_ratio, normalised_rmse, relative_error
from .optimisation import diagnose_inverse_problem, solve_currents
from .pareto import knee_index, sweep
from .physics import influence_matrix
from .reporting import (
    plot_current_map,
    plot_field_map,
    plot_field_profile,
    plot_pareto,
    plot_robustness,
    plot_singular_spectrum,
    write_field_profile,
    write_json,
)
from .robustness import monte_carlo
from .validation import run_validation


def build_problem(config: dict) -> dict:
    geometry = config["geometry"]
    field = config["field"]
    coils = planar_array(
        geometry["rows"],
        geometry["columns"],
        geometry["spacing_m"],
        geometry["radius_m"],
        turns=geometry["turns"],
        segments=geometry["segments"],
        wire_radius=geometry["wire_radius_m"],
        resistivity=geometry["resistivity_ohm_m"],
    )
    x = np.linspace(-field["profile_half_width_m"], field["profile_half_width_m"], field["samples"])
    points = np.column_stack([x, np.zeros_like(x), np.full_like(x, field["target_depth_m"])])
    influence = influence_matrix(points, coils, component=2)
    target = field["target_tesla"] * np.exp(-0.5 * (x / field["target_sigma_m"]) ** 2)
    weights = np.full_like(x, float(config["optimisation"]["leakage_weight"]))
    weights[np.abs(x) <= field["target_sigma_m"]] = config["optimisation"]["target_weight"]
    target_point = np.array([[0.0, 0.0, field["target_depth_m"]]])
    target_row = influence_matrix(target_point, coils, component=2)[0]
    return {
        "coils": coils,
        "x": x,
        "points": points,
        "influence": influence,
        "target": target,
        "weights": weights,
        "target_point": target_point,
        "target_row": target_row,
    }


def run_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path = "results",
    backend: str = "scipy",
) -> dict:
    config = load_config(config_path)
    problem = build_problem(config)
    resistance = resistance_matrix(problem["coils"])
    inductance = approximate_inductance_matrix(problem["coils"])
    options = config["optimisation"]
    inverse_diagnostics = diagnose_inverse_problem(
        problem["influence"],
        weights=problem["weights"],
    )
    result = solve_currents(
        problem["influence"],
        problem["target"],
        problem["weights"],
        resistance,
        target_row=problem["target_row"],
        target_value=config["field"]["target_tesla"],
        tolerance=options["target_tolerance"],
        regularisation=options["regularisation"],
        max_current=options["max_current_a"],
        backend=backend,
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    realised = problem["influence"] @ result.currents
    target_index = int(np.argmin(np.abs(problem["x"])))
    robustness_options = config["robustness"]
    robust = monte_carlo(
        problem["coils"],
        problem["target_point"],
        result.currents,
        resistance,
        config["field"]["target_tesla"],
        draws=robustness_options["draws"],
        current_sigma_fraction=robustness_options["current_sigma_fraction"],
        resistance_sigma_fraction=robustness_options["resistance_sigma_fraction"],
        position_sigma_m=robustness_options["position_sigma_m"],
        tilt_sigma_deg=robustness_options["tilt_sigma_deg"],
        seed=config["experiment"]["seed"],
    )
    regularisations = np.logspace(-10, -5, 11)
    pareto = sweep(
        regularisations,
        problem["influence"],
        problem["target"],
        problem["weights"],
        resistance,
        target_row=problem["target_row"],
        target_value=config["field"]["target_tesla"],
        tolerance=options["target_tolerance"],
        max_current=options["max_current_a"],
        backend=backend,
    )
    knee = knee_index(pareto)
    validation = run_validation(segments=max(192, config["geometry"]["segments"]))
    summary = {
        "evidence_level": "reproduced-by-this-repository",
        "experiment": config["experiment"],
        "config_path": str(config_path),
        "solver": {
            "backend": result.backend,
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
        },
        "inverse_problem": inverse_diagnostics.to_dict(),
        "validation": validation.to_dict(),
        "geometry": config["geometry"],
        "field": {
            "target_tesla": config["field"]["target_tesla"],
            "achieved_tesla": float(realised[target_index]),
            "target_relative_error": relative_error(
                realised[target_index], config["field"]["target_tesla"]
            ),
            "profile_nrmse": normalised_rmse(realised, problem["target"]),
            "fwhm_m": fwhm(problem["x"], realised),
            "leakage_ratio_fixed_40mm_guard": leakage_ratio(problem["x"], realised, 0.040),
        },
        "electrical": {
            "ohmic_power_w": ohmic_power(result.currents, resistance),
            "stored_magnetic_energy_j": stored_energy(result.currents, inductance),
            "peak_current_a": float(np.max(np.abs(result.currents))),
            "currents_a": result.currents.tolist(),
        },
        "robustness": {
            "draws": robustness_options["draws"],
            "median_target_error": robust.median_error,
            "p95_target_error": robust.p95_error,
            "assumptions": robustness_options,
        },
        "pareto": {
            "points": len(pareto),
            "knee_index": knee,
            "knee_regularisation": pareto[knee].regularisation,
        },
    }
    output = Path(output_dir)
    write_json(output / "reproduction_summary.json", summary)
    write_field_profile(output / "field_profile.csv", problem["x"], problem["target"], realised)
    pd.DataFrame(
        {
            "regularisation": [p.regularisation for p in pareto],
            "power_w": [p.power_w for p in pareto],
            "profile_nrmse": [p.nrmse for p in pareto],
            "peak_current_a": [p.peak_current_a for p in pareto],
        }
    ).to_csv(output / "pareto_points.csv", index=False)
    figure_dir = output / "figures"
    plot_field_profile(figure_dir / "field_profile.png", problem["x"], problem["target"], realised)
    plot_current_map(
        figure_dir / "current_map.png",
        result.currents,
        config["geometry"]["rows"],
        config["geometry"]["columns"],
    )
    plot_field_map(
        figure_dir / "field_map.png",
        problem["coils"],
        result.currents,
        target_depth=config["field"]["target_depth_m"],
    )
    plot_pareto(
        figure_dir / "pareto_front.png",
        np.asarray([p.power_w for p in pareto]),
        np.asarray([p.nrmse for p in pareto]),
        knee,
    )
    plot_robustness(figure_dir / "robustness.png", robust.target_errors)
    plot_singular_spectrum(
        figure_dir / "singular_spectrum.png",
        np.asarray(inverse_diagnostics.relative_singular_values),
        inverse_diagnostics.relative_rank_tolerance,
    )
    return summary

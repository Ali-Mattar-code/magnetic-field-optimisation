"""Deterministic result serialisation and plotting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import Coil
from .physics import superposed_field


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_field_profile(
    path: str | Path, coordinate: np.ndarray, target: np.ndarray, realised: np.ndarray
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"x_m": coordinate, "target_t": target, "realised_t": realised}
    ).to_csv(path, index=False)


def plot_field_profile(
    path: str | Path, coordinate: np.ndarray, target: np.ndarray, realised: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(coordinate * 1e3, target * 1e6, "--", color="#8491A3", label="Target")
    ax.plot(coordinate * 1e3, realised * 1e6, color="#12355B", lw=2.2, label="Optimised")
    ax.set(xlabel="Lateral position (mm)", ylabel="Axial field (uT)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_current_map(path: str | Path, currents: np.ndarray, rows: int, columns: int) -> None:
    matrix = np.asarray(currents).reshape(rows, columns)
    bound = max(float(np.max(np.abs(matrix))), 1e-12)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound)
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    ax.set(xlabel="Coil column", ylabel="Coil row", title="Optimised current (A)")
    fig.colorbar(image, ax=ax, label="Current (A)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pareto(path: str | Path, power: np.ndarray, error: np.ndarray, knee: int) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(power, error * 100, "o-", color="#12355B")
    ax.scatter([power[knee]], [error[knee] * 100], s=100, color="#E4572E", label="Knee")
    ax.set(xlabel="Ohmic power (W)", ylabel="Profile NRMSE (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_robustness(path: str | Path, errors: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.hist(errors * 100, bins=min(18, max(6, len(errors) // 5)), color="#12355B", alpha=0.85)
    ax.axvline(np.quantile(errors, 0.95) * 100, color="#E4572E", ls="--", label="95th percentile")
    ax.set(xlabel="Target-field error (%)", ylabel="Monte Carlo draws")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_singular_spectrum(
    path: str | Path,
    relative_singular_values: np.ndarray,
    relative_tolerance: float,
) -> None:
    """Plot controllable modes of the weighted inverse-design operator."""
    values = np.asarray(relative_singular_values, dtype=float)
    modes = np.arange(1, len(values) + 1)
    floor = max(relative_tolerance / 100.0, np.finfo(float).tiny)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.semilogy(modes, np.maximum(values, floor), "o-", color="#12355B", lw=2.0)
    ax.axhline(relative_tolerance, color="#E4572E", ls="--", label="Rank tolerance")
    ax.set(
        xlabel="Singular mode",
        ylabel="Singular value / largest singular value",
        title="Weighted inverse-problem spectrum",
    )
    ax.set_xticks(modes)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_field_map(
    path: str | Path,
    coils: list[Coil],
    currents: np.ndarray,
    *,
    target_depth: float,
) -> None:
    """Render a signed axial-field map through the central XZ plane."""
    x = np.linspace(-0.09, 0.09, 141)
    z = np.linspace(0.004, 0.105, 111)
    xx, zz = np.meshgrid(x, z)
    points = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])
    bz = superposed_field(points, coils, currents)[:, 2].reshape(xx.shape) * 1e6
    target_index = np.unravel_index(
        np.argmin((xx - 0.0) ** 2 + (zz - target_depth) ** 2), xx.shape
    )
    target_field = max(abs(float(bz[target_index])), 1.0)
    limit = 1.5 * target_field

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    image = ax.pcolormesh(
        xx * 1e3,
        zz * 1e3,
        np.clip(bz, -limit, limit),
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    positive = np.linspace(0.2 * target_field, target_field, 5)
    ax.contour(
        xx * 1e3,
        zz * 1e3,
        bz,
        levels=positive,
        colors="white",
        linewidths=0.65,
        alpha=0.75,
    )
    ax.scatter(
        [0.0],
        [target_depth * 1e3],
        marker="x",
        s=80,
        lw=2.0,
        color="#FFCB47",
        label="Target",
    )
    ax.axhline(0.0, color="#1B263B", lw=2.0, alpha=0.8)
    ax.set(
        xlabel="Lateral position (mm)",
        ylabel="Height above coil plane (mm)",
        title="Optimised axial magnetic field",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.colorbar(image, ax=ax, label="Axial field (uT, clipped for visibility)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)

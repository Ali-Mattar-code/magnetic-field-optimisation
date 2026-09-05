"""Coil geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Coil:
    """A thin circular coil represented by its centre, normal, radius, and turns."""

    centre: tuple[float, float, float]
    radius: float
    turns: int = 1
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    segments: int = 128
    wire_radius: float = 0.00075
    resistivity: float = 1.68e-8

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.wire_radius <= 0:
            raise ValueError("Coil and wire radii must be positive")
        if self.turns < 1 or self.segments < 16:
            raise ValueError("turns must be >=1 and segments >=16")
        normal = np.asarray(self.normal, dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            raise ValueError("normal must be finite and non-zero")

    @property
    def unit_normal(self) -> np.ndarray:
        normal = np.asarray(self.normal, dtype=float)
        return normal / np.linalg.norm(normal)

    @property
    def conductor_length(self) -> float:
        return float(2.0 * np.pi * self.radius * self.turns)

    def moved(self, offset: np.ndarray) -> "Coil":
        centre = tuple(np.asarray(self.centre) + np.asarray(offset))
        return replace(self, centre=centre)


def planar_array(
    rows: int,
    columns: int,
    spacing: float,
    radius: float,
    *,
    turns: int = 1,
    segments: int = 128,
    wire_radius: float = 0.00075,
    resistivity: float = 1.68e-8,
) -> list[Coil]:
    """Build an XY-plane array centred on the origin."""
    if rows < 1 or columns < 1 or spacing <= 0:
        raise ValueError("rows, columns, and spacing must be positive")
    xs = (np.arange(columns) - (columns - 1) / 2) * spacing
    ys = (np.arange(rows) - (rows - 1) / 2) * spacing
    return [
        Coil(
            centre=(float(x), float(y), 0.0),
            radius=radius,
            turns=turns,
            segments=segments,
            wire_radius=wire_radius,
            resistivity=resistivity,
        )
        for y in ys
        for x in xs
    ]


def tilted_normal(normal: np.ndarray, tilt_x: float, tilt_y: float) -> np.ndarray:
    """Apply small rotations around global x and y axes."""
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx @ np.asarray(normal, dtype=float)


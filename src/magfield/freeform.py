"""Exploratory alternative conductor bases."""

from __future__ import annotations

from .geometry import Coil


def multi_scale_basis(
    spacing: float,
    *,
    turns: int,
    segments: int,
    wire_radius: float,
    resistivity: float,
) -> list[Coil]:
    """Create an exploratory nested-loop basis at several positions and scales.

    The basis is not conductor-cost-equivalent to a regular lattice and must be labelled
    exploratory in any comparison.
    """
    centres = [
        (0.0, 0.0, 0.0),
        (-spacing, 0.0, 0.0),
        (spacing, 0.0, 0.0),
        (0.0, -spacing, 0.0),
        (0.0, spacing, 0.0),
    ]
    coils: list[Coil] = []
    for scale, radius in enumerate((0.012, 0.020, 0.030)):
        selected = centres if scale < 2 else centres[:1]
        coils.extend(
            Coil(
                centre=centre,
                radius=radius,
                turns=turns,
                segments=segments,
                wire_radius=wire_radius,
                resistivity=resistivity,
            )
            for centre in selected
        )
    return coils


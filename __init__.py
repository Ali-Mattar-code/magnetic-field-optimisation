"""Physics-constrained magnetic-field inverse design."""

from .geometry import Coil, planar_array
from .optimisation import OptimisationResult, solve_currents
from .physics import influence_matrix, loop_field

__all__ = [
    "Coil",
    "OptimisationResult",
    "influence_matrix",
    "loop_field",
    "planar_array",
    "solve_currents",
]
__version__ = "0.1.0"


import numpy as np

from magfield.electrical import (
    approximate_inductance_matrix,
    ohmic_power,
    resistance_matrix,
    stored_energy,
)
from magfield.geometry import planar_array


def test_power_and_energy_are_nonnegative():
    coils = planar_array(2, 2, 0.025, 0.015, turns=5)
    current = np.array([1.0, -2.0, 0.5, 0.0])
    resistance = resistance_matrix(coils)
    inductance = approximate_inductance_matrix(coils)
    assert ohmic_power(current, resistance) >= 0
    assert stored_energy(current, inductance) >= 0
    assert np.linalg.eigvalsh(inductance).min() >= -1e-14


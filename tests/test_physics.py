import numpy as np
import pytest

from magfield.geometry import Coil
from magfield.physics import analytic_on_axis, influence_matrix, loop_field, superposed_field


def test_discretised_loop_matches_analytical_axis_field():
    coil = Coil((0.0, 0.0, 0.0), 0.04, segments=384)
    z = np.array([0.01, 0.04, 0.08])
    points = np.column_stack([np.zeros(3), np.zeros(3), z])
    actual = loop_field(points, coil)[:, 2]
    expected = analytic_on_axis(z, 0.04, 1.0)
    np.testing.assert_allclose(actual, expected, rtol=5e-4)


def test_turns_scale_field_linearly():
    point = np.array([[0.0, 0.0, 0.05]])
    one = loop_field(point, Coil((0.0, 0.0, 0.0), 0.03, turns=1))[0, 2]
    ten = loop_field(point, Coil((0.0, 0.0, 0.0), 0.03, turns=10))[0, 2]
    assert ten == pytest.approx(10 * one, rel=1e-12)


def test_influence_matrix_matches_direct_superposition():
    coils = [Coil((-0.02, 0.0, 0.0), 0.015), Coil((0.02, 0.0, 0.0), 0.015)]
    points = np.array([[0.0, 0.0, 0.04], [0.01, 0.0, 0.04]])
    currents = np.array([2.0, -0.5])
    matrix_result = (influence_matrix(points, coils) @ currents).reshape(-1, 3)
    np.testing.assert_allclose(matrix_result, superposed_field(points, coils, currents))

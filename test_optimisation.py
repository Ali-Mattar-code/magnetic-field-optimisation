import numpy as np

from magfield.optimisation import solve_currents


def test_solver_hits_target_and_limits_current():
    influence = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.array([0.5, 0.5, 1.0])
    weights = np.ones(3)
    resistance = np.eye(2)
    result = solve_currents(
        influence,
        target,
        weights,
        resistance,
        target_row=np.array([1.0, 1.0]),
        target_value=1.0,
        tolerance=0.01,
        regularisation=0.01,
        max_current=1.0,
    )
    assert result.success
    assert 0.99 <= np.sum(result.currents) <= 1.01
    assert np.max(np.abs(result.currents)) <= 1.0 + 1e-9


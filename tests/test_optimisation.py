import numpy as np
import pytest

from magfield.optimisation import diagnose_inverse_problem, solve_currents


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


def test_inverse_diagnostics_recover_known_spectrum():
    diagnostics = diagnose_inverse_problem(np.diag([4.0, 2.0, 1.0]))
    assert diagnostics.numerical_rank == 3
    assert diagnostics.nullity == 0
    assert diagnostics.condition_number == pytest.approx(4.0)
    assert diagnostics.retained_condition_number == pytest.approx(4.0)
    assert diagnostics.relative_singular_values == pytest.approx((1.0, 0.5, 0.25))


def test_inverse_diagnostics_expose_numerical_null_space():
    diagnostics = diagnose_inverse_problem(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert diagnostics.numerical_rank == 1
    assert diagnostics.nullity == 1
    assert diagnostics.condition_number is None
    assert diagnostics.to_dict()["rank_deficient_at_tolerance"] is True
    zero_operator = diagnose_inverse_problem(np.zeros((3, 2)))
    assert zero_operator.numerical_rank == 0
    assert zero_operator.retained_condition_number is None


@pytest.mark.parametrize(
    "matrix,weights",
    [(np.ones(3), None), (np.ones((2, 2)), np.ones(3))],
)
def test_inverse_diagnostics_reject_invalid_shapes(matrix, weights):
    with pytest.raises(ValueError):
        diagnose_inverse_problem(matrix, weights=weights)

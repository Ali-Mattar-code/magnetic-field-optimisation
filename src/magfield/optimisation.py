"""Constrained inverse current design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, minimize


@dataclass(frozen=True)
class OptimisationResult:
    currents: np.ndarray
    objective: float
    success: bool
    message: str
    backend: str
    iterations: int | None = None


def _quadratic_terms(
    influence: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    resistance: np.ndarray,
    regularisation: float,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = weights[:, None] * influence
    weighted_target = weights * target
    hessian = 2.0 * (weighted.T @ weighted + regularisation * resistance)
    linear = -2.0 * weighted.T @ weighted_target
    return hessian, linear


def solve_currents(
    influence: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    resistance: np.ndarray,
    *,
    target_row: np.ndarray,
    target_value: float,
    tolerance: float = 0.02,
    regularisation: float = 1e-8,
    max_current: float = 12.0,
    backend: str = "scipy",
) -> OptimisationResult:
    """Solve a convex quadratic field-shaping problem."""
    influence = np.asarray(influence, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    resistance = np.asarray(resistance, dtype=float)
    target_row = np.asarray(target_row, dtype=float)
    n = influence.shape[1]
    if target.shape != (influence.shape[0],) or weights.shape != target.shape:
        raise ValueError("influence, target, and weights have incompatible shapes")
    if resistance.shape != (n, n) or target_row.shape != (n,):
        raise ValueError("resistance or target_row has incompatible shape")
    if backend == "cvxpy":
        return _solve_cvxpy(
            influence,
            target,
            weights,
            resistance,
            target_row,
            target_value,
            tolerance,
            regularisation,
            max_current,
        )
    if backend != "scipy":
        raise ValueError("backend must be 'scipy' or 'cvxpy'")

    hessian, linear = _quadratic_terms(
        influence, target, weights, resistance, regularisation
    )
    ridge = np.linalg.solve(hessian + np.eye(n) * 1e-18, -linear)
    denominator = float(target_row @ ridge)
    if abs(denominator) > 1e-20:
        ridge *= target_value / denominator
    ridge = np.clip(ridge, -max_current, max_current)

    lower = target_value * (1.0 - tolerance)
    upper = target_value * (1.0 + tolerance)

    def objective(current: np.ndarray) -> float:
        return float(0.5 * current @ hessian @ current + linear @ current)

    def jacobian(current: np.ndarray) -> np.ndarray:
        return hessian @ current + linear

    constraints = [
        {"type": "ineq", "fun": lambda x: float(target_row @ x - lower)},
        {"type": "ineq", "fun": lambda x: float(upper - target_row @ x)},
    ]
    result = minimize(
        objective,
        ridge,
        jac=jacobian,
        method="SLSQP",
        bounds=Bounds(-max_current, max_current),
        constraints=constraints,
        options={"ftol": 1e-18, "maxiter": 1000},
    )
    return OptimisationResult(
        currents=np.asarray(result.x),
        objective=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        backend="scipy",
        iterations=int(result.nit),
    )


def _solve_cvxpy(
    influence: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    resistance: np.ndarray,
    target_row: np.ndarray,
    target_value: float,
    tolerance: float,
    regularisation: float,
    max_current: float,
) -> OptimisationResult:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("Install the 'optimisation' extra to use CVXPY") from exc
    current = cp.Variable(influence.shape[1])
    loss = cp.sum_squares(cp.multiply(weights, influence @ current - target))
    objective = cp.Minimize(loss + regularisation * cp.quad_form(current, resistance))
    constraints = [
        cp.abs(current) <= max_current,
        target_row @ current >= target_value * (1.0 - tolerance),
        target_row @ current <= target_value * (1.0 + tolerance),
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    return OptimisationResult(
        currents=np.asarray(current.value).ravel(),
        objective=float(problem.value),
        success=problem.status in {"optimal", "optimal_inaccurate"},
        message=str(problem.status),
        backend="cvxpy",
    )


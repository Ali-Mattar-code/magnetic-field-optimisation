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


@dataclass(frozen=True)
class InverseProblemDiagnostics:
    observations: int
    control_channels: int
    numerical_rank: int
    nullity: int
    relative_rank_tolerance: float
    condition_number: float | None
    retained_condition_number: float | None
    relative_singular_values: tuple[float, ...]

    def to_dict(self) -> dict[str, int | float | bool | None | list[float]]:
        return {
            "observations": self.observations,
            "control_channels": self.control_channels,
            "numerical_rank": self.numerical_rank,
            "nullity": self.nullity,
            "rank_deficient_at_tolerance": self.numerical_rank
            < min(self.observations, self.control_channels),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "condition_number": self.condition_number,
            "retained_condition_number": self.retained_condition_number,
            "relative_singular_values": list(self.relative_singular_values),
        }


def diagnose_inverse_problem(
    influence: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    relative_tolerance: float = 1e-10,
) -> InverseProblemDiagnostics:
    """Diagnose controllable modes of an influence matrix with an SVD.

    Row weights may be supplied to analyse the same weighted operator used by
    the objective. Only singular values above ``relative_tolerance`` times the
    largest are treated as numerically retained modes.
    """
    influence = np.asarray(influence, dtype=float)
    if influence.ndim != 2 or min(influence.shape) == 0:
        raise ValueError("influence must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(influence)):
        raise ValueError("influence must contain only finite values")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1)")

    operator = influence
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (influence.shape[0],):
            raise ValueError("weights must have one value per influence row")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        scale = max(float(np.max(weights)), np.finfo(float).tiny)
        operator = (weights / scale)[:, None] * influence

    singular_values = np.linalg.svd(operator, compute_uv=False)
    largest = float(singular_values[0])
    if largest == 0.0:
        relative = np.zeros_like(singular_values)
    else:
        relative = singular_values / largest
    retained = relative > relative_tolerance
    numerical_rank = int(retained.sum())
    smallest_retained = float(relative[retained][-1]) if numerical_rank else None
    full_rank = numerical_rank == min(influence.shape)
    condition_number = float(1.0 / relative[-1]) if full_rank and relative[-1] > 0 else None
    return InverseProblemDiagnostics(
        observations=influence.shape[0],
        control_channels=influence.shape[1],
        numerical_rank=numerical_rank,
        nullity=max(0, influence.shape[1] - numerical_rank),
        relative_rank_tolerance=relative_tolerance,
        condition_number=condition_number,
        retained_condition_number=float(1.0 / smallest_retained) if smallest_retained else None,
        relative_singular_values=tuple(float(value) for value in relative),
    )


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

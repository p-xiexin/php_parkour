"""Numerical QP steps for the trajectory solver's elastic contact model.

This module does not change pose increments, constraints, or physical acceptance
tolerances. An inexact step is only a direction for the caller's nonlinear merit
line search; its Clarabel termination status remains part of the report.
"""

import warnings

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def _fixed_variables(A, rhs, equality_count, base_size, tolerance):
    """Eliminate singleton hard equalities, auditing dependent-row roundoff."""
    A = A.tocsr(copy=True)
    A.eliminate_zeros()
    hard = np.asarray(A[:, base_size:].getnnz(axis=1)).ravel() == 0
    singleton = np.flatnonzero(
        hard & (np.arange(len(rhs)) < equality_count) & (np.diff(A.indptr) == 1)
    )
    fixed_value = np.zeros(A.shape[1])
    representatives = {}
    duplicate_error = 0.0
    for row in singleton:
        entry = A.indptr[row]
        column, coefficient = int(A.indices[entry]), A.data[entry]
        if column in representatives:
            duplicate_error = max(
                duplicate_error, abs(coefficient * fixed_value[column] - rhs[row])
            )
        else:
            fixed_value[column] = rhs[row] / coefficient
            representatives[column] = int(row)
    if duplicate_error > tolerance:
        raise ValueError("Inconsistent singleton hard equalities")
    free = np.ones(A.shape[1], dtype=bool)
    free[list(representatives)] = False
    reduced_A = A[:, free].tocsr()
    reduced_A.eliminate_zeros()
    reduced_rhs = rhs - A @ fixed_value
    zero = np.diff(reduced_A.indptr) == 0
    zero_equal = zero & (np.arange(len(rhs)) < equality_count)
    zero_inequal = zero & ~zero_equal
    equal_error = float(np.max(abs(reduced_rhs[zero_equal]), initial=0.0))
    inequal_error = float(np.max(-reduced_rhs[zero_inequal], initial=0.0))
    if max(equal_error, inequal_error) > tolerance:
        raise ValueError("Fixed variables violate a hard QP constraint")
    keep = ~zero
    return dict(
        A=A,
        hard=hard,
        free=free,
        fixed_value=fixed_value,
        representatives=representatives,
        keep=keep,
        reduced_A=reduced_A[keep].tocsc(),
        reduced_rhs=reduced_rhs[keep],
        equality_count=int(np.count_nonzero(keep[:equality_count])),
        report=dict(
            original_variables=A.shape[1],
            fixed_variables=len(representatives),
            reduced_variables=int(np.count_nonzero(free)),
            original_rows=A.shape[0],
            reduced_rows=int(np.count_nonzero(keep)),
            dependent_singletons=len(singleton) - len(representatives),
            removed_equality_roundoff=equal_error,
            removed_inequality_violation=inequal_error,
        ),
    )


def _elastic_model(A, rhs, equality_count, base_size, increment):
    """Evaluate the exact elastic epigraph at unchanged physical increments."""
    value = np.zeros(A.shape[1])
    value[:base_size] = increment[:base_size]
    residual = rhs - A @ value
    elastic = A[:, base_size:].tocsr()
    for row in range(equality_count):
        start, stop = elastic.indptr[row : row + 2]
        columns, coefficients = elastic.indices[start:stop], elastic.data[start:stop]
        if not len(columns):
            continue
        if len(columns) != 2 or set(coefficients) != {-1.0, 1.0}:
            raise ValueError("Expected two signed contact elastic variables")
        selected = np.flatnonzero(
            coefficients > 0 if residual[row] >= 0 else coefficients < 0
        )[0]
        value[base_size + columns[selected]] = abs(residual[row])
    for row in range(equality_count, A.shape[0]):
        start, stop = elastic.indptr[row : row + 2]
        columns, coefficients = elastic.indices[start:stop], elastic.data[start:stop]
        if not len(columns):
            continue
        if len(columns) != 1 or coefficients[0] >= 0:
            raise ValueError("Expected a nonnegative shared collision elastic variable")
        column = base_size + columns[0]
        # All manifold rows of a pair share this maximum, never their sum.
        value[column] = max(value[column], residual[row] / coefficients[0])
    return value


def _residuals(A, rhs, equality_count, value, hard):
    residual = A @ value - rhs
    violation = np.maximum(residual, 0.0)
    violation[:equality_count] = abs(residual[:equality_count])
    return dict(
        max_equality_residual=float(
            np.max(abs(residual[:equality_count]), initial=0.0)
        ),
        max_inequality_violation=float(np.max(residual[equality_count:], initial=0.0)),
        max_hard_constraint_violation=float(np.max(violation[hard], initial=0.0)),
    )


def _inexact_certificate(
    P,
    c,
    rhs,
    equality_count,
    base_size,
    state,
    value,
    dual,
    scale,
    *,
    feasibility_tol,
    absolute_gap_tol,
    relative_model_gap,
    hessian_lower_bound,
):
    A, hard = state["A"], state["hard"]
    raw = _residuals(A, rhs, equality_count, value, hard)
    baseline = _elastic_model(A, rhs, equality_count, base_size, state["fixed_value"])
    exact = _elastic_model(A, rhs, equality_count, base_size, value)
    exact_residual = _residuals(A, rhs, equality_count, exact, hard)
    baseline_residual = _residuals(A, rhs, equality_count, baseline, hard)

    def objective(x):
        return float(0.5 * x @ (P @ x) + c @ x)

    baseline_objective, model_objective = objective(baseline), objective(exact)
    raw_objective = objective(value)
    if not np.isfinite([baseline_objective, model_objective, raw_objective]).all():
        return dict(accepted=False, rejection="nonfinite_model_objective")
    report = dict(
        raw_primal=raw,
        exact_model_primal=exact_residual,
        baseline_primal=baseline_residual,
        raw_objective=raw_objective,
        model_objective=model_objective,
        baseline_model=baseline_objective,
        model_decrease=baseline_objective - model_objective,
        inexact_relative_model_gap=relative_model_gap,
        accepted=False,
    )
    if not np.isfinite(dual).all() or np.min(dual[equality_count:], initial=0.0) < 0.0:
        report["rejection"] = "invalid_dual_cone"
        return report
    if max(*exact_residual.values(), *baseline_residual.values()) > feasibility_tol:
        report["rejection"] = "original_constraint_residual"
        return report
    if model_objective >= baseline_objective:
        report["rejection"] = "no_model_decrease"
        return report
    stationarity = P @ value + c + A.T @ dual
    if not np.isfinite(stationarity).all():
        report["rejection"] = "nonfinite_dual_stationarity"
        return report
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        correction = spsolve(P, stationarity)
    if not np.isfinite(correction).all():
        report["rejection"] = "nonfinite_dual_linear_solve"
        return report
    error = stationarity - P @ correction
    error_bound = float(error @ error / hessian_lower_bound)
    # P >= mu I is supplied by the trajectory model's quadratic regularizer.
    # This upper bound on r.T @ inv(P) @ r makes the dual bound conservative,
    # including the numerical residual of the sparse linear solve.
    inverse_quadratic_upper = float(
        2 * stationarity @ correction - correction @ (P @ correction) + error_bound
    )
    dual_lower = float(
        -0.5 * value @ (P @ value)
        - rhs @ dual
        + value @ stationarity
        - 0.5 * inverse_quadratic_upper
    )
    if not np.isfinite([dual_lower, error_bound]).all():
        report["rejection"] = "nonfinite_dual_bound"
        return report
    gap = model_objective - dual_lower
    available_decrease = baseline_objective - dual_lower
    # Clarabel's absolute gap tolerance is in scaled objective units.
    allowed_gap = relative_model_gap * available_decrease + scale * absolute_gap_tol
    report.update(
        dual_lower_bound=dual_lower,
        dual_gap=gap,
        available_model_decrease=available_decrease,
        relative_gap_to_available_decrease=gap / available_decrease
        if available_decrease > 0
        else None,
        allowed_inexact_gap=allowed_gap,
        original_dual_stationarity_max=float(np.max(abs(stationarity), initial=0.0)),
        dual_linear_solve_residual=float(np.max(abs(error), initial=0.0)),
        dual_bound_error_allowance=0.5 * error_bound,
        hessian_lower_bound=hessian_lower_bound,
        accepted=bool(
            np.isfinite(dual_lower)
            and available_decrease > 0.0
            and 0.0 <= gap <= allowed_gap
        ),
    )
    if not report["accepted"]:
        report["rejection"] = "insufficient_inexact_model_accuracy"
    return report


def solve_elastic_qp(
    P,
    c,
    canonical,
    rhs,
    equality_count,
    base_size,
    *,
    penalty,
    feasibility_tol=1e-7,
    absolute_gap_tol=1e-7,
    relative_gap_tol=1e-7,
    inexact_relative_gap=1e-4,
    hessian_lower_bound=1e-6,
):
    """Return ``(full_increment_or_None, JSON_report)`` for one SQP model.

    ``P`` must be the full symmetric Hessian from TrajectorySQP.linearize, whose
    diagonal regularization proves ``P >= 1e-6 I``. Canonical rows are equalities
    first, followed by ``A x <= rhs``. Failed numerical statuses remain visible;
    an inexact accepted direction still requires the caller's original merit
    line search and final physical acceptance checks.
    """
    P = sp.csc_matrix(P)
    c, rhs = np.asarray(c), np.asarray(rhs)
    report = dict(
        accepted=False, attempts=[], inexact_relative_gap=inexact_relative_gap
    )
    if (
        not np.isfinite(P.data).all()
        or not np.isfinite(c).all()
        or not np.isfinite(canonical.data).all()
        or not np.isfinite(rhs).all()
        or hessian_lower_bound <= 0
    ):
        raise ValueError("Invalid trajectory QP coefficients or Hessian lower bound")
    elastic_hessian = P[base_size:, base_size:]
    elastic_diagonal = elastic_hessian.diagonal()
    elastic_cross = elastic_hessian - sp.diags(elastic_diagonal)
    elastic_cross.eliminate_zeros()
    if (
        P[:base_size, base_size:].nnz
        or P[base_size:, :base_size].nnz
        or elastic_cross.nnz
        or np.any(elastic_diagonal < hessian_lower_bound)
        or np.any(c[base_size:] < 0)
    ):
        raise ValueError(
            "Elastic objective must be independent, positive diagonal, "
            "and nonnegative linear"
        )
    try:
        state = _fixed_variables(
            canonical, rhs, equality_count, base_size, feasibility_tol
        )
    except ValueError as error:
        report["rejection"] = str(error)
        return None, report
    report["fixed_variable_elimination"] = state["report"]
    free, fixed_value = state["free"], state["fixed_value"]
    reduced_P = P[free][:, free].tocsc()
    reduced_c = (c + P @ fixed_value)[free]
    report["fixed_objective_constant"] = float(
        0.5 * fixed_value @ (P @ fixed_value) + c @ fixed_value
    )
    settings = clarabel.DefaultSettings()
    settings.direct_solve_method = "qdldl"
    settings.max_threads, settings.verbose, settings.max_iter = 1, False, 200
    settings.tol_feas = feasibility_tol
    settings.tol_gap_abs, settings.tol_gap_rel = absolute_gap_tol, relative_gap_tol
    cones = [
        clarabel.ZeroConeT(state["equality_count"]),
        clarabel.NonnegativeConeT(len(state["reduced_rhs"]) - state["equality_count"]),
    ]
    for scale in dict.fromkeys((100.0, max(100.0, float(penalty)))):
        result = clarabel.DefaultSolver(
            sp.triu(reduced_P / scale, format="csc"),
            reduced_c / scale,
            state["reduced_A"],
            state["reduced_rhs"],
            cones,
            settings,
        ).solve()
        status = str(result.status)
        attempt = dict(
            status=status, objective_scale=scale, iterations=int(result.iterations)
        )
        report["attempts"].append(attempt)
        value = fixed_value.copy()
        value[free] = np.asarray(result.x)
        if (
            not np.isfinite(value).all()
            or not np.isfinite(P @ value).all()
            or not np.isfinite(state["A"] @ value).all()
        ):
            attempt["rejection"] = "nonfinite_direction"
            continue
        raw_objective = float(0.5 * value @ (P @ value) + c @ value)
        if not np.isfinite(raw_objective):
            attempt["rejection"] = "nonfinite_model_objective"
            continue
        attempt["raw_objective"] = raw_objective
        attempt["original_primal"] = _residuals(
            state["A"], rhs, equality_count, value, state["hard"]
        )
        if status in ("Solved", "AlmostSolved"):
            report.update(accepted=True, acceptance="solver_status", status=status)
            return value, report
        if status not in ("NumericalError", "MaxIterations", "InsufficientProgress"):
            continue
        dual = np.zeros(len(rhs))
        dual[state["keep"]] = np.asarray(result.z) * scale
        stationarity = P @ value + c + state["A"].T @ dual
        for column, row in state["representatives"].items():
            coefficient = state["A"].data[state["A"].indptr[row]]
            dual[row] -= stationarity[column] / coefficient
        try:
            certificate = _inexact_certificate(
                P,
                c,
                rhs,
                equality_count,
                base_size,
                state,
                value,
                dual,
                scale,
                feasibility_tol=feasibility_tol,
                absolute_gap_tol=absolute_gap_tol,
                relative_model_gap=inexact_relative_gap,
                hessian_lower_bound=hessian_lower_bound,
            )
        except (ValueError, RuntimeError, MatrixRankWarning) as error:
            attempt["rejection"] = f"dual_certificate_failed: {error}"
            continue
        attempt["inexact_certificate"] = certificate
        if certificate["accepted"]:
            report.update(
                accepted=True, acceptance="inexact_model_decrease", status=status
            )
            return value, report
    report["rejection"] = "no_acceptable_qp_direction"
    return None, report

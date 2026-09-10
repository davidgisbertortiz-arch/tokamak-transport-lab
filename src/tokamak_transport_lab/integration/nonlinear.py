"""Stationary Picard iteration with residual line search and Newton rescue."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve as dense_solve
from scipy.optimize import root

from tokamak_transport_lab.solver.steady import balance_residual, stationary_solve


def gradient_length(t, rho):
    """a/L_T = -d(log T)/d rho; enforce axis symmetry, retain outward drive."""
    grad = np.gradient(t, rho, edge_order=2)
    grad[0] = 0.0
    return np.maximum(-grad / np.maximum(t, 1e-10), 0.0)


def iterate(
    *,
    rho,
    sources,
    pedestals,
    vp,
    initial,
    models,
    params,
    fallback_model,
    fallback_params,
    exchange,
    max_iters,
    tol,
    alpha0,
    alpha_min,
    alpha_max,
):
    start = time.perf_counter()
    if len(rho) < 3:
        raise ValueError("At least three grid points are required")
    if max_iters < 1 or tol <= 0 or not 0 < alpha_min <= alpha0 <= alpha_max <= 1:
        raise ValueError("Invalid iteration limit, tolerance or relaxation bounds")
    t = np.asarray(initial, dtype=float).copy()
    if t.shape != np.asarray(sources).shape or not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("Initial profiles must be finite, positive and match the grid")
    if not np.isfinite(sources).all() or np.any(np.asarray(pedestals) <= 0):
        raise ValueError("Sources must be finite and pedestal temperatures positive")
    t[:, -1] = pedestals
    active = list(models)
    active_params = [dict(p) for p in params]
    used_fallback, used_newton = False, False
    evaluations = 0
    hybrid_evaluations = 0

    def coefficients(x):
        nonlocal used_fallback, evaluations
        evaluations += 1
        gradients = [gradient_length(tc, rho) for tc in x]
        result = []
        for c, grad in enumerate(gradients):
            kw = dict(active_params[c])
            if c == 1 and hasattr(active[c], "electron_model"):
                kw["_a_over_lte"] = gradients[0]
            try:
                value = np.broadcast_to(
                    np.asarray(active[c](grad, **kw), dtype=float), rho.shape
                ).copy()
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("Transport returned invalid diffusivity")
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                if fallback_model is None:
                    raise ValueError(f"Transport channel {c} failed: {exc}") from exc
                active[c] = fallback_model
                active_params[c] = dict(fallback_params[c])
                used_fallback = True
                value = np.broadcast_to(fallback_model(grad, **active_params[c]), rho.shape).copy()
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("Fallback transport returned invalid diffusivity") from exc
            result.append(value)
        return np.array(result)

    def defect(x):
        chi = coefficients(x)
        return balance_residual(x, rho, chi, vp, sources, pedestals, exchange)

    def norm(f):
        return float(np.max(np.linalg.norm(f, axis=1)))

    residuals, alphas = [], []
    alpha = alpha0
    failure = "maximum_iterations"
    for _ in range(max_iters):
        chi = coefficients(t)
        f = balance_residual(t, rho, chi, vp, sources, pedestals, exchange)
        res = norm(f)
        if res <= tol:
            residuals.append(res)
            alphas.append(0.0)
            failure = "converged"
            break
        candidate = stationary_solve(rho, chi, vp, sources, pedestals, exchange)
        direction = candidate - t
        accepted = False
        # Picard may not be a descent direction for a stiff closure. Newton
        # then changes the numerical method, never silently the physics model.
        for method in ("newton",) if used_newton or len(residuals) >= 8 else ("picard", "newton"):
            if method == "newton":
                used_newton = True
                flat = t.ravel()
                jac = np.zeros((flat.size, flat.size))
                n = len(rho)
                # Local transport plus centered gradients has a five-point
                # dependency stencil in each channel. Color columns at stride
                # five: their affected rows cannot overlap. Ten evaluations
                # suffice for two channels, instead of 2*n evaluations.
                for channel in range(len(t)):
                    for color in range(5):
                        radial = np.arange(color, n, 5)
                        columns = channel * n + radial
                        dx = 1e-8 * np.maximum(np.abs(flat[columns]), 1.0)
                        trial = flat.copy()
                        trial[columns] += dx
                        difference = (defect(trial.reshape(t.shape)) - f).ravel()
                        for col, j, delta in zip(columns, radial, dx, strict=True):
                            local = np.arange(max(0, j - 2), min(n, j + 3))
                            rows = np.concatenate([c * n + local for c in range(len(t))])
                            jac[rows, col] = difference[rows] / delta
                try:
                    direction = dense_solve(jac, -f.ravel()).reshape(t.shape)
                except np.linalg.LinAlgError:
                    break
            step = alpha if method == "picard" else 1.0
            for _backtrack in range(18):
                trial = t + step * direction
                trial[:, -1] = pedestals
                if np.isfinite(trial).all() and np.all(trial > 0):
                    new_res = norm(defect(trial))
                    if (new_res < res * (0.95 if method == "picard" else 1.0)) or new_res <= tol:
                        accepted = True
                        break
                step *= 0.5
            if accepted:
                break
        if not accepted:
            # Powell's hybrid method can cross a nonsmooth active-set change
            # where a strict residual line search stalls. It solves the same
            # equations. Acceptance still requires the independently measured
            # PDE defect, not the optimizer's success flag.
            def hybrid_defect(flat, shape=t.shape):
                x = flat.reshape(shape)
                if not np.isfinite(x).all() or np.any(x <= 0):
                    return np.full_like(flat, 1e12)
                return defect(x).ravel()

            rescue = root(
                hybrid_defect,
                t.ravel(),
                method="hybr",
                options={
                    "xtol": 1e-10,
                    "maxfev": (t.size + 1) * max(1, max_iters - len(residuals)),
                },
            )
            hybrid_evaluations += rescue.nfev
            trial = rescue.x.reshape(t.shape)
            if np.isfinite(trial).all() and np.all(trial > 0):
                trial[:, -1] = pedestals
                new_res = norm(defect(trial))
                if new_res < res:
                    t = trial
                    res = new_res
            failure = "converged" if res <= tol else "nonlinear_rescue_failed"
            residuals.append(res)
            alphas.append(0.0)
            break
        t = trial
        residuals.append(new_res)
        alphas.append(step)
        alpha = min(alpha_max, max(alpha_min, step * 1.2))
        if new_res <= tol:
            failure = "converged"
            break
    chi = coefficients(t)
    final_residual = norm(balance_residual(t, rho, chi, vp, sources, pedestals, exchange))
    return (
        t,
        chi,
        residuals,
        alphas,
        {
            "n_iters": len(residuals),
            "converged": final_residual <= tol,
            "final_residual": final_residual,
            "residual_kind": "source_scaled_pde_l2",
            "used_fallback": used_fallback,
            "used_newton": used_newton,
            "transport_evaluations": evaluations,
            "hybrid_function_evaluations": hybrid_evaluations,
            "termination_reason": failure,
            "wall_time_s": time.perf_counter() - start,
            "temperature_clips": 0,
        },
    )

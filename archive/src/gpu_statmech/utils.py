"""
Utility helpers for analysis and diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def local_polynomial_second_derivative(
    x_values: Sequence[float],
    y_values: Sequence[float],
    window: int = 21,
    poly_order: int = 4,
) -> np.ndarray:
    """
    Estimate a smooth second derivative from local polynomial fits.

    Each output point is computed from an independent centered fit over a local
    window, which is useful for diagnostic curves derived from noisy
    finite-difference thermodynamic paths.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x_values and y_values must be 1-D sequences of equal length")
    if len(x) < 3:
        raise ValueError("at least three sample points are required")
    if window < 3:
        raise ValueError("window must be at least 3")
    if poly_order < 2:
        raise ValueError("poly_order must be at least 2")

    n = len(x)
    half_window = max(window // 2, 1)
    values = np.zeros(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        xw = x[lo:hi] - x[i]
        yw = y[lo:hi]
        degree = min(poly_order, len(xw) - 1)
        coeff = np.polyfit(xw, yw, degree)
        poly = np.poly1d(coeff)
        values[i] = float(poly.deriv(2)(0.0)) if degree >= 2 else 0.0

    return values


def smoothed_specific_heat_from_log_z(
    betas: Sequence[float],
    log_z_values: Sequence[float],
    n_dof: float,
    window: int = 21,
    poly_order: int = 4,
) -> list[float]:
    """
    Estimate Cv from a smoothed local curvature of ln Z(beta).

    The thermodynamic definition is unchanged:

        Cv(beta) = beta^2 * d^2(ln Z) / d beta^2 / n_dof
    """
    if n_dof <= 0.0:
        raise ValueError("n_dof must be positive")

    betas_arr = np.asarray(betas, dtype=float)
    d2_ln_z = local_polynomial_second_derivative(
        betas_arr,
        log_z_values,
        window=window,
        poly_order=poly_order,
    )
    return (betas_arr**2 * d2_ln_z / n_dof).tolist()

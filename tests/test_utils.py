import numpy as np
import pytest

from gpu_statmech.utils import (
    local_polynomial_second_derivative,
    smoothed_specific_heat_from_log_z,
)


def test_local_polynomial_second_derivative_recovers_quadratic():
    x = np.linspace(-2.0, 2.0, 41)
    y = 3.0 * x**2 + 2.0 * x + 1.0
    d2 = local_polynomial_second_derivative(x, y, window=9, poly_order=2)
    assert np.allclose(d2, 6.0, atol=1e-9)


def test_smoothed_specific_heat_matches_quadratic_log_z():
    betas = np.linspace(0.5, 3.0, 31)
    log_z = 2.5 * betas**2 - 0.5 * betas + 1.0
    cv = smoothed_specific_heat_from_log_z(betas, log_z, n_dof=4.0, window=9, poly_order=2)
    expected = 2.0 * 2.5 * betas**2 / 4.0
    assert np.allclose(cv, expected, atol=1e-9)


@pytest.mark.parametrize(
    ("x_values", "y_values", "window", "poly_order", "n_dof"),
    [
        ([0.0, 1.0], [0.0, 1.0], 5, 4, 1.0),
        ([0.0, 1.0, 2.0], [0.0, 1.0], 5, 4, 1.0),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1, 4, 1.0),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 5, 1, 1.0),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 5, 4, 0.0),
    ],
)
def test_utils_validate_inputs(x_values, y_values, window, poly_order, n_dof):
    if n_dof <= 0.0:
        with pytest.raises(ValueError):
            smoothed_specific_heat_from_log_z(x_values, y_values, n_dof=n_dof)
        return

    with pytest.raises(ValueError):
        local_polynomial_second_derivative(x_values, y_values, window=window, poly_order=poly_order)

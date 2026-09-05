import numpy as np
import pytest

from magfield.metrics import fwhm, leakage_ratio, normalised_rmse, relative_error


def test_fwhm_for_gaussian():
    sigma = 0.02
    x = np.linspace(-0.1, 0.1, 4001)
    profile = np.exp(-0.5 * (x / sigma) ** 2)
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert fwhm(x, profile) == pytest.approx(expected, abs=1e-4)


def test_error_metrics():
    assert relative_error(9.0, 10.0) == pytest.approx(0.1)
    assert normalised_rmse(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0
    assert leakage_ratio(
        np.array([-2, -1, 0, 1, 2]), np.array([1, 2, 5, 2, 1]), 1
    ) == pytest.approx(0.2)

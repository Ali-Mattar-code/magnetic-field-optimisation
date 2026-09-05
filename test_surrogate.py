import numpy as np
import pytest

from magfield.surrogate import physics_features


def test_physics_features_are_finite_and_scale_invariant_ratios_are_correct():
    raw = np.array([[0.02, 0.03, 0.06], [0.01, 0.02, 0.04]])
    features = physics_features(raw)
    assert features.shape == (2, 9)
    np.testing.assert_allclose(features[:, 3], raw[:, 0] / raw[:, 2])
    assert np.isfinite(features).all()


def test_physics_features_reject_nonphysical_geometry():
    with pytest.raises(ValueError):
        physics_features(np.array([[0.02, 0.03, 0.0]]))

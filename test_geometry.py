import numpy as np
import pytest

from magfield.geometry import Coil, planar_array


def test_planar_array_is_centred():
    coils = planar_array(3, 5, 0.02, 0.01)
    centres = np.asarray([coil.centre for coil in coils])
    np.testing.assert_allclose(centres.mean(axis=0), 0.0, atol=1e-15)
    assert len(coils) == 15


def test_invalid_coil_rejected():
    with pytest.raises(ValueError):
        Coil((0.0, 0.0, 0.0), -1.0)


def test_conductor_length_includes_turns():
    coil = Coil((0.0, 0.0, 0.0), radius=0.02, turns=7)
    assert coil.conductor_length == pytest.approx(2 * np.pi * 0.02 * 7)


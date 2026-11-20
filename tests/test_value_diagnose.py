"""Diagnostic summaries compare outcome and blended-target errors explicitly."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('value_diagnose', Path(__file__).parents[1] / 'tools/value_diagnose.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_mse_against_outcome_and_blend_are_different():
    z = np.array([1., -1., 1., -1.])
    v = np.array([0., 0., 0., 0.])
    prediction = np.array([0.5, -0.5, 0.5, -0.5])
    assert diag.mse(prediction, z) == pytest.approx(0.25)
    assert diag.mse(prediction, 0.5*z + 0.5*v) == 0
    assert diag.mse(v, z) == 1
    assert diag.mse(prediction, z, np.array([True, False, False, False])) == 0.25

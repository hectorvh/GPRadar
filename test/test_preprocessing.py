"""The first three processing steps keep a pulse and leave the input alone."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import dewow, remove_background, time_zero


def test_time_zero_aligns_a_known_shift():
    data = np.zeros((30, 1), dtype=np.float64)
    data[12, 0] = 1.0
    data[13, 0] = 0.4

    shifted, picks = time_zero(data, dt=1e-9, method="threshold", threshold=0.5)

    assert picks.tolist() == [12]
    assert shifted[0, 0] == 1.0
    assert shifted[1, 0] == 0.4
    assert shifted[2, 0] == 0.0


def test_dewow_removes_dc_and_drift_but_keeps_a_short_pulse():
    n_samples = 500
    sample = np.arange(n_samples, dtype=np.float64)
    pulse = np.zeros(n_samples, dtype=np.float64)
    pulse[250:253] = 8.0
    data = (3.0 + 0.01 * sample + pulse)[:, np.newaxis]

    result = dewow(data, dt=1e-9, window_ns=20)

    assert abs(result[:200, 0].mean()) < 0.05
    assert abs(result[400:, 0].mean()) < 0.05
    assert result[250, 0] > 4.0


def test_remove_background_drops_a_horizontal_band_and_keeps_a_hyperbola():
    data = np.full((40, 11), 2.0, dtype=np.float64)
    for trace in range(data.shape[1]):
        data[10 + trace, trace] += 5.0

    result = remove_background(data)

    assert abs(result[0, 0]) < 1e-12
    assert result[10, 0] > 4.0
    assert result[20, 10] > 4.0


def test_shapes_dtypes_and_inputs_are_preserved():
    data = np.zeros((16, 4), dtype=np.float32)
    data[5, :] = 1.5
    data[8, 2] = 4.0
    original = data.copy()

    shifted, picks = time_zero(data, dt=2e-10, threshold=0.2)
    dewowed = dewow(data, dt=2e-10, window_ns=1.0)
    background = remove_background(data, window_traces=3)

    for result in (shifted, dewowed, background):
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert not np.shares_memory(result, data)
    assert picks.shape == (4,)
    assert picks.dtype == int
    np.testing.assert_array_equal(data, original)

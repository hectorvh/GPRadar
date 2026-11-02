"""Processing steps keep a pulse and leave the input alone."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (
    amplitude_spectrum,
    bandpass,
    default_band,
    dewow,
    gain_agc,
    gain_tpow,
    remove_background,
    time_zero,
)


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


def _tone_amplitude(trace, dt, frequency):
    sample = np.arange(trace.size) * dt
    sine = np.sin(2 * np.pi * frequency * sample)
    cosine = np.cos(2 * np.pi * frequency * sample)
    design = np.column_stack((sine, cosine))
    coefficients, *_ = np.linalg.lstsq(design, trace, rcond=None)
    return float(np.hypot(*coefficients))


def test_bandpass_keeps_the_in_band_tone_and_rejects_the_others():
    n_samples = 10000
    dt = 2e-11
    sample = np.arange(n_samples) * dt
    low, mid, high = 50e6, 1.5e9, 6e9
    data = (
        np.sin(2 * np.pi * low * sample)
        + np.sin(2 * np.pi * mid * sample)
        + np.sin(2 * np.pi * high * sample)
    )[:, np.newaxis]

    filtered = bandpass(data, dt, low_hz=0.5e9, high_hz=3e9)

    assert _tone_amplitude(filtered[:, 0], dt, mid) > 0.9
    assert _tone_amplitude(filtered[:, 0], dt, low) < 0.05
    assert _tone_amplitude(filtered[:, 0], dt, high) < 0.05


def test_bandpass_does_not_shift_a_symmetric_pulse():
    n_samples = 4001
    dt = 2e-11
    centre = n_samples // 2
    time = (np.arange(n_samples) - centre) * dt
    pulse = np.exp(-0.5 * (time / 0.3e-9) ** 2) * np.cos(2 * np.pi * 1.5e9 * time)
    data = pulse[:, np.newaxis]

    filtered = bandpass(data, dt, low_hz=0.5e9, high_hz=3e9)

    assert int(np.argmax(filtered[:, 0])) == centre


def test_gain_tpow_scales_later_samples_by_time():
    data = np.ones((5, 2), dtype=np.float64)
    unchanged = gain_tpow(data, dt=1e-9, power=0)
    gained = gain_tpow(data, dt=1e-9, power=1)

    np.testing.assert_array_equal(unchanged, data)
    np.testing.assert_allclose(gained[:, 0], [0, 1, 2, 3, 4])


def test_gain_agc_flattens_an_exponential_decay():
    n_samples = 2000
    dt = 1e-9
    time = np.arange(n_samples) * dt
    data = np.exp(-time / 200e-9)[:, np.newaxis]

    gained = gain_agc(data, dt, window_ns=20)
    chunks = np.array_split(gained[100:-100, 0], 6)
    levels = [np.sqrt(np.mean(chunk**2)) for chunk in chunks]

    assert max(levels) / min(levels) < 1.3


def test_invalid_filter_and_gain_parameters_raise():
    data = np.ones((64, 1))
    dt = 2e-11
    nyquist = 0.5 / dt
    with pytest.raises(ValueError):
        bandpass(data, dt, low_hz=0, high_hz=1e9)
    with pytest.raises(ValueError):
        bandpass(data, dt, low_hz=2e9, high_hz=1e9)
    with pytest.raises(ValueError):
        bandpass(data, dt, low_hz=1e9, high_hz=nyquist)
    with pytest.raises(ValueError):
        bandpass(data, dt, low_hz=1e9, high_hz=2e9, order=0)
    with pytest.raises(ValueError):
        gain_tpow(data, dt, power=-1)
    with pytest.raises(ValueError):
        gain_agc(data, dt, window_ns=0)
    low, high = default_band(1.5e9, nyquist)
    assert low == pytest.approx(0.75e9)
    assert high == pytest.approx(3e9)


def test_later_steps_preserve_shape_dtype_and_input():
    n_samples = 256
    data = np.zeros((n_samples, 3), dtype=np.float32)
    data[40:80, :] = 1.0
    original = data.copy()
    dt = 1e-10

    filtered = bandpass(data, dt, low_hz=0.2e9, high_hz=1e9)
    tpow = gain_tpow(data, dt, power=1)
    agc = gain_agc(data, dt, window_ns=2)
    frequencies, amplitude = amplitude_spectrum(data, dt)

    for result in (filtered, tpow, agc):
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert not np.shares_memory(result, data)
    assert frequencies.shape == amplitude.shape == (n_samples // 2 + 1,)
    np.testing.assert_array_equal(data, original)

"""First processing steps for a GPR B-scan.

Each function takes a copy of a ``(n_samples, n_traces)`` array and returns a
new array. Inputs are left unchanged. Sample interval ``dt`` is in seconds.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def time_zero(data, dt, method="threshold", threshold=0.05):
    """Align each trace so its first break sits at sample zero.

    The direct air wave and ground wave are the time reference for the record.
    Picking that arrival on every trace and shifting it to the start puts
    two-way time on a common origin, so later reflections can be compared.

    ``method="threshold"`` takes the first sample whose absolute amplitude
    reaches ``threshold`` times that trace's peak. ``method="peak"`` takes the
    first local maximum of the absolute amplitude that also reaches that
    level. Traces are shifted up and the vacated samples at the end are zero.
    ``dt`` is the sample interval in seconds; the shift itself is in samples.

    Returns the shifted array and the picked sample index of each trace.
    """

    array = _scan(data)
    _require_dt(dt)
    if method not in ("threshold", "peak"):
        raise ValueError("method must be 'threshold' or 'peak'")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    picks = np.empty(array.shape[1], dtype=int)
    magnitude = np.abs(array)
    for index in range(array.shape[1]):
        picks[index] = _first_break(magnitude[:, index], method, float(threshold))

    shifted = np.zeros(array.shape, dtype=np.float64)
    n_samples = array.shape[0]
    for index, pick in enumerate(picks):
        shifted[: n_samples - pick, index] = array[pick:, index]
    return _like(data, shifted), picks


def dewow(data, dt, window_ns):
    """Remove the low-frequency wow from each trace.

    Coupling between the antennas, and induction in the ground, add a slowly
    varying bias to a trace. Subtracting a running mean along time removes
    that bias. A window longer than the pulse leaves the reflection in place.

    ``window_ns`` is the averaging length in nanoseconds. It is converted to
    samples with ``dt`` (seconds).
    """

    array = _scan(data)
    _require_dt(dt)
    if not np.isfinite(window_ns) or window_ns <= 0:
        raise ValueError("window_ns must be positive")
    window = max(int(round(window_ns * 1e-9 / dt)), 1)
    values = np.asarray(array, dtype=np.float64)
    trend = uniform_filter1d(values, size=window, axis=0, mode="nearest")
    return _like(data, values - trend)


def remove_background(data, window_traces=None):
    """Remove energy that is constant along the profile.

    The direct wave and system ringing arrive at nearly the same time on every
    trace, so they form a horizontal band. Subtracting the mean trace removes
    that band. A target such as a pipe changes from trace to trace and remains.
    With ``window_traces``, the mean is taken over that many neighbouring
    traces instead of the whole profile, which keeps a slowly dipping event.
    """

    array = _scan(data)
    values = np.asarray(array, dtype=np.float64)
    if window_traces is None:
        background = values.mean(axis=1, keepdims=True)
    else:
        if not np.isfinite(window_traces) or window_traces < 1:
            raise ValueError("window_traces must be at least 1")
        background = uniform_filter1d(
            values, size=int(window_traces), axis=1, mode="nearest"
        )
    return _like(data, values - background)


def _scan(data):
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("data must have shape (samples, traces) with neither axis empty")
    return array


def _require_dt(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive sample interval in seconds")


def _first_break(magnitude, method, threshold):
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak == 0:
        return 0
    level = threshold * peak
    if method == "threshold":
        above = np.flatnonzero(magnitude >= level)
        return int(above[0]) if above.size else 0

    for index, value in enumerate(magnitude):
        left = magnitude[index - 1] if index else -np.inf
        right = magnitude[index + 1] if index + 1 < magnitude.size else -np.inf
        if value >= level and value >= left and value >= right and value > left:
            return index
    above = np.flatnonzero(magnitude >= level)
    return int(above[0]) if above.size else 0


def _like(data, values):
    return np.asarray(values, dtype=np.asarray(data).dtype)

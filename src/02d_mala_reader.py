"""Load a MALA profile (.rd3 samples + .rad text header) into one scan.

The ``.rad`` header is one ``NAME:value`` pair per line. The ``.rd3`` file is
int16 samples, one trace after another, ``SAMPLES`` values per trace. The
result is the same :class:`GprScan` that ``load_gprmax`` in 02a returns, so
the processing in 02b and the plots in 03a work on either.

Two header fields are easy to mix up. ``FREQUENCY`` is the *sampling*
frequency in MHz (1280 MHz for Dinner Point Avenue, so dt = 0.78125 ns,
which equals ``TIMEWINDOW / SAMPLES``). The antenna centre frequency is
written in the ``ANTENNAS`` field, e.g. ``HDR-80 MHz``.
"""

import importlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

_loader = importlib.import_module("02a_gprmax_loader")
GprScan = _loader.GprScan
GprSource = _loader.GprSource

_BYTES_PER_SAMPLE = 2  # int16
_ANTENNA_MHZ = re.compile(r"(\d+(?:\.\d+)?)\s*MHz", re.IGNORECASE)


def read_header(path):
    """Read a ``.rad`` header file into a dict of stripped strings."""

    return read_header_text(Path(path).read_text(errors="replace"))


def read_header_text(text):
    """Parse ``.rad`` header text into a dict of stripped strings.

    Lines without a colon are skipped. Only the first colon splits the line,
    so a value may itself contain colons.
    """

    info = {}
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if sep:
            info[name.strip()] = value.strip()
    return info


def load_mala(path):
    """Load a MALA ``.rd3`` profile and its ``.rad`` header as a GprScan.

    ``path`` may name either file; the other is found by its suffix.
    ``data`` is float64, shape ``(samples, traces)``. ``dt`` is in seconds,
    from the sampling ``FREQUENCY``. ``trace_spacing`` is ``DISTANCE
    INTERVAL`` in metres (1 m with a warning when the header gives zero).
    ``frequency`` is the antenna centre frequency in Hz from ``ANTENNAS``,
    or ``None`` if that field has no MHz value. ``source.name`` is the
    ``.rd3`` file name.
    """

    rd3 = Path(path).with_suffix(".rd3")
    rad = rd3.with_suffix(".rad")
    for item in (rd3, rad):
        if not item.is_file():
            raise FileNotFoundError(item)
    info = read_header(rad)
    samples = _header_int(info, "SAMPLES", rad)
    data = _read_samples(rd3, samples)
    _check_trace_count(info, data.shape[1], rad)

    dt = _sample_interval(info, rad)
    spacing = _trace_spacing(info, rad)
    frequency = _antenna_frequency(info)
    positions = np.zeros((data.shape[1], 3))
    positions[:, 0] = spacing * np.arange(data.shape[1])
    source = GprSource(
        name=rd3.name,
        kind=info.get("ANTENNAS") or None,
        polarisation=None,
        waveform_id=None,
        waveform_type=None,
        amplitude=None,
        frequency=frequency,
        position=positions[0],
        positions=positions,
    )
    return GprScan(
        data=data,
        dt=dt,
        trace_spacing=spacing,
        frequency=frequency,
        source=source,
        component="rd3",
        positions=positions,
    )


def _header_float(info, name, path):
    """A required numeric header field."""

    if name not in info:
        raise ValueError(f"{path} header has no {name}")
    try:
        return float(info[name])
    except ValueError as exc:
        raise ValueError(f"{path} header {name} is not a number: {info[name]!r}") from exc


def _header_int(info, name, path):
    """A required positive integer header field."""

    value = _header_float(info, name, path)
    if value <= 0 or value != int(value):
        raise ValueError(f"{path} header {name} must be a positive integer, got {value}")
    return int(value)


def _read_samples(rd3, samples):
    """Read int16 traces as a float (samples x traces) array, checking the size."""

    size = rd3.stat().st_size
    trace_bytes = samples * _BYTES_PER_SAMPLE
    if size == 0 or size % trace_bytes:
        raise ValueError(
            f"{rd3} has {size} bytes, not a multiple of SAMPLES x 2 = {trace_bytes}; "
            "the header does not match the data file"
        )
    values = np.fromfile(rd3, dtype="<i2")
    return values.reshape(-1, samples).T.astype(np.float64)


def _check_trace_count(info, n_traces, path):
    """Warn when LAST TRACE disagrees with the traces in the data file."""

    if "LAST TRACE" in info:
        try:
            expected = int(float(info["LAST TRACE"]))
        except ValueError:
            return
        if expected != n_traces:
            warnings.warn(
                f"{path}: LAST TRACE is {expected} but the data file holds {n_traces} traces",
                stacklevel=3,
            )


def _sample_interval(info, path):
    """dt in seconds from the sampling FREQUENCY (MHz)."""

    sampling_mhz = _header_float(info, "FREQUENCY", path)
    if not np.isfinite(sampling_mhz) or sampling_mhz <= 0:
        raise ValueError(f"{path} header FREQUENCY must be positive")
    return 1.0 / (sampling_mhz * 1e6)


def _trace_spacing(info, path):
    """DISTANCE INTERVAL in metres; zero becomes 1 m, as in the old notebook."""

    spacing = _header_float(info, "DISTANCE INTERVAL", path)
    if abs(spacing) < np.finfo(float).eps:
        warnings.warn(
            f"{path}: DISTANCE INTERVAL is 0 (time-triggered line); using 1 m per trace",
            stacklevel=3,
        )
        return 1.0
    return spacing


def _antenna_frequency(info):
    """Antenna centre frequency in Hz from ANTENNAS (e.g. 'HDR-80 MHz'), or None."""

    match = _ANTENNA_MHZ.search(info.get("ANTENNAS", ""))
    return float(match.group(1)) * 1e6 if match else None

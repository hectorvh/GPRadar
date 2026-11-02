"""Load gprMax HDF5 outputs (.h5, or the older .out extension) into one scan.

A scan is a time-by-trace array plus the sample interval, the distance between
traces, the source centre frequency, and the source description. Pass one
merged B-scan, one A-scan, a directory of per-trace files, or a list of those
files. gprMax drops the waveform group when it merges traces, so frequency and
waveform fields are present on per-trace files and absent on a merged file.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_OUTPUT_SUFFIXES = {".h5", ".out"}


@dataclass(frozen=True)
class GprSource:
    """Transmitter that produced the scan."""

    name: str
    kind: str | None
    polarisation: str | None
    waveform_id: str | None
    waveform_type: str | None
    amplitude: float | None
    frequency: float | None
    position: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class GprScan:
    """One GPR record loaded from gprMax output.

    ``positions`` has shape ``(n_traces, 3)`` and is the receiver coordinate
    of each column, in metres. ``trace_spacing`` is the step between those
    coordinates.
    """

    data: np.ndarray
    dt: float
    trace_spacing: float | None
    frequency: float | None
    source: GprSource
    component: str
    positions: np.ndarray


def load_gprmax(source, component="Ez", receiver="rx1"):
    """Load gprMax output into a :class:`GprScan`.

    ``source`` is a ``.h5`` or ``.out`` path, a directory of per-trace outputs,
    or a sequence of those paths. ``data`` has shape ``(n_samples, n_traces)``.
    ``trace_spacing`` is in metres and ``frequency`` is in hertz. Both are
    ``None`` when the file does not record them: a single A-scan has no trace
    spacing, and a merged B-scan does not keep the source waveform.
    """

    files = _resolve_inputs(source)
    if len(files) == 1:
        return _load_file(files[0], component, receiver)

    scans = [_load_file(path, component, receiver) for path in files]
    widths = [scan.data.shape[1] for scan in scans]
    if any(width != 1 for width in widths):
        raise ValueError("stack single-trace outputs, or pass one merged B-scan")
    sample_counts = {scan.data.shape[0] for scan in scans}
    if len(sample_counts) != 1:
        raise ValueError("trace files do not have the same number of samples")
    intervals = {scan.dt for scan in scans}
    if len(intervals) != 1:
        raise ValueError("trace files do not have the same dt")
    frequencies = {scan.frequency for scan in scans}
    if len(frequencies) != 1:
        raise ValueError("trace files do not have the same source frequency")

    data = np.column_stack([scan.data[:, 0] for scan in scans])
    positions = np.vstack([scan.positions for scan in scans])
    source_positions = np.vstack([scan.source.positions for scan in scans])
    first = scans[0].source
    source_info = GprSource(
        name=first.name,
        kind=first.kind,
        polarisation=first.polarisation,
        waveform_id=first.waveform_id,
        waveform_type=first.waveform_type,
        amplitude=first.amplitude,
        frequency=first.frequency,
        position=source_positions[0],
        positions=source_positions,
    )
    return GprScan(
        data=data,
        dt=scans[0].dt,
        trace_spacing=_trace_spacing(positions),
        frequency=first.frequency,
        source=source_info,
        component=component,
        positions=positions,
    )


def _resolve_inputs(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = [
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in _OUTPUT_SUFFIXES and not item.name.startswith(".")
            ]
            traces = [item for item in files if "_merged" not in item.stem]
            files = traces or files
            if not files:
                raise FileNotFoundError(f"no gprMax output files in {path}")
            return tuple(sorted(files, key=_natural_key))
        if not path.is_file():
            raise FileNotFoundError(path)
        return (path,)

    files = tuple(Path(item) for item in source)
    if not files:
        raise ValueError("no gprMax output files were given")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("gprMax output file(s) not found: " + ", ".join(missing))
    return files


def _load_file(path, component, receiver):
    try:
        output = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError(f"{path} is not a gprMax HDF5 output") from exc

    with output:
        if "dt" not in output.attrs:
            raise ValueError(f"{path} has no gprMax dt attribute")
        dt = float(output.attrs["dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"{path} has an invalid dt")

        rx_group = _receiver_group(output, receiver, path)
        data = _component_array(rx_group, component, path)
        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError(f"{path} contains an empty {component} record")

        source_info = _read_source(output, data.shape[1])
        positions = _receiver_positions(output, receiver, data.shape[1])
        return GprScan(
            data=data,
            dt=dt,
            trace_spacing=_trace_spacing(positions),
            frequency=source_info.frequency,
            source=source_info,
            component=component,
            positions=positions,
        )


def _receiver_group(output, receiver, path):
    rxs = output.get("rxs")
    if rxs is None or receiver not in rxs:
        found = [] if rxs is None else list(rxs.keys())
        names = ", ".join(found) if found else "none"
        raise KeyError(f"{path} has no receiver {receiver!r} (found {names})")
    return rxs[receiver]


def _component_array(rx_group, component, path):
    if component not in rx_group or not isinstance(rx_group[component], h5py.Dataset):
        available = [name for name, item in rx_group.items() if isinstance(item, h5py.Dataset)]
        names = ", ".join(available) if available else "none"
        raise KeyError(f"{path} receiver has no {component} dataset (found {names})")
    data = np.asarray(rx_group[component], dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError(f"{path} {component} must be a trace or a time-by-trace array")
    return data


def _read_source(output, n_traces):
    group, name = _source_group(output)
    excitation = group.get("excitation") if group is not None else None
    attrs = excitation.attrs if isinstance(excitation, h5py.Group) else {}
    group_attrs = group.attrs if group is not None else {}
    positions = _source_positions(output, name, n_traces)
    frequency = _optional_float(attrs.get("WaveformFrequency"))
    return GprSource(
        name=name or "",
        kind=_text(attrs.get("SourceType", group_attrs.get("Type"))),
        polarisation=_text(attrs.get("Polarisation", group_attrs.get("Polarisation"))),
        waveform_id=_text(attrs.get("WaveformID", group_attrs.get("Waveform"))),
        waveform_type=_text(attrs.get("WaveformType")),
        amplitude=_optional_float(attrs.get("WaveformAmplitude")),
        frequency=frequency,
        position=positions[0],
        positions=positions,
    )


def _source_group(output):
    """Return the first source group and its name.

    Merged B-scans keep the name and positions under ``trace_metadata`` and
    drop the source group itself.
    """

    sources = output.get("srcs")
    if sources is not None and len(sources):
        names = sorted(sources.keys(), key=_natural_key)
        return sources[names[0]], names[0]
    recorded = output.get("trace_metadata/srcs")
    if recorded is not None and len(recorded):
        names = sorted(recorded.keys(), key=_natural_key)
        return None, names[0]
    return None, None


def _receiver_positions(output, receiver, n_traces):
    recorded = output.get(f"trace_metadata/rxs/{receiver}/Position")
    if isinstance(recorded, h5py.Dataset):
        positions = np.asarray(recorded, dtype=float)
        if positions.shape == (n_traces, 3):
            return positions

    origin = np.asarray(output[f"rxs/{receiver}"].attrs["Position"], dtype=float)
    step = _cell_step(output, "rxsteps")
    return origin + np.arange(n_traces, dtype=float)[:, np.newaxis] * step


def _source_positions(output, name, n_traces):
    if name is not None:
        recorded = output.get(f"trace_metadata/srcs/{name}/Position")
        if isinstance(recorded, h5py.Dataset):
            positions = np.asarray(recorded, dtype=float)
            if positions.shape == (n_traces, 3):
                return positions

    origin = None
    if name is not None and name in output.get("srcs", {}):
        if "Position" in output["srcs"][name].attrs:
            origin = np.asarray(output["srcs"][name].attrs["Position"], dtype=float)
    if origin is None:
        return np.full((n_traces, 3), np.nan)

    step = _cell_step(output, "srcsteps")
    return origin + np.arange(n_traces, dtype=float)[:, np.newaxis] * step


def _cell_step(output, attribute):
    dx = np.asarray(output.attrs.get("dx_dy_dz", [0.0, 0.0, 0.0]), dtype=float)
    steps = np.asarray(output.attrs.get(attribute, [0, 0, 0]), dtype=float)
    return dx * steps


def _trace_spacing(positions):
    if positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
        return None
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if distances.size == 0 or np.any(distances <= 0) or not np.isfinite(distances).all():
        return None
    spacing = float(np.median(distances))
    if not np.allclose(distances, spacing, rtol=1e-4, atol=1e-6):
        raise ValueError("trace positions are not equally spaced")
    return spacing


def _optional_float(value):
    if value is None:
        return None
    number = float(np.asarray(value))
    if not np.isfinite(number):
        return None
    return number


def _text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    elif isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value
        if isinstance(value, bytes):
            value = value.decode()
    text = str(value)
    return text or None


def _natural_key(value):
    text = value.name if isinstance(value, Path) else str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]

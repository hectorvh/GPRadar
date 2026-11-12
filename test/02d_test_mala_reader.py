"""Checks for the MALA .rd3/.rad reader on a tiny synthetic pair and Dinner Point Avenue."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

mala = importlib.import_module("02d_mala_reader")
loader = importlib.import_module("02a_gprmax_loader")

PROFILE = ROOT / "data" / "raw" / "usgs_forsythe" / "DinnerPointAve.rd3"
SAMPLE_WINDOW = ROOT / "data" / "sample" / "usgs" / "DinnerPointAve_traces0000-0299.npz"


def _write_pair(folder, traces, header):
    """Write int16 traces (samples x traces) and a NAME:value header."""

    rd3 = folder / "line.rd3"
    np.asarray(traces, dtype="<i2").T.tofile(rd3)
    lines = [f"{name}:{value}" for name, value in header.items()]
    (folder / "line.rad").write_text("\n".join(lines) + "\n")
    return rd3


def _header(samples, traces, distance="0.250000"):
    return {
        "SAMPLES": samples,
        "FREQUENCY": "1000.000000",
        "DISTANCE INTERVAL": f" {distance}",
        "ANTENNAS": "HDR-250 MHz M1=1",
        "TIMEWINDOW": f"{samples * 1.0:.6f}",
        "LAST TRACE": traces,
    }


def test_synthetic_pair_shape_dt_spacing_frequency(tmp_path):
    traces = np.arange(5 * 3, dtype=np.int16).reshape(5, 3)  # 5 samples, 3 traces
    rd3 = _write_pair(tmp_path, traces, _header(5, 3))
    scan = mala.load_mala(rd3)
    assert isinstance(scan, loader.GprScan)
    assert scan.data.shape == (5, 3)
    assert np.array_equal(scan.data, traces.astype(float))
    assert scan.dt == pytest.approx(1e-9)  # 1000 MHz sampling
    assert scan.trace_spacing == pytest.approx(0.25)
    assert scan.positions[:, 0] == pytest.approx([0.0, 0.25, 0.5])
    assert scan.frequency == pytest.approx(250e6)  # from ANTENNAS, not FREQUENCY
    assert scan.source.name == "line.rd3"
    # The .rad path works too.
    assert mala.load_mala(tmp_path / "line.rad").data.shape == (5, 3)


def test_size_must_be_a_multiple_of_samples(tmp_path):
    rd3 = _write_pair(tmp_path, np.zeros((5, 3)), _header(5, 3))
    with rd3.open("ab") as handle:
        handle.write(b"\x00\x00")  # one stray sample
    with pytest.raises(ValueError, match="multiple of SAMPLES x 2"):
        mala.load_mala(rd3)


def test_zero_distance_interval_warns_and_uses_one_metre(tmp_path):
    rd3 = _write_pair(tmp_path, np.zeros((4, 2)), _header(4, 2, distance="0.000000"))
    with pytest.warns(UserWarning, match="DISTANCE INTERVAL is 0"):
        scan = mala.load_mala(rd3)
    assert scan.trace_spacing == 1.0


def test_dinner_point_avenue():
    if not PROFILE.is_file():
        pytest.skip("DinnerPointAve.rd3 is not in data/raw/usgs_forsythe/")
    scan = mala.load_mala(PROFILE)
    assert scan.data.shape == (1017, 4188)
    assert scan.trace_spacing == pytest.approx(0.15, abs=0.005)
    assert scan.dt == pytest.approx(0.78125e-9)
    assert scan.frequency == pytest.approx(80e6)


def test_dinner_point_avenue_sample_window():
    """The committed 300-trace window: same header facts, and the same samples as the full file."""

    if not SAMPLE_WINDOW.is_file():
        pytest.skip("data/sample/ has not been built (src/00_make_sample_data.py)")
    with np.load(SAMPLE_WINDOW) as window:
        data = window["data"]
        header = mala.read_header_text(str(window["header"]))
        dt, spacing, frequency = (float(window[k]) for k in ("dt", "trace_spacing", "frequency"))
    assert data.shape == (1017, 300)
    assert int(header["SAMPLES"]) == 1017 and int(header["LAST TRACE"]) == 4188
    assert spacing == pytest.approx(0.15, abs=0.005)
    assert dt == pytest.approx(0.78125e-9)
    assert frequency == pytest.approx(80e6)
    if PROFILE.is_file():
        assert np.array_equal(data, mala.load_mala(PROFILE).data[:, :300])

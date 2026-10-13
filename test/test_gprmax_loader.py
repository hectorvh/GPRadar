"""Loader checks against synthetic outputs and the pipe B-scan already run."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gprmax_loader import load_gprmax

ROOT = Path(__file__).resolve().parents[1]
ASCAN = ROOT / "data" / "sim" / "cylinder_Ascan_2D.h5"
BSCAN = ROOT / "data" / "sim" / "pipe_study" / "f1.5_e6_s0_d9" / "pipe_merged.h5"


def _write_trace(path, values, position, frequency=1.5e9):
    with h5py.File(path, "w") as output:
        output.attrs["dt"] = 1e-11
        output.attrs["dx_dy_dz"] = np.array([0.002, 0.002, 0.002])
        output.attrs["srcsteps"] = np.array([1, 0, 0])
        output.attrs["rxsteps"] = np.array([1, 0, 0])
        rx = output.create_group("rxs/rx1")
        rx.attrs["Position"] = np.asarray(position, dtype=float) + np.array([0.04, 0.0, 0.0])
        rx.create_dataset("Ez", data=np.asarray(values, dtype=np.float32))
        src = output.create_group("srcs/src1")
        src.attrs["Position"] = np.asarray(position, dtype=float)
        src.attrs["Type"] = "HertzianDipole"
        src.attrs["Polarisation"] = "z"
        excitation = src.create_group("excitation")
        excitation.attrs["SourceType"] = "HertzianDipole"
        excitation.attrs["Polarisation"] = "z"
        excitation.attrs["WaveformID"] = "my_ricker"
        excitation.attrs["WaveformType"] = "ricker"
        excitation.attrs["WaveformAmplitude"] = 1.0
        excitation.attrs["WaveformFrequency"] = frequency


def test_stacks_trace_files_and_reads_waveform(tmp_path):
    _write_trace(tmp_path / "pipe1.out", [1.0, 2.0, 3.0], [0.040, 0.170, 0.0])
    _write_trace(tmp_path / "pipe2.out", [4.0, 5.0, 6.0], [0.042, 0.170, 0.0])
    _write_trace(tmp_path / "pipe10.out", [7.0, 8.0, 9.0], [0.044, 0.170, 0.0])

    scan = load_gprmax(tmp_path)

    assert scan.data.shape == (3, 3)
    assert scan.data[0, 0] == pytest.approx(1.0)
    assert scan.data[0, 2] == pytest.approx(7.0)
    assert scan.dt == pytest.approx(1e-11)
    assert scan.trace_spacing == pytest.approx(0.002)
    assert scan.frequency == pytest.approx(1.5e9)
    assert scan.component == "Ez"
    assert scan.source.kind == "HertzianDipole"
    assert scan.source.waveform_id == "my_ricker"
    assert scan.source.waveform_type == "ricker"
    assert scan.source.amplitude == pytest.approx(1.0)
    assert scan.source.polarisation == "z"
    assert scan.source.positions.shape == (3, 3)
    assert scan.positions[0, 0] == pytest.approx(0.080)


def test_single_ascan_has_no_trace_spacing(tmp_path):
    _write_trace(tmp_path / "one.h5", [1.0, 0.0], [0.100, 0.170, 0.0])

    scan = load_gprmax(tmp_path / "one.h5")

    assert scan.data.shape == (2, 1)
    assert scan.trace_spacing is None
    assert scan.frequency == pytest.approx(1.5e9)


def test_merged_bscan_uses_recorded_positions(tmp_path):
    with h5py.File(tmp_path / "pipe_merged.h5", "w") as output:
        output.attrs["dt"] = 5e-12
        output.attrs["dx_dy_dz"] = np.array([0.002, 0.002, 0.002])
        output.attrs["ntraces"] = 3
        rx = output.create_group("rxs/rx1")
        rx.attrs["Position"] = np.array([0.080, 0.170, 0.0])
        rx.create_dataset("Ez", data=np.ones((4, 3), dtype=np.float32))
        meta = output.create_group("trace_metadata/srcs/src1")
        meta.create_dataset(
            "Position",
            data=np.array([[0.040, 0.170, 0.0], [0.042, 0.170, 0.0], [0.044, 0.170, 0.0]]),
        )
        rx_meta = output.create_group("trace_metadata/rxs/rx1")
        rx_meta.create_dataset(
            "Position",
            data=np.array([[0.080, 0.170, 0.0], [0.082, 0.170, 0.0], [0.084, 0.170, 0.0]]),
        )

    scan = load_gprmax(tmp_path / "pipe_merged.h5")

    assert scan.data.shape == (4, 3)
    assert scan.trace_spacing == pytest.approx(0.002)
    assert scan.frequency is None
    assert scan.source.name == "src1"
    assert scan.source.kind is None
    assert scan.source.positions[0, 0] == pytest.approx(0.040)
    assert scan.positions[-1, 0] == pytest.approx(0.084)


@pytest.mark.skipif(not ASCAN.is_file(), reason="cylinder A-scan output is not in data/sim")
def test_cylinder_ascan_output():
    scan = load_gprmax(ASCAN)

    assert scan.data.shape == (637, 1)
    assert scan.dt == pytest.approx(4.717308673499368e-12)
    assert scan.trace_spacing is None
    assert scan.frequency == pytest.approx(1.5e9)
    assert scan.source.kind == "HertzianDipole"
    assert scan.source.waveform_id == "my_ricker"
    assert scan.source.position.tolist() == pytest.approx([0.1, 0.17, 0.0])


@pytest.mark.skipif(not BSCAN.is_file(), reason="pipe B-scan output is not in data/sim")
def test_pipe_bscan_output():
    scan = load_gprmax(BSCAN)

    assert scan.data.shape == (1061, 60)
    assert scan.dt == pytest.approx(4.717308673499368e-12)
    assert scan.trace_spacing == pytest.approx(0.002)
    assert scan.frequency is None
    assert scan.positions.shape == (60, 3)
    assert scan.positions[0, 0] == pytest.approx(0.08)
    assert scan.source.positions[0, 0] == pytest.approx(0.04)
    assert scan.source.positions[1, 0] - scan.source.positions[0, 0] == pytest.approx(0.002)

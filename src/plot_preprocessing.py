"""Show the processing sequence and the band-pass spectrum of one pipe B-scan."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gprmax_loader import load_gprmax
from preprocessing import (
    amplitude_spectrum,
    bandpass,
    default_band,
    dewow,
    gain_agc,
    remove_background,
    time_zero,
)

SCAN = ROOT / "data" / "sim" / "pipe_study" / "f1.5_e6_s0_d5" / "pipe_merged.h5"
FIGURE = ROOT / "figures" / "preprocessing_steps_sim.png"
SPECTRUM = ROOT / "figures" / "preprocessing_spectrum_sim.png"
# First break at 5% of the peak. Dewow window is longer than the 1.5 GHz
# pulse and shorter than the 5 ns record. The AGC window covers the wavelet
# envelope rather than a single half-cycle.
THRESHOLD = 0.05
DEWOW_WINDOW_NS = 2.0
AGC_WINDOW_NS = 2.0


def _panel(ax, data, extent, title):
    limit = np.percentile(np.abs(data), 99)
    if limit == 0:
        limit = 1
    image = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Trace position [m]")
    ax.set_ylabel("Two-way time [ns]")
    return image


def _center_hz(scan):
    """Antenna frequency. A merged file does not keep the waveform, so use the study table."""

    if scan.frequency is not None:
        return float(scan.frequency)
    with (SCAN.parent.parent / "cases.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["tag"] == SCAN.parent.name:
                return float(row["freq_hz"])
    raise ValueError(f"no centre frequency recorded for {SCAN.parent.name}")


def _save_steps(panels, extent):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (data, title) in zip(axes.ravel(), panels):
        image = _panel(ax, data, extent, title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("1.5 GHz pipe, εr = 6, centre depth 5 cm")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140)
    plt.close(fig)
    print(f"wrote {FIGURE}")


def _save_spectrum(before, after, dt, low_hz, high_hz):
    frequency_before, amplitude_before = amplitude_spectrum(before, dt)
    frequency_after, amplitude_after = amplitude_spectrum(after, dt)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axvspan(low_hz / 1e9, high_hz / 1e9, color="0.85", label="Pass band")
    ax.plot(frequency_before / 1e9, amplitude_before, label="Before band-pass")
    ax.plot(frequency_after / 1e9, amplitude_after, label="After band-pass")
    ax.set_xlim(0, 6)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Mean amplitude spectrum")
    ax.legend()
    fig.savefig(SPECTRUM, dpi=140)
    plt.close(fig)
    print(f"wrote {SPECTRUM}")


def main():
    scan = load_gprmax(SCAN)
    raw = scan.data
    zeroed, _picks = time_zero(raw, scan.dt, method="threshold", threshold=THRESHOLD)
    dewowed = dewow(zeroed, scan.dt, DEWOW_WINDOW_NS)
    cleaned = remove_background(dewowed)
    low_hz, high_hz = default_band(_center_hz(scan), nyquist_hz=0.5 / scan.dt)
    filtered = bandpass(cleaned, scan.dt, low_hz, high_hz)
    gained = gain_agc(filtered, scan.dt, AGC_WINDOW_NS)

    time_ns = np.arange(raw.shape[0]) * scan.dt * 1e9
    position = scan.positions[:, 0]
    extent = [position[0], position[-1], time_ns[-1], time_ns[0]]
    panels = (
        (raw, "Raw"),
        (zeroed, "Time zero"),
        (dewowed, "Dewow"),
        (cleaned, "Background removal"),
        (filtered, "Band-pass"),
        (gained, "AGC"),
    )
    _save_steps(panels, extent)
    _save_spectrum(cleaned, filtered, scan.dt, low_hz, high_hz)


if __name__ == "__main__":
    main()

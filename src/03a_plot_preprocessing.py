"""Show the processing sequence and the band-pass spectrum of one B-scan.

``--profile sim`` (default) uses the 1.5 GHz simulated pipe. ``--profile
dinnerpoint`` uses the MALA field line Dinner Point Avenue (80 MHz antenna),
read with ``02d_mala_reader``. Both run the same 02b functions with the same
window rule: dewow and AGC windows of three periods of the antenna centre
frequency, and a band-pass from half to twice that frequency.

    MPLBACKEND=Agg python src/03a_plot_preprocessing.py --profile dinnerpoint
"""

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_gprmax = importlib.import_module("02a_gprmax_loader").load_gprmax
load_mala = importlib.import_module("02d_mala_reader").load_mala
processing = importlib.import_module("02b_preprocessing")
amplitude_spectrum = processing.amplitude_spectrum
bandpass = processing.bandpass
default_band = processing.default_band
dewow = processing.dewow
gain_agc = processing.gain_agc
remove_background = processing.remove_background
time_zero = processing.time_zero

SCAN = ROOT / "data" / "sim" / "pipe_study" / "f1.5_e6_s0_d5" / "pipe_merged.h5"
FIELD_SCAN = ROOT / "data" / "raw" / "usgs_forsythe" / "DinnerPointAve.rd3"
FIGURE_WIDTH_IN = 12
# First break at 5% of the peak. Dewow and AGC windows are three periods of
# the antenna centre frequency: longer than the pulse, so dewow leaves the
# reflection in place, and long enough that AGC follows the wavelet envelope
# rather than a single half-cycle. At 1.5 GHz that is 2 ns; at 80 MHz, 37.5 ns.
THRESHOLD = 0.05
WINDOW_PERIODS = 3
# On the field line the direct wave fills the top ~4% of samples, so a 99th
# percentile over the whole panel sits on it and hides the reflections. The
# colour limit there is taken from samples later than this time instead.
FIELD_CLIM_FROM_NS = 100.0


@dataclass(frozen=True)
class _Profile:
    """Where a profile comes from and where its figures go."""

    title: str
    steps_figure: Path
    spectrum_figure: Path
    grid: tuple
    height_in: float
    distance_label: str


PROFILES = {
    "sim": _Profile(
        title="1.5 GHz pipe, εr = 6, centre depth 5 cm",
        steps_figure=ROOT / "figures" / "preprocessing_steps_sim.png",
        spectrum_figure=ROOT / "figures" / "preprocessing_spectrum_sim.png",
        grid=(2, 3),
        height_in=8,
        distance_label="Trace position [m]",
    ),
    "dinnerpoint": _Profile(
        title="Dinner Point Avenue (MALA, 80 MHz)",
        steps_figure=ROOT / "figures" / "preprocessing_steps_real.png",
        spectrum_figure=ROOT / "figures" / "preprocessing_spectrum_real.png",
        grid=(3, 2),
        height_in=11,
        distance_label="Distance [m]",
    ),
}


def _panel(ax, data, extent, title, xlabel, interpolation, clim_rows=slice(None)):
    """One B-scan panel with a robust symmetric colour limit.

    The limit is the 99th percentile of |amplitude| over ``clim_rows`` (all
    samples by default); larger values saturate.
    """

    limit = np.percentile(np.abs(data[clim_rows]), 99)
    if limit == 0:
        limit = 1
    image = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        interpolation=interpolation,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
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


def _load(name):
    """Load the scan for a profile name."""

    if name == "sim":
        return load_gprmax(SCAN)
    return load_mala(FIELD_SCAN)


def _windows_ns(center_hz):
    """Dewow and AGC window, both three periods of the centre frequency."""

    window = WINDOW_PERIODS / float(center_hz) * 1e9
    return window, window


def _process(scan, center_hz, detailed_titles):
    """Run the 02b chain; return the panels, the spectra inputs and the windows."""

    dewow_ns, agc_ns = _windows_ns(center_hz)
    raw = scan.data
    zeroed, _picks = time_zero(raw, scan.dt, method="threshold", threshold=THRESHOLD)
    dewowed = dewow(zeroed, scan.dt, dewow_ns)
    cleaned = remove_background(dewowed)
    low_hz, high_hz = default_band(center_hz, nyquist_hz=0.5 / scan.dt)
    filtered = bandpass(cleaned, scan.dt, low_hz, high_hz)
    gained = gain_agc(filtered, scan.dt, agc_ns)
    if detailed_titles:
        titles = (
            "Raw",
            "Time zero",
            f"Dewow ({dewow_ns:g} ns)",
            "Background removal (mean trace)",
            f"Band-pass ({low_hz / 1e6:g}-{high_hz / 1e6:g} MHz)",
            f"AGC ({agc_ns:g} ns)",
        )
    else:
        titles = ("Raw", "Time zero", "Dewow", "Background removal", "Band-pass", "AGC")
    panels = tuple(zip((raw, zeroed, dewowed, cleaned, filtered, gained), titles))
    return panels, cleaned, filtered, (low_hz, high_hz), (dewow_ns, agc_ns)


def _save_steps(profile, panels, extent, n_traces, real, dt):
    """Write the six-panel processing figure."""

    rows, cols = profile.grid
    fig, axes = plt.subplots(
        rows, cols, figsize=(FIGURE_WIDTH_IN, profile.height_in), constrained_layout=True
    )
    # The field line has far more traces than pixels per panel. All traces
    # are plotted; matplotlib's antialiased resampling reduces them for display.
    interpolation = "antialiased" if real else "nearest"
    clim_rows = slice(int(round(FIELD_CLIM_FROM_NS * 1e-9 / dt)), None) if real else slice(None)
    for ax, (data, title) in zip(axes.ravel(), panels):
        image = _panel(
            ax, data, extent, title, profile.distance_label, interpolation, clim_rows
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    title = profile.title
    if real:
        title += (
            f": all {n_traces} traces, image resampled for display.\n"
            f"Colour limit per panel: 99th percentile of |amplitude| after "
            f"{FIELD_CLIM_FROM_NS:g} ns (earlier samples saturate)"
        )
    fig.suptitle(title)
    profile.steps_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(profile.steps_figure, dpi=140)
    plt.close(fig)
    print(f"wrote {profile.steps_figure}")


def _save_spectrum(profile, before, after, dt, low_hz, high_hz, real):
    """Mean amplitude spectrum before and after the band-pass."""

    frequency_before, amplitude_before = amplitude_spectrum(before, dt)
    frequency_after, amplitude_after = amplitude_spectrum(after, dt)
    scale, unit = (1e6, "MHz") if real else (1e9, "GHz")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axvspan(low_hz / scale, high_hz / scale, color="0.85", label="Pass band")
    ax.plot(frequency_before / scale, amplitude_before, label="Before band-pass")
    ax.plot(frequency_after / scale, amplitude_after, label="After band-pass")
    ax.set_xlim(0, 3 * high_hz / scale if real else 6)
    ax.set_xlabel(f"Frequency [{unit}]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Mean amplitude spectrum" + (f": {profile.title}" if real else ""))
    ax.legend()
    fig.savefig(profile.spectrum_figure, dpi=140)
    plt.close(fig)
    print(f"wrote {profile.spectrum_figure}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="sim")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    real = args.profile != "sim"

    scan = _load(args.profile)
    center_hz = _center_hz(scan)
    panels, cleaned, filtered, (low_hz, high_hz), (dewow_ns, agc_ns) = _process(
        scan, center_hz, detailed_titles=real
    )
    n_samples, n_traces = scan.data.shape
    print(
        f"scan {scan.source.name}: {n_samples} samples x {n_traces} traces, "
        f"dt {scan.dt * 1e9:.5f} ns, spacing {scan.trace_spacing} m, "
        f"centre {center_hz / 1e6:g} MHz"
    )
    print(
        f"windows: time zero at {THRESHOLD:.0%} of peak, dewow {dewow_ns:g} ns, "
        f"band-pass {low_hz / 1e6:g}-{high_hz / 1e6:g} MHz, AGC {agc_ns:g} ns"
    )

    time_ns = np.arange(n_samples) * scan.dt * 1e9
    position = scan.positions[:, 0]
    extent = [position[0], position[-1], time_ns[-1], time_ns[0]]
    _save_steps(profile, panels, extent, n_traces, real, scan.dt)
    _save_spectrum(profile, cleaned, filtered, scan.dt, low_hz, high_hz, real)


if __name__ == "__main__":
    main()

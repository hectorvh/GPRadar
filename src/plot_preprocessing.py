"""Show the first three processing steps on one simulated pipe B-scan."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gprmax_loader import load_gprmax
from preprocessing import dewow, remove_background, time_zero

SCAN = ROOT / "data" / "sim" / "pipe_study" / "f1.5_e6_s0_d5" / "pipe_merged.h5"
FIGURE = ROOT / "figures" / "preprocessing_steps_sim.png"
# First break at 5% of the peak. Dewow window is longer than the 1.5 GHz
# pulse and shorter than the 5 ns record.
THRESHOLD = 0.05
DEWOW_WINDOW_NS = 2.0


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


def main():
    scan = load_gprmax(SCAN)
    raw = scan.data
    zeroed, _picks = time_zero(raw, scan.dt, method="threshold", threshold=THRESHOLD)
    dewowed = dewow(zeroed, scan.dt, DEWOW_WINDOW_NS)
    cleaned = remove_background(dewowed)

    time_ns = np.arange(raw.shape[0]) * scan.dt * 1e9
    position = scan.positions[:, 0]
    extent = [position[0], position[-1], time_ns[-1], time_ns[0]]
    panels = (
        (raw, "Raw"),
        (zeroed, "Time zero"),
        (dewowed, "Dewow"),
        (cleaned, "Background removal"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (data, title) in zip(axes.ravel(), panels):
        image = _panel(ax, data, extent, title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("1.5 GHz pipe, εr = 6, centre depth 5 cm")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140)
    plt.close(fig)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()

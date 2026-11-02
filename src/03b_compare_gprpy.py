"""Compare dewow and AGC on a field profile with GPRPy 1.0.14.

GPRPy's ``dewow`` and ``agcGain`` (``gprpy/toolbox/gprpyTools.py``) are not
the same operators as ``dewow`` and ``gain_agc`` in ``02b_preprocessing.py``.
The notes below are those differences. The functions in this repository
are not changed to remove them.

Window. GPRPy sets ``halfwid = ceil(window / 2)`` and, away from the ends,
averages samples ``[i - halfwid, i + halfwid]`` inclusive. For an even
requested length that window is one sample longer than asked: 33 samples
when dewow is given 32, and 65 when AGC is given 64, centred on sample i.
``dewow`` and ``gain_agc`` call ``scipy.ndimage.uniform_filter1d`` with the
requested length, 32 and 64. An even SciPy window is not symmetric. At
sample i it covers ``[i - size // 2, i + size // 2)``, one more sample
before i than after it.

Edges. For a requested length of 32, ``halfwid`` is 16. GPRPy fills
samples ``0 .. halfwid - 1`` with one mean (or one L2 norm) of samples
``0 .. halfwid``, which is ``halfwid + 1`` samples. It fills samples
``n - halfwid .. n - 1`` with one window over those same ``halfwid`` tail
samples. The sliding window does write the sample at ``halfwid`` and the
sample at ``n - halfwid``, but the tail assignment then overwrites
``n - halfwid``. ``uniform_filter1d(..., mode="nearest")`` instead repeats
the end sample and keeps the window moving.

AGC normalisation. GPRPy divides by ``max(||window||_2, 1e-8)``, the
square root of the sum of squares. ``gain_agc`` divides by the RMS, the
square root of the mean of squares, plus the same ``1e-8``. Where the
windows agree, the gains differ by about ``sqrt(N)``. A relative RMS near
7 is that scale factor, not a time shift. Pearson correlation ignores it.

Order. The reference file applies ``agcGain`` to the dewowed section.
This script does the same: ``gain_agc`` is called on the output of
``dewow``, not on the raw section.

The sample interval that turns 25 ns and 50 ns into 32 and 64 samples is
0.78125 ns, the MALA sampling interval (1/1280 MHz). GPRPy stores ``twtt``
as ``linspace(0, TIMEWINDOW, n_samples)``, about 0.782 ns per sample. The
comparison is sample by sample. The figure uses the stored ``twtt`` and
``pos`` as its axes.

Run from the repository root, in the gprMax-v4 environment:

    MPLBACKEND=Agg python src/03b_compare_gprpy.py

``src/02c_gprpy_reference.py`` writes the ``.npz`` and has to be run first, in
the gprpy environment. This script does not import GPRPy.
"""

import csv
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

processing = importlib.import_module("02b_preprocessing")
dewow = processing.dewow
gain_agc = processing.gain_agc

NPZ = ROOT / "data" / "raw" / "usgs_forsythe" / "DinnerPointAve_gprpy_steps.npz"
FIGURE = ROOT / "figures" / "gprpy_comparison.png"
TABLE = ROOT / "figures" / "gprpy_comparison.csv"
# MALA sampling interval for this profile: FREQUENCY 1280 MHz.
DT_S = 0.78125e-9
DEWOW_WINDOW_NS = 25.0
AGC_WINDOW_NS = 50.0


def window_samples(window_ns):
    """Samples corresponding to ``window_ns`` at ``DT_S``, as the filters count them."""

    return max(int(round(window_ns * 1e-9 / DT_S)), 1)


def load_reference(path):
    """Load the arrays written by ``src/02c_gprpy_reference.py``."""

    with np.load(path) as archive:
        raw = np.asarray(archive["raw"], dtype=np.float64).copy()
        dewowed = np.asarray(archive["dewow"], dtype=np.float64).copy()
        gained = np.asarray(archive["agc"], dtype=np.float64).copy()
        twtt = np.asarray(archive["twtt"], dtype=np.float64).copy()
        pos = np.asarray(archive["pos"], dtype=np.float64).copy()
        stored = {
            "dewow": int(archive["dewow_samples"]),
            "agc": int(archive["agc_samples"]),
        }
    return raw, dewowed, gained, twtt, pos, stored


def pearson(left, right):
    """Pearson correlation of two arrays, flattened in the same order."""

    coefficient = np.corrcoef(
        np.asarray(left, dtype=np.float64).ravel(),
        np.asarray(right, dtype=np.float64).ravel(),
    )[0, 1]
    return float(coefficient)


def record_metrics(mine, reference):
    """Max absolute difference, relative RMS, and correlation."""

    difference = np.asarray(mine, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    scale = np.linalg.norm(reference)
    if scale == 0:
        raise ValueError("GPRPy reference is all zeros")
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_rms": float(np.linalg.norm(difference) / scale),
        "correlation": pearson(mine, reference),
    }


def compare(raw, gprpy_dewow, gprpy_agc):
    """Dewow the raw section, then apply AGC, and score both against GPRPy."""

    mine_dewow = dewow(raw, DT_S, DEWOW_WINDOW_NS)
    mine_agc = gain_agc(mine_dewow, DT_S, AGC_WINDOW_NS)
    steps = (
        ("dewow", mine_dewow, gprpy_dewow, window_samples(DEWOW_WINDOW_NS)),
        ("agc", mine_agc, gprpy_agc, window_samples(AGC_WINDOW_NS)),
    )
    rows = []
    images = {}
    for name, mine, reference, trim in steps:
        if mine.shape != reference.shape:
            raise ValueError(f"{name} shape {mine.shape} != GPRPy shape {reference.shape}")
        if trim * 2 >= mine.shape[0]:
            raise ValueError(f"{name} window leaves no interior samples")
        rows.append({"step": name, "region": "full", "trim_samples": 0, **record_metrics(mine, reference)})
        interior = slice(trim, -trim)
        rows.append(
            {
                "step": name,
                "region": "interior",
                "trim_samples": trim,
                **record_metrics(mine[interior], reference[interior]),
            }
        )
        images[name] = (reference, mine, mine - reference)
    return rows, images


def _write_table(rows):
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "region", "trim_samples", "max_abs", "relative_rms", "correlation"]
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = f"{'step':<6} {'region':<9} {'trim':>5} {'max_abs':>12} {'relative_rms':>13} {'correlation':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['step']:<6} {row['region']:<9} {row['trim_samples']:>5} "
            f"{row['max_abs']:12.4f} {row['relative_rms']:13.6f} {row['correlation']:12.6f}"
        )
    print(f"wrote {TABLE}")


def _limit(*arrays):
    limit = max(float(np.percentile(np.abs(array), 99)) for array in arrays)
    if limit == 0:
        return 1.0
    return limit


def _panel(ax, data, extent, title, vmin, vmax):
    image = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Trace position [m]")
    ax.set_ylabel("Two-way time [ns]")
    return image


def _save_figure(images, twtt, pos):
    extent = [pos[0], pos[-1], twtt[-1], twtt[0]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    row_names = ("dewow", "agc")
    column_names = ("GPRPy", "preprocessing.py", "Difference")
    for row, name in enumerate(row_names):
        reference, mine, difference = images[name]
        shared = _limit(reference, mine)
        own = _limit(difference)
        limits = ((-shared, shared), (-shared, shared), (-own, own))
        for column, (data, (vmin, vmax)) in enumerate(zip((reference, mine, difference), limits)):
            image = _panel(axes[row, column], data, extent, column_names[column], vmin, vmax)
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(f"{name}\nTwo-way time [ns]")
    fig.suptitle("DinnerPointAve, dewow 25 ns then AGC 50 ns")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140)
    plt.close(fig)
    print(f"wrote {FIGURE}")


def main():
    if not NPZ.is_file():
        print(
            f"Missing {NPZ.relative_to(ROOT)}.\n"
            "Create it first, in the gprpy environment, from the repository root:\n"
            "    conda activate gprpy\n"
            "    python src/02c_gprpy_reference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    raw, gprpy_dewow, gprpy_agc, twtt, pos, stored = load_reference(NPZ)
    expected = {"dewow": window_samples(DEWOW_WINDOW_NS), "agc": window_samples(AGC_WINDOW_NS)}
    if stored != expected:
        raise ValueError(f"reference windows {stored} do not match {expected} samples at dt = {DT_S} s")
    rows, images = compare(raw, gprpy_dewow, gprpy_agc)
    _write_table(rows)
    _save_figure(images, twtt, pos)


if __name__ == "__main__":
    main()

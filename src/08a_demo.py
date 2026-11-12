"""Quick demo on the committed sample data (data/sample/ only).

``run`` (gprMax-v4): processes the three sample scans with the 02b chain
(time zero, dewow, mean-trace removal, band-pass, AGC; windows of three
periods of the centre frequency, as in 05a) and draws raw beside processed:

- figures/demo/pipe_bscan.png (1.5 GHz simulated pipe)
- figures/demo/scene_field_400mhz.png (400 MHz simulated field scene)
- figures/demo/dinnerpoint_window.png (80 MHz field data, 300 traces)
- figures/demo/mendeley_samples.png (7 real images with their hand labels)

``detect`` (gprdet): runs the ONNX detector (06f) on the Mendeley samples and
writes figures/demo/mendeley_detections.png. Skipped if best.onnx is missing.

    python src/08a_demo.py run
    python src/08a_demo.py detect
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ROOT / "data" / "sample"
OUT = ROOT / "figures" / "demo"
ONNX = ROOT / "runs" / "detect" / "A_f100_seed2" / "weights" / "best.onnx"


def _robust_imshow(ax, data, extent, title, xlabel):
    """B-scan panel with a symmetric 99th-percentile colour limit."""

    limit = float(np.percentile(np.abs(data), 99)) or 1.0
    ax.imshow(data, extent=extent, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Two-way time [ns]")


def _raw_vs_processed(name, data, dt, spacing, frequency_hz, title):
    """Process one scan and save raw beside processed."""

    render = importlib.import_module("05a_render_synthetic")
    processed, _picks, windows = render.process_scene_array(
        np.asarray(data, dtype=np.float64), dt, frequency_hz
    )
    n_samples, n_traces = processed.shape
    extent = [0.0, (n_traces - 1) * spacing, n_samples * dt * 1e9, 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    _robust_imshow(axes[0], data, extent, "Raw", "Distance [m]")
    _robust_imshow(
        axes[1],
        processed,
        extent,
        f"Processed: dewow {windows['dewow_ns']:g} ns, band-pass "
        f"{windows['band_low_hz'] / 1e6:g}-{windows['band_high_hz'] / 1e6:g} MHz, "
        f"AGC {windows['agc_ns']:g} ns",
        "Distance [m]",
    )
    fig.suptitle(title)
    path = OUT / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _pipe_scan():
    """The sample pipe B-scan and its centre frequency from cases.csv."""

    loader = importlib.import_module("02a_gprmax_loader")
    study = SAMPLE / "sim" / "pipe_study"
    with (study / "cases.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    scan = loader.load_gprmax(study / row["tag"] / "pipe_merged.h5")
    return scan, float(row["freq_hz"]), row["tag"]


def _npz(path):
    """Load an .npz into a plain dict."""

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def mendeley_samples():
    """(id, class, image path, label path) for every Mendeley sample."""

    folder = SAMPLE / "mendeley"
    with (folder / "samples.csv").open(newline="", encoding="utf-8") as handle:
        return [
            (r["id"], r["class"], folder / "images" / r["file"], folder / "labels" / f"{r['id']}.txt")
            for r in csv.DictReader(handle)
        ]


def _yolo_boxes(label_path, width, height):
    """YOLO label rows as pixel xyxy boxes."""

    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            cx, cy, w, h = (float(v) for v in parts[1:])
            boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def _draw_box(ax, box, colour, label=None):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=2))
    if label:
        ax.text(x0 + 2, y0 + 2, label, color=colour, fontsize=8, va="top")


def _mendeley_grid(name, detections=None):
    """Grid of the Mendeley samples: labels in green, detections in red."""

    samples = mendeley_samples()
    cols = 4
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (image_id, cls, image_path, label_path) in zip(axes.ravel(), samples):
        grey = np.asarray(Image.open(image_path).convert("L"))
        h, w = grey.shape
        ax.imshow(grey, cmap="gray", vmin=0, vmax=255)
        for box in _yolo_boxes(label_path, w, h):
            _draw_box(ax, box, "lime")
        if detections is not None:
            for box, score in zip(*detections[image_id]):
                _draw_box(ax, box, "red", f"{score:.2f}")
        ax.set_title(f"{image_id} ({cls})", fontsize=9)
    legend = "green: hand labels" + ("; red: ONNX detections (conf >= 0.25)" if detections else "")
    fig.suptitle(f"Mendeley train samples (CC BY 4.0, Mojahid et al. 2025). {legend}")
    path = OUT / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def run():
    """Process the three sample scans and draw the Mendeley samples."""

    OUT.mkdir(parents=True, exist_ok=True)
    scan, pipe_hz, tag = _pipe_scan()
    _raw_vs_processed(
        "pipe_bscan.png", scan.data, scan.dt, scan.trace_spacing, pipe_hz,
        f"Simulated pipe B-scan ({tag}, {pipe_hz / 1e9:g} GHz, gprMax)",
    )
    scene_path = next((SAMPLE / "sim").glob("field_400mhz_seed*.npz"))
    scene = _npz(scene_path)
    _raw_vs_processed(
        "scene_field_400mhz.png", scene["data"], float(scene["dt"]), float(scene["trace_spacing"]),
        float(scene["frequency"]), f"Simulated field scene ({scene_path.stem}, gprMax)",
    )
    dinner_path = next((SAMPLE / "usgs").glob("DinnerPointAve_*.npz"))
    dinner = _npz(dinner_path)
    _raw_vs_processed(
        "dinnerpoint_window.png", dinner["data"], float(dinner["dt"]), float(dinner["trace_spacing"]),
        float(dinner["frequency"]), f"Dinner Point Avenue, {dinner_path.stem.split('_')[1]} (USGS, MALA 80 MHz)",
    )
    _mendeley_grid("mendeley_samples.png")


def detect(onnx_path=ONNX):
    """ONNX detections on the Mendeley samples (gprdet)."""

    if not Path(onnx_path).is_file():
        print(f"detect: {onnx_path} not found; skipped")
        return
    onnx_mod = importlib.import_module("06f_export_onnx")
    session = onnx_mod.open_session(onnx_path)
    detections = {}
    for image_id, _cls, image_path, _label in mendeley_samples():
        detections[image_id] = onnx_mod.detect(session, onnx_mod.load_grey(image_path), min_conf=0.25)
        print(f"  {image_id}: {len(detections[image_id][0])} detection(s)")
    OUT.mkdir(parents=True, exist_ok=True)
    _mendeley_grid("mendeley_detections.png", detections)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "detect"), nargs="?", default="run")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    run() if args.command == "run" else detect()
    print(f"{args.command}: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

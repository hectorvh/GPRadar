"""Choose and export small demo samples for the GPRadar-API web service.

Writes ``exports/demo_samples/`` (gitignored):

- two Mendeley originals from the TRAIN split (one utility, one void), bytes
  unchanged, with their YOLO label files;
- one field_400mhz gprMax scene from the val seeds (180 to 199) as a
  compressed ``.npz``;
- ``manifest.json`` describing the three samples;
- ``golden.json`` with the ONNX detections (conf >= 0.25) the API should
  reproduce, if ``best.onnx`` exists.

Two subcommands, because the inputs and the ONNX path live in different
environments (no new dependencies in either):

    conda activate gprMax-v4; python src/07a_export_demo_samples.py export
    conda activate gprdet;    python src/07a_export_demo_samples.py golden

``export`` needs h5py and SciPy (gprMax loader, 05a processing). ``golden``
needs onnxruntime and OpenCV (06f) and reads only the exported files.
Test-split ids are refused: the split is checked against the test lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")

DEFAULT_SPLIT = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_LOCK = ROOT / "configs" / "mendeley_test_lock.txt"
DEFAULT_RAW = ROOT / "data" / "raw" / "mendeley_gpr"
DEFAULT_LABELS = ROOT / "data" / "labels" / "mendeley_originals"
DEFAULT_SCENES = ROOT / "data" / "sim" / "scenes" / "field_400mhz"
DEFAULT_OUT = ROOT / "exports" / "demo_samples"
DEFAULT_ONNX = ROOT / "runs" / "detect" / "A_f100_seed2" / "weights" / "best.onnx"
VAL_SEEDS = range(180, 200)
# A demo scene should show whole hyperbolas: no visible apex this close to a
# profile end (seed 185, apex 0.26 m from the start, shows only one flank).
EDGE_MARGIN_M = 1.0
MAX_NPZ_BYTES = 5 * 1024 * 1024
IMAGE_SIZE = 224
CONF = 0.25
IOU = 0.5
CITATION_MENDELEY = (
    "Mojahid et al. (2025), Intelligent recognition of subsurface utilities and voids "
    "using ground-penetrating radar and deep learning techniques: Data of GPR profiles "
    "images, Mendeley Data, DOI 10.17632/ww7fd9t325.1. Licence: CC BY 4.0."
)
CITATION_GPRMAX = "Simulated with gprMax (Warren et al. 2016)"
MANIFEST_FIELDS = (
    "id",
    "name",
    "kind",
    "file",
    "default_mode",
    "citation",
    "description",
)


# ---------------------------------------------------------------- Mendeley


def checked_train_rows(split_csv=DEFAULT_SPLIT, lock_path=DEFAULT_LOCK):
    """Train rows after verifying the test lock; also the set of test ids."""

    test_ids, rows = prep.verify_test_lock(split_csv, lock_path)
    test_set = set(test_ids)
    train = [row for row in rows if row["split"] == "train"]
    for row in train:
        prep.assert_not_test(row["id"], test_set)
    return train, test_set


def read_yolo(label_path):
    """YOLO rows (class, cx, cy, w, h) from a label file; [] if missing."""

    path = Path(label_path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def _largest_box_area(boxes):
    """Largest normalised box area (w * h) in a label file."""

    return max(w * h for _c, _x, _y, w, h in boxes)


def choose_mendeley(train_rows, test_ids, labels_dir, class_name, override=None):
    """Pick one train image of ``class_name``.

    Rule: among train images of the class with 1 or 2 boxes, take the one
    whose largest box covers the biggest share of the image (ties: lowest
    id). ``override`` picks a given id instead, after the same checks.
    Returns (row, boxes).
    """

    by_id = {row["id"]: row for row in train_rows}
    if override is not None:
        prep.assert_not_test(override, test_ids)
        row = by_id.get(override)
        if row is None:
            raise ValueError(f"{override} is not a train-split id")
        if row["class"] != class_name:
            raise ValueError(f"{override} is class {row['class']}, not {class_name}")
        boxes = read_yolo(Path(labels_dir) / f"{override}.txt")
        if not boxes:
            raise ValueError(f"{override} has no boxes")
        return row, boxes

    candidates = []
    for row in train_rows:
        if row["class"] != class_name:
            continue
        boxes = read_yolo(Path(labels_dir) / f"{row['id']}.txt")
        if 1 <= len(boxes) <= 2:
            candidates.append((-_largest_box_area(boxes), row["id"], row, boxes))
    if not candidates:
        raise ValueError(f"no train {class_name} image with 1 to 2 boxes")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _neg, _id, row, boxes = candidates[0]
    return row, boxes


# ---------------------------------------------------------------- gprMax


def _flagged_apex(run):
    """Names of visible targets whose apex check failed (|error| > T/4 or edge)."""

    return [
        row["name"]
        for row in run.get("apex", [])
        if row.get("visible")
        and (abs(row["error_ns"]) > row["quarter_period_ns"] or row["on_window_edge"])
    ]


def scene_summary(scene_dir):
    """Visible targets, labelled rock boxes and flagged apexes of one scene."""

    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    run = json.loads((scene_dir / "run.json").read_text(encoding="utf-8"))
    boxes_path = scene_dir / "boxes_physical.json"
    rock_boxes = None
    if boxes_path.is_file():
        rock_boxes = json.loads(boxes_path.read_text(encoding="utf-8")).get("n_rock_boxes")
    targets = [t for t in scene["ground_truth"]["targets"] if t["visible"]]
    x0, x1 = float(scene["midpoint_x0"]), float(scene["midpoint_x1"])
    edge = min((min(t["x"] - x0, x1 - t["x"]) for t in targets), default=float("inf"))
    return {
        "seed": int(scene["seed"]),
        "visible": [t["name"] for t in targets],
        "edge_distance_m": float(edge),
        "rock_boxes": rock_boxes if rock_boxes is not None else len(scene.get("rocks", [])),
        "flagged": _flagged_apex(run),
    }


def choose_scene(scenes_root=DEFAULT_SCENES, seeds=VAL_SEEDS, override=None):
    """Pick one val scene.

    Rule: 1 or 2 visible targets, no flagged apex check, and every visible
    apex at least ``EDGE_MARGIN_M`` from both profile ends; then the fewest
    labelled rock boxes, then more visible targets, then the lowest seed.
    ``override`` picks a given val seed instead. Returns the summary dict.
    """

    if override is not None:
        if override not in seeds:
            raise ValueError(f"seed {override} is not a val seed ({seeds.start}-{seeds.stop - 1})")
        return scene_summary(Path(scenes_root) / f"seed{override}")
    candidates = []
    for seed in seeds:
        folder = Path(scenes_root) / f"seed{seed}"
        if not (folder / "scene_merged.h5").is_file():
            continue
        info = scene_summary(folder)
        if (
            1 <= len(info["visible"]) <= 2
            and not info["flagged"]
            and info["edge_distance_m"] >= EDGE_MARGIN_M
        ):
            candidates.append(info)
    if not candidates:
        raise ValueError("no val scene with 1 to 2 visible targets")
    candidates.sort(key=lambda s: (s["rock_boxes"], -len(s["visible"]), s["seed"]))
    return candidates[0]


def processed_image(data, dt, frequency_hz):
    """05a processing of the full record, robust 8-bit, resized to 224 x 224.

    Same chain and scaling as the training crops (time zero, dewow, mean-trace
    removal, band-pass, AGC; bilinear resize), without crop or realism steps.
    """

    render = importlib.import_module("05a_render_synthetic")
    processed, _picks, _windows = render.process_scene_array(data, dt, frequency_hz)
    grey = render.robust_to_uint8(processed)
    image = Image.fromarray(grey, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.uint8)


def write_scene_npz(path, data, dt, trace_spacing, frequency, scene_json, processed_u8=None):
    """Write the compressed scene archive and return its size in bytes."""

    arrays = {
        "data": np.asarray(data, dtype=np.float32),
        "dt": np.float64(dt),
        "trace_spacing": np.float64(trace_spacing),
        "frequency": np.float64(frequency),
        "scene_json": np.array(scene_json),
    }
    if processed_u8 is not None:
        arrays["processed_u8_224"] = np.asarray(processed_u8, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def read_scene_npz(path):
    """Load the scene archive as a dict (scene JSON parsed)."""

    with np.load(path, allow_pickle=False) as archive:
        out = {key: archive[key] for key in archive.files}
    for key in ("dt", "trace_spacing", "frequency"):
        out[key] = float(out[key])
    out["scene"] = json.loads(str(out.pop("scene_json")))
    return out


def export_scene(scene_dir, out_path):
    """Load a merged B-scan and write it as a compressed .npz (< 5 MB)."""

    loader = importlib.import_module("02a_gprmax_loader")
    scene_dir = Path(scene_dir)
    scene_text = (scene_dir / "scene.json").read_text(encoding="utf-8")
    scene = json.loads(scene_text)
    scan = loader.load_gprmax(scene_dir / "scene_merged.h5")
    # A merged gprMax file drops the waveform, so the frequency comes from the scene.
    frequency = scan.frequency if scan.frequency is not None else float(scene["frequency_hz"])
    spacing = scan.trace_spacing if scan.trace_spacing is not None else float(scene["trace_step_m"])
    size = write_scene_npz(
        out_path,
        scan.data,
        scan.dt,
        spacing,
        frequency,
        scene_text,
        processed_u8=processed_image(scan.data, scan.dt, frequency),
    )
    if size >= MAX_NPZ_BYTES:
        raise ValueError(f"{out_path} is {size} bytes, over the 5 MB limit")
    return size, scan.data.shape, scan.dt, spacing, frequency


# ---------------------------------------------------------------- manifest


def manifest_entry(sample_id, name, kind, file, description):
    """One manifest record with the mode and citation set from ``kind``."""

    if kind not in ("image", "gprmax"):
        raise ValueError("kind must be 'image' or 'gprmax'")
    return {
        "id": sample_id,
        "name": name,
        "kind": kind,
        "file": file,
        "default_mode": "detector" if kind == "image" else "full",
        "citation": CITATION_MENDELEY if kind == "image" else CITATION_GPRMAX,
        "description": description,
    }


def check_manifest(entries):
    """Raise if an entry misses a field, has an empty one, or ids repeat."""

    ids = [entry.get("id") for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest ids are not unique")
    for entry in entries:
        missing = [key for key in MANIFEST_FIELDS if not entry.get(key)]
        if missing:
            raise ValueError(f"manifest entry {entry.get('id')} misses {missing}")
    return True


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _image_entry(row, boxes, raw_root, labels_dir, out_dir, class_word):
    """Copy one Mendeley image and its labels; return the manifest entry."""

    src = Path(raw_root) / row["path"]
    if not src.is_file():
        raise FileNotFoundError(src)
    sample_id = f"mendeley_{row['id']}"
    image_name = f"{sample_id}{src.suffix.lower()}"
    shutil.copyfile(src, out_dir / image_name)
    shutil.copyfile(Path(labels_dir) / f"{row['id']}.txt", out_dir / f"{sample_id}.txt")
    with Image.open(src) as image:
        size = image.size
    n = len(boxes)
    description = (
        f"Real {class_word} B-scan image ({size[0]} x {size[1]} px) from the Mendeley "
        f"GPR dataset, train split, with {n} hand-labelled target box{'es' if n > 1 else ''}."
    )
    return manifest_entry(sample_id, f"Mendeley {class_word} ({row['id']})", "image", image_name, description)


def run_export(args):
    """Choose the samples and write files, labels and manifest.json."""

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, test_ids = checked_train_rows(args.split, args.lock)
    print("test lock: ok")
    print("Mendeley rule: train images of the class with 1 to 2 boxes; largest single box area wins")
    utility, u_boxes = choose_mendeley(train_rows, test_ids, args.labels, "utility", args.utility_id)
    void, v_boxes = choose_mendeley(train_rows, test_ids, args.labels, "void", args.void_id)
    for row, boxes in ((utility, u_boxes), (void, v_boxes)):
        print(
            f"  {row['class']}: {row['id']} ({row['path']}), {len(boxes)} box(es), "
            f"largest box {100 * _largest_box_area(boxes):.1f}% of the image"
        )

    print(
        "Scene rule: val seeds 180-199 with 1 to 2 visible targets, no flagged apex, "
        f"every apex at least {EDGE_MARGIN_M:g} m from the profile ends; fewest labelled "
        "rock boxes, then more visible targets, then lowest seed"
    )
    info = choose_scene(args.scenes, override=args.seed)
    seed = info["seed"]
    print(
        f"  seed {seed}: visible {info['visible']}, labelled rock boxes {info['rock_boxes']}, "
        f"nearest apex {info['edge_distance_m']:.2f} m from a profile end"
    )

    entries = [
        _image_entry(utility, u_boxes, args.raw, args.labels, out_dir, "utility"),
        _image_entry(void, v_boxes, args.raw, args.labels, out_dir, "void"),
    ]
    scene_id = f"gprmax_field_400mhz_seed{seed}"
    npz_name = f"{scene_id}.npz"
    size, shape, dt, spacing, frequency = export_scene(
        Path(args.scenes) / f"seed{seed}", out_dir / npz_name
    )
    n_vis = len(info["visible"])
    entries.append(
        manifest_entry(
            scene_id,
            f"gprMax 400 MHz scene (seed {seed})",
            "gprmax",
            npz_name,
            f"Simulated 400 MHz B-scan over a 5 m profile with {n_vis} visible buried "
            f"target{'s' if n_vis > 1 else ''}, from the validation scenes.",
        )
    )
    check_manifest(entries)
    _write_json(out_dir / "manifest.json", {"samples": entries})
    print(
        f"scene npz: shape {shape}, dt {dt * 1e12:.3f} ps, spacing {spacing:g} m, "
        f"frequency {frequency / 1e6:g} MHz"
    )
    for entry in entries:
        path = out_dir / entry["file"]
        print(f"  {entry['file']}: {path.stat().st_size / 1024:.1f} KiB")
    print(f"wrote {out_dir / 'manifest.json'}")
    return entries


# ---------------------------------------------------------------- golden


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _box_rows(boxes, scores):
    return [
        {"xyxy": [round(float(v), 1) for v in box], "score": round(float(score), 4)}
        for box, score in zip(boxes, scores)
    ]


def run_golden(args):
    """Run the 06f ONNX path on every manifest sample; write golden.json."""

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"golden: {onnx_path} not found; skipped")
        return None
    onnx_mod = importlib.import_module("06f_export_onnx")
    out_dir = Path(args.out)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    session = onnx_mod.open_session(onnx_path)
    results = []
    for entry in manifest["samples"]:
        path = out_dir / entry["file"]
        if entry["kind"] == "image":
            grey = onnx_mod.load_grey(path)
            input_desc = "image file read as 8-bit greyscale"
        else:
            grey = read_scene_npz(path)["processed_u8_224"]
            input_desc = (
                "processed_u8_224 from the npz: 05a processing of the full record "
                "(time zero, dewow, mean-trace removal, band-pass, AGC), robust 8-bit, "
                "bilinear resize to 224 x 224"
            )
        boxes, scores = onnx_mod.detect(session, grey, min_conf=CONF, iou=IOU)
        results.append(
            {
                "id": entry["id"],
                "input": input_desc,
                "input_size_wh": [int(grey.shape[1]), int(grey.shape[0])],
                "boxes": _box_rows(boxes, scores),
            }
        )
        print(f"  {entry['id']}: {len(boxes)} box(es)")
    payload = {
        "model": onnx_path.name,
        "model_sha256": _sha256(onnx_path),
        "conf": CONF,
        "iou": IOU,
        "box_format": "xyxy in pixels of the input image",
        "note": "Scores and coordinates are rounded; compare with a tolerance.",
        "samples": results,
    }
    _write_json(out_dir / "golden.json", payload)
    print(f"wrote {out_dir / 'golden.json'}")
    return payload


def _has_onnx_runtime():
    """True when onnxruntime and OpenCV are importable (gprdet)."""

    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=("export", "golden"), default="export")
    parser.add_argument("--utility-id", default=None)
    parser.add_argument("--void-id", default=None)
    parser.add_argument("--seed", type=int, default=None, help="override the val scene seed")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.command == "export":
        run_export(args)
        if not Path(args.onnx).is_file():
            print(f"golden: {args.onnx} not found; skipped")
        elif _has_onnx_runtime():
            run_golden(args)
        else:
            print(
                "golden: onnxruntime/OpenCV are not in this environment; run\n"
                "  conda activate gprdet; python src/07a_export_demo_samples.py golden"
            )
    else:
        run_golden(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

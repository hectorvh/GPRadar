"""Build the small committed sample dataset in data/sample/ from the full data.

The sample lets a reviewer run the tests and ``make demo`` without the large,
gitignored data. It holds:

- ``sim/``: the cylinder A-scan and the merged pipe B-scan the loader tests
  use (with their ``pipe.in``, ``ground_truth.json`` and ``cases.csv`` row),
  and one field_400mhz val scene as ``.npz`` (chosen by the 07a rule).
- ``mendeley/``: 5 train-split originals with boxes (3 utility, 2 void) and 2
  intact train originals as negatives, bytes unchanged, with YOLO labels
  (empty for intact). Test ids are refused (checked against the test lock).
- ``usgs/``: traces 0 to 299 of Dinner Point Avenue as ``.npz``.
- ``CITATION.txt``: sources and licences.

Run in the gprMax-v4 environment from the repository root:

    python src/00_make_sample_data.py

It prints the total size and stops if it is over 15 MB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")
demo = importlib.import_module("07a_export_demo_samples")

DEFAULT_OUT = ROOT / "data" / "sample"
SIM = ROOT / "data" / "sim"
PIPE_TAG = "f1.5_e6_s0_d9"  # the B-scan used by test/02a_test_gprmax_loader.py
PIPE_FILES = ("pipe_merged.h5", "pipe.in", "ground_truth.json")
ASCAN = "cylinder_Ascan_2D.h5"
DINNER = ROOT / "data" / "raw" / "usgs_forsythe" / "DinnerPointAve.rd3"
DINNER_TRACES = (0, 300)
N_UTILITY, N_VOID, N_INTACT = 3, 2, 2
MAX_BYTES = 15 * 1024 * 1024

CITATION = """\
GPRadar sample data (data/sample/)
==================================

Built by src/00_make_sample_data.py from the full datasets. Small subsets,
for tests and the demo only.

mendeley/  Mendeley GPR images (CC BY 4.0)
  Mojahid, K. et al. (2025). Intelligent recognition of subsurface utilities
  and voids using ground-penetrating radar and deep learning techniques: Data
  of GPR profiles images. Mendeley Data, V1. DOI 10.17632/ww7fd9t325.1
  Licence: Creative Commons Attribution 4.0 (CC BY 4.0). The images are copied
  unchanged. The YOLO box labels are this project's own hand labels (one class,
  "target"); intact images are negatives with empty label files. All 7 images
  are from the train split of data/splits/mendeley_originals.csv; no image from
  the locked test split is included.

usgs/  Dinner Point Avenue, traces {t0} to {t1} (U.S. Geological Survey)
  Fiore, A.R. (2018). Raw ground-penetrating radar data, Edwin B. Forsythe
  National Wildlife Refuge, New Jersey, 2014-15. U.S. Geological Survey data
  release. https://doi.org/10.5066/F7JH3KBD
  Related report: Fiore, A.R., Voronin, L.M. and Wieben, C.M. (2018), USGS
  Scientific Investigations Report 2017-5135, https://doi.org/10.3133/sir20175135
  Redistribution: the release metadata lists no access or use constraints
  (accconst and useconst are "none"), and data produced by the USGS are in
  the U.S. public domain. USGS asks that the data be cited as above. The
  window is the raw int16 samples of DinnerPointAve.rd3 with no processing;
  "header" in the .npz is the original .rad text.

sim/  gprMax simulations (this project)
  Produced by this project with gprMax 4.0.1. Please cite:
  Warren, C., Giannopoulos, A. and Giannakis, I. (2016). gprMax: Open source
  software to simulate electromagnetic wave propagation for Ground Penetrating
  Radar. Computer Physics Communications 209, 163-170.
  https://doi.org/10.1016/j.cpc.2016.08.020
"""


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _copy(src, dst):
    """Copy one file, creating the parent folder."""

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _pipe_case_row(tag):
    """The cases.csv row of one pipe-study case."""

    with (SIM / "pipe_study" / "cases.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["tag"] == tag]
        fields = reader.fieldnames
    if not rows:
        raise ValueError(f"{tag} not in cases.csv")
    return fields, rows[0]


def copy_sim(out):
    """Cylinder A-scan, one pipe B-scan with its model files, and cases.csv."""

    _copy(SIM / ASCAN, out / "sim" / ASCAN)
    for name in PIPE_FILES:
        _copy(SIM / "pipe_study" / PIPE_TAG / name, out / "sim" / "pipe_study" / PIPE_TAG / name)
    fields, row = _pipe_case_row(PIPE_TAG)
    with (out / "sim" / "pipe_study" / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(f"sim: {ASCAN}, pipe_study/{PIPE_TAG}")


def export_scene_sample(out):
    """One field_400mhz val scene, chosen and written as in 07a."""

    info = demo.choose_scene()
    seed = info["seed"]
    name = f"field_400mhz_seed{seed}.npz"
    size, *_ = demo.export_scene(demo.DEFAULT_SCENES / f"seed{seed}", out / "sim" / name)
    print(f"sim: {name} ({size / 1e6:.2f} MB), visible {info['visible']}")


def _ranked(train_rows, labels_dir, class_name):
    """Train ids of a class with 1 to 2 boxes, largest box first (07a rule)."""

    ranked = []
    for row in train_rows:
        if row["class"] != class_name:
            continue
        boxes = demo.read_yolo(Path(labels_dir) / f"{row['id']}.txt")
        if 1 <= len(boxes) <= 2:
            ranked.append((-max(w * h for *_r, w, h in boxes), row["id"], row))
    return [row for *_k, row in sorted(ranked, key=lambda item: item[:2])]


def _first_distinct_groups(rows, n):
    """First ``n`` rows that do not share a near-duplicate group."""

    picked, groups = [], set()
    for row in rows:
        if row["group_id"] not in groups:
            picked.append(row)
            groups.add(row["group_id"])
        if len(picked) == n:
            break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} rows from distinct groups")
    return picked


def choose_mendeley(train_rows, labels_dir=demo.DEFAULT_LABELS):
    """3 utility + 2 void (07a rule, distinct groups) and 2 intact (lowest ids, no copies)."""

    utility = _first_distinct_groups(_ranked(train_rows, labels_dir, "utility"), N_UTILITY)
    void = _first_distinct_groups(_ranked(train_rows, labels_dir, "void"), N_VOID)
    # "N (2).JPG" files are second copies of "N.JPG" in the dataset; skip them so
    # the two negatives are different scans even when dHash split them apart.
    intact_rows = sorted(
        (r for r in train_rows if r["class"] == "intact" and " (" not in r["path"]),
        key=lambda r: r["id"],
    )
    intact = _first_distinct_groups(intact_rows, N_INTACT)
    return utility + void + intact


def copy_mendeley(out, rows, test_ids, raw_root=demo.DEFAULT_RAW, labels_dir=demo.DEFAULT_LABELS):
    """Copy images and labels; write samples.csv. Refuses test ids."""

    folder = out / "mendeley"
    table = []
    for row in rows:
        prep.assert_not_test(row["id"], test_ids)
        src = Path(raw_root) / row["path"]
        image = _copy(src, folder / "images" / f"{row['id']}{src.suffix.lower()}")
        label = folder / "labels" / f"{row['id']}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        source_label = Path(labels_dir) / f"{row['id']}.txt"
        if row["class"] == "intact":
            label.write_text("", encoding="utf-8")
        else:
            shutil.copyfile(source_label, label)
        if _md5(image) != row["md5"]:
            raise ValueError(f"{row['id']}: md5 differs from the split table")
        n_boxes = len(demo.read_yolo(label))
        table.append({"id": row["id"], "class": row["class"], "file": image.name,
                      "boxes": n_boxes, "original_path": row["path"], "md5": row["md5"]})
    with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("mendeley: " + ", ".join(f"{r['id']} ({r['boxes']})" for r in table))


def export_dinner_window(out, path=DINNER, traces=DINNER_TRACES):
    """Raw int16 traces [t0, t1) of Dinner Point Avenue, with dt, spacing, frequency."""

    mala = importlib.import_module("02d_mala_reader")
    scan = mala.load_mala(path)
    t0, t1 = traces
    raw = np.fromfile(Path(path).with_suffix(".rd3"), dtype="<i2").reshape(-1, scan.data.shape[0]).T
    window = raw[:, t0:t1]
    if not np.array_equal(window.astype(np.float64), scan.data[:, t0:t1]):
        raise ValueError("raw window does not match load_mala")
    target = out / "usgs" / f"DinnerPointAve_traces{t0:04d}-{t1 - 1:04d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        data=window,
        dt=np.float64(scan.dt),
        trace_spacing=np.float64(scan.trace_spacing),
        frequency=np.float64(scan.frequency),
        first_trace=np.int64(t0),
        header=np.array(Path(path).with_suffix(".rad").read_text(errors="replace")),
    )
    print(f"usgs: {target.name} {window.shape}")


def total_size(folder):
    """Bytes of every file under ``folder``."""

    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.out
    if (out / "CITATION.txt").is_file():
        # A previous build of this script: clear its folders so no stale file stays.
        for name in ("sim", "mendeley", "usgs"):
            shutil.rmtree(out / name, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    train_rows, test_ids = demo.checked_train_rows()
    print("test lock: ok")
    copy_sim(out)
    export_scene_sample(out)
    copy_mendeley(out, choose_mendeley(train_rows), test_ids)
    export_dinner_window(out)
    t0, t1 = DINNER_TRACES
    (out / "CITATION.txt").write_text(CITATION.format(t0=t0, t1=t1 - 1), encoding="utf-8")

    size = total_size(out)
    print(f"total: {size / 1e6:.2f} MB in {out}")
    if size > MAX_BYTES:
        raise SystemExit(f"sample data is over {MAX_BYTES / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Carve real YOLO train/val/test folders from the locked Mendeley split.

Validation is 20% of train groups by class (seed 0). Nested 25% and 100%
subsets of the remaining train pool are drawn by group for seeds 0, 1, 2.
Test images are written only under ``data/real_yolo/test/`` for ``06c``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

split_mod = importlib.import_module("04b_split_mendeley")

DEFAULT_SPLIT_CSV = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_LOCK = ROOT / "configs" / "mendeley_test_lock.txt"
DEFAULT_RAW = ROOT / "data" / "raw" / "mendeley_gpr"
DEFAULT_LABELS = ROOT / "data" / "labels" / "mendeley_originals"
DEFAULT_OUT = ROOT / "data" / "real_yolo"
DEFAULT_SUBSET_CSV = ROOT / "data" / "real_yolo" / "subset_membership.csv"
VAL_FRACTION = 0.20
FRACTIONS = (0.25, 1.0)
SUBSET_SEEDS = (0, 1, 2)
CLASSES = ("void", "utility", "intact")


class TestSplitError(RuntimeError):
    """Raised when a locked test id would be used for training or selection."""


def verify_test_lock(split_csv=DEFAULT_SPLIT_CSV, lock_path=DEFAULT_LOCK):
    """Return sorted test ids after checking the committed SHA-256."""

    rows = split_mod.read_split_csv(split_csv)
    test_ids = [row["id"] for row in rows if row["split"] == "test"]
    digest = split_mod.test_lock_hash(test_ids)
    expected = Path(lock_path).read_text(encoding="utf-8").strip()
    if digest != expected:
        raise RuntimeError(
            f"test lock mismatch: got {digest}, expected {expected}. Stop."
        )
    return sorted(test_ids), rows


def assert_not_test(image_id, test_ids):
    """Fail if ``image_id`` belongs to the locked test split."""

    if image_id in test_ids:
        raise TestSplitError(f"refusing to use test-split id: {image_id}")


def groups_by_class(rows, split_name="train"):
    """Map class -> list of (group_id, [rows]) for one split."""

    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] != split_name:
            continue
        buckets[row["class"]][row["group_id"]].append(row)
    out = {}
    for class_name in CLASSES:
        items = [
            (group_id, sorted(members, key=lambda r: r["id"]))
            for group_id, members in sorted(buckets[class_name].items())
        ]
        out[class_name] = items
    return out


def stratified_group_sample(groups_by_cls, fraction, seed, exclude=None):
    """Sample a fraction of groups per class. Returns set of group ids."""

    exclude = set(exclude or [])
    rng = random.Random(int(seed))
    chosen = set()
    for class_name in CLASSES:
        available = [
            group_id
            for group_id, _members in groups_by_cls[class_name]
            if group_id not in exclude
        ]
        rng.shuffle(available)
        n_take = int(round(len(available) * float(fraction)))
        if fraction > 0 and available and n_take == 0:
            n_take = 1
        if fraction >= 1.0:
            n_take = len(available)
        chosen.update(available[:n_take])
    return chosen


def carve_validation(train_groups, seed=0, fraction=VAL_FRACTION):
    """Return (val_group_ids, pool_group_ids)."""

    val_ids = stratified_group_sample(train_groups, fraction, seed)
    all_ids = {
        group_id
        for class_name in CLASSES
        for group_id, _ in train_groups[class_name]
    }
    pool_ids = all_ids - val_ids
    return val_ids, pool_ids


def nested_fraction_subsets(train_groups, pool_ids, seeds=SUBSET_SEEDS):
    """For each seed, 25% subset contained in the 100% pool."""

    pool_groups = {
        class_name: [
            (gid, members)
            for gid, members in train_groups[class_name]
            if gid in pool_ids
        ]
        for class_name in CLASSES
    }
    result = {}
    full = set(pool_ids)
    for seed in seeds:
        subset_25 = stratified_group_sample(pool_groups, 0.25, seed)
        if not subset_25.issubset(full):
            raise RuntimeError("25% subset escaped the 100% pool")
        result[int(seed)] = {0.25: subset_25, 1.0: set(full)}
    return result


def count_images(rows, group_ids):
    """Image counts by class for the given groups."""

    counts = Counter()
    for row in rows:
        if row["group_id"] in group_ids:
            counts[row["class"]] += 1
    return counts


def count_groups(group_ids, train_groups):
    """Group counts by class."""

    counts = Counter()
    for class_name in CLASSES:
        for group_id, _ in train_groups[class_name]:
            if group_id in group_ids:
                counts[class_name] += 1
    return counts


def to_grey_rgb_png(src_path, dst_path):
    """Convert any image to 8-bit greyscale saved as 3-channel PNG."""

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        grey = image.convert("L")
        rgb = Image.merge("RGB", (grey, grey, grey))
        rgb.save(dst_path, format="PNG")


def write_yolo_yaml(path, root, train_rel, val_rel, names=None):
    """Write a minimal Ultralytics dataset.yaml."""

    names = names or {0: "target"}
    lines = [
        f"path: {Path(root).resolve().as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        "names:",
    ]
    for key, value in names.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_label(src_labels, image_id, dst_label_path, class_name):
    """Write YOLO label file; intact images get an empty file."""

    dst_label_path = Path(dst_label_path)
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)
    if class_name == "intact":
        dst_label_path.write_text("", encoding="utf-8")
        return
    src = Path(src_labels) / f"{image_id}.txt"
    if not src.is_file():
        raise FileNotFoundError(src)
    dst_label_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def export_split_images(
    rows,
    group_ids,
    raw_root,
    labels_dir,
    images_dir,
    labels_out_dir,
    test_ids,
):
    """Write greyscale PNG images and labels for the given groups."""

    images_dir = Path(images_dir)
    labels_out_dir = Path(labels_out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        if row["group_id"] not in group_ids:
            continue
        assert_not_test(row["id"], test_ids)
        src = Path(raw_root) / row["path"]
        if not src.is_file():
            raise FileNotFoundError(src)
        dst_img = images_dir / f"{row['id']}.png"
        dst_lab = labels_out_dir / f"{row['id']}.txt"
        to_grey_rgb_png(src, dst_img)
        copy_label(labels_dir, row["id"], dst_lab, row["class"])
        written.append(row["id"])
    return written


def export_test_set(rows, raw_root, labels_dir, out_root, test_ids):
    """Write the locked test set under out_root/test (images + labels)."""

    test_rows = [row for row in rows if row["id"] in set(test_ids)]
    images_dir = Path(out_root) / "test" / "images"
    labels_dir_out = Path(out_root) / "test" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir_out.mkdir(parents=True, exist_ok=True)
    for row in test_rows:
        src = Path(raw_root) / row["path"]
        if not src.is_file():
            raise FileNotFoundError(src)
        to_grey_rgb_png(src, images_dir / f"{row['id']}.png")
        copy_label(labels_dir, row["id"], labels_dir_out / f"{row['id']}.txt", row["class"])
    write_yolo_yaml(
        Path(out_root) / "test" / "dataset.yaml",
        Path(out_root) / "test",
        "images",
        "images",
    )
    return len(test_rows)


def prepare_real_yolo(
    split_csv=DEFAULT_SPLIT_CSV,
    lock_path=DEFAULT_LOCK,
    raw_root=DEFAULT_RAW,
    labels_dir=DEFAULT_LABELS,
    out_root=DEFAULT_OUT,
    subset_csv=DEFAULT_SUBSET_CSV,
):
    """Build validation, nested subsets, YOLO folders and membership CSV."""

    test_ids, rows = verify_test_lock(split_csv, lock_path)
    test_id_set = set(test_ids)
    train_rows = [row for row in rows if row["split"] == "train"]
    for row in train_rows:
        assert_not_test(row["id"], test_id_set)

    train_groups = groups_by_class(rows, "train")
    val_ids, pool_ids = carve_validation(train_groups, seed=0)
    subsets = nested_fraction_subsets(train_groups, pool_ids)

    out_root = Path(out_root)
    # Shared validation once.
    val_img_dir = out_root / "val" / "images"
    val_lab_dir = out_root / "val" / "labels"
    export_split_images(
        train_rows, val_ids, raw_root, labels_dir, val_img_dir, val_lab_dir, test_id_set
    )
    write_yolo_yaml(out_root / "val" / "dataset.yaml", out_root / "val", "images", "images")

    membership = []
    for seed, fracs in subsets.items():
        for fraction, group_ids in sorted(fracs.items()):
            tag = f"seed{seed}_f{int(fraction * 100):03d}"
            img_dir = out_root / tag / "images"
            lab_dir = out_root / tag / "labels"
            export_split_images(
                train_rows,
                group_ids,
                raw_root,
                labels_dir,
                img_dir,
                lab_dir,
                test_id_set,
            )
            write_yolo_yaml(
                out_root / tag / "dataset.yaml",
                out_root,
                f"{tag}/images",
                "val/images",
            )
            for row in train_rows:
                if row["group_id"] not in group_ids:
                    continue
                membership.append(
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "class": row["class"],
                        "group_id": row["group_id"],
                        "role": "train_subset",
                        "seed": seed,
                        "fraction": fraction,
                        "subset_tag": tag,
                    }
                )

    for row in train_rows:
        if row["group_id"] in val_ids:
            membership.append(
                {
                    "id": row["id"],
                    "path": row["path"],
                    "class": row["class"],
                    "group_id": row["group_id"],
                    "role": "val",
                    "seed": "",
                    "fraction": "",
                    "subset_tag": "val",
                }
            )

    n_test = export_test_set(rows, raw_root, labels_dir, out_root, test_ids)

    subset_csv = Path(subset_csv)
    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "path",
        "class",
        "group_id",
        "role",
        "seed",
        "fraction",
        "subset_tag",
    ]
    with subset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(membership)

    summary = {
        "n_train_images": len(train_rows),
        "n_test_images": n_test,
        "val_images": dict(count_images(train_rows, val_ids)),
        "val_groups": dict(count_groups(val_ids, train_groups)),
        "pool_images": dict(count_images(train_rows, pool_ids)),
        "pool_groups": dict(count_groups(pool_ids, train_groups)),
        "subsets": {
            seed: {
                fraction: {
                    "images": dict(count_images(train_rows, gids)),
                    "groups": dict(count_groups(gids, train_groups)),
                    "n_images": sum(count_images(train_rows, gids).values()),
                    "n_groups": len(gids),
                }
                for fraction, gids in fracs.items()
            }
            for seed, fracs in subsets.items()
        },
        "subset_csv": str(subset_csv),
        "out_root": str(out_root),
        "test_lock_ok": True,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset-csv", type=Path, default=DEFAULT_SUBSET_CSV)
    args = parser.parse_args(argv)
    summary = prepare_real_yolo(
        split_csv=args.split_csv,
        lock_path=args.lock,
        raw_root=args.raw,
        labels_dir=args.labels,
        out_root=args.out,
        subset_csv=args.subset_csv,
    )
    print("test lock: ok")
    print(
        "val images",
        summary["val_images"],
        "groups",
        summary["val_groups"],
        "n=",
        sum(summary["val_images"].values()),
    )
    print(
        "pool images",
        summary["pool_images"],
        "groups",
        summary["pool_groups"],
        "n=",
        sum(summary["pool_images"].values()),
    )
    for seed, fracs in summary["subsets"].items():
        for fraction, info in sorted(fracs.items()):
            print(
                f"seed {seed} f={fraction}: images={info['n_images']} "
                f"groups={info['n_groups']} by_class={info['images']}"
            )
    print(f"test images written: {summary['n_test_images']}")
    print(f"subset csv: {summary['subset_csv']}")
    return summary


if __name__ == "__main__":
    main()

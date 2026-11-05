"""Locked split, labelling export and label import for Mendeley originals.

Uses only the 285 originals. Near-duplicates (dHash <= 6 under the same
transforms as ``04a_inventory_mendeley``) stay in one group and one split.
Augmented folders are not used. Subcommands: ``split``, ``export``, ``import``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

inventory = importlib.import_module("04a_inventory_mendeley")

DEFAULT_DATA = ROOT / "data" / "raw" / "mendeley_gpr"
DEFAULT_SPLIT_CSV = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_LOCK = ROOT / "configs" / "mendeley_test_lock.txt"
DEFAULT_EXPORT = ROOT / "data" / "raw" / "mendeley_labelling"
DEFAULT_LABELS = ROOT / "data" / "labels" / "mendeley_originals"
DEFAULT_REVIEW = ROOT / "figures" / "mendeley_split_review.png"
DEFAULT_GUIDE = ROOT / "figures" / "mendeley_labelling_guide.png"

ORIGINAL_FOLDERS = ("cavities", "Utilities", "intact")
NEAR_DUP_DIST = inventory.NEAR_DUP_DIST  # 6
REVIEW_LO = 7
REVIEW_HI = 10
TEST_FRACTION = 0.30
SEED = 0
MIN_BOX_PX = 4.0


class UnionFind:
    """Disjoint-set for near-duplicate groups."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            self.parent[ri] = rj
        elif self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1


def make_id(class_label, rel_path):
    """Stable id from class and original filename, e.g. ``void_0014``."""

    stem = Path(rel_path).stem.strip()
    match = re.fullmatch(r"(\d+)(?:\s*\((\d+)\))?", stem)
    if not match:
        safe = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
        return f"{class_label}_{safe}"
    number = int(match.group(1))
    suffix = match.group(2)
    if suffix:
        return f"{class_label}_{number:04d}_{int(suffix)}"
    return f"{class_label}_{number:04d}"


def list_originals(root):
    """Original images only, with class from the top folder."""

    root = Path(root)
    rows = []
    for folder in ORIGINAL_FOLDERS:
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"missing original folder: {directory}")
        label, kind, _ = inventory._folder_meta((folder,))
        if kind != "original":
            raise RuntimeError(f"expected original folder, got {folder}")
        for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in inventory.IMAGE_EXT:
                continue
            rel = path.relative_to(root).as_posix()
            image_id = make_id(label, rel)
            rows.append(
                {
                    "id": image_id,
                    "path": rel,
                    "class": label,
                    "abs_path": path,
                }
            )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        raise RuntimeError(f"non-unique ids: {dupes}")
    return rows


def pair_distance(hashes_a, hashes_b):
    """Minimum Hamming distance treating one image as a transform of the other.

    Compares the identity hash of each image to every transform hash of the
    other (the same idea as the augmented-to-original match in ``04a``).
    Comparing every transform to every transform is too weak on these
    thumbnails and merges unrelated classes.
    """

    best = 65
    identity = "identity"
    for name, hb in hashes_b.items():
        best = min(best, inventory.hamming64(hashes_a[identity], hb))
    for name, ha in hashes_a.items():
        best = min(best, inventory.hamming64(hashes_b[identity], ha))
    return best


def build_groups(originals):
    """Union-find near-duplicates at dHash <= 6. Return groups and review pairs.

    A group that mixes classes raises ``ValueError``.
    """

    n = len(originals)
    all_hashes = []
    for row in originals:
        grey = inventory.load_grey(row["abs_path"])
        all_hashes.append(
            {name: inventory.transform_hash(grey, name) for name in inventory.TRANSFORMS}
        )

    uf = UnionFind(n)
    review_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = pair_distance(all_hashes[i], all_hashes[j])
            if dist <= NEAR_DUP_DIST:
                uf.union(i, j)
            elif REVIEW_LO <= dist <= REVIEW_HI:
                review_pairs.append((i, j, dist))

    buckets = defaultdict(list)
    for index, row in enumerate(originals):
        buckets[uf.find(index)].append(index)

    groups = []
    cross = []
    for root_index, members in sorted(buckets.items(), key=lambda item: min(item[1])):
        classes = {originals[i]["class"] for i in members}
        if len(classes) > 1:
            cross.append(
                {
                    "members": [originals[i]["path"] for i in members],
                    "classes": sorted(classes),
                }
            )
            continue
        groups.append(
            {
                "class": next(iter(classes)),
                "member_indices": sorted(members),
                "paths": [originals[i]["path"] for i in sorted(members)],
                "ids": [originals[i]["id"] for i in sorted(members)],
            }
        )

    if cross:
        lines = [
            f"cross-class near-duplicate group: classes={g['classes']} members={g['members']}"
            for g in cross
        ]
        raise ValueError("\n".join(lines))

    # Stable group ids: by class then first id.
    groups.sort(key=lambda g: (g["class"], g["ids"][0]))
    for index, group in enumerate(groups):
        group["group_id"] = f"g{index:04d}"

    review_pairs.sort(key=lambda item: (item[2], originals[item[0]]["id"], originals[item[1]]["id"]))
    return groups, review_pairs


def stratified_split(groups, seed=SEED, test_fraction=TEST_FRACTION):
    """Assign each group to train or test. Stratify by class."""

    rng = random.Random(seed)
    assignment = {}  # group_id -> split
    by_class = defaultdict(list)
    for group in groups:
        by_class[group["class"]].append(group)

    for class_label in sorted(by_class):
        pool = list(by_class[class_label])
        rng.shuffle(pool)
        n_test = int(round(len(pool) * test_fraction))
        n_test = min(max(n_test, 1 if len(pool) > 1 else 0), len(pool) - (1 if len(pool) > 1 else 0))
        for index, group in enumerate(pool):
            assignment[group["group_id"]] = "test" if index < n_test else "train"
    return assignment


def test_lock_hash(test_ids):
    """SHA-256 of the sorted test ids, one per line."""

    payload = "\n".join(sorted(test_ids)) + ("\n" if test_ids else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_csv(path, rows):
    """Write the committed split table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "path", "class", "group_id", "split", "md5"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def read_split_csv(path):
    """Load the split table."""

    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_review_pairs(originals, review_pairs, out_path, max_pairs=40):
    """Side-by-side originals with dHash distance 7-10."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = review_pairs[:max_pairs]
    if not pairs:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.set_title("No original pairs with dHash 7-10")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return 0

    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(7, 1.5 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    for index, (i, j, dist) in enumerate(pairs):
        for col, member in ((0, i), (1, j)):
            ax = axes[index, col]
            path = originals[member]["abs_path"]
            with Image.open(path) as image:
                ax.imshow(np.asarray(image.convert("L")), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            title = f"{originals[member]['id']}  d={dist}" if col == 0 else originals[member]["id"]
            ax.set_title(title, fontsize=8)
    fig.suptitle("Original pairs with dHash 7-10 (not merged)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return len(pairs)


def split_counts(rows, groups, assignment):
    """Images and groups per class and split."""

    image_counts = Counter((row["class"], row["split"]) for row in rows)
    group_counts = Counter()
    for group in groups:
        group_counts[(group["class"], assignment[group["group_id"]])] += 1
    return image_counts, group_counts


def format_count_table(image_counts, group_counts):
    """Markdown-ish table for the console."""

    lines = [
        "| Class | Split | Images | Groups |",
        "| --- | --- | ---: | ---: |",
    ]
    for class_label in ("void", "utility", "intact"):
        for split_name in ("train", "test"):
            lines.append(
                f"| {class_label} | {split_name} | "
                f"{image_counts[(class_label, split_name)]} | "
                f"{group_counts[(class_label, split_name)]} |"
            )
    return "\n".join(lines)


def cmd_split(args):
    """Build the locked original split."""

    root = Path(args.root)
    originals = list_originals(root)
    groups, review_pairs = build_groups(originals)
    assignment = stratified_split(groups, seed=args.seed)

    id_to_group = {}
    for group in groups:
        for image_id in group["ids"]:
            id_to_group[image_id] = group["group_id"]

    rows = []
    for row in originals:
        group_id = id_to_group[row["id"]]
        rows.append(
            {
                "id": row["id"],
                "path": row["path"],
                "class": row["class"],
                "group_id": group_id,
                "split": assignment[group_id],
                "md5": inventory.md5_file(row["abs_path"]),
            }
        )
    rows.sort(key=lambda item: item["id"])

    test_ids = [row["id"] for row in rows if row["split"] == "test"]
    digest = test_lock_hash(test_ids)
    lock_path = Path(args.lock)
    if lock_path.is_file():
        existing = lock_path.read_text(encoding="utf-8").strip()
        if existing != digest:
            raise SystemExit(
                f"refusing to overwrite locked test set at {lock_path}: "
                f"existing={existing} new={digest}"
            )
        print(f"Lock unchanged: {digest}")
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(digest + "\n", encoding="utf-8")
        print(f"Wrote lock {lock_path}: {digest}")

    write_split_csv(args.split_csv, rows)
    n_review = plot_review_pairs(originals, review_pairs, args.review_figure)
    image_counts, group_counts = split_counts(rows, groups, assignment)
    print(format_count_table(image_counts, group_counts))
    print(f"Groups: {len(groups)}; review pairs (dHash 7-10): {len(review_pairs)} "
          f"(shown {n_review})")
    print(f"Wrote {args.split_csv}")
    print(f"Wrote {args.review_figure}")
    return rows, groups, review_pairs, digest


def cmd_export(args):
    """Copy cavity and utility originals for blind labelling."""

    root = Path(args.root)
    export_root = Path(args.export_dir)
    images_dir = export_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_split_csv(args.split_csv)
    targets = [row for row in split_rows if row["class"] in ("void", "utility")]
    if len(targets) != 210 and root == DEFAULT_DATA:
        print(f"warning: expected 210 cavity/utility originals, found {len(targets)}")

    for row in targets:
        src = root / row["path"]
        if not src.is_file():
            raise FileNotFoundError(src)
        dest = images_dir / f"{row['id']}.jpg"
        shutil.copyfile(src, dest)
        if inventory.md5_file(src) != inventory.md5_file(dest):
            raise RuntimeError(f"copy changed bytes: {row['id']}")

    labels_txt = export_root / "labels.txt"
    labels_txt.write_text("target\n", encoding="utf-8")
    plot_labelling_guide(root, Path(args.guide_figure))
    print(f"Copied {len(targets)} images to {images_dir}")
    print(f"Wrote {labels_txt}")
    print(f"Wrote {args.guide_figure}")


def plot_labelling_guide(root, out_path, n_per_class=12):
    """Augmented void and utility images with the authors' boxes."""

    root = Path(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks = []
    for folder, label in (("augmented_cavities", "void"), ("augmented_utilities", "utility")):
        directory = root / folder
        images = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in inventory.IMAGE_EXT
        )
        for path in images:
            yolo, voc = inventory.find_annotations(path, root)
            if not yolo and not voc:
                continue
            picks.append((path, yolo, voc, label))
            if sum(1 for item in picks if item[3] == label) >= n_per_class:
                break

    if len(picks) < 2:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.set_title("No augmented labelled images found for the guide")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for index, (path, yolo, voc, label) in enumerate(picks):
        ax = axes[index // cols, index % cols]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            draw = ImageDraw.Draw(rgb)
            boxes = []
            if voc is not None:
                boxes = inventory.parse_voc(voc)
            elif yolo is not None:
                boxes = inventory.parse_yolo(yolo, width, height)
            for box in boxes:
                draw.rectangle(
                    [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                    outline=(255, 40, 40),
                    width=2,
                )
            ax.imshow(np.asarray(rgb))
        ax.set_title(f"{label} {path.name}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for index in range(len(picks), rows * cols):
        axes[index // cols, index % cols].axis("off")
    fig.suptitle(
        "Labelling guide: authors' boxes on augmented images (one target per box)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def validate_yolo_labels(images_dir, labels_dir):
    """Return a list of failure strings. Empty means valid."""

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    failures = []
    images = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in inventory.IMAGE_EXT
    }
    label_files = {
        p.stem: p
        for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }

    for stem, path in sorted(label_files.items()):
        if stem not in images:
            failures.append(f"label without image: {path.name}")

    for stem, image_path in sorted(images.items()):
        label_path = label_files.get(stem)
        if label_path is None:
            failures.append(f"missing label file for {image_path.name}")
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            failures.append(f"empty label file: {label_path.name}")
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                class_id, xmin, ymin, xmax, ymax = inventory.yolo_to_pixels(line, width, height)
            except ValueError as exc:
                failures.append(f"{label_path.name}:{line_no}: {exc}")
                continue
            if class_id != "0":
                failures.append(f"{label_path.name}:{line_no}: class id {class_id!r} is not 0")
            box_w = xmax - xmin
            box_h = ymax - ymin
            if xmin < -1e-6 or ymin < -1e-6 or xmax > width + 1e-6 or ymax > height + 1e-6:
                failures.append(
                    f"{label_path.name}:{line_no}: box outside image "
                    f"({xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}) vs {width}x{height}"
                )
            if box_w < MIN_BOX_PX or box_h < MIN_BOX_PX:
                failures.append(
                    f"{label_path.name}:{line_no}: box too small "
                    f"{box_w:.2f}x{box_h:.2f} px (min {MIN_BOX_PX})"
                )
        # At least one box.
        boxes = [ln for ln in text.splitlines() if ln.strip()]
        if not boxes:
            failures.append(f"no boxes in {label_path.name}")
    return failures


def cmd_import(args):
    """Validate makesense YOLO labels and write committed label files."""

    export_root = Path(args.export_dir)
    images_dir = export_root / "images"
    labels_dir = Path(args.labels_yolo)
    if not images_dir.is_dir():
        raise SystemExit(f"export images not found: {images_dir} (run export first)")
    if not labels_dir.is_dir():
        raise SystemExit(f"YOLO labels not found: {labels_dir}")

    failures = validate_yolo_labels(images_dir, labels_dir)
    if failures:
        print("Import validation failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    out_dir = Path(args.labels_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {row["id"]: row for row in read_split_csv(args.split_csv)}

    stats = defaultdict(list)  # (class, split) -> list of (n_boxes, w_fracs, h_fracs)
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in inventory.IMAGE_EXT:
            continue
        image_id = image_path.stem
        src = labels_dir / f"{image_id}.txt"
        dest = out_dir / f"{image_id}.txt"
        text = src.read_text(encoding="utf-8")
        dest.write_text(text, encoding="utf-8")
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = inventory.parse_yolo(src, width, height)
        w_fracs = [(b["xmax"] - b["xmin"]) / width * 100.0 for b in boxes]
        h_fracs = [(b["ymax"] - b["ymin"]) / height * 100.0 for b in boxes]
        meta = split_rows.get(image_id, {"class": "?", "split": "?"})
        stats[(meta["class"], meta["split"])].append((len(boxes), w_fracs, h_fracs))

    print(f"Wrote {len(list(out_dir.glob('*.txt')))} label files to {out_dir}")
    print("| Class | Split | Images | Boxes | Median box W (% ) | Median box H (% ) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for key in sorted(stats):
        class_label, split_name = key
        entries = stats[key]
        n_images = len(entries)
        n_boxes = sum(item[0] for item in entries)
        all_w = [w for item in entries for w in item[1]]
        all_h = [h for item in entries for h in item[2]]
        med_w = float(np.median(all_w)) if all_w else float("nan")
        med_h = float(np.median(all_h)) if all_h else float("nan")
        print(
            f"| {class_label} | {split_name} | {n_images} | {n_boxes} | "
            f"{med_w:.1f} | {med_h:.1f} |"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    split_p = sub.add_parser("split", help="Group near-duplicates and write the locked split")
    split_p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    split_p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    split_p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    split_p.add_argument("--review-figure", type=Path, default=DEFAULT_REVIEW)
    split_p.add_argument("--seed", type=int, default=SEED)

    export_p = sub.add_parser("export", help="Copy cavity/utility originals for labelling")
    export_p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    export_p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    export_p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    export_p.add_argument("--guide-figure", type=Path, default=DEFAULT_GUIDE)

    import_p = sub.add_parser("import", help="Validate and store YOLO labels")
    import_p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    import_p.add_argument(
        "--labels-yolo",
        type=Path,
        default=DEFAULT_EXPORT / "labels_yolo",
    )
    import_p.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS)
    import_p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "split":
        cmd_split(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()

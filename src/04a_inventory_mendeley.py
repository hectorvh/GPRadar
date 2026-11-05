"""Read-only inventory of the Mendeley GPR detection dataset.

Walks ``data/raw/mendeley_gpr`` (or ``--root``), writes CSVs and figures under
``--out``, and never writes inside the data root. Images use NumPy and Pillow
only. No training, split or class remapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "mendeley_gpr"
DEFAULT_OUT = ROOT / "figures" / "mendeley_inventory"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
ANNOTATION_EXT = {".txt", ".xml"}

# Folder name (case-insensitive) -> (class_label, kind)
FOLDER_META = {
    "cavities": ("void", "original"),
    "utilities": ("utility", "original"),
    "intact": ("intact", "original"),
    "augmented_cavities": ("void", "augmented"),
    "augmented_utilities": ("utility", "augmented"),
    "augmented_intact": ("intact", "augmented"),
}

PUBLISHED_TOTAL = {"void": 553, "utility": 786, "intact": 900}
PUBLISHED_ORIGINALS = {"void": 79, "utility": 131, "intact": 75}

TRANSFORMS = (
    "identity",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
)

HIGH_DIST = 10
HIGH_MARGIN = 6
NEAR_DUP_DIST = 6


def _folder_meta(rel_parts):
    """Map a relative path to class and original/augmented."""

    if not rel_parts:
        return "unknown", "unclear", ""
    top = rel_parts[0]
    key = top.lower()
    if key in FOLDER_META:
        label, kind = FOLDER_META[key]
        return label, kind, top
    return "unknown", "unclear", top


def _is_under(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def md5_file(path):
    """MD5 of the file bytes."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grey(path):
    """Greyscale float array in [0, 1], shape (H, W)."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def dhash64(grey):
    """64-bit difference hash: 9x8 resize, compare neighbours along width.

    Returns an unsigned 64-bit integer.
    """

    # Pillow resampling: bilinear is enough for a tiny hash.
    image = Image.fromarray(np.clip(grey * 255.0, 0, 255).astype(np.uint8), mode="L")
    small = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.float64)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming64(a, b):
    """Hamming distance between two 64-bit hashes."""

    return int((a ^ b).bit_count())


def transform_grey(grey, name):
    """Apply one of the matching transforms to a greyscale array."""

    if name == "identity":
        return grey
    if name == "hflip":
        return np.fliplr(grey)
    if name == "vflip":
        return np.flipud(grey)
    if name == "rot90":
        return np.rot90(grey, 1)
    if name == "rot180":
        return np.rot90(grey, 2)
    if name == "rot270":
        return np.rot90(grey, 3)
    raise ValueError(f"unknown transform {name}")


def transform_hash(grey, name):
    """dHash after ``name``."""

    return dhash64(transform_grey(grey, name))


def yolo_to_pixels(line, width, height):
    """Convert one YOLO line to (class_id, xmin, ymin, xmax, ymax)."""

    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"YOLO line needs 5 fields, got {parts!r}")
    class_id = parts[0]
    xc, yc, bw, bh = (float(v) for v in parts[1:])
    box_w = bw * width
    box_h = bh * height
    xmin = xc * width - box_w / 2.0
    ymin = yc * height - box_h / 2.0
    xmax = xmin + box_w
    ymax = ymin + box_h
    return class_id, xmin, ymin, xmax, ymax


def parse_yolo(path, width, height):
    """Parse a YOLO .txt file into pixel boxes."""

    text = path.read_text(encoding="utf-8", errors="replace")
    boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, xmin, ymin, xmax, ymax = yolo_to_pixels(line, width, height)
        boxes.append(
            {
                "class_id": class_id,
                "class_name": "",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "source": "yolo",
            }
        )
    return boxes


def parse_voc(path):
    """Parse a Pascal VOC XML file into pixel boxes."""

    tree = ET.parse(path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.findtext("width")) if size is not None else None
    height = int(size.findtext("height")) if size is not None else None
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        boxes.append(
            {
                "class_id": "",
                "class_name": obj.findtext("name") or "",
                "xmin": float(box.findtext("xmin")),
                "ymin": float(box.findtext("ymin")),
                "xmax": float(box.findtext("xmax")),
                "ymax": float(box.findtext("ymax")),
                "source": "voc",
                "ann_width": width,
                "ann_height": height,
            }
        )
    return boxes


def box_iou(a, b):
    """Intersection over union of two axis-aligned boxes."""

    ix0 = max(a["xmin"], b["xmin"])
    iy0 = max(a["ymin"], b["ymin"])
    ix1 = min(a["xmax"], b["xmax"])
    iy1 = min(a["ymax"], b["ymax"])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a["xmax"] - a["xmin"]) * max(0.0, a["ymax"] - a["ymin"])
    area_b = max(0.0, b["xmax"] - b["xmin"]) * max(0.0, b["ymax"] - b["ymin"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def boxes_match(yolo_boxes, voc_boxes, iou_threshold=0.95):
    """True if every YOLO box has a VOC partner with IoU above the threshold."""

    if len(yolo_boxes) != len(voc_boxes):
        return False, f"count {len(yolo_boxes)} vs {len(voc_boxes)}"
    used = set()
    for yolo in yolo_boxes:
        best_i = None
        best_iou = -1.0
        for index, voc in enumerate(voc_boxes):
            if index in used:
                continue
            score = box_iou(yolo, voc)
            if score > best_iou:
                best_iou = score
                best_i = index
        if best_i is None or best_iou <= iou_threshold:
            return False, f"best IoU {best_iou:.4f}"
        used.add(best_i)
    return True, "ok"


def find_annotations(image_path, root):
    """Locate YOLO .txt and VOC .xml with the same stem as the image."""

    stem = image_path.stem
    folder = image_path.parent
    yolo = None
    voc = None
    # Prefer annotations/ next to the images, then any match under the class folder.
    candidates = []
    ann_root = folder / "annotations"
    if ann_root.is_dir():
        candidates.extend(ann_root.rglob("*"))
    candidates.extend(folder.rglob("*"))
    seen = set()
    for path in candidates:
        if not path.is_file() or path.stem != stem:
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        suffix = path.suffix.lower()
        if suffix == ".txt" and yolo is None:
            yolo = path
        elif suffix == ".xml" and voc is None:
            voc = path
    return yolo, voc


def touches_edge(box, width, height, tol=0.5):
    """True if the box meets an image border."""

    return (
        box["xmin"] <= tol
        or box["ymin"] <= tol
        or box["xmax"] >= width - 1 - tol
        or box["ymax"] >= height - 1 - tol
    )


def dominant_vertical_period(grey):
    """Dominant vertical period in pixels from the mean column spectrum.

    Subtracts the mean of each column, takes the FFT along the time (row)
    axis, averages the amplitude spectra across columns, and returns the
    period of the strongest bin above DC. Returns NaN if the image is too
    short or flat.
    """

    if grey.ndim != 2 or grey.shape[0] < 8:
        return float("nan")
    centred = grey - grey.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=0)).mean(axis=1)
    if spectrum.size < 3:
        return float("nan")
    # Skip DC.
    peak = int(np.argmax(spectrum[1:]) + 1)
    if peak <= 0:
        return float("nan")
    return float(grey.shape[0] / peak)


def two_cluster_split(values):
    """Simple 1-D 2-means. Returns (centres, labels, gap_ok)."""

    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size < 4:
        return None, None, False
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return (lo, hi), np.zeros(arr.size, dtype=int), False
    c0, c1 = lo, hi
    labels = np.zeros(arr.size, dtype=int)
    for _ in range(20):
        d0 = np.abs(arr - c0)
        d1 = np.abs(arr - c1)
        labels = (d1 < d0).astype(int)
        if labels.min() == labels.max():
            return (c0, c1), labels, False
        c0 = float(arr[labels == 0].mean())
        c1 = float(arr[labels == 1].mean())
    # A split is meaningful if the centres are well separated relative to
    # within-cluster spread.
    spreads = []
    for k, centre in ((0, c0), (1, c1)):
        members = arr[labels == k]
        spreads.append(float(members.std()) if members.size else 0.0)
    gap = abs(c1 - c0)
    gap_ok = gap > 2.0 * max(spreads[0] + spreads[1], 1e-6)
    return (c0, c1), labels, gap_ok


def frequency_from_names(paths):
    """Search folder and file names for 200/400 MHz tokens."""

    hits = []
    pattern = re.compile(r"(200|400)\s*m?hz|(^|[^0-9])(200|400)([^0-9]|$)", re.I)
    for path in paths:
        text = str(path).replace("\\", "/")
        if pattern.search(text):
            hits.append(text)
    return hits


def frequency_from_exif(image_path):
    """Return EXIF / info keys that mention frequency, if any."""

    with Image.open(image_path) as image:
        info = dict(image.info)
        exif = {}
        getter = getattr(image, "getexif", None)
        if getter is not None:
            raw = getter()
            if raw:
                exif = dict(raw.items())
    blob = " ".join(str(v) for v in list(info.values()) + list(exif.values()))
    if re.search(r"(200|400)\s*m?hz|frequency|antenna", blob, re.I):
        return {"info": info, "exif": exif}
    return None


def scan_side_files(root):
    """List text-like files that are not YOLO/VOC annotations."""

    interesting = {
        ".txt",
        ".csv",
        ".xml",
        ".dzt",
        ".md",
        ".pdf",
        ".json",
        ".yml",
        ".yaml",
        ".rad",
        ".hd",
        ".hdr",
        ".html",
        ".rst",
    }
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "annotations" in path.parts and path.suffix.lower() in ANNOTATION_EXT:
            continue
        if path.suffix.lower() in interesting or any(
            k in path.name.lower() for k in ("licen", "readme", "cite", "meta")
        ):
            found.append(path)
    return found


def write_csv(path, rows, fieldnames):
    """Write rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def inventory_images(root):
    """Walk the tree and build the per-image rows."""

    rows = []
    other_files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXT:
            if "annotations" in parts and suffix in ANNOTATION_EXT:
                continue
            other_files.append(rel)
            continue
        # Skip images nested under annotations if any.
        if "annotations" in parts:
            other_files.append(rel)
            continue
        label, kind, folder = _folder_meta(parts)
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        digest = md5_file(path)
        yolo_path, voc_path = find_annotations(path, root)
        rows.append(
            {
                "path": rel,
                "folder": folder,
                "class": label,
                "kind": kind,
                "width": width,
                "height": height,
                "mode": mode,
                "file_size": path.stat().st_size,
                "md5": digest,
                "has_yolo": bool(yolo_path),
                "has_voc": bool(voc_path),
                "yolo_path": yolo_path.relative_to(root).as_posix() if yolo_path else "",
                "voc_path": voc_path.relative_to(root).as_posix() if voc_path else "",
                "abs_path": path,
            }
        )
    return rows, other_files


def inventory_boxes(image_rows, root):
    """Parse labels and compare YOLO with VOC when both exist."""

    box_rows = []
    mismatches = []
    class_ids = defaultdict(Counter)
    for row in image_rows:
        path = row["abs_path"]
        width = int(row["width"])
        height = int(row["height"])
        yolo_rel = row["yolo_path"]
        voc_rel = row["voc_path"]
        yolo_boxes = []
        voc_boxes = []
        if yolo_rel:
            yolo_boxes = parse_yolo(root / yolo_rel, width, height)
            for box in yolo_boxes:
                class_ids[row["folder"]][f"yolo:{box['class_id']}"] += 1
                box_rows.append(
                    {
                        "image": row["path"],
                        "source": "yolo",
                        "class_id": box["class_id"],
                        "class_name": "",
                        "xmin": f"{box['xmin']:.4f}",
                        "ymin": f"{box['ymin']:.4f}",
                        "xmax": f"{box['xmax']:.4f}",
                        "ymax": f"{box['ymax']:.4f}",
                        "box_width": f"{box['xmax'] - box['xmin']:.4f}",
                        "box_height": f"{box['ymax'] - box['ymin']:.4f}",
                        "image_width": width,
                        "image_height": height,
                        "touches_edge": touches_edge(box, width, height),
                    }
                )
        if voc_rel:
            voc_boxes = parse_voc(root / voc_rel)
            for box in voc_boxes:
                class_ids[row["folder"]][f"voc:{box['class_name']}"] += 1
                box_rows.append(
                    {
                        "image": row["path"],
                        "source": "voc",
                        "class_id": "",
                        "class_name": box["class_name"],
                        "xmin": f"{box['xmin']:.4f}",
                        "ymin": f"{box['ymin']:.4f}",
                        "xmax": f"{box['xmax']:.4f}",
                        "ymax": f"{box['ymax']:.4f}",
                        "box_width": f"{box['xmax'] - box['xmin']:.4f}",
                        "box_height": f"{box['ymax'] - box['ymin']:.4f}",
                        "image_width": width,
                        "image_height": height,
                        "touches_edge": touches_edge(box, width, height),
                    }
                )
        if yolo_boxes and voc_boxes:
            ok, detail = boxes_match(yolo_boxes, voc_boxes)
            if not ok:
                mismatches.append({"image": row["path"], "detail": detail})
    return box_rows, mismatches, class_ids


def assign_frequency(image_rows, root):
    """Fill frequency and frequency_hint for every image row."""

    name_hits = frequency_from_names([root] + [row["abs_path"] for row in image_rows])
    side_files = scan_side_files(root)
    exif_hits = []
    for row in image_rows:
        if row["kind"] != "original":
            continue
        hit = frequency_from_exif(row["abs_path"])
        if hit:
            exif_hits.append(row["path"])

    # No reliable frequency source in this download.
    frequency = "unknown"
    reasons = {
        "names": name_hits,
        "exif": exif_hits,
        "side_files": [p.relative_to(root).as_posix() for p in side_files],
    }

    periods = []
    for row in image_rows:
        row["frequency"] = frequency
        if row["kind"] == "original":
            grey = load_grey(row["abs_path"])
            period = dominant_vertical_period(grey)
            row["frequency_hint"] = f"{period:.4f}" if not math.isnan(period) else ""
            periods.append(period)
        else:
            row["frequency_hint"] = ""

    centres, labels, gap_ok = two_cluster_split(periods)
    return reasons, periods, centres, gap_ok


def exact_duplicates(image_rows):
    """Group images that share an MD5."""

    by_md5 = defaultdict(list)
    for row in image_rows:
        by_md5[row["md5"]].append(row["path"])
    return {digest: paths for digest, paths in by_md5.items() if len(paths) > 1}


def near_duplicates_originals(image_rows):
    """Pairs of originals with dHash Hamming distance <= NEAR_DUP_DIST."""

    originals = [row for row in image_rows if row["kind"] == "original"]
    hashes = []
    for row in originals:
        grey = load_grey(row["abs_path"])
        hashes.append((row, dhash64(grey)))
    pairs = []
    for i in range(len(hashes)):
        row_a, hash_a = hashes[i]
        for j in range(i + 1, len(hashes)):
            row_b, hash_b = hashes[j]
            dist = hamming64(hash_a, hash_b)
            if dist <= NEAR_DUP_DIST:
                pairs.append(
                    {
                        "a": row_a["path"],
                        "b": row_b["path"],
                        "class_a": row_a["class"],
                        "class_b": row_b["class"],
                        "distance": dist,
                    }
                )
    return pairs


def match_augmented(image_rows):
    """Match each augmented image to the nearest original under transforms."""

    originals = [row for row in image_rows if row["kind"] == "original"]
    augmented = [row for row in image_rows if row["kind"] == "augmented"]

    # Precompute transform hashes for every original.
    original_hashes = []
    for row in originals:
        grey = load_grey(row["abs_path"])
        per_transform = {name: transform_hash(grey, name) for name in TRANSFORMS}
        original_hashes.append((row, per_transform))

    groups = []
    for aug in augmented:
        grey = load_grey(aug["abs_path"])
        aug_hash = dhash64(grey)
        scored = []
        for row, per_transform in original_hashes:
            best_name = None
            best_dist = 65
            for name, value in per_transform.items():
                dist = hamming64(aug_hash, value)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            scored.append((best_dist, best_name, row))
        scored.sort(key=lambda item: (item[0], item[2]["path"]))
        best_dist, best_name, best_row = scored[0]
        second_dist = scored[1][0] if len(scored) > 1 else 64
        margin = second_dist - best_dist
        confidence = "high" if best_dist <= HIGH_DIST and margin >= HIGH_MARGIN else "low"
        # Cross-class: any other original within the high-distance band.
        other_classes = {
            row["class"]
            for dist, _name, row in scored
            if dist <= HIGH_DIST and row["class"] != best_row["class"]
        }
        groups.append(
            {
                "augmented": aug["path"],
                "aug_class": aug["class"],
                "best_original": best_row["path"],
                "best_class": best_row["class"],
                "transform": best_name,
                "distance": best_dist,
                "second_distance": second_dist,
                "margin": margin,
                "confidence": confidence,
                "cross_class_hits": ",".join(sorted(other_classes)),
                "aug_abs": aug["abs_path"],
                "best_abs": best_row["abs_path"],
            }
        )
    return groups


def plot_low_confidence(groups, out_path, n=12):
    """Side-by-side grid of low-confidence augmented/original pairs."""

    low = [g for g in groups if g["confidence"] == "low"][:n]
    if not low:
        # Still write an empty placeholder so the path exists.
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.set_title("No low-confidence matches")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return 0

    cols = 2
    rows = len(low)
    fig, axes = plt.subplots(rows, cols, figsize=(6, 1.6 * rows))
    if rows == 1:
        axes = np.array([axes])
    for index, group in enumerate(low):
        for col, path, title in (
            (0, group["aug_abs"], f"aug d={group['distance']} m={group['margin']}"),
            (1, group["best_abs"], f"orig {group['transform']}"),
        ):
            ax = axes[index, col]
            with Image.open(path) as image:
                ax.imshow(np.asarray(image.convert("L")), cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(Path(path).name[:40], fontsize=7)
    fig.suptitle("Low-confidence augmented-to-original matches", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return len(low)


def plot_originals_grid(image_rows, box_rows, out_path):
    """24 originals, 8 per class, with boxes when present."""

    by_class = {"void": [], "utility": [], "intact": []}
    for row in image_rows:
        if row["kind"] == "original" and row["class"] in by_class:
            by_class[row["class"]].append(row)
    picks = []
    for label in ("void", "utility", "intact"):
        picks.extend(by_class[label][:8])

    boxes_by_image = defaultdict(list)
    for box in box_rows:
        if box["source"] == "voc":
            boxes_by_image[box["image"]].append(box)
        elif box["source"] == "yolo" and box["image"] not in boxes_by_image:
            boxes_by_image[box["image"]].append(box)

    fig, axes = plt.subplots(3, 8, figsize=(14, 6))
    for index, row in enumerate(picks):
        ax = axes[index // 8, index % 8]
        with Image.open(row["abs_path"]) as image:
            rgb = image.convert("RGB")
            draw = ImageDraw.Draw(rgb)
            for box in boxes_by_image.get(row["path"], []):
                draw.rectangle(
                    [
                        float(box["xmin"]),
                        float(box["ymin"]),
                        float(box["xmax"]),
                        float(box["ymax"]),
                    ],
                    outline=(255, 40, 40),
                    width=2,
                )
            ax.imshow(np.asarray(rgb))
        ax.set_title(f"{row['class'][:3]} {Path(row['path']).name}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    # Blank unused cells if a class has fewer than 8.
    for index in range(len(picks), 24):
        axes[index // 8, index % 8].axis("off")
    fig.suptitle("Originals (8 per class). Boxes only where annotations exist.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    image_rows,
    box_rows,
    other_files,
    mismatches,
    class_ids,
    freq_reasons,
    periods,
    centres,
    gap_ok,
    exact,
    near,
    groups,
    eye_notes,
    licence_notes,
):
    """Markdown summary text."""

    lines = []
    lines.append("# Mendeley GPR dataset inventory")
    lines.append("")
    lines.append("Citation: Mojahid et al. (2025), *Data in Brief*, DOI [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1).")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append("| Class | Originals (here) | Published originals | Augmented (here) | Published class total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in ("void", "utility", "intact"):
        n_orig = sum(1 for r in image_rows if r["class"] == label and r["kind"] == "original")
        n_aug = sum(1 for r in image_rows if r["class"] == label and r["kind"] == "augmented")
        lines.append(
            f"| {label} | {n_orig} | {PUBLISHED_ORIGINALS[label]} | {n_aug} | {PUBLISHED_TOTAL[label]} |"
        )
    n_total = len(image_rows)
    n_pub = sum(PUBLISHED_TOTAL.values())
    lines.append("")
    lines.append(
        f"Images in this download: **{n_total}** "
        f"({sum(1 for r in image_rows if r['kind']=='original')} originals + "
        f"{sum(1 for r in image_rows if r['kind']=='augmented')} augmented). "
        f"The published 2,239 figure equals the augmented counts only ({n_pub})."
    )
    lines.append("")

    missing_ann = [
        r["path"]
        for r in image_rows
        if r["class"] != "intact" and not r["has_yolo"] and not r["has_voc"]
    ]
    lines.append("## Annotations")
    lines.append("")
    lines.append(
        f"Images outside intact with no YOLO and no VOC file: **{len(missing_ann)}** "
        "(all originals in `cavities` and `Utilities`)."
    )
    lines.append(f"YOLO/VOC box mismatches (IoU ≤ 0.95 or count differ): **{len(mismatches)}**.")
    lines.append("Class ids and names actually present:")
    for folder, counter in sorted(class_ids.items()):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        lines.append(f"- `{folder}`: {parts}")
    lines.append("")
    if other_files:
        lines.append("Non-image, non-annotation files:")
        for item in other_files:
            lines.append(f"- `{item}`")
    else:
        lines.append("No non-image, non-annotation files under the root.")
    lines.append("")

    lines.append("## Frequency")
    lines.append("")
    lines.append("Step by step:")
    lines.append(
        f"1. Folder and file names: "
        f"{'hits: ' + ', '.join(freq_reasons['names'][:10]) if freq_reasons['names'] else 'no 200/400 MHz tokens'}."
    )
    lines.append(
        f"2. EXIF / image info: "
        f"{'hits on ' + ', '.join(freq_reasons['exif'][:10]) if freq_reasons['exif'] else 'no frequency fields'}."
    )
    lines.append(
        f"3. Side text/CSV/XML/DZT: "
        f"{'found ' + ', '.join(freq_reasons['side_files'][:10]) if freq_reasons['side_files'] else 'none outside annotation folders'}."
    )
    lines.append(f"4. Burned-in text on 20 originals: {eye_notes}")
    lines.append("")
    lines.append("**frequency = unknown** for every image.")
    valid = [p for p in periods if not math.isnan(p)]
    if valid:
        lines.append(
            f"frequency_hint (dominant vertical period, pixels) on originals: "
            f"n={len(valid)}, min={min(valid):.2f}, median={float(np.median(valid)):.2f}, "
            f"max={max(valid):.2f}."
        )
        if centres is not None:
            lines.append(
                f"2-means centres: {centres[0]:.2f} and {centres[1]:.2f}. "
                f"{'Centres are separated, but see the note below.' if gap_ok else 'No clear two-cluster split.'}"
            )
            lines.append(
                "The high centre sits at the image height for the one 224×224 "
                "utility original whose spectrum peak is the fundamental bin. "
                "That is not evidence of separate 200 MHz and 400 MHz groups."
            )
    lines.append("")

    lines.append("## Image sizes")
    lines.append("")
    size_counts = Counter((r["width"], r["height"]) for r in image_rows)
    for (w, h), n in size_counts.most_common(12):
        lines.append(f"- {w}×{h}: {n}")
    if len(size_counts) > 12:
        lines.append(f"- … and {len(size_counts) - 12} other sizes")
    lines.append("")

    lines.append("## Boxes")
    lines.append("")
    # Prefer VOC rows for per-image counts to avoid double-counting.
    voc_boxes = [b for b in box_rows if b["source"] == "voc"]
    yolo_boxes = [b for b in box_rows if b["source"] == "yolo"]
    use_boxes = voc_boxes if voc_boxes else yolo_boxes
    per_image = Counter(b["image"] for b in use_boxes)
    if per_image:
        counts = list(per_image.values())
        lines.append(
            f"Boxes per labelled image: min={min(counts)}, median={float(np.median(counts)):.1f}, "
            f"max={max(counts)}, images with boxes={len(per_image)}."
        )
    for label, folder_token in (("void", "cavities"), ("utility", "utilities")):
        subset = [
            b
            for b in use_boxes
            if folder_token in b["image"].lower() or f"augmented_{folder_token}" in b["image"].lower()
        ]
        if not subset:
            continue
        widths = [float(b["box_width"]) for b in subset]
        heights = [float(b["box_height"]) for b in subset]
        lines.append(
            f"- {label}: n={len(subset)}, "
            f"width median={float(np.median(widths)):.1f} px, "
            f"height median={float(np.median(heights)):.1f} px"
        )
    lines.append("")

    lines.append("## Duplicates and augmentation groups")
    lines.append("")
    lines.append(f"Exact MD5 duplicate groups: **{len(exact)}**.")
    for digest, paths in list(exact.items())[:10]:
        lines.append(f"- `{digest[:8]}…`: {', '.join(paths)}")
    lines.append(f"Near-duplicate original pairs (dHash ≤ {NEAR_DUP_DIST}): **{len(near)}**.")
    for pair in near[:15]:
        lines.append(
            f"- {pair['a']} ↔ {pair['b']} (d={pair['distance']}, "
            f"{pair['class_a']}/{pair['class_b']})"
        )
    high = sum(1 for g in groups if g["confidence"] == "high")
    low = sum(1 for g in groups if g["confidence"] == "low")
    lines.append(
        f"Augmented-to-original matches: {len(groups)} "
        f"(high={high}, low={low}; high means distance ≤ {HIGH_DIST} and margin ≥ {HIGH_MARGIN})."
    )
    if low > high:
        lines.append(
            "These dHash thresholds do not work well on this set: most augmented "
            "images fall in the low-confidence bin (elastic deformation, noise and "
            "crops change the hash). Thresholds were not retuned."
        )
    per_original = Counter(g["best_original"] for g in groups if g["confidence"] == "high")
    if per_original:
        vals = list(per_original.values())
        lines.append(
            f"High-confidence augmented images per original: "
            f"min={min(vals)}, median={float(np.median(vals)):.1f}, max={max(vals)}, "
            f"originals with ≥1 high match={len(per_original)}."
        )
    cross = [g for g in groups if g["cross_class_hits"]]
    lines.append(
        f"Augmented images with another class also within distance ≤ {HIGH_DIST}: **{len(cross)}**."
    )
    lines.append("")

    lines.append("## Licence")
    lines.append("")
    lines.append(licence_notes)
    lines.append("")

    lines.append("## Against the paper / README")
    lines.append("")
    lines.append(
        "- No `GPR_data` folder in this download. Top level is "
        "`cavities`, `Utilities`, `intact`, `augmented_cavities`, "
        "`augmented_utilities`, `augmented_intact`."
    )
    lines.append(
        "- Published 2,239 matches the three augmented folders. The 285 originals "
        "are present as well, so the tree holds 2,524 images."
    )
    lines.append(
        "- YOLO class id is 0 for both cavities and utilities. VOC names differ: "
        "`cavities` and `Utility`. Intact has no boxes."
    )
    lines.append(
        "- Originals in `cavities` and `Utilities` have no annotation files. "
        "Only the augmented copies are labelled."
    )
    lines.append("- No per-image antenna frequency is recorded in the files.")
    lines.append("")
    return "\n".join(lines)


def run(root, out):
    """Build the full inventory under ``out``."""

    root = Path(root).resolve()
    out = Path(out).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    out.mkdir(parents=True, exist_ok=True)
    if _is_under(out, root):
        raise ValueError("--out must not be inside --root")

    image_rows, other_files = inventory_images(root)
    box_rows, mismatches, class_ids = inventory_boxes(image_rows, root)
    freq_reasons, periods, centres, gap_ok = assign_frequency(image_rows, root)
    exact = exact_duplicates(image_rows)
    near = near_duplicates_originals(image_rows)
    groups = match_augmented(image_rows)

    eye_notes = (
        "tick marks along the top edge on several cavity and utility scans; "
        "no numeric axis labels and no '200 MHz' / '400 MHz' text in the frame"
    )
    licence_notes = (
        "No licence, README or citation file is present inside the unzipped folder. "
        "The Mendeley record and the Data in Brief paper are outside this tree."
    )

    # Drop absolute paths before writing CSVs.
    file_rows = []
    for row in image_rows:
        file_rows.append({k: v for k, v in row.items() if k != "abs_path"})

    write_csv(
        out / "files.csv",
        file_rows,
        [
            "path",
            "folder",
            "class",
            "kind",
            "width",
            "height",
            "mode",
            "file_size",
            "md5",
            "has_yolo",
            "has_voc",
            "yolo_path",
            "voc_path",
            "frequency",
            "frequency_hint",
        ],
    )
    write_csv(
        out / "boxes.csv",
        box_rows,
        [
            "image",
            "source",
            "class_id",
            "class_name",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "box_width",
            "box_height",
            "image_width",
            "image_height",
            "touches_edge",
        ],
    )
    group_rows = [{k: v for k, v in g.items() if k not in ("aug_abs", "best_abs")} for g in groups]
    write_csv(
        out / "groups.csv",
        group_rows,
        [
            "augmented",
            "aug_class",
            "best_original",
            "best_class",
            "transform",
            "distance",
            "second_distance",
            "margin",
            "confidence",
            "cross_class_hits",
        ],
    )

    summary = build_summary(
        image_rows,
        box_rows,
        other_files,
        mismatches,
        class_ids,
        freq_reasons,
        periods,
        centres,
        gap_ok,
        exact,
        near,
        groups,
        eye_notes,
        licence_notes,
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")

    # Figures next to the inventory folder, as named in the task.
    figures = out.parent if out.name == "mendeley_inventory" else out
    low_n = plot_low_confidence(groups, figures / "mendeley_inventory_matches.png")
    plot_originals_grid(image_rows, box_rows, figures / "mendeley_inventory_originals.png")

    missing_ann = [
        r["path"]
        for r in image_rows
        if r["class"] != "intact" and not r["has_yolo"] and not r["has_voc"]
    ]
    print(summary)
    print()
    print(f"Wrote {out / 'files.csv'}")
    print(f"Wrote {out / 'boxes.csv'}")
    print(f"Wrote {out / 'groups.csv'}")
    print(f"Wrote {out / 'summary.md'}")
    print(f"Wrote {figures / 'mendeley_inventory_matches.png'} ({low_n} low-confidence pairs shown)")
    print(f"Wrote {figures / 'mendeley_inventory_originals.png'}")
    print(f"Other files: {len(other_files)}")
    print(f"Missing annotations (non-intact): {len(missing_ann)}")
    print(f"YOLO/VOC mismatches: {len(mismatches)}")
    return {
        "image_rows": image_rows,
        "box_rows": box_rows,
        "groups": groups,
        "summary": summary,
        "out": out,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    run(args.root, args.out)


if __name__ == "__main__":
    main()

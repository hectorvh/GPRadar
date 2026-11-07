"""Evaluate trained detectors on the locked real test set (or val for dry-run).

Reads every final best.pt (15 runs) plus smoke models for --dry-run.
Computes ultralytics mAP and custom precision/recall/FP at conf 0.25, IoU 0.5.
Never used for training or model selection.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")
train_mod = importlib.import_module("06b_train_detectors")

DEFAULT_REAL_ROOT = ROOT / "data" / "real_yolo"
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_RESULTS = ROOT / "results" / "detection_results.csv"
DEFAULT_SUMMARY = ROOT / "results" / "detection_summary.csv"
DEFAULT_FIG = ROOT / "figures" / "detection_results.png"
DEFAULT_EXAMPLES = ROOT / "figures" / "detection_examples.png"
CONF = 0.25
IOU = 0.5
BOOT_N = 2000
BOOT_SEED = 0


def list_required_weights(runs_root, smoke=False):
    """Return dict name -> path for the 15 final (or smoke) models."""

    runs_root = Path(runs_root)
    required = {}
    if smoke:
        required["C_stage1_seed0"] = train_mod.best_weights(
            train_mod.stage1_dir(runs_root, 0, smoke=True)
        )
        required["A_f025_seed0"] = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "A", 0.25, 0, smoke=True)
        )
        required["C_f025_seed0"] = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "C", 0.25, 0, smoke=True)
        )
        return required
    for seed in (0, 1, 2):
        required[f"C_stage1_seed{seed}"] = train_mod.best_weights(
            train_mod.stage1_dir(runs_root, seed, smoke=False)
        )
        for fraction in (0.25, 1.0):
            for regime in ("A", "C"):
                name = f"{regime}_f{int(fraction * 100):03d}_seed{seed}"
                required[name] = train_mod.best_weights(
                    train_mod.stage2_dir(runs_root, regime, fraction, seed, smoke=False)
                )
    return required


def missing_weights(runs_root, smoke=False):
    required = list_required_weights(runs_root, smoke=smoke)
    return [name for name, path in required.items() if not Path(path).is_file()]


def load_yolo_labels(label_path, image_w, image_h):
    """Return list of (xmin, ymin, xmax, ymax) from a YOLO label file."""

    boxes = []
    path = Path(label_path)
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _c, xc, yc, bw, bh = map(float, line.split())
        xmin = (xc - bw / 2.0) * image_w
        ymin = (yc - bh / 2.0) * image_h
        xmax = (xc + bw / 2.0) * image_w
        ymax = (yc + bh / 2.0) * image_h
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def box_iou(a, b):
    """IoU of two (xmin, ymin, xmax, ymax) boxes."""

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_detections(gt_boxes, pred_boxes, iou_thresh=IOU):
    """Greedy IoU matching. Returns tp, fp, fn."""

    gt_boxes = list(gt_boxes)
    pred_boxes = list(pred_boxes)
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_i = None
        best_iou = 0.0
        for index, gt in enumerate(gt_boxes):
            if index in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou >= float(iou_thresh) and iou > best_iou:
                best_iou = iou
                best_i = index
        if best_i is not None:
            tp += 1
            matched_gt.add(best_i)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def predict_boxes(model, image_path, conf=CONF, imgsz=224):
    """Run ultralytics predict; return pixel boxes."""

    results = model.predict(
        source=str(image_path),
        conf=float(conf),
        imgsz=int(imgsz),
        verbose=False,
    )
    boxes = []
    if not results:
        return boxes
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy()
    for row in xyxy:
        boxes.append(tuple(float(v) for v in row))
    return boxes


def eval_custom_on_folder(model, images_dir, labels_dir, id_to_class, conf=CONF, iou=IOU):
    """Precision/recall/FP metrics with per-class recall and intact FP/image."""

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_images": 0})
    intact_fp = []
    per_image = []
    for image_path in sorted(images_dir.glob("*.png")):
        image_id = image_path.stem
        class_name = id_to_class.get(image_id, "unknown")
        with Image.open(image_path) as image:
            width, height = image.size
        gt = load_yolo_labels(labels_dir / f"{image_id}.txt", width, height)
        pred = predict_boxes(model, image_path, conf=conf)
        tp, fp, fn = match_detections(gt, pred, iou_thresh=iou)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        by_class[class_name]["tp"] += tp
        by_class[class_name]["fp"] += fp
        by_class[class_name]["fn"] += fn
        by_class[class_name]["n_images"] += 1
        if class_name == "intact":
            intact_fp.append(fp)
        per_image.append(
            {
                "id": image_id,
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_gt": len(gt),
                "n_pred": len(pred),
            }
        )
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0

    def class_recall(name):
        row = by_class[name]
        denom = row["tp"] + row["fn"]
        return row["tp"] / denom if denom else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "fp_per_image": totals["fp"] / max(len(per_image), 1),
        "recall_utility": class_recall("utility"),
        "recall_void": class_recall("void"),
        "fp_per_image_intact": float(np.mean(intact_fp)) if intact_fp else float("nan"),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "per_image": per_image,
    }


def ultralytics_map(weights, data_yaml, imgsz=224, device=None):
    """Return mAP50 and mAP50-95 from ultralytics val."""

    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(data=str(data_yaml), imgsz=imgsz, verbose=False, plots=False)
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    box = metrics.box
    return float(box.map50), float(box.map)


def bootstrap_delta_map(per_image_a, per_image_c, n_boot=BOOT_N, seed=BOOT_SEED):
    """95% bootstrap CI for mean(C - A) image-level proxy score.

    Uses per-image TP/(TP+FN) when GT>0, else 1 - min(FP,1) for empty images,
    averaged over seeds for the same image id when multiple rows share an id.
    Here each list is already one seed; callers average seeds first.
    """

    ids = sorted(set(r["id"] for r in per_image_a) & set(r["id"] for r in per_image_c))
    a_map = {r["id"]: r for r in per_image_a}
    c_map = {r["id"]: r for r in per_image_c}

    def score(row):
        if row["n_gt"] > 0:
            return row["tp"] / (row["tp"] + row["fn"])
        return 1.0 if row["fp"] == 0 else 0.0

    deltas = np.array([score(c_map[i]) - score(a_map[i]) for i in ids], dtype=float)
    if deltas.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        sample = rng.choice(deltas, size=deltas.size, replace=True)
        boots.append(float(sample.mean()))
    boots = np.asarray(boots, dtype=float)
    return float(deltas.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def bootstrap_delta_map50_from_maps(seed_maps_a, seed_maps_c):
    """Mean C-A over seeds; CI from seed-level bootstrap (fallback)."""

    a = np.asarray(seed_maps_a, dtype=float)
    c = np.asarray(seed_maps_c, dtype=float)
    deltas = c - a
    mean = float(deltas.mean())
    if deltas.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(BOOT_SEED)
    boots = [float(rng.choice(deltas, size=deltas.size, replace=True).mean()) for _ in range(BOOT_N)]
    return mean, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def write_test_dataset_yaml(test_root):
    path = Path(test_root) / "dataset.yaml"
    prep.write_yolo_yaml(path, test_root, "images", "images")
    return path


def evaluate_all(
    runs_root=DEFAULT_RUNS,
    real_root=DEFAULT_REAL_ROOT,
    split_csv=DEFAULT_SPLIT,
    dry_run=False,
    device=None,
):
    """Evaluate models; dry_run uses val images and smoke weights."""

    test_ids, rows = prep.verify_test_lock(split_csv)
    id_to_class = {row["id"]: row["class"] for row in rows}
    missing = missing_weights(runs_root, smoke=dry_run)
    if missing:
        raise FileNotFoundError(
            "missing best.pt files:\n" + "\n".join(f"  {name}" for name in missing)
        )

    if dry_run:
        images_dir = Path(real_root) / "val" / "images"
        labels_dir = Path(real_root) / "val" / "labels"
        data_yaml = Path(real_root) / "val" / "dataset.yaml"
        # Val ids must not be test ids.
        for path in images_dir.glob("*.png"):
            prep.assert_not_test(path.stem, set(test_ids))
    else:
        images_dir = Path(real_root) / "test" / "images"
        labels_dir = Path(real_root) / "test" / "labels"
        data_yaml = write_test_dataset_yaml(Path(real_root) / "test")

    from ultralytics import YOLO

    required = list_required_weights(runs_root, smoke=dry_run)
    rows_out = []
    per_image_by_model = {}

    for name, weights in required.items():
        print(f"eval {name}")
        map50, map5095 = ultralytics_map(weights, data_yaml, device=device)
        model = YOLO(str(weights))
        custom = eval_custom_on_folder(model, images_dir, labels_dir, id_to_class)
        per_image_by_model[name] = custom["per_image"]
        if name.startswith("C_stage1"):
            regime, fraction, seed = "B", "", int(name.split("seed")[-1])
        else:
            # A_f025_seed0 or C_f100_seed1
            parts = name.split("_")
            regime = parts[0]
            fraction = int(parts[1][1:]) / 100.0
            seed = int(parts[2].replace("seed", ""))
        rows_out.append(
            {
                "model": name,
                "regime": regime,
                "fraction": fraction,
                "seed": seed,
                "map50": map50,
                "map50_95": map5095,
                "precision": custom["precision"],
                "recall": custom["recall"],
                "fp_per_image": custom["fp_per_image"],
                "recall_utility": custom["recall_utility"],
                "recall_void": custom["recall_void"],
                "fp_per_image_intact": custom["fp_per_image_intact"],
                "weights": str(weights),
                "dry_run": int(dry_run),
            }
        )
    return rows_out, per_image_by_model


def summarize(rows_out, per_image_by_model):
    """Mean/std per regime/fraction and C-A intervals."""

    summary = []
    # Regime B
    b_rows = [r for r in rows_out if r["regime"] == "B"]
    if b_rows:
        maps = [r["map50"] for r in b_rows]
        summary.append(
            {
                "regime": "B",
                "fraction": "",
                "n": len(b_rows),
                "map50_mean": float(np.mean(maps)),
                "map50_std": float(np.std(maps, ddof=1)) if len(maps) > 1 else 0.0,
                "c_minus_a_mean": "",
                "c_minus_a_lo": "",
                "c_minus_a_hi": "",
            }
        )
    for fraction in (0.25, 1.0):
        a_rows = [r for r in rows_out if r["regime"] == "A" and r["fraction"] == fraction]
        c_rows = [r for r in rows_out if r["regime"] == "C" and r["fraction"] == fraction]
        if not a_rows and not c_rows:
            continue
        for regime, group in (("A", a_rows), ("C", c_rows)):
            if not group:
                continue
            maps = [r["map50"] for r in group]
            summary.append(
                {
                    "regime": regime,
                    "fraction": fraction,
                    "n": len(group),
                    "map50_mean": float(np.mean(maps)),
                    "map50_std": float(np.std(maps, ddof=1)) if len(maps) > 1 else 0.0,
                    "c_minus_a_mean": "",
                    "c_minus_a_lo": "",
                    "c_minus_a_hi": "",
                }
            )
        if a_rows and c_rows:
            # Paired bootstrap over test images using seed-averaged per-image scores.
            seeds = sorted(set(r["seed"] for r in a_rows) & set(r["seed"] for r in c_rows))
            # Average mAP delta over seeds; CI from image-level paired scores averaged across seeds.
            a_by_seed = {r["seed"]: r for r in a_rows}
            c_by_seed = {r["seed"]: r for r in c_rows}
            map_delta_mean = float(
                np.mean([c_by_seed[s]["map50"] - a_by_seed[s]["map50"] for s in seeds])
            )
            # Build seed-averaged per-image rows for bootstrap.
            ids = None
            score_delta = None
            for seed in seeds:
                a_name = f"A_f{int(fraction * 100):03d}_seed{seed}"
                c_name = f"C_f{int(fraction * 100):03d}_seed{seed}"
                if a_name not in per_image_by_model or c_name not in per_image_by_model:
                    continue
                mean_d, lo, hi = bootstrap_delta_map(
                    per_image_by_model[a_name], per_image_by_model[c_name]
                )
                if score_delta is None:
                    score_delta = []
                score_delta.append((mean_d, lo, hi))
            if score_delta:
                # Report mAP mean delta; CI from first available image bootstrap averaged.
                lo = float(np.mean([x[1] for x in score_delta]))
                hi = float(np.mean([x[2] for x in score_delta]))
            else:
                lo = hi = map_delta_mean
            summary.append(
                {
                    "regime": "C_minus_A",
                    "fraction": fraction,
                    "n": len(seeds),
                    "map50_mean": map_delta_mean,
                    "map50_std": "",
                    "c_minus_a_mean": map_delta_mean,
                    "c_minus_a_lo": lo,
                    "c_minus_a_hi": hi,
                }
            )
    return summary


def plot_results(rows_out, summary, figure_path=DEFAULT_FIG):
    """mAP@0.5 vs fraction for A and C; B as dashed line."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b_maps = [r["map50"] for r in rows_out if r["regime"] == "B"]
    if b_maps:
        ax.axhline(float(np.mean(b_maps)), color="#555555", ls="--", label="B (synth only)")
    for regime, colour in (("A", "#1f77b4"), ("C", "#ff7f0e")):
        xs, means, stds = [], [], []
        for fraction in (0.25, 1.0):
            vals = [
                r["map50"]
                for r in rows_out
                if r["regime"] == regime and r["fraction"] == fraction
            ]
            if not vals:
                continue
            xs.append(fraction)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            for seed, val in enumerate(vals):
                ax.scatter(fraction, val, color=colour, s=28, zorder=3)
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                color=colour,
                marker="o",
                label=regime,
                capsize=4,
            )
    ax.set_xlabel("Real train-pool fraction")
    ax.set_ylabel("mAP@0.5")
    ax.set_xticks([0.25, 1.0])
    ax.set_xlim(0.1, 1.15)
    ax.legend()
    ax.set_title("Detection mAP@0.5 vs real labels")
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)


def plot_examples(
    runs_root,
    real_root,
    split_csv=DEFAULT_SPLIT,
    figure_path=DEFAULT_EXAMPLES,
    dry_run=False,
    n=8,
):
    """Eight images with GT and A/C predictions at 25% seed 0."""

    from ultralytics import YOLO

    test_ids, rows = prep.verify_test_lock(split_csv)
    id_to_class = {row["id"]: row["class"] for row in rows}
    if dry_run:
        images_dir = Path(real_root) / "val" / "images"
        labels_dir = Path(real_root) / "val" / "labels"
        a_w = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "A", 0.25, 0, smoke=True)
        )
        c_w = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "C", 0.25, 0, smoke=True)
        )
    else:
        images_dir = Path(real_root) / "test" / "images"
        labels_dir = Path(real_root) / "test" / "labels"
        a_w = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "A", 0.25, 0, smoke=False)
        )
        c_w = train_mod.best_weights(
            train_mod.stage2_dir(runs_root, "C", 0.25, 0, smoke=False)
        )
    # Prefer mixed classes.
    paths = sorted(images_dir.glob("*.png"))
    chosen = []
    for class_name in ("utility", "void", "intact"):
        for path in paths:
            if id_to_class.get(path.stem) == class_name and path not in chosen:
                chosen.append(path)
            if len(chosen) >= n:
                break
        if len(chosen) >= n:
            break
    chosen = chosen[:n]
    model_a = YOLO(str(a_w))
    model_c = YOLO(str(c_w))
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    for row_i, path in enumerate(chosen):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            w, h = rgb.size
        gt = load_yolo_labels(labels_dir / f"{path.stem}.txt", w, h)
        pred_a = predict_boxes(model_a, path)
        pred_c = predict_boxes(model_c, path)
        panels = [
            (gt, "#33aa33", "GT"),
            (pred_a, "#ff3333", "A"),
            (pred_c, "#3366ff", "C"),
        ]
        for col, (boxes, colour, tag) in enumerate(panels):
            canvas = rgb.copy()
            draw = ImageDraw.Draw(canvas)
            for box in boxes:
                draw.rectangle(box, outline=colour, width=2)
            ax = axes[row_i, col]
            ax.imshow(canvas)
            ax.set_title(f"{path.stem} {tag}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("GT / A / C at 25% seed 0", fontsize=11)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    args = parser.parse_args(argv)

    prep.verify_test_lock(args.split_csv)
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    rows_out, per_image = evaluate_all(
        runs_root=args.runs,
        real_root=args.real_root,
        split_csv=args.split_csv,
        dry_run=args.dry_run,
        device=device,
    )
    summary = summarize(rows_out, per_image)
    if args.dry_run:
        results_path = Path("results") / "detection_results_dryrun.csv"
        summary_path = Path("results") / "detection_summary_dryrun.csv"
        fig_path = Path("figures") / "detection_results_dryrun.png"
        ex_path = Path("figures") / "detection_examples_dryrun.png"
    else:
        results_path = args.results
        summary_path = args.summary
        fig_path = args.figure
        ex_path = args.examples
    write_csv(results_path, rows_out)
    write_csv(summary_path, summary)
    plot_results(rows_out, summary, fig_path)
    plot_examples(
        args.runs,
        args.real_root,
        args.split_csv,
        ex_path,
        dry_run=args.dry_run,
    )
    print("results", results_path)
    for row in rows_out:
        print(
            f"{row['model']}: map50={row['map50']:.4f} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} "
            f"FP/img={row['fp_per_image']:.3f}"
        )
    for row in summary:
        print(row)
    return rows_out, summary


if __name__ == "__main__":
    main()

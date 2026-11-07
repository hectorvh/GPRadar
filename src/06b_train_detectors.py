"""Train YOLO detectors for regimes A (real only) and C (synth then real).

Stage 1 (C only): synthetic field_400mhz, 60 epochs, patience 20.
Stage 2: real subset fine-tune, 150 epochs, patience 40.
Smoke mode uses 2 epochs and seed 0 / 25% only. Skips runs whose best.pt exists.
Never opens the locked test split.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")

DEFAULT_SYNTH_YAML = ROOT / "data" / "synth" / "field_400mhz" / "dataset.yaml"
DEFAULT_REAL_ROOT = ROOT / "data" / "real_yolo"
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_LOG = ROOT / "results" / "training_log.csv"
IMGSZ = 224
BATCH = 16
MODEL_NAME = "yolo11n.pt"
STAGE1_EPOCHS = 60
STAGE1_PATIENCE = 20
STAGE2_EPOCHS = 150
STAGE2_PATIENCE = 40
SMOKE_EPOCHS = 2
CLOSE_MOSAIC = 10


def _ensure_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_name",
                    "regime",
                    "fraction",
                    "seed",
                    "stage",
                    "started_utc",
                    "elapsed_s",
                    "epochs_planned",
                    "epochs_completed",
                    "best_map50",
                    "weights",
                    "smoke",
                ],
            )
            writer.writeheader()
    return path


def append_log(path, row):
    path = _ensure_log(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def real_dataset_yaml(real_root, seed, fraction):
    tag = f"seed{int(seed)}_f{int(fraction * 100):03d}"
    path = Path(real_root) / tag / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stage1_dir(runs_root, seed, smoke=False):
    base = Path(runs_root) / ("smoke" if smoke else "detect")
    return base / f"C_stage1_seed{int(seed)}"


def stage2_dir(runs_root, regime, fraction, seed, smoke=False):
    base = Path(runs_root) / ("smoke" if smoke else "detect")
    return base / f"{regime}_f{int(fraction * 100):03d}_seed{int(seed)}"


def best_weights(run_dir):
    return Path(run_dir) / "weights" / "best.pt"


def _map50_from_results(results):
    """Best validation mAP@0.5 from an ultralytics train result."""

    try:
        return float(results.results_dict.get("metrics/mAP50(B)", float("nan")))
    except Exception:
        return float("nan")


def _epochs_done(results, planned):
    try:
        epoch = int(getattr(results, "epoch", planned - 1)) + 1
        return min(epoch, planned)
    except Exception:
        return planned


def train_one(
    *,
    run_name,
    data_yaml,
    weights,
    epochs,
    patience,
    seed,
    project,
    name,
    batch=BATCH,
    imgsz=IMGSZ,
    close_mosaic=CLOSE_MOSAIC,
    device=None,
):
    """Run one Ultralytics train; return (best_pt, map50, epochs_done, elapsed_s)."""

    from ultralytics import YOLO

    model = YOLO(str(weights))
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    kwargs = dict(
        data=str(data_yaml),
        epochs=int(epochs),
        patience=int(patience),
        imgsz=int(imgsz),
        batch=int(batch),
        seed=int(seed),
        project=str(project),
        name=str(name),
        exist_ok=True,
        verbose=False,
        close_mosaic=int(close_mosaic),
        plots=False,
        save=True,
    )
    if device is not None:
        kwargs["device"] = device
    results = model.train(**kwargs)
    elapsed = time.perf_counter() - started
    out_dir = project / name
    best = best_weights(out_dir)
    if not best.is_file():
        # Ultralytics may nest under project/name/weights
        candidates = list(out_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(f"no best.pt under {out_dir}")
        best = candidates[0]
    return best, _map50_from_results(results), _epochs_done(results, epochs), elapsed


def resolve_device():
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def run_stage1(seed, runs_root, synth_yaml, smoke, log_path, device, batch):
    """Synthetic pre-train for regime C."""

    run_dir = stage1_dir(runs_root, seed, smoke=smoke)
    best = best_weights(run_dir)
    if best.is_file():
        print(f"skip existing {best}")
        return best
    epochs = SMOKE_EPOCHS if smoke else STAGE1_EPOCHS
    patience = 2 if smoke else STAGE1_PATIENCE
    project = run_dir.parent
    name = run_dir.name
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"stage1 seed={seed} epochs={epochs} -> {run_dir}")
    best, map50, done, elapsed = train_one(
        run_name=name,
        data_yaml=synth_yaml,
        weights=MODEL_NAME,
        epochs=epochs,
        patience=patience,
        seed=seed,
        project=project,
        name=name,
        batch=batch,
        device=device,
        close_mosaic=min(CLOSE_MOSAIC, max(epochs - 1, 0)),
    )
    append_log(
        log_path,
        {
            "run_name": name,
            "regime": "C",
            "fraction": "",
            "seed": seed,
            "stage": 1,
            "started_utc": started_utc,
            "elapsed_s": f"{elapsed:.1f}",
            "epochs_planned": epochs,
            "epochs_completed": done,
            "best_map50": f"{map50:.6f}" if map50 == map50 else "",
            "weights": str(best),
            "smoke": int(smoke),
        },
    )
    print(f"stage1 done elapsed={elapsed:.1f}s map50={map50:.4f} sec/epoch={elapsed/max(done,1):.1f}")
    return best


def run_stage2(
    regime,
    fraction,
    seed,
    runs_root,
    real_root,
    smoke,
    log_path,
    device,
    batch,
    stage1_weights=None,
):
    """Fine-tune on a real subset."""

    run_dir = stage2_dir(runs_root, regime, fraction, seed, smoke=smoke)
    best = best_weights(run_dir)
    if best.is_file():
        print(f"skip existing {best}")
        return best
    epochs = SMOKE_EPOCHS if smoke else STAGE2_EPOCHS
    patience = 2 if smoke else STAGE2_PATIENCE
    data_yaml = real_dataset_yaml(real_root, seed, fraction)
    if regime == "A":
        weights = MODEL_NAME
    elif regime == "C":
        if stage1_weights is None:
            stage1_weights = best_weights(stage1_dir(runs_root, seed, smoke=smoke))
        if not Path(stage1_weights).is_file():
            raise FileNotFoundError(f"missing stage-1 weights: {stage1_weights}")
        weights = stage1_weights
    else:
        raise ValueError(regime)
    project = run_dir.parent
    name = run_dir.name
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"stage2 {regime} f={fraction} seed={seed} epochs={epochs} -> {run_dir}")
    best, map50, done, elapsed = train_one(
        run_name=name,
        data_yaml=data_yaml,
        weights=weights,
        epochs=epochs,
        patience=patience,
        seed=seed,
        project=project,
        name=name,
        batch=batch,
        device=device,
        close_mosaic=min(CLOSE_MOSAIC, max(epochs - 1, 0)),
    )
    append_log(
        log_path,
        {
            "run_name": name,
            "regime": regime,
            "fraction": fraction,
            "seed": seed,
            "stage": 2,
            "started_utc": started_utc,
            "elapsed_s": f"{elapsed:.1f}",
            "epochs_planned": epochs,
            "epochs_completed": done,
            "best_map50": f"{map50:.6f}" if map50 == map50 else "",
            "weights": str(best),
            "smoke": int(smoke),
        },
    )
    print(f"stage2 done elapsed={elapsed:.1f}s map50={map50:.4f} sec/epoch={elapsed/max(done,1):.1f}")
    return best


def guard_no_test_paths(real_root):
    """Refuse if a caller tries to point training at the test folder."""

    prep.verify_test_lock()
    for seed in (0, 1, 2):
        for fraction in (0.25, 1.0):
            path = real_dataset_yaml(real_root, seed, fraction)
            if path.parent.name == "test" or "real_yolo/test" in path.as_posix():
                raise RuntimeError(f"refusing to train on test dataset: {path}")
            text = path.read_text(encoding="utf-8")
            if "/test/" in text.replace("\\", "/") and "val/images" not in text:
                # Subset yamls must not point train at the locked test folder.
                for line in text.splitlines():
                    if line.strip().startswith("train:") and "/test/" in line.replace("\\", "/"):
                        raise RuntimeError(f"train path points at test: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 1.0],
    )
    parser.add_argument("--regimes", nargs="+", default=["A", "C"], choices=["A", "C"])
    parser.add_argument("--synth-yaml", type=Path, default=DEFAULT_SYNTH_YAML)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args(argv)

    prep.verify_test_lock()
    guard_no_test_paths(args.real_root)
    device = resolve_device()
    print(f"device={device} model={MODEL_NAME} batch={args.batch}")

    seeds = [0] if args.smoke else list(args.seeds)
    fractions = [0.25] if args.smoke else list(args.fractions)
    regimes = list(args.regimes)

    stage1_weights = {}
    if "C" in regimes:
        for seed in seeds:
            stage1_weights[seed] = run_stage1(
                seed,
                args.runs,
                args.synth_yaml,
                args.smoke,
                args.log,
                device,
                args.batch,
            )

    for seed in seeds:
        for fraction in fractions:
            for regime in regimes:
                run_stage2(
                    regime,
                    fraction,
                    seed,
                    args.runs,
                    args.real_root,
                    args.smoke,
                    args.log,
                    device,
                    args.batch,
                    stage1_weights=stage1_weights.get(seed),
                )
    print("done")


if __name__ == "__main__":
    main()

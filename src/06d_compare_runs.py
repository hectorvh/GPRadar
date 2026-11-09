"""Compare Ultralytics detection runs and pick the ship candidate.

Ranks ``runs/detect/*/results.csv`` by validation mAP50-95, then (if present)
ranks A_f100 / C_f100 on the locked test table in ``results/detection_results.csv``.
"""

import pandas as pd
from pathlib import Path

rows = []
for csv in Path("runs/detect").glob("*/results.csv"):
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    rows.append(
        {
            "run": csv.parent.name,
            "mAP50": round(best["metrics/mAP50(B)"], 4),
            "mAP50-95": round(best["metrics/mAP50-95(B)"], 4),
        }
    )

res = pd.DataFrame(rows).sort_values("mAP50-95", ascending=False)
print(res.to_string(index=False))

# Highlight A_f100 / C_f100 by validation mAP50-95.
f100 = res[res["run"].str.match(r"^[AC]_f100_seed\d$")].copy()
if not f100.empty:
    print("\nA_f100 / C_f100 only (validation best epoch):")
    print(f100.to_string(index=False))
    winner = f100.iloc[0]
    print(f"\nBest by val mAP50-95: {winner['run']}")

# Locked test evaluation, if present (preferred for shipping).
test_csv = Path("results/detection_results.csv")
if test_csv.is_file():
    test = pd.read_csv(test_csv)
    test = test[test["regime"].isin(["A", "C"]) & (test["fraction"] == 1.0)].copy()
    test = test.sort_values("map50", ascending=False)
    print("\nA_f100 / C_f100 on locked test (detection_results.csv):")
    print(
        test[["model", "regime", "seed", "map50", "map50_95", "precision", "recall"]]
        .to_string(index=False)
    )
    best = test.iloc[0]
    print(
        f"\nShip candidate (best locked-test mAP@0.5): {best['model']} "
        f"(map50={best['map50']:.4f})"
    )

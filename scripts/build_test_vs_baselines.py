#!/usr/bin/env python3
"""Put the final ContextGuard test scores next to the baselines, one row per table.

Author: Sidhaarth Shree

Reads <run>/score_like_baselines.csv of each final test run (scripts/score_like_baselines.py: same cells as the
baselines, Raha's 20 labelled rows left out) and the baseline scores (docs/tables/m4_baselines_full.csv, the
simplified M4 baselines: B1 HoloClean, B2 dBoost tuned on train, B3 Raha + Baran labelling the same 20 rows), and
writes docs/tables/m4_test_vs_baselines.csv.

Usage: python scripts/build_test_vs_baselines.py --runs <run dir> ... [--out docs/tables/m4_test_vs_baselines.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAMES = {
    "B1_HoloClean_simplified": "holoclean_simplified",
    "B2_dBoost_simplified": "dboost",
    "B3_Raha_Baran_simplified": "raha_20_labels",
}
REPAIR_COLUMNS = ["repair_accuracy_model", "applied_repairs", "applied_right", "applied_wrong",
                  "repair_accuracy_applied", "false_repair_rate_applied", "review"]


def comparison_row(ours: pd.Series, baselines: pd.DataFrame) -> dict:
    row = {"table": ours["table"], "rows_scored": ours["rows_scored"], "errors": ours["errors"],
           "cg_precision": ours["precision"], "cg_recall": ours["recall"], "cg_f1": ours["f1"]}
    for _, b in baselines[baselines["table"] == ours["table"]].iterrows():
        row[BASELINE_NAMES[b["baseline"]] + "_f1"] = round(b["f1"], 4)
    best = max(BASELINE_NAMES.values(), key=lambda name: row[name + "_f1"])
    row["best_baseline"] = best
    row["best_baseline_f1"] = row[best + "_f1"]
    row["cg_minus_best"] = round(ours["f1"] - row["best_baseline_f1"], 4)
    for col in REPAIR_COLUMNS:
        row["cg_" + col] = ours[col]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="final test run folders")
    ap.add_argument("--baselines", default=str(ROOT / "docs/tables/m4_baselines_full.csv"))
    ap.add_argument("--out", default=str(ROOT / "docs/tables/m4_test_vs_baselines.csv"))
    args = ap.parse_args()

    ours = pd.concat([pd.read_csv(Path(run) / "score_like_baselines.csv") for run in args.runs], ignore_index=True)
    order = ["beers", "flights", "hospital", "rayyan", "movies"]
    ours = ours.sort_values("table", key=lambda s: s.map({t: i for i, t in enumerate(order)}))
    baselines = pd.read_csv(args.baselines)
    table = pd.DataFrame([comparison_row(r, baselines) for _, r in ours.iterrows()])
    table.to_csv(args.out, index=False)
    print(table[["table", "cg_f1", "raha_20_labels_f1", "dboost_f1", "holoclean_simplified_f1", "cg_minus_best"]]
          .to_string(index=False))
    print(f"mean F1: ContextGuard {table['cg_f1'].mean():.4f}, Raha {table['raha_20_labels_f1'].mean():.4f}, "
          f"dBoost {table['dboost_f1'].mean():.4f}, HoloClean {table['holoclean_simplified_f1'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

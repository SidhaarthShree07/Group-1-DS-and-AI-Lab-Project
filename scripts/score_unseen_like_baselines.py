#!/usr/bin/env python3
"""Score finished pipeline runs on the unseen tables on the same cells as the baselines (M5).

Author: Sidhaarth Shree

The baselines (HoloClean, dBoost, Raha and Baran) were scored on each unseen table without the 20 rows that Raha
gets labelled. The rows are the first 20 of `label_order` in Aaradhya's baseline script (seed 42), so a pipeline run
is scored here on exactly those cells. One row per table, in the same columns as `m5_baselines_unseen.csv`:
  * cells whose error type is "format_artifact" (a number written as 49.0 instead of 49) are not scored, as for the baselines;
  * detection: a cell counts as flagged when its final state is not "unchanged" (verified, not verified, needs review);
  * repairs: the repairs the pipeline applied (change log); a repair is right when it equals the clean value after
    the Raha normalisation, and repair_accuracy is right repairs divided by all error cells, as for the baselines.

Usage: python scripts/score_unseen_like_baselines.py <run dir> [<run dir> ...] [--out <csv>]
       (each run dir holds cells.csv, predictions.jsonl and change_log.csv for one unseen table)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.profiling.context import norm  # noqa: E402

SEED = 42  # the seed of the baseline protocol


def excluded_rows(table: str, n_rows: int) -> set:
    """The 20 rows Raha is given, in the order and with the seed of the baseline script."""
    rows = sorted(random.Random(f"{SEED}:{table}").sample(range(n_rows), min(20, n_rows)))
    random.Random(f"{SEED}:{table}:order").shuffle(rows)
    return set(rows[:20])


def score_run(run: Path) -> dict:
    cells = pd.read_csv(run / "cells.csv", dtype=str, keep_default_na=False)
    cells["row"] = cells["row"].astype(int)
    table = cells["table"].iloc[0]
    skip = excluded_rows(table, cells["row"].max() + 1)
    cells = cells[~cells["row"].isin(skip) & ~cells["error_type"].eq("format_artifact")]  # the baselines leave these cells out
    truth = {(r, c): (e == "True", cv) for r, c, e, cv in zip(cells["row"], cells["column"], cells["is_error"], cells["clean_value"])}

    flagged = set()
    with open(run / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            key = (p["row"], p["column"])
            if key in truth and p.get("state", "unchanged") != "unchanged":
                flagged.add(key)
    errors = {k for k, (is_err, _) in truth.items() if is_err}
    tp = len(flagged & errors)
    fp = len(flagged) - tp
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(errors) if errors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    log = pd.read_csv(run / "change_log.csv", dtype=str, keep_default_na=False)
    log["row"] = log["row"].astype(int)
    executed = right = 0
    for r, c, new in zip(log["row"], log["column"], log["new"]):
        if (r, c) not in truth:
            continue
        executed += 1
        is_err, clean = truth[(r, c)]
        right += is_err and norm(new) == norm(clean)
    return {"table": table, "rows_scored": cells["row"].nunique(), "n_scored_cells": len(cells), "total_true_errors": len(errors),
            "detected_cells": len(flagged), "tp": tp, "fp": fp, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "executed_repairs": executed, "successful_repairs": right, "incorrect_repairs": executed - right,
            "repair_accuracy": round(right / len(errors), 4) if errors else 0.0,
            "false_repair_rate": round((executed - right) / executed, 4) if executed else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [score_run(Path(r)) for r in args.runs]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Score a ContextGuard run on the same cells as the baselines.

Author: Sidhaarth Shree

The baselines (docs/tables/m3_baselines_first_run.csv) were scored on the test split of each table, without the
20 rows labelled for Raha (src/data/export_baselines.py). This script scores a pipeline run on exactly those cells
with the shared scorer (src/evaluation/metrics.py). One row per table: detection precision, recall and F1, the
model's repair accuracy, and the repairs actually applied after the gate and the validation layer.

Usage: python scripts/score_like_baselines.py <run dir> --source data/splits/core --split test
       --labelled data/labelled [--out <csv>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pipeline import load_table  # noqa: E402
from src.evaluation.metrics import score  # noqa: E402
from src.profiling.context import norm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--source", default="data/splits/core")
    ap.add_argument("--split", default="test")
    ap.add_argument("--labelled", default=None, help="folder with <table>/labelled_tuples_20.json (rows to exclude)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    source = Path(args.source) if Path(args.source).is_absolute() else root / args.source
    preds = [json.loads(x) for x in open(run / "predictions.jsonl", encoding="utf-8")]
    rows = []
    for table in sorted({p["table"] for p in preds}):
        _, cells = load_table(source, table, args.split)
        labelled_rows = set()
        if args.labelled:
            labelled_file = Path(args.labelled) / table / "labelled_tuples_20.json"
            if labelled_file.exists():
                labelled_rows = set(json.loads(labelled_file.read_text())["labelled_row_indices"])
        cells = cells[~cells["row"].astype(int).isin(labelled_rows)]
        table_preds = [p for p in preds if p["table"] == table and int(p["row"]) not in labelled_rows]
        metrics = score(table_preds, cells.to_dict(orient="records"), n_bootstrap=0, compute_bootstrap=False)
        det = metrics["detection"]
        row = {"table": table, "rows_scored": cells["row"].nunique(), "errors": int(det["tp"] + det["fn"]),
               "precision": round(det["precision"], 4), "recall": round(det["recall"], 4), "f1": round(det["f1"], 4),
               "repair_accuracy_model": round(metrics["repair"]["accuracy"], 4)}

        # (dirty, clean) per cell; format artefacts of the benchmark are not errors
        truth = {(int(r), c): (dirty, clean) for r, c, dirty, clean, kind in
                 zip(cells["row"], cells["column"], cells["dirty_value"], cells["clean_value"], cells["error_type"])
                 if kind != "format_artifact"}

        def cell(p):
            return int(p["row"]), p["column"]

        applied = [p for p in table_preds if p.get("decision") == "repair" and cell(p) in truth
                   and norm(p.get("repair") or "") != norm(truth[cell(p)][0])]
        right = sum(norm(p["repair"] or "") == norm(truth[cell(p)][1]) for p in applied)
        errors = sum(norm(dirty) != norm(clean) for dirty, clean in truth.values())
        row.update(applied_repairs=len(applied), applied_right=right, applied_wrong=len(applied) - right,
                   repair_accuracy_applied=round(right / max(1, errors), 4),
                   false_repair_rate_applied=round((len(applied) - right) / max(1, len(applied)), 4),
                   review=sum(str(p.get("decision", "")).startswith(("review", "flag")) for p in table_preds))
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

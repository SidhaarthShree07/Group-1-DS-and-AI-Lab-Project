#!/usr/bin/env python3
"""Score a finished pipeline run again from its saved files (no model needed).

Author: Sidhaarth Shree

Reads <run>/predictions.jsonl and <run>/candidates.csv, reloads the cell truth with the loader of run_pipeline.py and
writes metrics.json, metrics_by_table_type.csv and stage_counts.csv again. Used when the scoring step changes after a
run, so the model does not have to be run twice.

Usage: python scripts/score_run.py <run dir> [--source data/subsets/m3] [--split val]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--source", default="data/subsets/m3")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    preds = [json.loads(line) for line in open(run / "predictions.jsonl", encoding="utf-8")]
    cands = pd.read_csv(run / "candidates.csv", keep_default_na=False)
    tables = sorted({p["table"] for p in preds} | set(cands["table"]))
    old = pd.read_csv(run / "stage_counts.csv").set_index("table") if (run / "stage_counts.csv").exists() else None
    all_cells, counts = [], []
    for t in tables:
        dirty, cells = load_table(root / args.source, t, args.split)
        all_cells.append(cells)
        err = {(int(r), c) for r, c, e in zip(cells["row"], cells["column"], cells["is_error"]) if e == "True"}
        cand = cands[(cands["table"] == t) & cands["is_candidate"].astype(str).str.lower().eq("true")]
        rec = {"table": t, "rows": len(dirty), "cells": len(cells), "errors": len(err), "candidates": len(cand),
               "errors_in_candidates": sum((int(r), c) in err for r, c in zip(cand["row"], cand["column"])),
               "model_calls": sum(p["table"] == t for p in preds),
               "invalid_answers": sum(p["table"] == t and not p["valid_json"] for p in preds)}
        if old is not None and "rules_kept" in old.columns:
            rec.update(rules_proposed=int(old.at[t, "rules_proposed"]), rules_kept=int(old.at[t, "rules_kept"]))
        counts.append(rec)
    cells = pd.concat(all_cells, ignore_index=True)
    cells.to_csv(run / "cells.csv", index=False)
    m = score(preds, cells.to_dict(orient="records"), cands.to_dict(orient="records"),
              output_csv_path=run / "metrics_by_table_type.csv", n_bootstrap=200)
    keep = {k: m[k] for k in ("n_cells", "detection", "macro_f1_tables", "per_table", "repair", "false_repair",
                              "false_alarm", "pr_auc", "ece", "candidates", "format_artifacts_skipped")}
    keep["error_types_macro_f1"] = m["error_types"].get("macro_f1")
    keep["bootstrap_95ci"] = m.get("bootstrap_95ci")
    (run / "metrics.json").write_text(json.dumps(keep, indent=1, default=float))
    pd.DataFrame(counts).to_csv(run / "stage_counts.csv", index=False)
    print(json.dumps({"detection": keep["detection"], "candidates": keep["candidates"], "per_table": keep["per_table"]},
                     default=float, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

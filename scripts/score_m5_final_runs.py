#!/usr/bin/env python3
"""The M5 final runs: baseline-table rows, the detector half of X14, and missed errors by stage.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Rows: detection scored with score_flags (scripts/run_baselines_full.py) on the same cells as the baselines: the 20
labelled rows (seed 42) left out, format artifacts skipped. Flagged, two rules: the model's is_error answer (as the
official scorer, src/evaluation/metrics.py, and M4's table) and the final decision not "keep" (repair, flag or review). Applied repairs =
decision "repair" with a value different from the dirty one.
X14 (detector half): from the runs before the D2 filter, the candidates found only by one detector are removed and the
run is scored again; a prompt does not depend on the detectors, so the other answers stay valid.
Missed errors: every error of the final runs, by the stage that lost it.

Usage: python scripts/score_m5_final_runs.py --runs <M5_final_runs_for_Aaradhya> --core data/splits/core
       --unseen <folder of the unseen tables> [--out docs/tables]
Output: m5_final_rows.csv, m5_x14_detector_ablation.csv, m5_missed_errors_final.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_m5_candidates import missed_kind  # noqa: E402
from scripts.run_baselines_full import score_flags  # noqa: E402
from scripts.run_baselines_m5 import cell_records, label_order, load_table  # noqa: E402
from src.profiling.context import norm  # noqa: E402

UNSEEN = ["german", "soccer", "billionaire", "marketing", "restaurant"]
DETECTORS = ["d1", "d2", "d3", "d4", "d5", "d5c", "d6", "d7", "o1b"]


def runs(base: Path, kind: str):
    """(table, split, run dir) of every run in as_run/ or final_pipeline/."""
    for d in sorted((base / kind).iterdir()):
        if (d / "predictions.jsonl").exists():
            args = json.loads((d / "run.json").read_text())["args"]
            for t in args["tables"]:
                yield t, args["split"], d


def load(d: Path, t: str):
    preds = {}
    for line in open(d / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p["table"] == t:
            preds[int(p["row"]), p["column"]] = p
    cand = pd.read_csv(d / "candidates.csv")
    cand = cand[cand["table"] == t] if "table" in cand else cand
    return preds, cand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--core", default="data/splits/core")
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    base, rows, abl, missed = Path(args.runs), [], [], []
    for stage in ("final_pipeline", "as_run"):
        for t, split, d in runs(base, stage):
            dirty, cells = load_table(args.unseen if t in UNSEEN else args.core, t, split)
            mine = pd.read_csv(d / "cells.csv", dtype=str, keep_default_na=False)
            mine = mine[mine["table"] == t] if "table" in mine else mine
            if len(mine) != len(cells) or (mine["dirty_value"].to_numpy() != cells["dirty_value"].to_numpy()).any() \
                    or (mine["clean_value"].to_numpy() != cells["clean_value"].to_numpy()).any():
                raise ValueError(f"{d.name}/{t}: cells differ from the data")
            recs, preds, cand = cell_records(dirty, cells), *load(d, t)
            order = label_order(t, len(dirty)) if split == "test" else []
            ex = set(order[:20])
            flags = {k for k, p in preds.items() if p.get("is_error")}
            applied = {k: p["repair"] for k, p in preds.items() if p.get("decision") == "repair" and p.get("repair") is not None
                       and norm(str(p["repair"])) != norm(dirty.at[k[0], k[1]])}
            decided = {k for k, p in preds.items() if p.get("decision", "keep") != "keep"}
            s = score_flags(flags, recs, ex, applied)
            for rule, fl in (("is_error", flags), ("decision not keep", decided)):
                r = score_flags(fl, recs, ex, applied)
                rows.append({"runs": stage, "table": t, "split": split, "flag_rule": rule, "left_out": "20 labelled rows" if ex else "none",
                             "model_calls": len(preds), **{k: r[k] for k in ("n_scored_cells", "total_true_errors", "tp", "fp", "precision", "recall", "f1",
                                                                              "executed_repairs", "successful_repairs", "incorrect_repairs", "false_repair_rate")}})
            if stage == "as_run":  # X14, detector half
                det = [x for x in DETECTORS if x in cand]
                on = cand[cand["is_candidate"]]
                for drop in ("d7", "d5c"):
                    if drop not in det:
                        continue
                    only = on[(on[drop] > 0) & (on[[x for x in det if x != drop]] > 0).sum(axis=1).eq(0)]
                    gone = set(zip(only["row"].astype(int), only["column"]))
                    r = score_flags(flags - gone, recs, ex, {k: v for k, v in applied.items() if k not in gone})
                    abl.append({"table": t, "split": split, "without": drop, "candidates_removed": len(gone),
                                "errors_removed": sum(norm(dirty.at[k[0], k[1]]) != norm(cells.at[k[0] * dirty.shape[1] + list(dirty.columns).index(k[1]), "clean_value"]) for k in gone),
                                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "delta_f1": round(r["f1"] - s["f1"], 4),
                                "applied_right": r["successful_repairs"], "applied_wrong": r["incorrect_repairs"]})
            else:  # missed errors by stage, final pipeline
                pre = load(base / "as_run" / d.name.replace("final_", ""), t)[0] if (base / "as_run" / d.name.replace("final_", "")).exists() else {}
                n, cols = dirty.shape[1], list(dirty.columns)
                values = cells["clean_value"].to_numpy()
                domain = {c: {norm(v) for v in values[j::n]} for j, c in enumerate(cols)}
                sent = set(zip(cand[cand["is_candidate"]]["row"].astype(int), cand[cand["is_candidate"]]["column"]))
                for i, r in enumerate(recs):
                    if r["error_type"] == "format_artifact" or norm(r["dirty_value"]) == norm(r["clean_value"]) or r["row"] in ex:
                        continue
                    k = (r["row"], r["column"])
                    where = ("flagged" if k in flags else "model said no error" if k in preds else
                             "withheld by the D2 filter" if k in pre else "never a candidate")
                    missed.append({"table": t, "split": split, "column": k[1], "stage": where,
                                   "kind": missed_kind(r["dirty_value"], r["clean_value"], domain[k[1]])})
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out / "m5_final_rows.csv", index=False)
    pd.DataFrame(abl).to_csv(out / "m5_x14_detector_ablation.csv", index=False)
    m = pd.DataFrame(missed)
    m.groupby(list(m.columns)).size().rename("errors").reset_index().to_csv(out / "m5_missed_errors_final.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(abl).to_string(index=False))
    print(m.groupby(["table", "stage"], sort=False).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

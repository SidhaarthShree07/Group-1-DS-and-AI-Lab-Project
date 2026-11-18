#!/usr/bin/env python3
"""Run the repair validation layer on a finished pipeline run and compare the pipeline with and without it.

Author: Sidhaarth Shree

Reads <run>/predictions.jsonl (with the gate's `decision`), the dirty tables and the cell truth, validates every gate
repair (src/validation/repair_validator.py) and writes into the run folder:
  validation_log.jsonl         one audit record per gate repair: every check, its source, before/after and the outcome
  change_log_validated.csv     the repairs that are applied, with the old value kept for reversal
  review_list_validated.csv    the cells referred for review, with the reason
  validation_comparison.csv    existing against enhanced pipeline, per table and in total
The truth is used only here, for the comparison; the layer itself never sees it.

Usage: python scripts/validate_repairs.py <run dir> [--source data/subsets/m3] [--split val] [--copy-to docs/tables/x.csv]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pipeline import load_table  # noqa: E402
from src.profiling.context import norm  # noqa: E402
from src.validation.repair_validator import validate_predictions  # noqa: E402

REVIEW = ("flag for review", "review (invalid answer)")


def repair_kind(p: dict, truth: dict) -> str:
    """correct / wrong (error cell) / wrong (correct cell) / no change, for a proposed repair."""
    dirty, clean_v, is_err = truth[(int(p["row"]), p["column"])]
    new = "" if p.get("repair") is None else str(p["repair"])
    if norm(new) == norm(dirty):
        return "no change"
    if is_err and norm(new) == norm(clean_v):
        return "correct"
    return "wrong (error cell)" if is_err else "wrong (correct cell)"


def compare(table: str, preds: list, truth: dict, seconds: float) -> dict:
    """One row of validation_comparison.csv: the existing pipeline (gate only) against the enhanced one."""
    gate = [p for p in preds if p["gate_decision"] == "repair"]
    kinds = {id(p): repair_kind(p, truth) for p in gate}
    applied_old = [p for p in gate if kinds[id(p)] != "no change"]
    applied_new = [p for p in gate if p["decision"] == "repair" and kinds[id(p)] != "no change"]

    def wrong(repairs):
        return sum(kinds[id(p)].startswith("wrong") for p in repairs)

    def right(repairs):
        return sum(kinds[id(p)] == "correct" for p in repairs)

    def outcome(name):
        return sum((p.get("validation") or {}).get("outcome") == name for p in gate)

    review_old = sum(p["gate_decision"] in REVIEW for p in preds)
    review_new = sum(p["decision"].startswith("review") or p["decision"] == "flag for review" for p in preds)
    return {
        "table": table, "gate_repairs": len(gate),
        "validated": outcome("validated"), "rejected": outcome("rejected"), "unvalidated": outcome("unvalidated"),
        "no_change": outcome("no change"),
        "existing_applied": len(applied_old), "existing_correct": right(applied_old), "existing_incorrect": wrong(applied_old),
        "enhanced_applied": len(applied_new), "enhanced_correct": right(applied_new), "enhanced_incorrect": wrong(applied_new),
        "incorrect_blocked": wrong(applied_old) - wrong(applied_new),
        "correct_rejected": right(applied_old) - right(applied_new),
        "existing_review": review_old, "enhanced_review": review_new,
        "existing_false_repair_rate": round(wrong(applied_old) / max(1, len(applied_old)), 4),
        "enhanced_false_repair_rate": round(wrong(applied_new) / max(1, len(applied_new)), 4),
        "validation_seconds": round(seconds, 3),
        "ms_per_repair": round(1000 * seconds / max(1, len(gate)), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--source", default="data/subsets/m3")
    ap.add_argument("--split", default="val")
    ap.add_argument("--copy-to", default=None, help="also write the comparison table here")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    source = Path(args.source) if Path(args.source).is_absolute() else root / args.source
    preds = [json.loads(x) for x in open(run / "predictions.jsonl", encoding="utf-8")]
    for p in preds:  # a re-run starts from the gate's decision again
        if "gate_decision" in p:
            p["decision"] = p["gate_decision"]
        p.pop("validation", None)
    rows, log, changes, review = [], [], [], []
    for table in sorted({p["table"] for p in preds}):
        dirty, cells = load_table(source, table, args.split)
        truth = {(int(r), c): (d, cl, norm(d) != norm(cl))
                 for r, c, d, cl, et in zip(cells["row"], cells["column"], cells["dirty_value"], cells["clean_value"],
                                            cells["error_type"]) if et != "format_artifact"}
        table_preds = [p for p in preds if p["table"] == table and (int(p["row"]), p["column"]) in truth]
        t0 = time.perf_counter()
        results = validate_predictions(table, dirty, table_preds)
        secs = time.perf_counter() - t0
        rows.append(compare(table, table_preds, truth, secs))
        by_key = {(r.row, r.column): r for r in results}
        for p in table_preds:
            r = by_key.get((int(p["row"]), p["column"]))
            if r is not None:
                log.append({"id": p["id"], "p_error": p.get("p_error"), "gate_decision": p["gate_decision"],
                            "final_decision": p["decision"], **r.as_dict()})
            if p["decision"] == "repair":
                changes.append({"table": table, "row": p["row"], "column": p["column"], "old": r.old, "new": r.new,
                                "p_error": round(p.get("p_error") or 0, 4), "validation": r.reason,
                                "evidence": p.get("evidence")})
            elif p["decision"].startswith("review") or p["decision"] == "flag for review":
                review.append({"table": table, "row": p["row"], "column": p["column"],
                               "value": dirty.at[int(p["row"]), p["column"]], "proposed": p.get("repair"),
                               "p_error": round(p.get("p_error") or 0, 4), "reason": p["decision"] +
                               (f": {r.reason}" if r is not None else "")})
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("table",) and not k.endswith("rate")
           and k != "ms_per_repair"}
    tot["table"] = "all"
    tot["existing_false_repair_rate"] = round(tot["existing_incorrect"] / max(1, tot["existing_applied"]), 4)
    tot["enhanced_false_repair_rate"] = round(tot["enhanced_incorrect"] / max(1, tot["enhanced_applied"]), 4)
    tot["validation_seconds"] = round(tot["validation_seconds"], 3)
    tot["ms_per_repair"] = round(1000 * tot["validation_seconds"] / max(1, tot["gate_repairs"]), 2)
    rows.append(tot)
    table = pd.DataFrame(rows)[list(rows[0].keys())]
    table.to_csv(run / "validation_comparison.csv", index=False)
    if args.copy_to:
        table.to_csv(args.copy_to, index=False)
    with open(run / "validation_log.jsonl", "w", encoding="utf-8") as f:
        for x in log:
            f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    pd.DataFrame(changes).to_csv(run / "change_log_validated.csv", index=False)
    pd.DataFrame(review).to_csv(run / "review_list_validated.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

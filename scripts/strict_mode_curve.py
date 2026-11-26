"""Strict-mode curve: applied repairs right / wrong as the repair-confidence threshold rises.

Author: Sidhaarth Shree

Rescoring only: it reads finished run folders (predictions.jsonl, cells.csv) and makes no model calls.
A repair counts at threshold t when the run applied it (final decision "repair") and its lowest token
probability p_repair_min is at least t; the others would go to review instead.

    python scripts/strict_mode_curve.py runs/m5_final_unseen_soccer runs/m5_final_unseen_german ... > curve.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

THRESHOLDS = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
norm = lambda s: re.sub(r"\s+", " ", str(s or "").strip())  # noqa: E731


def curve(run: Path):
    truth = {(r["table"], int(r["row"]), r["column"]): (r["dirty_value"], r["clean_value"])
             for r in csv.DictReader(open(run / "cells.csv", encoding="utf-8"))}
    applied = {}  # table -> list of (p_repair_min, right)
    for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8")):
        dirty, clean = truth[(p["table"], int(p["row"]), p["column"])]
        if p["decision"] != "repair" or norm(p["repair"]) == norm(dirty):
            continue
        conf = 1.0 if p.get("p_repair_min") is None else p["p_repair_min"]
        applied.setdefault(p["table"], []).append((conf, norm(p["repair"]) == norm(clean)))
    for table, reps in applied.items():
        for t in THRESHOLDS:
            kept = [ok for c, ok in reps if c >= t]
            yield {"table": table, "threshold": t, "applied_right": sum(kept), "applied_wrong": len(kept) - sum(kept),
                   "false_repair_rate": round((len(kept) - sum(kept)) / max(1, len(kept)), 4),
                   "to_review_instead": len(reps) - len(kept)}


if __name__ == "__main__":
    rows = [r for d in sys.argv[1:] for r in curve(Path(d))]
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

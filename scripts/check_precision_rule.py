#!/usr/bin/env python3
"""X7 precision rule without the model: compare the M4 and M5 candidate sets cell by cell (validation splits).

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

A prompt depends on the dirty table and the cell, not on the detector, so where M5 only removes non-errors or only adds
errors or format artifacts (not scored), precision cannot fall. With --e3 the same question is replayed on the run that
has an answer for every cell of the M3 subset (m4_e3_all_ft). Flagged = the model's is_error answer (official scorer).

Usage: python scripts/check_precision_rule.py [--root .] [--e3 <m4_e3_all_ft folder>] [--out docs/tables]
Output: m5_precision_rule.csv, m5_e3_replay.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_m5_candidates import CORE, DIRS, EXTERNAL  # noqa: E402
from src.detectors import DEFAULT_DETECTORS, build_candidates, d5_entity_consensus, d7_format_mix, m5_detectors  # noqa: E402
from src.llm.build_examples import load_split  # noqa: E402


def candidates(df: pd.DataFrame, table: str) -> tuple:
    m4 = {**DEFAULT_DETECTORS, "d7": d7_format_mix.detect, "d5c": lambda x: d5_entity_consensus.detect(x, table)}
    out = []
    for detectors in (m4, m5_detectors(table)):
        k = build_candidates(df, table, detectors=detectors)
        out.append(set(k[k["is_candidate"]][["row", "column"]].itertuples(index=False, name=None)))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--e3")
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    rows = []
    for t in CORE + EXTERNAL:
        d, c = load_split(Path(args.root).resolve(), DIRS, t, "val")
        n = d.shape[1]
        art = {(i // n, d.columns[i % n]) for i in c.index[c["error_type"] == "format_artifact"]}
        err = {(i // n, d.columns[i % n]) for i in c.index[c["is_error"]]} - art
        a, b = candidates(d, t)
        gone, new = a - b, b - a
        verdict = ("pass: identical candidates" if a == b else
                   "pass: only non-errors or artifacts change, errors only added" if not (gone & err) and not (new - err - art)
                   else "not decided: needs the model")
        rows.append({"table": t, "m4_candidates": len(a), "m5_candidates": len(b), "removed_errors": len(gone & err),
                     "removed_non_errors": len(gone - err), "added_errors": len(new & err), "added_non_errors": len(new - err - art),
                     "added_format_artifacts": len(new & art), "precision_rule": verdict})
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "m5_precision_rule.csv", index=False)
    print(df.to_string(index=False))
    if args.e3:
        preds, out = {}, []
        for line in open(Path(args.e3) / "predictions.jsonl", encoding="utf-8"):
            p = json.loads(line)
            preds[p["table"], int(p["row"]), p["column"]] = p["is_error"]
        cells = pd.read_csv(Path(args.e3) / "cells.csv", dtype=str, keep_default_na=False)
        for t, g in cells.groupby("table", sort=False):
            cols = list(dict.fromkeys(g["column"]))
            d = pd.DataFrame(g["dirty_value"].to_numpy().reshape(-1, len(cols)), columns=cols)
            err = {(int(r), k) for r, k, e, y in zip(g["row"], g["column"], g["is_error"], g["error_type"]) if e == "True" and y != "format_artifact"}
            a, b = candidates(d, t)
            for name, cand in (("M4", a), ("M5", b), ("every cell", {x[1:] for x in preds if x[0] == t})):
                flagged = {x for x in cand if preds[(t, *x)]}
                tp = len(flagged & err)
                out.append({"table": t, "detectors": name, "candidates": len(cand), "flagged": len(flagged), "tp": tp, "fp": len(flagged) - tp,
                            "precision": round(tp / max(1, len(flagged)), 4), "recall": round(tp / len(err), 4)})
        pd.DataFrame(out).to_csv(Path(args.out) / "m5_e3_replay.csv", index=False)
        print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

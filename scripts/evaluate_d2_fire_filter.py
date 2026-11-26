#!/usr/bin/env python3
"""X7, extra arm: Sidhaarth's D2 fire-rate filter on top of the M5 detector set, on the eight validation (or train) splits.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

The filter (src/detectors/d2_fire_gate.py) drops a candidate that only D2 flagged, in a column where D2 flags more than
`max_fire_rate` of the cells. No model: candidate recall and candidate count, against arm A and against M4, with the
rules of the X7 check (no table loses more than 1 point of recall, at most 35% of the cells per table).

Usage: python scripts/evaluate_d2_fire_filter.py [--root .] [--split val|train] [--rates 0.5] [--out docs/tables]
Output: m5_d2_fire_filter.csv (or m5_d2_fire_filter_train.csv)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_m5_candidates import CAP, CORE, DIRS, EXTERNAL, FLOOR  # noqa: E402
from src.detectors import DEFAULT_DETECTORS, build_candidates, d2_fire_gate, d5_entity_consensus, d7_format_mix, m5_detectors  # noqa: E402
from src.llm.build_examples import load_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--rates", nargs="*", type=float, default=[d2_fire_gate.DEFAULT_MAX_FIRE_RATE])
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    rows = []
    for t in CORE + EXTERNAL:
        d, c = load_split(Path(args.root).resolve(), DIRS, t, args.split)
        n = d.shape[1]
        err = {(i // n, d.columns[i % n]) for i in c.index[c["is_error"] & (c["error_type"] != "format_artifact")]}
        m4 = build_candidates(d, t, args.split, {**DEFAULT_DETECTORS, "d7": d7_format_mix.detect, "d5c": lambda x: d5_entity_consensus.detect(x, t)})
        arm = build_candidates(d, t, args.split, m5_detectors(t))
        sets = {"M4": m4, "arm A (M5)": arm, **{f"arm A + D2 filter {r}": d2_fire_gate.apply(arm, r) for r in args.rates}}
        base = {}
        for name, k in sets.items():
            cand = set(k[k["is_candidate"]][["row", "column"]].itertuples(index=False, name=None))
            base.setdefault(name, cand)
            rec = 100 * len(cand & err) / len(err)
            rows.append({"table": t, "detectors": name, "candidates": len(cand), "pass_through_pct": round(100 * len(cand) / d.size, 2),
                         "recall_pct": round(rec, 2), "errors_found": len(cand & err), "errors": len(err)})
        a = rows[-len(sets) + 1]  # arm A row of this table
        for x in rows[-len(sets):]:
            x["recall_vs_arm_a"] = round(x["recall_pct"] - a["recall_pct"], 2)
            x["candidates_vs_arm_a_pct"] = round(100 * (x["candidates"] / a["candidates"] - 1), 2)
            m4r = rows[-len(sets)]["recall_pct"]
            x["within_rules_vs_m4"] = x["recall_pct"] >= m4r - FLOOR and x["pass_through_pct"] <= CAP
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / ("m5_d2_fire_filter.csv" if args.split == "val" else "m5_d2_fire_filter_train.csv"), index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

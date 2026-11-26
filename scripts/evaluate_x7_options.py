#!/usr/bin/env python3
"""X7 report for the pipeline owner: candidates, recall and what each detector adds, before and after each change.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Validation (or train) splits only, no model. Steps: M4 set -> M5 set (arm A) -> arm A plus one option at a time ->
arm A plus every option that passes. Rule for an option, against arm A: recall up on at least one table, candidates grow
by less than 10% on every table, no table above 35%. Format artifacts are not errors.
"new" = the cells only this detector flags; "new errors" = the true errors among them.

Usage: python scripts/evaluate_x7_options.py [--root .] [--split val|train] [--out docs/tables]
Output: m5_detector_contribution.csv, m5_x7_options.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_m5_candidates import CAP, CORE, DIRS, EXTERNAL, cells_of  # noqa: E402
from src.detectors import DEFAULT_DETECTORS, d5_entity_consensus, d7_format_mix, m5_detectors  # noqa: E402
from src.detectors.x7_options import OPTIONS  # noqa: E402
from src.llm.build_examples import load_split  # noqa: E402

GROWTH = 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    tables, flags, truth, size, contrib, steps = CORE + EXTERNAL, {}, {}, {}, [], []
    for t in tables:
        d, c = load_split(Path(args.root).resolve(), DIRS, t, args.split)
        n = d.shape[1]
        err = c["is_error"] & (c["error_type"] != "format_artifact")
        truth[t], size[t] = {(i // n, d.columns[i % n]) for i in c.index[err]}, d.size
        m4 = {**DEFAULT_DETECTORS, "d7": d7_format_mix.detect, "d5c": lambda x, t=t: d5_entity_consensus.detect(x, t)}
        flags[t] = {"M4": {k: cells_of(f(d)) for k, f in m4.items()}, "M5": {k: cells_of(f(d)) for k, f in m5_detectors(t).items()},
                    "options": {k: cells_of(f(d, t)) for k, f in OPTIONS.items()}}
        for name in ("M4", "M5"):
            sets = flags[t][name]
            for k, s in sorted(sets.items()):
                new = s - set().union(*[v for j, v in sets.items() if j != k])
                contrib.append({"detector_set": name, "table": t, "detector": k, "flagged": len(s), "errors_in_flagged": len(s & truth[t]),
                                "share_errors_pct": round(100 * len(s & truth[t]) / len(s), 1) if s else "", "new": len(new), "new_errors": len(new & truth[t])})
    pd.DataFrame(contrib).to_csv(Path(args.out) / "m5_detector_contribution.csv", index=False)

    arm = {t: set().union(*flags[t]["M5"].values()) for t in tables}
    m4c = {t: set().union(*flags[t]["M4"].values()) for t in tables}

    def step(label, cand, base):
        out = []
        for t in tables:
            new = cand[t] - base[t]
            out.append({"step": label, "table": t, "errors": len(truth[t]), "candidates": len(cand[t]), "pass_through_pct": round(100 * len(cand[t]) / size[t], 2),
                        "recall_pct": round(100 * len(cand[t] & truth[t]) / len(truth[t]), 2), "added_cells": len(new), "added_errors": len(new & truth[t]),
                        "growth_pct": round(100 * (len(cand[t]) / len(base[t]) - 1), 2)})
        ok = (any(x["added_errors"] > 0 for x in out) and all(x["growth_pct"] < 100 * GROWTH and x["pass_through_pct"] <= CAP for x in out))
        for x in out:
            x["passes_rule"] = ok
        steps.extend(out)
        return ok

    step("M4", m4c, m4c)
    step("M5 (arm A), against M4", arm, m4c)
    passed = [o for o in OPTIONS if step(f"arm A + {o}", {t: arm[t] | flags[t]["options"][o] for t in tables}, arm)]
    if len(passed) > 1:
        step("arm A + " + " + ".join(passed), {t: arm[t].union(*[flags[t]["options"][o] for o in passed]) for t in tables}, arm)
    df = pd.DataFrame(steps)
    df["passes_rule"] = df["passes_rule"].astype(object)
    df.loc[df["step"] == "M4", "passes_rule"] = ""
    df.to_csv(Path(args.out) / "m5_x7_options.csv", index=False)
    print(df.assign(s=lambda x: x.added_cells.astype(str) + " (" + x.added_errors.astype(str) + " err, +" + x.growth_pct.astype(str) + "%)")
          .pivot(index="step", columns="table", values="s").loc[list(dict.fromkeys(df.step)), tables].to_string())
    print(df.groupby("step", sort=False).passes_rule.first().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

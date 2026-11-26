#!/usr/bin/env python3
"""Replay the flag gate on finished unseen runs with a prior-shift correction of P(error) (X11, M5).

Author: Sidhaarth Shree

M4 asked whether the error-rate prior of a new table should be used inside the gate, compared with the fixed default.
For each run, the flag decision "P(error) >= 0.5" is recomputed from the stored model probabilities with the prior
(a) left at the training value 0.5 (the final pipeline), (b) estimated without labels by EM (Saerens et al., 2002) and
(c) set to the true error rate of the candidate cells (an oracle). Scored on the baseline cells (the 20 rows of Raha and
the number-format cells left out). No model calls are made.

Usage: python scripts/em_prior_gate_replay.py <runs dir> <run folder> [<run folder> ...] [--out <csv>]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_unseen_like_baselines import excluded_rows  # noqa: E402
from src.evaluation.calibrate import adjust_prior, estimate_prior_em  # noqa: E402


def replay(run: Path) -> list:
    cells = pd.read_csv(run / "cells.csv", dtype=str, keep_default_na=False)
    cells["row"] = cells["row"].astype(int)
    table = cells["table"].iloc[0]
    all_truth = {(r, c): e == "True" for r, c, e in zip(cells["row"], cells["column"], cells["is_error"])}
    scored = cells[~cells["row"].isin(excluded_rows(table, cells["row"].max() + 1)) & ~cells["error_type"].eq("format_artifact")]
    truth = {(r, c): e == "True" for r, c, e in zip(scored["row"], scored["column"], scored["is_error"])}
    errors = sum(truth.values())
    preds = [json.loads(line) for line in open(run / "predictions.jsonl", encoding="utf-8")]
    probs = [p["p_error"] for p in preds]
    em, _ = estimate_prior_em(probs)
    true_rate = sum(all_truth.get((p["row"], p["column"]), False) for p in preds) / len(preds)
    clip = lambda x: min(max(x, 1e-4), 1 - 1e-4)  # noqa: E731
    out = []
    for name, pi in (("training prior 0.5 (final pipeline)", 0.5), ("EM estimate", clip(em)), ("true rate (oracle)", clip(true_rate))):
        flagged = {(p["row"], p["column"]) for p, a in zip(preds, adjust_prior(probs, pi)) if a >= 0.5} & set(truth)
        tp = sum(truth[k] for k in flagged)
        prec = tp / len(flagged) if flagged else 0.0
        rec = tp / errors if errors else 0.0
        out.append({"table": table, "prior": name, "pi": round(pi, 3), "precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [r for run in args.runs for r in replay(Path(args.runs_dir) / run)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(df.groupby("prior").f1.mean().round(3).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

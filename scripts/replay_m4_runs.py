#!/usr/bin/env python3
"""What the M4 final runs say about the M5 detectors, without new model calls (X7) and where errors are lost (M5 missed errors).

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

A prompt depends on the dirty table and the cell, not on the detector, so an M4 answer stays valid for every cell that
both candidate sets contain. Cells only M5 sends have no answer: precision is bounded (new errors missed and new
non-errors flagged = worst case). Flagged = the model's is_error answer, as in src/evaluation/metrics.py.
The official test scores (docs/tables/m4_test_vs_baselines.csv) are reproduced first; the script stops if they differ.

Usage: python scripts/replay_m4_runs.py --runs <folder with the m4_final* runs> [--root .] [--out docs/tables]
Output: m5_precision_replay.csv, m5_missed_errors_by_stage.csv, m5_contextguard_x9_cells.csv,
m5_candidate_recall_by_type.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_m5_candidates import DIRS, missed_kind  # noqa: E402
from scripts.run_baselines_m5 import label_order  # noqa: E402
from src.detectors import build_candidates, m5_detectors  # noqa: E402
from src.llm.build_examples import TYPE_MAP, load_split  # noqa: E402
from src.profiling.context import norm  # noqa: E402

RUNS = [("m4_final2_val_all", "val", ["hospital", "flights", "beers"]), ("m4_final2_test_hfb", "test", ["hospital", "flights", "beers"]),
        ("m4_final_test_rayyan", "test", ["rayyan"]), ("m4_final_test_movies", "test", ["movies"])]


def prf(flags: set, errors: set) -> tuple:
    tp = len(flags & errors)
    p, r = tp / max(1, len(flags)), tp / max(1, len(errors))
    return p, r, 2 * p * r / max(1e-9, p + r), tp, len(flags) - tp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    ref = pd.read_csv(ROOT / "docs/tables/m4_test_vs_baselines.csv").set_index("table")
    replay, missed, curve, by_type = [], [], [], []
    for run, split, tables in RUNS:
        run_cells = pd.read_csv(Path(args.runs) / run / "cells.csv", dtype=str, keep_default_na=False)  # TableEG types
        preds = defaultdict(dict)
        for line in open(Path(args.runs) / run / "predictions.jsonl", encoding="utf-8"):
            p = json.loads(line)
            preds[p["table"]][int(p["row"]), p["column"]] = p
        for t in tables:
            d, c = load_split(Path(args.root).resolve(), DIRS, t, split)
            n, cols = d.shape[1], list(d.columns)
            values = c["clean_value"].to_numpy()
            clean = {(i // n, cols[i % n]): v for i, v in enumerate(values)}
            errors = {(i // n, cols[i % n]) for i in c.index[c["is_error"]]}
            m4 = set(preds[t])
            flag = {x for x, p in preds[t].items() if p["is_error"]}
            k = build_candidates(d, t, split, detectors=m5_detectors(t))
            m5 = set(k[k["is_candidate"]][["row", "column"]].itertuples(index=False, name=None))
            order = label_order(t, len(d)) if split == "test" else []
            if split == "test":
                keep = {x for x in errors if x[0] not in order[:20]}
                p, r, f1, *_ = prf({x for x in flag if x[0] not in order[:20]}, keep)
                if (round(p, 4), round(r, 4), round(f1, 4)) != tuple(ref.loc[t, ["cg_precision", "cg_recall", "cg_f1"]]):
                    raise ValueError(f"{t}: the run does not reproduce the M4 test score")
                for left, rows in (("20 rows", order[:20]), ("50 rows", order)):
                    p, r, f1, *_ = prf({x for x in flag if x[0] not in rows}, {x for x in errors if x[0] not in rows})
                    curve.append({"table": t, "left_out": left, "configuration": "M4", "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
                k20 = {x for x in errors if x[0] not in order[:20]}
                have, ne, nn = [{x for x in y if x[0] not in order[:20]} for y in (flag & m5, (m5 - m4) & errors, (m5 - m4) - errors)]
                for label, extra in (("M5, worst case (no new error flagged, every new non-error flagged)", nn),
                                     ("M5, best case (every new error flagged, no new non-error flagged)", ne)):
                    p, r, f1, *_ = prf(have | extra, k20)
                    curve.append({"table": t, "left_out": "20 rows", "configuration": label, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
            # bounds on all cells of the split
            p4, r4, _, tp4, fp4 = prf(flag, errors)
            known, new_e, new_n = flag & m5, (m5 - m4) & errors, (m5 - m4) - errors
            tp, fp = len(known & errors), len(known - errors)
            worst = tp / max(1, tp + fp + len(new_n))
            need = next((x for x in range(len(new_e) + 1) if (tp + x) / max(1, tp + x + fp + len(new_n)) >= p4 - 0.01), None)
            # how the model answered the errors M4 did send, in the columns that hold M5's new errors
            nc = Counter(x[1] for x in new_e)
            seen = [x for x in m4 & errors if x[1] in nc]
            per_col = {col: (sum(x in flag for x in seen if x[1] == col), sum(x[1] == col for x in seen)) for col in nc}
            replay.append({"split": split, "table": t, "errors": len(errors), "m4_candidates": len(m4), "m5_candidates": len(m5),
                           "m4_recall_candidates": round(100 * len(m4 & errors) / len(errors), 2), "m5_recall_candidates": round(100 * len(m5 & errors) / len(errors), 2),
                           "m4_precision": round(p4, 4), "m4_recall": round(r4, 4), "m4_tp": tp4, "m4_fp": fp4,
                           "fp_removed_by_m5": fp4 - fp, "tp_lost_by_m5": tp4 - tp, "new_errors": len(new_e), "new_non_errors": len(new_n),
                           "m5_precision_worst_case": round(worst, 4), "new_errors_needed_for_rule": "never" if need is None else need,
                           "m5_recall_if_all_new_errors_flagged": round((tp + len(new_e)) / len(errors), 4),
                           "model_flag_rate_on_same_column_errors": "; ".join(f"{col} {a}/{b}" for col, (a, b) in sorted(per_col.items()))})
            e = run_cells[(run_cells["table"] == t) & (run_cells["is_error"] == "True")]
            for kind, g in e.groupby("error_type"):
                x = set(zip(g["row"].astype(int), g["column"]))
                by_type.append({"split": split, "table": t, "error_type": TYPE_MAP.get(kind, kind), "errors": len(x),
                                "m4_recall_pct": round(100 * len(x & m4) / len(x), 2), "m5_recall_pct": round(100 * len(x & m5) / len(x), 2)})
            domain = {col: {norm(v) for v in values[j::n]} for j, col in enumerate(cols)}
            for (r, col) in sorted(errors):
                kind = missed_kind(d.at[r, col], clean[r, col], domain[col])
                stage = ("flagged" if (r, col) in flag else "model said no error" if (r, col) in m4 else "never sent")
                missed.append({"split": split, "table": t, "column": col, "kind": kind, "m4": stage,
                               "m5_candidate_stage": "sent" if (r, col) in m5 else "never sent"})
    out = Path(args.out)
    rp = pd.DataFrame(replay)
    rp.to_csv(out / "m5_precision_replay.csv", index=False)
    ms = pd.DataFrame(missed)
    ms.groupby(list(ms.columns)).size().rename("errors").reset_index().to_csv(out / "m5_missed_errors_by_stage.csv", index=False)
    pd.DataFrame(by_type).to_csv(out / "m5_candidate_recall_by_type.csv", index=False)
    pd.DataFrame(curve).to_csv(out / "m5_contextguard_x9_cells.csv", index=False)
    print(rp.drop(columns=["model_flag_rate_on_same_column_errors"]).T.to_string())
    for x in replay:
        print(x["split"], x["table"], "|", x["model_flag_rate_on_same_column_errors"])
    print(ms.groupby(["split", "table", "m4", "m5_candidate_stage"], sort=False).size().unstack(fill_value=0).to_string())
    print(pd.DataFrame(curve).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

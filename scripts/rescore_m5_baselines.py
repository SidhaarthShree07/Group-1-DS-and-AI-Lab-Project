#!/usr/bin/env python3
"""Cross-check: score the saved outputs of the M5 baselines again and compare them with the published tables.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Needs no Raha or HoloClean environment: it reads the cells each system flagged and the repairs it made, scores them with
score_flags on the same cells, and compares tp, fp and repair counts with m5_baselines_full.csv, m5_baselines_unseen.csv,
m5_raha_label_budgets.csv and m5_raha_unseen_labelling.csv. Prints every difference; exit code 1 if there is one.

Usage: python scripts/rescore_m5_baselines.py --raw <folder with core/ and unseen/> [--core data/splits/core]
       [--unseen <folder of the unseen tables>] [--tables-dir docs/tables]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baselines_full import score_flags  # noqa: E402
from scripts.run_baselines_m5 import CORE, cell_records, label_order, load_table  # noqa: E402

UNSEEN = ["restaurant", "marketing", "soccer", "billionaire", "german"]
KEYS = ["tp", "fp", "executed_repairs", "successful_repairs"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--core", default="data/splits/core")
    ap.add_argument("--unseen")
    ap.add_argument("--tables-dir", default="docs/tables")
    args = ap.parse_args()
    bad = checked = 0
    for group, tables, splits, full, curve in (("core", CORE, args.core, "m5_baselines_full.csv", "m5_raha_label_budgets.csv"),
                                               ("unseen", UNSEEN, args.unseen, "m5_baselines_unseen.csv", "m5_raha_unseen_labelling.csv")):
        if not splits:
            print(group, "skipped: no data folder given")
            continue
        pub, cur = pd.read_csv(Path(args.tables_dir) / full), pd.read_csv(Path(args.tables_dir) / curve)
        for t in tables:
            dirty, cells = load_table(splits, t, "test")
            recs, cols, order, raw = cell_records(dirty, cells), list(dirty.columns), label_order(t, len(dirty)), Path(args.raw) / group / t
            ex20, name = set(order[:20]), {f"c{i}": c for i, c in enumerate(cols)}
            col = lambda c: c if c in cols else name[c]  # noqa: E731
            got = []
            h = json.loads((raw / "holoclean.json").read_text())
            rep = {(r, col(c)): v for r, c, v in h["repairs"]}
            got.append(("B1_HoloClean", score_flags(list(rep), recs, ex20, rep)))
            got.append(("B1_HoloClean_noisy_cells", score_flags([(r, col(c)) for r, c in h["noisy"]], recs, ex20)))
            for f, b in (("dboost.json", "B2_dBoost"), ("dboost36.json", "B2_dBoost_union36")):
                if (raw / f).exists():
                    got.append((b, score_flags([(i, cols[j]) for i, j in json.loads((raw / f).read_text())[0]["cells"]], recs, ex20)))
            for r in json.loads((raw / "raha.json").read_text()):
                flags, rep = [tuple(x) for x in r["flags"]], {(a, b): v for a, b, v in r.get("repairs", [])}
                if (r["mode"], r["budget"], r["seed"]) == ("fixed", 20, 0):
                    got.append(("B3_Raha_Baran", score_flags(flags, recs, ex20, rep)))
                for left, ex in (("50 rows", set(order)), ("20 rows", ex20)):
                    ex = ex | set(r["labelled_rows"]) if r["mode"] == "own" else ex
                    x = cur[(cur.table == t) & (cur.budget == r["budget"]) & (cur.labelling == r["mode"])
                            & (cur.worker_seed == r["seed"]) & cur.left_out.str.startswith(left)]
                    if len(x):
                        s, checked = score_flags(flags, recs, ex, rep), checked + 1
                        if any(int(x.iloc[0][k]) != s[k] for k in KEYS):
                            bad += 1
                            print("DIFF", t, r["mode"], r["budget"], r["seed"], left)
            for b, s in got:
                x = pub[(pub.table == t) & (pub.baseline == b) & (pub.status == "completed")].iloc[0]
                checked += 1
                if any(int(x[k]) != s[k] for k in KEYS):
                    bad += 1
                    print("DIFF", t, b, {k: (int(x[k]), s[k]) for k in KEYS})
        print(group, "done")
    print(f"{checked} published rows checked, {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

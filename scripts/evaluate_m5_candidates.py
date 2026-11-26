#!/usr/bin/env python3
"""X7: candidate recall and pass-through of the M5 detectors, the sweep behind them and the ablation (no model calls).

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Detectors read the dirty table only; the cell truth is used here, for counting. Format artifacts are not errors.
Pass rule (written before the sweep): recall +2 points on at least one table, no table loses more than 1 point,
at most 35% of the cells of every table sent to the model, total calls at most M4's. (Precision needs the model.)

Usage: python scripts/evaluate_m5_candidates.py [--root .] [--split val] [--out docs/tables]
Output: m5_candidate_recall.csv, m5_x7_sweep.csv, m5_detector_ablation.csv, m5_missed_errors.csv (validation only)
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_missed_errors import MISSING_TOKENS, levenshtein_distance  # noqa: E402
from src.detectors import d1_missing, d2_pattern, d3_robust_z, d4_multivariate, d5_entity_consensus, d5_fd, d6_typo, d7_format_mix  # noqa: E402
from src.detectors.candidates import M5_D2  # noqa: E402
from src.llm.build_examples import load_split  # noqa: E402
from src.profiling.context import norm  # noqa: E402

CORE, EXTERNAL = ["hospital", "flights", "beers", "rayyan", "movies"], ["tax", "inpatient", "facilities"]
DIRS = ["data/splits/core", "data/splits/external"]
CAP, FLOOR, GAIN = 35.0, 1.0, 2.0
GATES = [("none", {}), ("unique>=0.9", {"min_unique_share": 0.9}), ("unique>=0.7", {"min_unique_share": 0.7}),
         ("fire>0.5", {"max_fire_rate": 0.5}), ("fire>0.3", {"max_fire_rate": 0.3}), ("fire>0.2", {"max_fire_rate": 0.2})]
FALLBACKS = [("none", {}), ("damage", {"fallback": "damage"}), ("rare_char 2%", {"fallback": "rare_char", "rare_char_share": 0.02}),
             ("rare_char 5%", {"fallback": "rare_char", "rare_char_share": 0.05})]


def cells_of(out: pd.DataFrame) -> set:
    return set(zip(out["row"].astype(int), out["column"])) if len(out) else set()


def missed_kind(dirty: str, clean: str, clean_values: set) -> str:
    d, c = norm(dirty), norm(clean)
    if d.lower() in MISSING_TOKENS:
        return "missing value"
    if re.sub(r"[,\s]", "", d) == re.sub(r"[,\s]", "", c) or re.sub(r"\.0+$", "", d) == c:
        return "same value, other format"
    if d in clean_values:
        return "valid value, wrong for this row"
    if re.fullmatch(r"\d+[/\-.]\d+[/\-.]\d+", d) and re.fullmatch(r"\d+[/\-.]\d+[/\-.]\d+", c):
        return "date order"
    if re.search(r"\d", d) and re.sub(r"\d+", "0", d) == re.sub(r"\d+", "0", c):
        return "same format, different number or time"
    if c and d and (c.startswith(d) or c.endswith(d)):
        return "truncated"
    if c and c in d:
        return "extra text"
    return "typo (1-3 edits)" if levenshtein_distance(d.lower(), c.lower()) <= 3 else "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder that holds data/splits")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    tables, data, truth, fix = CORE + EXTERNAL, {}, {}, {}
    never = re.compile(r"(?!)")
    for t in tables:
        d, c = load_split(Path(args.root).resolve(), DIRS, t, args.split)
        n = d.shape[1]
        err = c["is_error"] & (c["error_type"] != "format_artifact")
        data[t], truth[t] = (d, c), {(i // n, d.columns[i % n]) for i in c.index[err]}
        f = {"d1": d1_missing.detect(d), "d2": d2_pattern.detect(d), "d3": d3_robust_z.detect(d), "d4": d4_multivariate.detect(d),
             "d5": d5_fd.detect(d), "d6": d6_typo.detect(d), "d5c": d5_entity_consensus.detect(d, t),
             "d7": d7_format_mix.detect(d), "d7x": d7_format_mix.detect(d, extended=True), "d2a": d2_pattern.detect(d, **M5_D2)}
        keep = d3_robust_z.ID_PATTERN
        d3_robust_z.ID_PATTERN = d4_multivariate.ID_PATTERN = never
        f["d3n"], f["d4n"] = d3_robust_z.detect(d), d4_multivariate.detect(d)
        d3_robust_z.ID_PATTERN = d4_multivariate.ID_PATTERN = keep
        fix[t] = {k: cells_of(v) for k, v in f.items()}

    def measure(t, cand):
        hit = len(cand & truth[t])
        return len(cand), 100 * len(cand) / data[t][0].size, hit, 100 * hit / len(truth[t])

    rest = ("d1", "d3", "d4", "d5", "d6")
    sets = {"D1-D6": rest + ("d2",), "D1-D6 + D5 consensus": rest + ("d2", "d5c"), "D1-D6 + D7": rest + ("d2", "d7"),
            "M4 (D1-D6, D7, D5 consensus)": rest + ("d2", "d5c", "d7"),
            "M4 + D7 extension, no D2 gate": rest + ("d2", "d5c", "d7x"), "M4 + D2 gate, no D7 extension": rest + ("d2a", "d5c", "d7"),
            "M5": rest + ("d2a", "d5c", "d7x"),
            "M5, D3/D4 without column names": ("d1", "d3n", "d4n", "d5", "d6", "d2a", "d5c", "d7x")}
    union = {(name, t): set().union(*[fix[t][k] for k in ks]) for name, ks in sets.items() for t in tables}
    ablation = []
    for name in sets:
        for t in tables:
            n, pt, hit, rec = measure(t, union[name, t])
            ablation.append({"detectors": name, "table": t, "candidates": n, "pass_through_pct": round(pt, 2), "errors_found": hit,
                             "errors": len(truth[t]), "recall_pct": round(rec, 2)})
    pd.DataFrame(ablation).to_csv(Path(args.out) / "m5_detector_ablation.csv", index=False)

    m4 = {t: measure(t, union["M4 (D1-D6, D7, D5 consensus)", t]) for t in tables}
    recall = []
    for t in tables:
        a, b = union["M4 (D1-D6, D7, D5 consensus)", t], union["M5", t]
        n, pt, hit, rec = measure(t, b)
        recall.append({"table": t, "role": "core" if t in CORE else "generality check", "cells": data[t][0].size, "errors": len(truth[t]),
                       "m4_candidates": m4[t][0], "m4_pass_through_pct": round(m4[t][1], 2), "m4_recall_pct": round(m4[t][3], 2),
                       "m5_candidates": n, "m5_pass_through_pct": round(pt, 2), "m5_recall_pct": round(rec, 2),
                       "delta_recall": round(rec - m4[t][3], 2), "errors_gained": len((b - a) & truth[t]),
                       "errors_lost": len((a - b) & truth[t]), "new_cells_to_model": len(b - a)})
    rec_df = pd.DataFrame(recall)
    rec_df.to_csv(Path(args.out) / "m5_candidate_recall.csv", index=False)

    sweep = []
    for (gn, gk), (fn, fk), ext in itertools.product(GATES, FALLBACKS, (False, True)):
        if gn == "none" and fn != "none":
            continue
        d2 = {t: cells_of(d2_pattern.detect(data[t][0], **gk, **fk)) for t in tables}
        for budget in ((False,) if gn == "none" else (False, True)):
            row, ms = {"d2_gate": gn, "fallback": fn, "d7_extension": ext, "only_tables_above_35pct": budget}, {}
            for t in tables:
                others = set().union(*[fix[t][k] for k in rest + ("d5c", "d7x" if ext else "d7")])
                full = others | fix[t]["d2"]
                gated = not budget or len(full) > CAP / 100 * data[t][0].size
                ms[t] = measure(t, others | d2[t] if gated else full)
                row[f"{t}_recall"], row[f"{t}_pass"] = round(ms[t][3], 2), round(ms[t][1], 2)
            row["total_candidates"] = sum(m[0] for m in ms.values())
            row["mean_recall"] = round(sum(m[3] for m in ms.values()) / len(ms), 2)
            row["passes_recall_cost_rules"] = (all(m[1] <= CAP and m[3] >= m4[t][3] - FLOOR for t, m in ms.items())
                                               and any(m[3] >= m4[t][3] + GAIN for t, m in ms.items())
                                               and sum(ms[t][0] for t in CORE) <= sum(m4[t][0] for t in CORE))
            row["breaks_floor_on"] = " ".join(t for t, m in ms.items() if m[3] < m4[t][3] - FLOOR)
            sweep.append(row)
    pd.DataFrame(sweep).to_csv(Path(args.out) / "m5_x7_sweep.csv", index=False)

    if args.split == "val":
        groups = defaultdict(list)
        for t in tables:
            d, c = data[t]
            n, values = d.shape[1], c["clean_value"].to_numpy()
            domain = {col: {norm(v) for v in values[j::n]} for j, col in enumerate(d.columns)}
            for r, col in sorted(truth[t] - union["M5", t]):
                cv = values[r * n + list(d.columns).index(col)]
                groups[t, col, missed_kind(d.at[r, col], cv, domain[col])].append((d.at[r, col], cv))
        missed = [{"table": t, "column": col, "kind": k, "missed": len(v), "sample_dirty": v[0][0], "sample_clean": v[0][1]}
                  for (t, col, k), v in groups.items()]
        pd.DataFrame(missed).sort_values("missed", ascending=False).to_csv(Path(args.out) / "m5_missed_errors.csv", index=False)
    print(rec_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

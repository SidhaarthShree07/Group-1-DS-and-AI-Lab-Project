#!/usr/bin/env python3
"""Export the input files of the baselines B1, B2 and B3 from the TEST split of each core table.

Author: Aaradhya (file prepared with Sidhaarth's review fixes)

Protocol (Decision E): every method, including ContextGuard, is scored on the test split of each table. Each baseline
receives the test-split dirty table as its table instance.
  B1 HoloClean : dirty.csv + constraints.txt. The constraints are the filtered approximate dependencies mined on that
                 same dirty table (src/profiling/dependencies.py), written in HoloClean's denial-constraint syntax
                 t1&t2&EQ(t1.X,t2.X)&IQ(t1.Y,t2.Y), so no rule is written by hand.
  B2 dBoost    : dirty.csv + config.json listing Raha's dBoost grid (histogram and gaussian); the configuration is chosen on
                 the train split of the same table, never on the test labels.
  B3 Raha+Baran: dirty.csv, clean.csv (same header, as Raha requires), labelled_tuples_20.json (20 rows drawn with a fixed
                 seed from the test table) and scoring_mask.csv. The 20 labelled rows are excluded from the scoring of ALL
                 methods (Rule D8). raha_baseline_manifest.json records the check.

Usage: python src/data/export_baselines.py [--root .] [--seed 42]
Output: data/processed/baselines/B1_HoloClean|B2_dBoost|B3_Raha_Baran/<table>/
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling.context import norm  # noqa: E402
from src.profiling.dependencies import scan_dependencies  # noqa: E402

TABLES = ["hospital", "flights", "beers", "rayyan", "movies"]
META = {"row_id", "tuple_id", "index", "id_col"}
DBOOST_GRID = ([["histogram", p, o] for p, o in itertools.product(["0.1", "0.3", "0.5", "0.7", "0.9"], repeat=2)] +
               [["gaussian", s] for s in ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]])


def load_test(root: Path, table: str):
    base = root / "data" / "splits" / "core" / table
    dirty = pd.read_csv(base / "test.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clean = pd.read_csv(base / "test_clean.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    keep = [i for i, c in enumerate(dirty.columns) if c.strip().lower() not in META]
    dirty, clean = dirty.iloc[:, keep].reset_index(drop=True), clean.iloc[:, keep].reset_index(drop=True)
    clean.columns = dirty.columns  # Raha needs identical headers
    return dirty, clean


def hc_constraint(lhs: str, rhs: str) -> str:
    parts = [c.strip() for c in lhs.split(",")]
    eq = "&".join(f"EQ(t1.{c},t2.{c})" for c in parts)
    return f"t1&t2&{eq}&IQ(t1.{rhs},t2.{rhs})"


def export(root: Path, seed: int, out: Path) -> list:
    manifest = []
    for table in TABLES:
        dirty, clean = load_test(root, table)
        # B1
        d = out / "B1_HoloClean" / table
        d.mkdir(parents=True, exist_ok=True)
        dirty.to_csv(d / "dirty.csv", index=False)
        deps = scan_dependencies(dirty.map(norm))
        lines = [hc_constraint(r.lhs, r.rhs) for r in deps.itertuples()]
        (d / "constraints.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        # B2
        d = out / "B2_dBoost" / table
        d.mkdir(parents=True, exist_ok=True)
        dirty.to_csv(d / "dirty.csv", index=False)
        (d / "config.json").write_text(json.dumps({
            "table": table, "input_file": "dirty.csv", "grid": DBOOST_GRID,
            "selection_rule": "pick the configuration with the best detection F1 on the train split of this table"},
            indent=2) + "\n", encoding="utf-8")
        # B3
        d = out / "B3_Raha_Baran" / table
        d.mkdir(parents=True, exist_ok=True)
        dirty.to_csv(d / "dirty.csv", index=False)
        clean.to_csv(d / "clean.csv", index=False)
        n = len(dirty)
        rows = sorted(random.Random(f"{seed}:{table}").sample(range(n), min(20, n)))
        (d / "labelled_tuples_20.json").write_text(json.dumps(
            {"table": table, "seed": seed, "label_budget": 20, "labelled_row_indices": rows}, indent=2) + "\n", encoding="utf-8")
        mask = pd.DataFrame({"row_id": range(n), "include_in_scoring": [i not in set(rows) for i in range(n)]})
        mask.to_csv(d / "scoring_mask.csv", index=False)
        manifest.append({"table": table, "test_rows": n, "labelled_rows": len(rows), "scoring_rows": int(mask["include_in_scoring"].sum()),
                         "constraints_b1": len(lines),
                         "rule_d8_verified": len(rows) + int(mask["include_in_scoring"].sum()) == n and
                         not mask.loc[rows, "include_in_scoring"].any()})
        print(f"{table:8s} test rows={n:5d} B1 constraints={len(lines):3d} B3 labelled={len(rows)}", flush=True)
    (out / "B3_Raha_Baran" / "raha_baseline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/processed/baselines")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    manifest = export(root, args.seed, root / args.out)
    return 0 if all(m["rule_d8_verified"] for m in manifest) else 1


if __name__ == "__main__":
    sys.exit(main())

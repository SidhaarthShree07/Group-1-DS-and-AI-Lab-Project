#!/usr/bin/env python3
"""Leakage checks for the split tables and the fine-tuning examples.

Author: Sidhaarth Shree

Checks (each prints PASS or FAIL and is written to docs/tables/m2_leakage_checks.csv; exit code 1 if any fails):
  1. Row overlap: no row key appears in two splits.
  2. Entity overlap: no real entity (hospital, flight, brewery) appears in two splits. The entity is read from the
     CLEAN table, because typos in the dirty key would hide shared entities.
  3. Exact duplicate rows: no data row of validation or test equals a data row of train.
  4. Duplicate prompts: no prompt text appears in two splits.
  5. Unseen tables: no example of an unseen test table, and no mention of one in a prompt, in any split.
  6. Synthetic examples appear only in train.

Usage: python src/data/check_leakage.py [--root .]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SPLITS = ["train", "val", "test"]
ENTITY_COLUMN = {"hospital": "ProviderNumber", "flights": "flight", "beers": "brewery_id", "rayyan": "id", "movies": "Id",
                 "tax": "row_id", "inpatient": "NPI", "facilities": "CCN"}
UNSEEN = ["restaurant", "marketing", "soccer", "billionaire", "germany", "credit"]
TRAINING_TABLES = {"hospital", "flights", "beers", "rayyan", "movies", "tax", "inpatient", "facilities",
                  "adult", "bike_day", "bike_hour", "wine_red", "wine_white"}  # the last five are synthetic (train only)
META = {"row_id", "tuple_id", "index", "id_col"}


def read_rows(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", nargs="*", default=["data/splits/core", "data/splits/external"])
    ap.add_argument("--examples", default="data/processed/examples")
    ap.add_argument("--out", default="docs/tables/m2_leakage_checks.csv")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    results = []

    def record(check, scope, ok, detail):
        results.append({"check": check, "scope": scope, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {check:28s} {scope:10s} {detail}")

    for table, entity_col in ENTITY_COLUMN.items():
        if not any((root / d / table / "train.csv").exists() for d in args.splits):
            continue  # table not built in this run
        keys, entities, data_rows = {}, {}, {}
        for split in SPLITS:
            base = next(root / d / table for d in args.splits if (root / d / table / "train.csv").exists())
            hdr, rows = read_rows(base / f"{split}.csv")
            keys[split] = {r[0] for r in rows}  # first column of the dirty file is the row key (index, tuple_id or id)
            keep = [i for i, h in enumerate(hdr) if h.strip().lower() not in META]
            data_rows[split] = {tuple(r[i] for i in keep) for r in rows}
            chdr, crows = read_rows(base / f"{split}_clean.csv")
            ei = [h.strip().lower() for h in chdr].index(entity_col.lower())
            entities[split] = {r[ei] for r in crows}
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        row_ov = sum(len(keys[a] & keys[b]) for a, b in pairs)
        ent_ov = sum(len(entities[a] & entities[b]) for a, b in pairs)
        dup = sum(len(data_rows[s] & data_rows["train"]) for s in ("val", "test"))
        n_ent = len(set().union(*entities.values()))
        record("row overlap", table, row_ov == 0, f"{row_ov} shared row keys")
        record("entity overlap (clean key)", table, ent_ov == 0, f"{ent_ov} of {n_ent} entities in more than one split")
        record("exact duplicate rows", table, dup == 0, f"{dup} val/test rows equal a train row")

    ex_dir = root / args.examples
    texts, bad_table, bad_prompt, synth_outside = {}, 0, 0, 0
    for split in SPLITS:
        path = ex_dir / f"{split}.jsonl"
        if not path.exists():
            record("examples present", split, False, f"missing {path}")
            continue
        seen = set()
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            seen.add(rec["prompt"])
            bad_table += rec["table"].lower() not in TRAINING_TABLES
            low = rec["prompt"].lower()
            bad_prompt += any(f"table: {u}" in low for u in UNSEEN)
            synth_outside += rec.get("source") == "synthetic" and split != "train"
        texts[split] = seen
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if a in texts and b in texts:
            record("duplicate prompts", f"{a}/{b}", not (texts[a] & texts[b]), f"{len(texts[a] & texts[b])} identical prompts")
    record("unseen tables in examples", "all", bad_table == 0 and bad_prompt == 0,
           f"{bad_table} examples from a non-training table, {bad_prompt} prompts naming an unseen table")
    record("synthetic only in train", "all", synth_outside == 0, f"{synth_outside} synthetic examples outside train")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "scope", "result", "detail"])
        w.writeheader()
        w.writerows(results)
    failed = [r for r in results if r["result"] == "FAIL"]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

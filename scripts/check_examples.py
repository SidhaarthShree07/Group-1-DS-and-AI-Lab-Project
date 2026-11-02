#!/usr/bin/env python3
"""Check that every fine-tuning example is consistent with its source cell and does not reveal the answer.

Author: Sidhaarth Shree

For every record in data/processed/examples/{train,val,test}.jsonl the script goes back to the split files and checks:
  1. target:    the cell marked <-- SUSPECT and the "Cell value" in the header are the DIRTY value of cell (row, column).
  2. label:     the answer's is_error and error_type equal the record's and the ground-truth file's.
  3. repair:    for an error the answer's repair is the clean value of that cell; for a correct cell it is null.
  4. no answer: for an error the shown value differs from the repair, and the clean value of the cell is not shown
                in the row of the prompt under the suspect column.
  5. format:    the answer is JSON with exactly is_error, error_type, evidence, repair.
It also counts, for information, error examples whose repair text also occurs somewhere else in the prompt. That is
allowed only when it comes from the dirty table (a dependency majority or a similar row); the clean table is never read
by the prompt builder (src/profiling/context.py receives the dirty table only).
Output: docs/tables/m2_example_checks.csv (one row per table and split). Exit code 1 if any hard check fails.

Usage: python scripts/check_examples.py [--root .]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import TYPE_MAP, load_split  # noqa: E402
from src.profiling.context import norm  # noqa: E402

SPLITS = ["train", "val", "test"]
SUSPECT = re.compile(r"^- (?P<col>.*?): (?P<val>'.*'|\".*\")   <-- SUSPECT$", re.M)
HEADER = re.compile(r"^Table: .*? \| Column: .*? \| Cell value: (?P<val>'.*'|\".*\")$", re.M)


def same(shown: str, value: str) -> bool:
    """`shown` is the value as printed in the prompt: whole, or cut with a trailing ellipsis."""
    return shown == value or (shown.endswith("…") and value.startswith(shown[:-1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--examples", default="data/processed/examples")
    ap.add_argument("--splits", nargs="*", default=["data/splits/core", "data/splits/external", "data/splits/synthetic"])
    ap.add_argument("--out", default="docs/tables/m2_example_checks.csv")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    stats = defaultdict(lambda: defaultdict(int))
    cache = {}
    problems = []
    for split in SPLITS:
        for line in open(root / args.examples / f"{split}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            table, row, col = rec["table"], rec["row"], rec["column"]
            key = (table, split)
            if key not in cache:
                dirty, cells = load_split(root, args.splits, table, split)
                cache[key] = (list(dirty.columns), cells)
            cols, cells = cache[key]
            truth = cells.iloc[row * len(cols) + cols.index(col)]
            s = stats[key]
            s["examples"] += 1
            dirty_value = norm(truth["dirty_value"])
            ok = {}
            sus = SUSPECT.search(rec["prompt"])
            head = HEADER.search(rec["prompt"])
            ok["target"] = bool(sus and head and sus.group("col") == col
                                and same(ast.literal_eval(sus.group("val")), dirty_value)
                                and same(ast.literal_eval(head.group("val")), dirty_value))
            try:
                ans = json.loads(rec["response"])
            except ValueError:
                ans = {}
            etype = TYPE_MAP.get(truth["error_type"], truth["error_type"]) if truth["is_error"] else "none"
            ok["format"] = set(ans) == {"is_error", "error_type", "evidence", "repair"}
            ok["label"] = (ans.get("is_error") == bool(truth["is_error"]) == rec["is_error"]
                           and ans.get("error_type") == etype == rec["error_type"])
            ok["repair"] = ans.get("repair") == (truth["clean_value"] if truth["is_error"] else None)
            clean = norm(truth["clean_value"])
            shown = ast.literal_eval(sus.group("val")) if sus else ""
            ok["no_answer"] = not (truth["is_error"] and (clean == shown or clean == norm(shown)))
            for name, good in ok.items():
                s[name + "_ok"] += good
                if not good and len(problems) < 10:
                    problems.append((rec["id"], name))
            if sus and truth["is_error"] and clean and clean in rec["prompt"].replace(sus.group(0), ""):
                s["repair_also_in_prompt"] += 1  # from dirty-table evidence; informational
            s["errors"] += bool(truth["is_error"])
    fields = ["table", "split", "examples", "errors", "target_ok", "format_ok", "label_ok", "repair_ok", "no_answer_ok",
              "repair_also_in_prompt"]
    rows = [{"table": t, "split": sp, **{f: stats[(t, sp)][f] for f in fields[2:]}} for (t, sp) in sorted(stats)]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    total = {f: sum(r[f] for r in rows) for f in fields[2:]}
    print(f"{total['examples']} examples, {total['errors']} errors")
    bad = False
    for name in ("target", "format", "label", "repair", "no_answer"):
        n_bad = total["examples"] - total[name + "_ok"]
        bad |= n_bad > 0
        print(f"[{'PASS' if n_bad == 0 else 'FAIL'}] {name:10s} {total[name + '_ok']} of {total['examples']} correct")
    print(f"info: {total['repair_also_in_prompt']} of {total['errors']} error examples have the repair text elsewhere in the "
          f"prompt (dirty-table evidence)")
    for p in problems:
        print("  problem:", p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

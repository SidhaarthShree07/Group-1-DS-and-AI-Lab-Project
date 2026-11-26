#!/usr/bin/env python3
"""Examples for cells that are empty in the clean table too (a blank that is a genuine value), with the M5 blank-context lines.

Author: Sidhaarth Shree

The training samples hold very few of them (124 of 5,516 empty-cell examples in M4), so the new lines would be a perfect
sign of an error. Taking them from the train splits only restores the balance between the two kinds of empty cell.

Usage: python scripts/blank_negatives.py --splits data/splits/core --out <file.jsonl> [--per-table 500] [--tokenizer <path>]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import CORE_TABLES, TokenCounter, build_prompt_within_budget, load_split, make_response  # noqa: E402
from src.profiling.context import MISSING, TableContext  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", nargs="*", default=["data/splits/core"])
    ap.add_argument("--tables", nargs="*", default=CORE_TABLES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-table", type=int, default=500)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root, count, out, total = Path(args.root).resolve(), TokenCounter(args.tokenizer), [], 0
    for table in args.tables:
        dirty, cells = load_split(root, args.splits, table, "train")
        ctx = TableContext(table, dirty, evidence_threshold=0.9, blank_context=True)
        n_cols = len(ctx.cols)
        pool = [i for i, (e, v) in enumerate(zip(cells["is_error"], cells["dirty_value"]))
                if not e and v.strip().lower() in MISSING]
        pick = random.Random(f"{args.seed}:{table}:blank").sample(pool, min(args.per_table, len(pool)))
        for i in sorted(pick):
            r, c = divmod(i, n_cols)
            col = ctx.cols[c]
            prompt, _, _ = build_prompt_within_budget(ctx, r, col, count, 1500, ["profile", "dependencies", "similar"])
            ev = ctx.evidence_sentence(r, col, False, "none")
            out.append({"id": f"{table}:train:{r}:{col}", "table": table, "split": "train", "row": int(r), "column": col,
                        "source": "benchmark", "is_error": False, "error_type": "none", "abstain": False,
                        "prompt": prompt, "response": make_response(False, "none", ev, None)})
        print(f"{table}: {len(pool)} genuine empty cells in train, {len(pick)} used", flush=True)
        total += len(pick)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

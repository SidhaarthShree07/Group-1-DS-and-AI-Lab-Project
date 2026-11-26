#!/usr/bin/env python3
"""Build the M5 training file: new error-family examples, a replay of the M4 examples and genuine-empty examples.

Author: Sidhaarth Shree

Three sources are mixed with a fixed seed:
  1. examples built from the augmented train splits (src/data/augment_m5.py), the same number per table;
  2. a random replay sample of the M4 training examples (rewritten with the blank-context lines by
     scripts/add_blank_context.py), so the model keeps what it learned in M4;
  3. examples of cells that are empty in the clean table too (scripts/blank_negatives.py).
The mixed file is shuffled. The report at the end counts the empty-cell examples, split into errors and genuine
blanks, and how many prompts carry the new blank-context lines.

Usage: python scripts/mix_m5_examples.py --aug <aug.jsonl> --replay <m4.jsonl> --blank <genuine.jsonl> --out <mix.jsonl>
                                         [--n-aug 9000] [--n-replay 13000] [--seed 42]
"""
import argparse
import collections
import json
import random


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug", required=True, help="examples built from the augmented train splits")
    ap.add_argument("--replay", required=True, help="M4 training examples, with the blank-context lines added")
    ap.add_argument("--blank", required=True, help="examples of genuinely empty cells")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-aug", type=int, default=9000, help="total augmented examples, split equally over the tables")
    ap.add_argument("--n-replay", type=int, default=13000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_table = collections.defaultdict(list)
    for row in read_jsonl(args.aug):
        by_table[row["table"]].append(row)
    mix = []
    for table, rows in sorted(by_table.items()):
        mix += rng.sample(rows, min(args.n_aug // len(by_table), len(rows)))
    mix += rng.sample(read_jsonl(args.replay), args.n_replay)
    mix += read_jsonl(args.blank)
    rng.shuffle(mix)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in mix:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    empty = [r for r in mix if "Cell value: ''" in r["prompt"]]
    print(f"{len(mix)} examples; empty-cell examples: {len(empty)} "
          f"({sum(r['is_error'] for r in empty)} errors, {sum(not r['is_error'] for r in empty)} genuine)")
    print("prompts with the closed-set line:", sum("an empty cell is not one of them" in r["prompt"] for r in mix),
          "| with the row-filled line:", sum("cells are filled" in r["prompt"] for r in mix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

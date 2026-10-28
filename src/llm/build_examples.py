#!/usr/bin/env python3
"""Build the instruction-response examples for fine-tuning from the split tables.

Author: Sidhaarth Shree

Input : data/splits/core/<table>/{train,val,test}.csv (dirty) and {train,val,test}_cells.csv (cell truth),
        written by src/data/split.py.
Output: data/processed/examples/{train,val,test}.jsonl, one record per cell, plus
        docs/tables/m2_final_dataset_summary.csv and docs/tables/m2_token_lengths.csv.

Each record: id, table, split, row, column, source, is_error, error_type, prompt, response.
The prompt is built from the DIRTY table of the same split only (src/profiling/context.py). The response is the
JSON answer the model must learn. Selection per table and split: every error cell, plus correct cells at a fixed
ratio, half of them "hard" (a rare value in a column that repeats values). Every random step is seeded.

Usage: python src/llm/build_examples.py [--root .] [--neg-ratio 2] [--tokenizer <hf name or path>] [--full-test]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling.context import ALL_BLOCKS, TableContext  # noqa: E402

CORE_TABLES = ["hospital", "flights", "beers", "rayyan", "movies"]
EXTERNAL_TABLES = ["tax", "inpatient", "facilities"]  # training tables from the external sources
SYNTHETIC_TABLES = ["adult", "bike_day", "bike_hour", "wine_red", "wine_white"]  # src/data/synthetic.py; train only
SYNTHETIC_MIX = {"missing": 0.20, "pattern": 0.25, "rule": 0.25, "outlier": 0.30}  # share of synthetic error examples per type
SPLITS = ["train", "val", "test"]
META_COLUMNS = {"row_id", "tuple_id", "index", "id_col"}
TYPE_MAP = {"missing_value": "missing", "pattern_violation": "pattern", "rule_violation": "rule",
            "outliers": "outlier", "outlier": "outlier", "": "none"}
CHARS_PER_TOKEN = 3.2  # used only when no tokenizer is given; the real count must be taken with the Gemma tokenizer


class TokenCounter:
    def __init__(self, name: Optional[str]):
        self.tok = None
        self.method = f"estimate (characters / {CHARS_PER_TOKEN})"
        if name:
            from transformers import AutoTokenizer  # optional dependency

            self.tok = AutoTokenizer.from_pretrained(name)
            self.method = f"tokenizer {name}"

    def __call__(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False)["input_ids"])
        return math.ceil(len(text) / CHARS_PER_TOKEN)


def find_table_dir(root: Path, splits_dirs, table: str) -> Path:
    for d in splits_dirs:
        if (root / d / table / "train.csv").exists():
            return root / d / table
    raise FileNotFoundError(f"no split files for table {table!r} in {list(splits_dirs)}")


def load_split(root: Path, splits_dirs, table: str, split: str):
    """Return (dirty DataFrame without metadata columns, truth DataFrame reshaped to rows x columns)."""
    base = find_table_dir(root, splits_dirs, table)
    dirty = pd.read_csv(base / f"{split}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    keep = [c for c in dirty.columns if c.strip().lower() not in META_COLUMNS]
    dirty = dirty[keep].reset_index(drop=True)
    cells = pd.read_csv(base / f"{split}_cells.csv", dtype=str, keep_default_na=False)
    n_rows, n_cols = len(dirty), len(keep)
    if len(cells) != n_rows * n_cols:
        raise ValueError(f"{table}/{split}: {len(cells)} cell records but {n_rows} rows x {n_cols} columns")
    cells["is_error"] = cells["is_error"].eq("True")
    return dirty, cells


def sample_cells(ctx: TableContext, cells: pd.DataFrame, neg_ratio: float, hard_share: float, seed_text: str,
                 everything: bool, error_subset: Optional[set] = None) -> List[tuple]:
    """List of (row, column index) to turn into examples."""
    n_cols = len(ctx.cols)
    idx = np.arange(len(cells))
    err = cells["is_error"].to_numpy()
    art = cells["error_type"].eq("format_artifact").to_numpy()  # same number in another format: not a label (M1 decision)
    rng = random.Random(seed_text)
    pos = [int(i) for i in idx[err & ~art]]
    if error_subset is not None:  # synthetic tables: the error cells were chosen by synthetic_quota()
        pos = [i for i in pos if i in error_subset]
    neg = [int(i) for i in idx[~err]]
    ctx.artifacts_skipped = int(art.sum())
    if everything:
        chosen = pos + neg
    else:
        k = min(len(neg), int(round(neg_ratio * len(pos))))
        hard = []
        for i in neg:
            r, c = divmod(i, n_cols)
            col = ctx.cols[c]
            prof = ctx.profile(col)
            if prof["distinct"] < 0.5 * ctx.n and prof["counts"].get(ctx.df.at[r, col], 0) <= 2:
                hard.append(i)
        hard_set = set(hard)
        n_hard = min(len(hard), int(round(hard_share * k)))
        pick = rng.sample(hard, n_hard)
        picked = set(pick)
        rest = [i for i in neg if i not in picked]
        pick += rng.sample(rest, k - n_hard)
        chosen = pos + pick
        ctx.hard_counts = {"hard_pool": len(hard_set), "hard_used": n_hard}
    chosen.sort()
    return [divmod(i, n_cols) for i in chosen]


def synthetic_quota(root: Path, splits_dirs, tables, n_real_train: int, share: float, seed: int) -> Dict[str, set]:
    """Choose the synthetic error cells so that synthetic examples are at most `share` of the training examples.

    With one correct example per error example, the synthetic error cells are at most (share / (1 - share)) * n_real / 2.
    The cells are drawn by error type according to SYNTHETIC_MIX, so that rare types (outlier, rule) are well covered.
    """
    budget = int(share / (1 - share) * n_real_train / 2)
    pool: Dict[str, list] = collections.defaultdict(list)
    for t in tables:
        base = find_table_dir(root, splits_dirs, t)
        cells = pd.read_csv(base / "train_cells.csv", dtype=str, keep_default_na=False, usecols=["is_error", "error_type"])
        for i in np.flatnonzero(cells["is_error"].eq("True").to_numpy()):
            etype = TYPE_MAP.get(cells.at[i, "error_type"], cells.at[i, "error_type"])
            if etype in SYNTHETIC_MIX:
                pool[etype].append((t, int(i)))
    rng = random.Random(f"{seed}:synthetic")
    chosen: Dict[str, set] = collections.defaultdict(set)
    for etype, share_e in SYNTHETIC_MIX.items():
        take = min(len(pool[etype]), int(budget * share_e))
        for t, i in rng.sample(pool[etype], take):
            chosen[t].add(i)
    return chosen


def make_response(is_error: bool, error_type: str, evidence: str, repair: Optional[str]) -> str:
    return json.dumps({"is_error": is_error, "error_type": error_type, "evidence": evidence, "repair": repair},
                      ensure_ascii=False)


def build_prompt_within_budget(ctx: TableContext, r: int, col: str, count: TokenCounter, budget: int, blocks):
    """Trim in a fixed order until the prompt fits: fewer similar rows, shorter values, no top-values line."""
    for level, (n_sim, chars, top) in enumerate([(3, 40, True), (1, 40, True), (0, 40, True), (0, 24, True), (0, 24, False)]):
        text = ctx.prompt(r, col, blocks=blocks, max_chars=chars, n_similar=n_sim, top_values=top)
        tokens = count(text)
        if tokens <= budget:
            return text, tokens, level
    return text, tokens, level


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", nargs="*", default=["data/splits/core", "data/splits/external", "data/splits/synthetic"],
                    help="folders with one sub-folder per table")
    ap.add_argument("--out", default="data/processed/examples")
    ap.add_argument("--tables-dir", default="docs/tables")
    ap.add_argument("--sample-dir", default="data/sample")
    ap.add_argument("--tables", nargs="*", default=CORE_TABLES + EXTERNAL_TABLES + SYNTHETIC_TABLES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="correct cells per error cell")
    ap.add_argument("--hard-share", type=float, default=0.5, help="share of the correct cells that are hard negatives")
    ap.add_argument("--synthetic-share", type=float, default=0.20, help="largest share of synthetic examples in train")
    ap.add_argument("--max-prompt-tokens", type=int, default=1500)
    ap.add_argument("--max-repair-chars", type=int, default=200,
                    help="error cells whose clean value is longer are left out (reported in the summary)")
    ap.add_argument("--tokenizer", default=None, help="Hugging Face name or path of the Gemma tokenizer")
    ap.add_argument("--blocks", nargs="*", default=list(ALL_BLOCKS), choices=ALL_BLOCKS)
    ap.add_argument("--full-test", action="store_true", help="also write test_full.jsonl with every test cell")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    count = TokenCounter(args.tokenizer)
    records: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    full_test: List[dict] = []
    summary, token_rows = [], []

    real_tables = [t for t in args.tables if t not in SYNTHETIC_TABLES]
    synth_tables = [t for t in args.tables if t in SYNTHETIC_TABLES]
    quota: Optional[Dict[str, set]] = None
    for table in real_tables + synth_tables:
        base = find_table_dir(root, args.splits, table)
        synthetic = table in SYNTHETIC_TABLES
        if synthetic and quota is None:
            n_real = len(records["train"])
            quota = synthetic_quota(root, args.splits, synth_tables, n_real, args.synthetic_share, args.seed)
            print(f"synthetic cap: {args.synthetic_share:.0%} of train -> {sum(len(v) for v in quota.values())} error cells "
                  f"(real train examples: {n_real})", flush=True)
        for split in SPLITS:
            if not (base / f"{split}.csv").exists():
                continue  # synthetic tables have a train split only
            dirty, cells = load_split(root, args.splits, table, split)
            ctx = TableContext(table, dirty)
            for mode in (["sampled", "full"] if (split == "test" and args.full_test) else ["sampled"]):
                chosen = sample_cells(ctx, cells, args.neg_ratio, args.hard_share, f"{args.seed}:{table}:{split}", mode == "full",
                                      quota.get(table, set()) if synthetic else None)
                tokens, trimmed, recs, excluded = [], 0, [], 0
                for r, c in chosen:
                    col = ctx.cols[c]
                    truth = cells.iloc[r * len(ctx.cols) + c]
                    is_error = bool(truth["is_error"])
                    if is_error and len(truth["clean_value"]) > args.max_repair_chars:
                        excluded += 1
                        continue
                    etype = TYPE_MAP.get(truth["error_type"], truth["error_type"]) if is_error else "none"
                    prompt, ntok, level = build_prompt_within_budget(ctx, r, col, count, args.max_prompt_tokens, args.blocks)
                    trimmed += level > 0
                    repair = truth["clean_value"] if is_error else None
                    answer = make_response(is_error, etype, ctx.evidence_sentence(r, col, is_error, etype), repair)
                    recs.append({"id": f"{table}:{split}:{r}:{col}", "table": table, "split": split, "row": int(r), "column": col,
                                 "source": "synthetic" if synthetic else "benchmark", "is_error": is_error, "error_type": etype,
                                 "prompt": prompt, "response": answer})
                    tokens.append(ntok + count(answer))
                if mode == "full":
                    full_test += recs
                    continue
                records[split] += recs
                n_err = sum(x["is_error"] for x in recs)
                types = collections.Counter(x["error_type"] for x in recs if x["is_error"])
                summary.append({"table": table, "split": split, "source": "synthetic" if synthetic else "benchmark", "rows": len(dirty), "cells": len(cells),
                                "examples": len(recs), "error_examples": n_err, "correct_examples": len(recs) - n_err,
                                "missing": types["missing"], "pattern": types["pattern"], "rule": types["rule"],
                                "outlier": types["outlier"], "hard_negatives": getattr(ctx, "hard_counts", {}).get("hard_used", 0),
                                "trimmed_prompts": trimmed, "excluded_long_repair": excluded,
                                "format_artifacts_skipped": getattr(ctx, "artifacts_skipped", 0)})
                a = np.array(tokens)
                token_rows.append({"table": table, "split": split, "examples": len(a), "median": int(np.median(a)),
                                   "p95": int(np.percentile(a, 95)), "max": int(a.max()), "method": count.method,
                                   "budget": args.max_prompt_tokens, "over_budget": int((a > args.max_prompt_tokens + 100).sum())})
                print(f"{table:8s} {split:5s} rows={len(dirty):5d} examples={len(recs):6d} errors={n_err:5d} "
                      f"median_tokens={int(np.median(a))} p95={int(np.percentile(a, 95))} trimmed={trimmed}", flush=True)

    order = random.Random(args.seed)
    for split in SPLITS:
        order.shuffle(records[split])
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in records[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if full_test:
        with open(out / "test_full.jsonl", "w", encoding="utf-8") as f:
            for rec in full_test:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tdir = root / args.tables_dir
    tdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("m2_final_dataset_summary.csv", summary), ("m2_token_lengths.csv", token_rows)):
        with open(tdir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # a 20-record sample for the repository: five tables, error and correct, all from Apache-2.0 sources
    sdir = root / args.sample_dir
    sdir.mkdir(parents=True, exist_ok=True)
    pick, rng = [], random.Random(args.seed)
    for table in [t for t in args.tables if t in CORE_TABLES]:  # Apache-2.0 sources only
        # keep the public sample free of third-party names that read like attributions
        pool = [x for x in records["train"] if x["table"] == table and "claude" not in x["prompt"].lower()]
        pool_e = [x for x in pool if x["is_error"]]
        pool_c = [x for x in pool if not x["is_error"]]
        pick += rng.sample(pool_e, 2) + rng.sample(pool_c, 2)
    with open(sdir / "m2_examples_sample.jsonl", "w", encoding="utf-8") as f:
        for rec in pick:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total = {s: len(records[s]) for s in SPLITS}
    print("examples:", total, "| tokens:", count.method)


if __name__ == "__main__":
    main()

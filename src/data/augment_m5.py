#!/usr/bin/env python3
"""M5 training augmentation: new error families injected into the CLEAN train splits of the core tables.

Author: Sidhaarth Shree

Why: the M4 training data has character typos only in columns that repeat values (synthetic.py skips near-unique
columns, and its typos are an 'x' substitution or a case flip), missing values at low rates only, and no
abbreviation, initialism or scrambled-category errors. These are standard error families of the error-generation
literature (BART, Jenga, TableEG). Only the train splits of the five core tables are used; validation, test and the
unseen tables are never read.

Families (label in brackets, decided by the Raha rule src/profiling/context.norm):
  typo        a realistic misspelling inside one word: keyboard-neighbour substitution, deletion, transposition or
              doubling, in any text column, near-unique columns included (pattern)
  abbreviate  one long word cut to its first 3 to 5 letters plus '.', or a multi-word value written as its
              initials (pattern)
  scramble    the letters of a value of a categorical column shuffled (pattern)
  blank       a high-rate copy only: one or two columns with no empty cell get 25 to 40% of their cells emptied
              (missing), so that "frequent" no longer implies "correct"

Two dirty copies per table: <table>_auglo (about 5% of cells corrupted) and <table>_aughi (about 20%, plus blanks).
Output: data/splits/aug/<name>/train{,_clean,_cells}.csv, read by src/llm/build_examples.py (--splits data/splits/aug).

Usage: python src/data/augment_m5.py [--root .] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling.context import norm  # noqa: E402

CORE = ["hospital", "flights", "beers", "rayyan", "movies"]
META = {"row_id", "tuple_id", "index", "id_col"}
RATES = {"lo": {"typo": 0.02, "abbreviate": 0.015, "scramble": 0.015},
         "hi": {"typo": 0.07, "abbreviate": 0.06, "scramble": 0.05}}
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBOURS = {}
for r, row in enumerate(ROWS):
    for i, ch in enumerate(row):
        near = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS):
                near += [ROWS[rr][j] for j in (i - 1, i) if 0 <= j < len(ROWS[rr])]
        NEIGHBOURS[ch] = near
WORD = re.compile(r"[A-Za-zÀ-ÿ]{4,}")


def is_text(values) -> bool:
    """A column whose filled values are mostly letters (no ids, numbers, dates or codes)."""
    filled = [v for v in values if v.strip()]
    if not filled:
        return False
    letters = sum(1 for v in filled if sum(ch.isalpha() for ch in v) >= 0.6 * len(v.replace(" ", "")) and WORD.search(v))
    return letters >= 0.8 * len(filled)


def typo(value: str, rng: random.Random) -> str:
    words = list(WORD.finditer(value))
    if not words:
        return value
    m = rng.choice(words)
    w, s = m.group(), m.start()
    i = rng.randrange(1, len(w) - 1)  # keep the first letter, as most real misspellings do
    op = rng.choice(["near", "del", "swap", "dup"])
    if op == "near" and w[i].lower() in NEIGHBOURS:
        ch = rng.choice(NEIGHBOURS[w[i].lower()])
        new = w[:i] + (ch.upper() if w[i].isupper() else ch) + w[i + 1:]
    elif op == "del":
        new = w[:i] + w[i + 1:]
    elif op == "swap" and i < len(w) - 1:
        new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    else:
        new = w[:i] + w[i] + w[i:]
    return value[:s] + new + value[s + len(w):]


def abbreviate(value: str, rng: random.Random) -> str:
    parts = value.split()
    if len(parts) >= 2 and rng.random() < 0.3 and all(p[:1].isalpha() for p in parts) and len(parts) <= 5:
        return "".join(p[0].upper() for p in parts)
    long_words = [m for m in WORD.finditer(value) if len(m.group()) >= 6]
    if not long_words:
        return value
    m = rng.choice(long_words)
    cut = m.group()[:rng.randint(3, min(5, len(m.group()) - 2))] + "."
    return value[:m.start()] + cut + value[m.end():]


def scramble(value: str, rng: random.Random) -> str:
    if len(value) < 4 or " " in value.strip():
        return value
    chars = list(value)
    for _ in range(5):
        rng.shuffle(chars)
        if "".join(chars) != value:
            return "".join(chars)
    return value


def corrupt(clean: pd.DataFrame, level: str, rng: random.Random):
    cols = [c for c in clean.columns if c.strip().lower() not in META]
    dirty = clean.copy()
    n = len(clean)
    text = {c for c in cols if is_text(clean[c].tolist())}
    categorical = {c for c in text if clean[c].nunique() <= max(30, 0.05 * n)}
    labels = {}
    for c in cols:
        families = [f for f in ("typo", "abbreviate", "scramble") if
                    (f == "typo" and c in text) or (f == "abbreviate" and c in text) or (f == "scramble" and c in categorical)]
        for i in range(n):
            old = clean.at[i, c]
            if not old.strip():
                continue
            for f in families:
                if rng.random() < RATES[level][f]:
                    new = {"typo": typo, "abbreviate": abbreviate, "scramble": scramble}[f](old, rng)
                    if norm(new) != norm(old):
                        dirty.at[i, c] = new
                        labels[(i, c)] = "pattern_violation"
                    break
    if level == "hi":
        # repeating columns only: in a near-unique column the blank cannot be repaired from the table, and training on
        # such repairs would teach the model to invent values
        full = [c for c in cols if (clean[c].str.strip() != "").all() and clean[c].nunique() <= 0.2 * n]
        low = [c for c in full if clean[c].nunique() <= 5]  # closed value sets: the blank-context lines list them
        chosen = rng.sample(low, min(len(low), 1))
        rest = [c for c in full if c not in chosen]
        chosen += rng.sample(rest, min(len(rest), 1))
        for c in chosen:
            share = rng.uniform(0.25, 0.40)
            for i in rng.sample(range(n), int(share * n)):
                if (i, c) not in labels:
                    dirty.at[i, c] = ""
                    labels[(i, c)] = "missing_value"
    rows = []
    for i in range(n):
        for c in cols:
            err = (i, c) in labels
            rows.append((i, c, dirty.at[i, c], clean.at[i, c], err, labels.get((i, c), "")))
    cells = pd.DataFrame(rows, columns=["row", "column", "dirty_value", "clean_value", "is_error", "error_type"])
    return dirty, cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--source", default="data/splits/core")
    ap.add_argument("--out", default="data/splits/aug")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)
    for t in CORE:
        clean = pd.read_csv(root / args.source / t / "train_clean.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for level in ("lo", "hi"):
            rng = random.Random(f"{args.seed}:{t}:{level}")
            dirty, cells = corrupt(clean, level, rng)
            out = root / args.out / f"{t}_aug{level}"
            out.mkdir(parents=True, exist_ok=True)
            dirty.to_csv(out / "train.csv", index=False)
            clean.to_csv(out / "train_clean.csv", index=False)
            cells.to_csv(out / "train_cells.csv", index=False)
            kinds = cells[cells.is_error].error_type.value_counts().to_dict()
            print(f"{t}_aug{level}: rows {len(clean)}, errors {int(cells.is_error.sum())} "
                  f"({cells.is_error.mean():.1%}) {kinds}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

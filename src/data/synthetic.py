#!/usr/bin/env python3
"""Generate synthetic dirty versions of clean UCI tables, with the same files as the benchmark tables.

Author: Sahilpreet (file prepared with Sidhaarth's review fixes)

Why: the five core benchmark tables contain no outlier errors and few rule violations, so training needs more
examples of those types. The synthetic data is used for training only, tagged source=synthetic, and capped at 20% of
the training examples (the cap is applied in src/llm/build_examples.py).

Inputs (data/raw/, see scripts/download_data.py): UCI Adult (adult.data only; adult.test is never read), Bike Sharing
(day.csv, hour.csv) and Wine Quality (red, white).

For each table a dirty copy is made by corrupting cells with four recipes, one per error type:
  missing  a placeholder (empty, N/A, NULL, ?, Not Available) replaces the value
  pattern  a character-level typo (x substitution, deleted, doubled or swapped character, case flip)
  rule     a value that breaks a dependency of the clean table (A -> B holds for at least 99.5% of rows) is replaced by
           another valid value of the column; dependencies are DISCOVERED from the clean table, none is hard-coded
  outlier  a numeric value is moved far outside the column's range (more than five interquartile ranges)
Each cell gets at most one corruption. Truth is decided by the Raha rule (src/profiling/context.norm), so a change that
the rule ignores is not an error. Output per table: data/processed/synthetic/<table>_cells.csv and
data/splits/synthetic/<table>/train{,_clean,_cells}.csv (train only), plus docs/tables/m2_synthetic_summary.csv.

Usage: python src/data/synthetic.py [--raw data/raw] [--seed 42]
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling.context import norm  # noqa: E402  (one comparison rule for the whole project)

SEED = 42
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship",
                 "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
PLACEHOLDERS = ["", "N/A", "NULL", "?", "Not Available"]
# probability that an eligible cell receives each corruption; the type mix is chosen to add what the real tables lack
RATES = {"missing_value": 0.015, "pattern_violation": 0.020, "rule_violation": 0.015, "outliers": 0.020}
MIN_DEP_STRENGTH = 0.995


def load_tables(raw: Path) -> dict:
    """Clean UCI tables as strings. adult.test is never read."""
    out = {}
    adult = raw / "adult" / "adult.data"
    out["adult"] = pd.read_csv(adult, header=None, names=ADULT_COLUMNS, skipinitialspace=True, dtype=str, keep_default_na=False)
    with ZipFile(raw / "bike+sharing+dataset.zip") as z:
        for name in ("day", "hour"):
            with z.open(f"{name}.csv") as f:
                out[f"bike_{name}"] = pd.read_csv(f, dtype=str, keep_default_na=False)
    with ZipFile(raw / "wine+quality.zip") as z:
        for color in ("red", "white"):
            with z.open(f"winequality-{color}.csv") as f:
                out[f"wine_{color}"] = pd.read_csv(f, sep=";", dtype=str, keep_default_na=False)
    return out


def is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def numeric_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        vals = df[c]
        if vals.map(is_number).mean() > 0.99 and vals.nunique() >= 20 and vals.nunique() < 0.95 * len(df):
            cols.append(c)
    return cols


def discover_dependencies(df: pd.DataFrame, max_deps: int = 6) -> list:
    """(A, B) pairs where B is determined by A in at least 99.5% of rows. A is not a key and B is not constant."""
    n = len(df)
    found = []
    for a in df.columns:
        if not 2 <= df[a].nunique() <= 0.5 * n:
            continue
        for b in df.columns:
            if a == b or df[b].nunique() < 2:
                continue
            g = df.groupby([a, b]).size().reset_index(name="k")
            strength = g.groupby(a)["k"].max().sum() / n
            if strength >= MIN_DEP_STRENGTH:
                found.append((strength, df[a].nunique(), a, b))
    found.sort(key=lambda t: (-t[0], t[1]))
    return [(a, b) for _, _, a, b in found[:max_deps]]


def typo(value: str, rng: random.Random) -> str:
    """A character-level corruption that changes the value under the Raha rule."""
    if not value:
        return value
    ops = ["sub", "del", "dup", "swap", "case"]
    for _ in range(10):
        op = rng.choice(ops)
        i = rng.randrange(len(value))
        if op == "sub":
            new = value[:i] + "x" + value[i + 1:]
        elif op == "del" and len(value) > 1:
            new = value[:i] + value[i + 1:]
        elif op == "dup":
            new = value[:i] + value[i] + value[i:]
        elif op == "swap" and len(value) > 1:
            i = min(i, len(value) - 2)
            new = value[:i] + value[i + 1] + value[i] + value[i + 2:]
        elif op == "case" and value != value.swapcase():
            new = value.swapcase()
        else:
            continue
        if norm(new) != norm(value):
            return new
    return value + "x"


def column_stats(col: pd.Series) -> dict:
    x = pd.to_numeric(col, errors="coerce").dropna()
    q1, q3 = np.percentile(x, [25, 75])
    return {"iqr": max(q3 - q1, 1e-9), "min": float(x.min()), "max": float(x.max()),
            "integer": bool(all(float(s).is_integer() for s in col.head(500) if is_number(s)))}


def outlier_value(value: str, st: dict, rng: random.Random) -> str:
    """Move a number far outside the range of its column (more than five IQRs beyond the minimum or maximum)."""
    far = rng.uniform(6, 12) * st["iqr"]
    up = rng.random() < 0.7 or st["min"] - far < 0 <= st["min"]  # a non-negative column is mostly pushed upwards
    new = st["max"] + far if up else st["min"] - far
    return str(int(round(new))) if st["integer"] else f"{new:.2f}"


def corrupt(name: str, clean: pd.DataFrame, rng: random.Random):
    """Return the dirty table and a list of injection records."""
    dirty = clean.copy()
    n = len(clean)
    key_like = {c for c in clean.columns if clean[c].nunique() >= 0.95 * n}
    num_cols = set(numeric_columns(clean))
    stats = {c: column_stats(clean[c]) for c in num_cols}
    deps = discover_dependencies(clean)
    deps_by_b = collections.defaultdict(list)
    for a, b in deps:
        deps_by_b[b].append(a)
    values = {c: sorted(set(clean[c])) for c in clean.columns}
    records = []
    probs = list(RATES.items())
    for c in clean.columns:
        if c in key_like:
            continue
        col = clean[c]
        for i in range(n):
            u = rng.random()
            acc, kind = 0.0, None
            for k, p in probs:
                acc += p
                if u < acc:
                    kind = k
                    break
            if kind is None:
                continue
            old = col.iat[i]
            if kind == "missing_value":
                new = rng.choice(PLACEHOLDERS)
            elif kind == "pattern_violation":
                new = typo(old, rng)
            elif kind == "rule_violation":
                if c not in deps_by_b:
                    continue
                pool = [v for v in values[c] if v != old]
                if not pool:
                    continue
                new = rng.choice(pool)
            else:  # outliers
                if c not in num_cols or not is_number(old):
                    continue
                new = outlier_value(old, stats[c], rng)
            if norm(new) == norm(old):
                continue
            dirty.iat[i, dirty.columns.get_loc(c)] = new
            records.append((i, c, kind, old, new))
    return dirty, records, deps


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    tables = load_tables(root / args.raw)
    summary = []
    for name, clean in tables.items():
        rng = random.Random(f"{args.seed}:{name}")
        dirty, records, deps = corrupt(name, clean, rng)
        kinds = {(r[0], r[1]): r[2] for r in records}
        cells, counts = [], collections.Counter()
        cols = list(clean.columns)
        for i in range(len(clean)):
            for c in cols:
                a, b = clean.at[i, c], dirty.at[i, c]
                err = norm(a) != norm(b)
                etype = kinds.get((i, c), "") if err else ""
                counts[etype] += 1 if err else 0
                cells.append([i, c, b, a, err, etype])
        header = ["row", "column", "dirty_value", "clean_value", "is_error", "error_type"]
        write_csv(root / "data/processed/synthetic" / f"{name}_cells.csv", header, cells)
        base = root / "data/splits/synthetic" / name
        write_csv(base / "train.csv", ["row_id"] + cols, [[i] + list(dirty.iloc[i]) for i in range(len(dirty))])
        write_csv(base / "train_clean.csv", ["row_id"] + cols, [[i] + list(clean.iloc[i]) for i in range(len(clean))])
        write_csv(base / "train_cells.csv", header, cells)
        for kind in RATES:
            ex = next((r for r in records if r[2] == kind), None)
            summary.append({"table": name, "error_type": kind, "rows": len(clean), "cells": len(clean) * len(cols),
                            "injected_cells": counts[kind], "dependencies_used": "; ".join(f"{a}->{b}" for a, b in deps) if kind == "rule_violation" else "",
                            "example_column": ex[1] if ex else "", "example_clean": ex[3] if ex else "",
                            "example_dirty": ex[4] if ex else "", "source": "synthetic", "seed": args.seed})
        print(f"{name:10s} rows={len(clean):6d} errors={sum(counts.values()):6d} " +
              " ".join(f"{k.split('_')[0]}={counts[k]}" for k in RATES) + f" deps={len(deps)}", flush=True)
    out = root / args.out
    write_csv(out / "m2_synthetic_summary.csv", list(summary[0]), [list(r.values()) for r in summary])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Approximate functional dependencies by the g3 measure (a TANE-style scan, not the TANE lattice algorithm).

Author: Aaradhya (file prepared with Sidhaarth's review fixes)

A dependency X -> Y holds approximately when g3 = 1 - (sum over X-groups of the size of the most common Y) / n is small;
strength = 1 - g3. This scan tries every single column and every pair of columns as X, so it is exhaustive for
determinants of up to two columns, but it is not the pruned lattice search of TANE (Huhtala et al. 1999).

Three filters remove dependencies that carry no information:
  * key-like determinants: X must not have more distinct values than `key_share` of the rows (a key determines everything);
  * support: at least `min_support` of the rows must sit in X-groups with two or more rows;
  * gain: strength must exceed the share of the most common Y value by at least `min_gain` (otherwise Y is nearly constant
    and the dependency says nothing).
A pair X is only reported if neither of its columns alone already determines Y (minimality).
"""
from __future__ import annotations

import itertools
from typing import List

import pandas as pd


def _g3(df: pd.DataFrame, lhs: List[str], rhs: str) -> float:
    g = df.groupby(lhs + [rhs]).size().reset_index(name="k")
    return 1.0 - g.groupby(lhs)["k"].max().sum() / len(df)


def _support(df: pd.DataFrame, lhs: List[str]) -> float:
    sizes = df.groupby(lhs).size()
    return float(sizes[sizes >= 2].sum() / len(df))


def strength_matrix(df: pd.DataFrame, key_share: float = 0.6) -> pd.DataFrame:
    """strength[A][B] for single-column determinants; NaN where A is key-like, constant or equal to B."""
    n = len(df)
    out = pd.DataFrame(float("nan"), index=df.columns, columns=df.columns)
    for a in df.columns:
        if not 2 <= df[a].nunique() <= key_share * n:
            continue
        for b in df.columns:
            if a != b:
                out.loc[a, b] = 1.0 - _g3(df, [a], b)
    return out


def scan_dependencies(df: pd.DataFrame, max_g3: float = 0.10, key_share: float = 0.6, min_support: float = 0.5,
                      min_gain: float = 0.05, pairs: bool = True) -> pd.DataFrame:
    n = len(df)
    nuniq = df.nunique()
    maj = {c: df[c].value_counts().iloc[0] / n for c in df.columns}
    cand = [c for c in df.columns if 2 <= nuniq[c] <= key_share * n]
    rows, single_ok = [], set()

    def keep(lhs, rhs, g3):
        strength = 1.0 - g3
        return g3 <= max_g3 and strength - maj[rhs] >= min_gain and _support(df, lhs) >= min_support

    for a in cand:
        for b in df.columns:
            if a == b or nuniq[b] < 2:
                continue
            g3 = _g3(df, [a], b)
            if keep([a], b, g3):
                single_ok.add((a, b))
                rows.append({"lhs": a, "rhs": b, "lhs_columns": 1, "lhs_distinct": int(nuniq[a]), "g3": round(g3, 4),
                             "strength": round(1 - g3, 4), "majority_share_of_rhs": round(maj[b], 4),
                             "support": round(_support(df, [a]), 3)})
    if pairs:
        for a1, a2 in itertools.combinations(cand, 2):
            joint = df[[a1, a2]].drop_duplicates().shape[0]
            if joint > key_share * n:
                continue
            for b in df.columns:
                if b in (a1, a2) or nuniq[b] < 2 or (a1, b) in single_ok or (a2, b) in single_ok:
                    continue
                g3 = _g3(df, [a1, a2], b)
                if keep([a1, a2], b, g3):
                    rows.append({"lhs": f"{a1}, {a2}", "rhs": b, "lhs_columns": 2, "lhs_distinct": int(joint), "g3": round(g3, 4),
                                 "strength": round(1 - g3, 4), "majority_share_of_rhs": round(maj[b], 4),
                                 "support": round(_support(df, [a1, a2]), 3)})
    cols = ["lhs", "rhs", "lhs_columns", "lhs_distinct", "g3", "strength", "majority_share_of_rhs", "support"]
    return pd.DataFrame(rows, columns=cols).sort_values(["strength", "lhs_columns"], ascending=[False, True]).reset_index(drop=True)

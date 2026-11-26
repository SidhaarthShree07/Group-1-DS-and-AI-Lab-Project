"""Optional X7 detectors (M5), off by default; not part of m5_detectors().

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree), from suggestions by Sidhaarth Shree

The suggestions came from labelled errors of unseen tables (Soccer, German, Billionaire). Results with these options on
those tables are therefore not held-out results. Each option reads the dirty table only and uses no column names.
Settings are fixed in advance (no sweep). Output: row, column, detector, score, as the other detectors.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

import pandas as pd

from src.detectors.d5_entity_consensus import MIN_DEPENDENCY, MIN_GROUP_ROWS, MIN_SUPPORT
from src.profiling.context import MISSING, TableContext

COLUMNS = ["row", "column", "detector", "score"]
MIN_GAIN = 0.05           # O1: the pair must beat either column alone and the column's majority share by this much
MIN_COVERAGE = 0.5        # O1: share of rows in groups of at least MIN_GROUP_ROWS
STRICT = (0.85, 0.70)     # O1b: plain D5's own dependency and group-support thresholds
DOMINANT_SHARE = 0.80     # O2
MIN_SPELLING_SHARE = 0.02  # O3
MAX_CATEGORIES, MIN_FILLED, MAX_COUNT = 50, 100, 2  # O4

# O3: the same word written out and abbreviated
SPELLINGS = [(re.compile(long + r"\b", re.I), re.compile(short, re.I)) for long, short in (
    (r"stra(?:ß|ss)e", r"str\.?(?=\s|$)"), (r"\bstreet", r"\bst\b\.?"), (r"\broad", r"\brd\b\.?"),
    (r"\bavenue", r"\bave?\b\.?"), (r"\bboulevard", r"\bblvd\b\.?"), (r"\bdrive", r"\bdr\b\.?"),
    (r"\blane", r"\bln\b\.?"), (r"\bhighway", r"\bhwy\b\.?"), (r"\bsuite", r"\bste\b\.?"))]


def _frame(flags: list, name: str) -> pd.DataFrame:
    out = pd.DataFrame(flags, columns=["row", "column", "score"])
    out["detector"] = name
    return out[COLUMNS]


def _filled(df: pd.DataFrame, col: str) -> list:
    return [(i, str(v).strip()) for i, v in enumerate(df[col]) if str(v).strip().lower() not in MISSING]


def composite_key(ctx: TableContext, min_dependency: float = MIN_DEPENDENCY):
    """The column pair whose groups determine the other columns best, and those columns; (None, []) if there is none."""
    n, best, best_score = ctx.n, (None, []), 0.0
    if n == 0:
        return best
    cols = [c for c in ctx.cols if 2 <= ctx.df[c].nunique() <= 0.5 * n
            and ctx.df[c].str.lower().isin(MISSING).mean() < 0.05]
    single = {}
    for a in cols:
        g = ctx.df.groupby(a)
        for b in ctx.cols:
            if b != a:
                single[a, b] = float(g[b].agg(lambda s: s.value_counts().iloc[0]).sum() / n)
    majority = {c: ctx.df[c].value_counts().iloc[0] / n for c in ctx.cols}
    for a, b in itertools.combinations(cols, 2):
        sizes = ctx.df.groupby([a, b]).size()
        if sizes[sizes >= MIN_GROUP_ROWS].sum() / n < MIN_COVERAGE or max(single[a, b], single[b, a]) >= 0.98:
            continue  # groups too small, or one column already determines the other: a single key
        targets, score = [], 0.0
        for c in ctx.cols:
            if c in (a, b):
                continue
            k = ctx.df.groupby([a, b, c]).size().groupby(level=[0, 1]).max().sum() / n
            gain = k - max(majority[c], single[a, c], single[b, c])
            if k >= min_dependency and gain >= MIN_GAIN:
                targets.append(c)
                score += gain
        if score > best_score:
            best, best_score = ((a, b), targets), score
    return best


def composite_consensus(dirty_df: pd.DataFrame, name: str = "table", min_dependency: float = MIN_DEPENDENCY,
                        min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """O1. D5 entity consensus with two columns as the entity; silent when one column is already the entity."""
    ctx = TableContext(name, dirty_df)
    if ctx.entity_key() is not None:
        return _frame([], "o1")
    pair, targets = composite_key(ctx, min_dependency)
    flags = []
    if pair:
        for key, ids in ctx.df.groupby(list(pair)).indices.items():
            rows = [int(i) for i in ids]
            if len(rows) < MIN_GROUP_ROWS or any(str(k).lower() in MISSING for k in key):
                continue
            for col in targets:
                weights, votes = ctx.row_reliability(rows, exclude=(*pair, col)), defaultdict(float)
                for r in rows:
                    if ctx.df.at[r, col].lower() not in MISSING:
                        votes[ctx.df.at[r, col]] += weights[r]
                if votes:
                    value = max(votes, key=votes.get)
                    support = votes[value] / sum(weights.values())
                    if support >= min_support:
                        flags += [(r, col, round(support, 3)) for r in rows if ctx.df.at[r, col] != value]
    return _frame(flags, "o1")


def dominant_value(dirty_df: pd.DataFrame) -> pd.DataFrame:
    """O2. One value fills at least 80% of a column: flag every other filled cell."""
    flags = []
    for col in dirty_df.columns:
        cells = _filled(dirty_df, col)
        if len(cells) >= 10:
            value, k = Counter(v for _, v in cells).most_common(1)[0]
            if k / len(cells) >= DOMINANT_SHARE:
                flags += [(i, col, round(k / len(cells), 4)) for i, v in cells if v != value]
    return _frame(flags, "o2")


def abbreviation_mix(dirty_df: pd.DataFrame) -> pd.DataFrame:
    """O3. A column uses both the long and the short spelling of a word: flag the rarer spelling."""
    flags = {}
    for col in dirty_df.columns:
        cells = _filled(dirty_df, col)
        for long, short in SPELLINGS:
            a, b = [i for i, v in cells if long.search(v)], [i for i, v in cells if short.search(v)]
            if cells and min(len(a), len(b)) / len(cells) >= MIN_SPELLING_SHARE:
                flags.update({(i, col): 1.0 for i in (a if len(a) < len(b) else b)})
    return _frame([(i, c, s) for (i, c), s in flags.items()], "o3")


def rare_category(dirty_df: pd.DataFrame) -> pd.DataFrame:
    """O4. A column with fewer than 50 distinct values: flag the values seen at most twice."""
    flags = []
    for col in dirty_df.columns:
        cells = _filled(dirty_df, col)
        counts = Counter(v for _, v in cells)
        if len(cells) >= MIN_FILLED and 2 <= len(counts) < MAX_CATEGORIES:
            flags += [(i, col, round(1 - counts[v] / len(cells), 4)) for i, v in cells if counts[v] <= MAX_COUNT]
    return _frame(flags, "o4")


OPTIONS = {"o1 composite key": composite_consensus,
           "o1b composite key, D5 thresholds": lambda df, name: composite_consensus(df, name, *STRICT),
           "o2 dominant value": lambda df, name: dominant_value(df),
           "o3 abbreviation mix": lambda df, name: abbreviation_mix(df), "o4 rare category": lambda df, name: rare_category(df)}

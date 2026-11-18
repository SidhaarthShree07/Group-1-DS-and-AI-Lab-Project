#!/usr/bin/env python3
"""D5 extension: cells that disagree with what the other rows say about the same entity.

Author: Sidhaarth Shree

D5 only uses a dependency A -> B that holds for at least 90% of the rows. When several sources describe the same
objects (Flights: many websites report the same flight) the sources conflict, so flight -> departure time holds
for only 55 to 80% of the rows and D5 never fires, although most of the errors are exactly these conflicts.

This detector works on the dirty table alone and uses no column names:
  1. find the entity column with TableContext.entity_key() (for Flights it finds `flight`). A table with one
     row per object has no entity column and gets no flags;
  2. for every column the entity determines for at least MIN_DEPENDENCY of the rows, and every entity with at
     least MIN_GROUP_ROWS rows, take the consensus value: each row votes with its reliability, i.e. how well its
     other columns agree with the group (truth discovery, as in Li et al., SIGMOD 2014);
  3. if the consensus is clear (at least MIN_SUPPORT of the group's vote), flag every cell of the group that
     differs from it, empty cells included.
Output has the same shape as D1 to D6; the score is the consensus support.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.profiling.context import MISSING, TableContext

COLUMNS = ["row", "column", "detector", "score"]
MIN_DEPENDENCY = 0.5
MIN_GROUP_ROWS = 3
MIN_SUPPORT = 0.4


def consensus_value(ctx: TableContext, rows: list, entity_col: str, col: str):
    """(value, support) of the reliability-weighted vote of `rows` on `col`; (None, 0) if every cell is empty."""
    weights = ctx.row_reliability(rows, exclude=(entity_col, col))
    votes = defaultdict(float)
    for row in rows:
        value = ctx.df.at[row, col]
        if value.lower() not in MISSING:
            votes[value] += weights[row]
    if not votes:
        return None, 0.0
    winner = max(votes, key=votes.get)
    return winner, votes[winner] / sum(weights.values())


def detect(dirty_df: pd.DataFrame, name: str = "table") -> pd.DataFrame:
    ctx = TableContext(name, dirty_df)
    entity_col = ctx.entity_key()
    if entity_col is None:
        return pd.DataFrame(columns=COLUMNS)

    strength = ctx.dependencies()
    target_cols = [c for c in ctx.cols if c != entity_col and strength.get((entity_col, c), 0.0) >= MIN_DEPENDENCY]
    flags = []
    for entity, row_ids in ctx.df.groupby(entity_col).indices.items():
        rows = [int(r) for r in row_ids]
        if entity.lower() in MISSING or len(rows) < MIN_GROUP_ROWS:
            continue
        for col in target_cols:
            value, support = consensus_value(ctx, rows, entity_col, col)
            if value is None or support < MIN_SUPPORT:
                continue
            flags += [(row, col, round(support, 3)) for row in rows if ctx.df.at[row, col] != value]

    out = pd.DataFrame(flags, columns=["row", "column", "score"])
    out["detector"] = "d5c"
    return out[COLUMNS]

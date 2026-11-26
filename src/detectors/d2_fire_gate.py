"""Candidate filter (M5): do not send a cell to the model when only D2 flagged it and D2 flags most of its column.

Author: Sidhaarth Shree

D2 flags a value whose shape is rare in its column. In a column of names, titles or addresses almost every value has
a rare shape, so D2 fires on most cells and the flag says nothing about whether a cell is wrong. A cell that only D2
flagged, in a column where D2 flagged more than `max_fire_rate` of the cells, is dropped from the candidates. A cell
that any other detector also flagged is kept. The choice of 0.5 was made on the validation splits of the five core
tables (scripts/d2_fire_gate_check.py); the rule reads the dirty table's own detector output only, nothing per table.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_MAX_FIRE_RATE = 0.5
OTHER_DETECTORS = ("d1", "d3", "d4", "d5", "d5c", "d6", "d7", "o1b")


def apply(candidates: pd.DataFrame, max_fire_rate: float = DEFAULT_MAX_FIRE_RATE) -> pd.DataFrame:
    """Return a copy of `candidates` (build_candidates output) with the gated cells set to is_candidate = False."""
    out = candidates.copy()
    if "d2" not in out or out.empty:
        return out
    others = [d for d in OTHER_DETECTORS if d in out]
    d2_only = (out["d2"] > 0) & ((out[others] > 0).sum(axis=1) == 0 if others else True)
    fire = (out["d2"] > 0).groupby(out["column"]).transform("mean")
    out.loc[d2_only & (fire > max_fire_rate), "is_candidate"] = False
    return out

"""Pattern and format rarity detector (D2) for ContextGuard Milestone 4.

Author: Aaradhya (Group 1 · DSAI Lab, IIT Madras BS Degree)

Evaluates value syntax, character shapes, and pattern distributions across
tabular columns in an unsupervised manner. Strictly operates on the dirty split
with zero clean-data access.
"""
from __future__ import annotations

import collections
import html
import re
import unicodedata
from typing import Optional, Set, Tuple
import pandas as pd

MISSING_TOKENS = {
    "", "nan", "null", "none", "n/a", "na", "-", "?", 
    "not available", "undefined", "empty"
}

def normalize(val: object) -> str:
    """Normalize whitespace and HTML entities."""
    if val is None or pd.isna(val):
        return ""
    text = html.unescape(str(val))
    text = re.sub(r"[\t\n ]+", " ", text)
    return text.strip("\t\n ")

def shape(value: str) -> str:
    """Map string characters to canonical shape representation (A=upper, a=lower, 9=digit)."""
    out, prev, run = [], None, 0
    for ch in value:
        key = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if key == prev:
            run += 1
        else:
            if prev is not None:
                out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
            prev, run = key, 1
    if prev is not None:
        out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
    return "".join(out)

def has_encoding_damage(value: str) -> bool:
    """True for a replacement, control, private-use or unassigned character, or a loose combining mark."""
    return any(ch == "\ufffd" or unicodedata.category(ch) in ("Cc", "Cn", "Co", "Mn") for ch in value)


def free_text_flags(norm_vals: list, valid_indices: list, fallback: Optional[str], rare_char_share: float) -> list:
    """Cells still flagged in a free-text column, where shape rarity says nothing: (row, score)."""
    if fallback is None:
        return []
    total = len(valid_indices)
    char_cells: collections.Counter = collections.Counter()
    if fallback == "rare_char":
        for i in valid_indices:
            char_cells.update(set(norm_vals[i].lower()))
    flags = []
    for i in valid_indices:
        if has_encoding_damage(norm_vals[i]):
            flags.append((i, 1.0))
        elif fallback == "rare_char":
            share = min(char_cells[ch] for ch in set(norm_vals[i].lower())) / total
            if share <= rare_char_share:
                flags.append((i, round(1.0 - share, 4)))
    return flags


def detect(dirty_df: pd.DataFrame, rarity_threshold: float = 0.02, max_fire_rate: Optional[float] = None,
           min_unique_share: Optional[float] = None, fallback: Optional[str] = None,
           rare_char_share: float = 0.02) -> pd.DataFrame:
    """Detect pattern and format anomalies across dirty columns.

    M5 (X7): a column where nearly every shape is rare is free text, and rarity carries no information there. With
    `max_fire_rate` (share of the column D2 itself would flag) or `min_unique_share` (share of distinct values) such
    a column gets no shape flags, only those of `fallback` (None, "damage" or "rare_char"). With all three left at
    None the detector is exactly the M4 one.

    Threshold Selection Note (Train/Validation):
    - rarity_threshold = 0.02 (2%): Selected on the train split to capture format
      outliers while preventing false alarms on columns with natural pattern diversity.
    - min_valid_samples = 10: Ensures sufficient support before profiling shape counts.
    - count_cutoff <= 2 when total >= 30: Catches isolated singleton formatting noise.

    Operates strictly without table-specific heuristics or clean data access.
    """
    records = []
    seen: Set[Tuple[int, str]] = set()
    columns = list(dirty_df.columns)
    
    # Generic pattern rarity detection based on character shape distributions
    for col in columns:
        series = dirty_df[col]
        norm_vals = [normalize(v) for v in series]
        valid_indices = [i for i, v in enumerate(norm_vals) if v.lower() not in MISSING_TOKENS and v]
        
        if len(valid_indices) < 10:
            continue
            
        shapes = [shape(norm_vals[i]) for i in valid_indices]
        counts = collections.Counter(shapes)
        total = len(valid_indices)
        
        flags = []
        for idx, s in zip(valid_indices, shapes):
            freq = counts[s] / total
            if freq <= rarity_threshold or (counts[s] <= 2 and total >= 30):
                flags.append((idx, round(1.0 - freq, 4)))

        unique_share = len({norm_vals[i] for i in valid_indices}) / total
        if ((max_fire_rate is not None and len(flags) / total > max_fire_rate)
                or (min_unique_share is not None and unique_share >= min_unique_share)):
            flags = free_text_flags(norm_vals, valid_indices, fallback, rare_char_share)

        for idx, score in flags:
            seen.add((idx, col))
            records.append({
                "row": idx,
                "column": col,
                "detector": "d2",
                "score": score
            })

    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)

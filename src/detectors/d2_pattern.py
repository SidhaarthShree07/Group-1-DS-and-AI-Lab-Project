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
from typing import Set, Tuple
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

def detect(dirty_df: pd.DataFrame, rarity_threshold: float = 0.02) -> pd.DataFrame:
    """Detect pattern and format anomalies across dirty columns.

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
        
        for idx in valid_indices:
            s = shape(norm_vals[idx])
            freq = counts[s] / total
            if freq <= rarity_threshold or (counts[s] <= 2 and total >= 30):
                score = round(1.0 - freq, 4)
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

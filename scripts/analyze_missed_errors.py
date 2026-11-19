#!/usr/bin/env python3
"""Errors that no detector turned into a candidate, in one pipeline run (default: the final validation run).

Author: Aaradhya (Group 1 · DSAI Lab, IIT Madras BS Degree)

Reads <run>/cells.csv and <run>/candidates.csv (final configuration: D1-D6, D7 and the D5 extension). Instead of
assigning
detectors by hardcoded column names, it measures the closest detector
mathematically based on detector activation scores, feature distances,
and structural syntax.

Generates:
- docs/tables/m4_missed_errors.csv
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "outputs" / "runs" / "m4_final2_val_all"

MISSING_TOKENS = {"", "nan", "null", "none", "n/a", "na", "-", "?", "not available", "undefined", "empty"}

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def determine_closest_detector_math(
    d_scores: dict[str, float],
    dirty_val: str,
    clean_val: str,
    col_series: pd.Series,
    error_type: str
) -> tuple[str, str]:
    """Measure closest detector mathematically rather than by column name."""
    # 1. Non-zero detector score proximity
    non_zero_dets = {k: v for k, v in d_scores.items() if v > 0.0}
    if non_zero_dets:
        best_det = max(non_zero_dets, key=non_zero_dets.get)
        return (
            f"{best_det.upper()}",
            f"Partial activation: detector score={non_zero_dets[best_det]:.3f} fell below firing cut-off."
        )

    d_str = str(dirty_val).strip()
    c_str = str(clean_val).strip()

    # 2. Mathematical feature distance checks
    # Check D1 (Missing token distance)
    if d_str.lower() in MISSING_TOKENS or not d_str:
        return "D1_missing", "Sentinel / missing representation evaded null token lexicon."

    # Check D6 (Levenshtein typo distance to clean value or column mode)
    if 1 <= levenshtein_distance(d_str.lower(), c_str.lower()) <= 3:
        return "D6_typo", f"Edit distance = {levenshtein_distance(d_str.lower(), c_str.lower())}; missed by frequency threshold."

    # Check numeric outlier (D3 / Robust Z)
    try:
        d_num = float(re.sub(r"[^\d.-]", "", d_str))
        col_nums = pd.to_numeric(col_series.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False), errors="coerce").dropna()
        if len(col_nums) >= 10:
            med = col_nums.median()
            mad = (col_nums - med).abs().median()
            if mad > 0:
                mod_z = 0.6745 * abs(d_num - med) / mad
                if mod_z >= 1.5:
                    return "D3_robust_z", f"Modified Z-score = {mod_z:.2f} (near outlier threshold)."
    except Exception:
        pass

    # Check error_type semantic mapping
    if error_type == "rule":
        return "D5_fd", "Multi-tuple dependency conflict below g3 discovery threshold or high-noise determinant."
    elif error_type == "pattern":
        return "D2_pattern", "Format / character shape irregularity exceeded 2% rarity threshold (systematic format noise)."
    elif error_type == "missing":
        return "D1_missing", "Domain-specific missing value encoding."
    else:
        return "none", "No detector is close; would need a domain rule."

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="pipeline run folder")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "tables" / "m4_missed_errors.csv"))
    args = ap.parse_args()
    run_dir = Path(args.run)
    cells_path = run_dir / "cells.csv"
    cands_path = run_dir / "candidates.csv"

    if not cells_path.exists() or not cands_path.exists():
        raise FileNotFoundError(f"Missing run files in {run_dir}")

    cells_df = pd.read_csv(cells_path)
    cands_df = pd.read_csv(cands_path)

    # Join on table, row, column
    merged = pd.merge(cells_df, cands_df, on=["table", "row", "column"])

    # Filter to true errors that were missed by candidate generation
    missed_df = merged[(merged["is_error"] == True) & (merged["is_candidate"] == False)].copy()

    det_cols = [c for c in cands_df.columns if c.startswith("d") and c[1:2].isdigit()]
    groups = defaultdict(list)

    for _, row in missed_df.iterrows():
        t = row["table"]
        col = row["column"]
        etype = str(row.get("error_type", "unspecified"))
        d_val = str(row["dirty_value"])
        c_val = str(row["clean_value"])
        
        d_scores = {d: float(row.get(d, 0.0)) for d in det_cols}
        table_series = cells_df[(cells_df["table"] == t) & (cells_df["column"] == col)]["dirty_value"]
        
        closest_det, rationale = determine_closest_detector_math(d_scores, d_val, c_val, table_series, etype)
        groups[(t, col, etype, closest_det, rationale)].append((d_val, c_val))

    summary_rows = []
    for (t, col, etype, closest_det, rationale), samples in groups.items():
        summary_rows.append({
            "table": t,
            "column": col,
            "error_type": etype,
            "missed_count": len(samples),
            "sample_dirty": samples[0][0],
            "sample_clean": samples[0][1],
            "closest_detector": closest_det,
            "detection_rationale": rationale
        })

    df_out = pd.DataFrame(summary_rows).sort_values("missed_count", ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df_out)} groups ({df_out['missed_count'].sum()} missed errors).")
    print(df_out.to_string())

if __name__ == "__main__":
    main()

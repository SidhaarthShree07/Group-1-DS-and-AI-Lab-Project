#!/usr/bin/env python3
"""
src/data/load_core.py

ContextGuard Milestone 2: Core Benchmark Tables Loader & Ground Truth Builder
Author: Jashan (Data Lead) - DSAI Lab, IIT Madras BS
Role: Loads the 5 core Raha benchmark tables (Hospital, Flights, Beers, Rayyan, Movies),
verifies SHA-256 checksums, aligns clean/dirty columns by positional index, excludes
metadata columns, applies Raha ground-truth normalisation, maps error types from TableEG
annotations (with deterministic M1 fallback rules), and exports cell-level truth files to
data/processed/core/<table_name>_cells.csv.
"""

import os
import sys
import re
import csv
import json
import html
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter

# Standard missing tokens recognized across data cleaning literature
MISSING_TOKENS: Set[str] = {
    '', 'nan', 'null', 'none', 'n/a', 'na', 'not available', '?', '-', 'undefined'
}

# Metadata columns excluded from evaluation
EXCLUDE_COLS: Tuple[str, ...] = ("row_id", "tuple_id", "index", "id_col")

# Pinned commit for reproducible clean-checkout fallback
PINNED_RAHA_COMMIT = "7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c"
BASE_RAHA_RAW_URL = f"https://raw.githubusercontent.com/BigDaMa/raha/{PINNED_RAHA_COMMIT}/datasets"

CORE_TABLE_CONFIGS = [
    {
        "name": "hospital",
        "raha_subpath": "hospital",
        "annotation_subpath": "hospital/hospital_annotation.jsonl",
        "expected_sha256_clean": "ea3ee44998455c0b491750c348509de176c758a3bbf58e4530c0a136bb248b4b",
        "expected_sha256_dirty": "dbc5575b915fe8b5e0ac6dc6172f38ba91e611fdb76d09a8f4a81cb7ea9925ac",
        "expected_errors": 509,
        "expected_type_counts": {"rule_violation": 408, "pattern_violation": 101}
    },
    {
        "name": "flights",
        "raha_subpath": "flights",
        "annotation_subpath": "flight/flight_annotation.csv",
        "expected_sha256_clean": "0acfcfd8985b06fdd363965c9e8d9522c43e7589a93d79ae7dc311e1c37fdf3b",
        "expected_sha256_dirty": "1b5c1afa10aa0e7c20fd7e14d05c56772715b2771aa0f5fa67ed1709e1eecd46",
        "expected_errors": 4920,
        "expected_type_counts": {"missing_value": 2312, "pattern_violation": 1459, "rule_violation": 1149}
    },
    {
        "name": "beers",
        "raha_subpath": "beers",
        "annotation_subpath": "beers/beers_annotation.csv",
        "expected_sha256_clean": "373227df59ad197e154dd5149125789e415019535c7223355e9486ee1b3b93de",
        "expected_sha256_dirty": "7110bf4931a9445a1675e544d6c996817c739136239f8a2b02e088c7ec0a1f68",
        "expected_errors": 4362,
        "expected_type_counts": {"pattern_violation": 3077, "missing_value": 1132, "rule_violation": 153}
    },
    {
        "name": "rayyan",
        "raha_subpath": "rayyan",
        "annotation_subpath": "rayyan/rayyan_annotation.jsonl",
        "expected_sha256_clean": "23159f43c0706782388ed8957ad0c74eb7b88bc98f34d65bd49296e186d4673f",
        "expected_sha256_dirty": "7e25e6db262b0c72ca2d9735d5959599cf5a582e1c705459507c7b45d0d1d174",
        "expected_errors": 948,
        "expected_type_counts": {"missing_value": 75, "pattern_violation": 254, "rule_violation": 619}
    },
    {
        "name": "movies",
        "raha_subpath": "movies_1",
        "annotation_subpath": "movie_metadata/movie_annotation.jsonl",
        "expected_sha256_clean": "22a84e70d153d809341c55cf8e8154beebe1836456f617bf8277b953a762bfca",
        "expected_sha256_dirty": "bed752115dd0a5925ea36db443ff6835d10a5a8d2f549299535b14f8ac324316",
        "expected_errors": 7675,
        "expected_type_counts": {"pattern_violation": 7568, "missing_value": 107}
    }
]


def normalize_raha(val: Any) -> str:
    """
    Raha / Baran ground-truth normalisation rule (from scripts/data_inventory.py):
    1. html.unescape
    2. collapse runs of tab, newline, and space into one space
    3. strip tab, newline, and space from both ends
    None is treated as empty string.
    """
    if val is None:
        return ""
    s = html.unescape(str(val))
    s = re.sub(r"[\t\n ]+", " ", s)
    return s.strip("\t\n ")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_or_download_raha_file(repo_root: Path, subpath: str, filename: str, expected_sha: str) -> Path:
    """
    Locates existing Raha benchmark table or automatically downloads fallback from pinned GitHub commit.
    """
    candidates = [
        repo_root / "data" / "raw" / "raha" / subpath / filename,
        repo_root / "raha-master" / "datasets" / subpath / filename,
        repo_root / "datasets" / subpath / filename
    ]
    for c in candidates:
        if c.exists():
            actual_sha = compute_sha256(c)
            if actual_sha == expected_sha:
                return c
            else:
                print(f"[WARN] Hash mismatch for {c}: expected {expected_sha}, got {actual_sha}")

    # Fallback: Download from pinned commit into data/raw/raha
    download_dir = repo_root / "data" / "raw" / "raha" / subpath
    download_dir.mkdir(parents=True, exist_ok=True)
    target_path = download_dir / filename
    url = f"{BASE_RAHA_RAW_URL}/{subpath}/{filename}"
    print(f"[*] Downloading fallback: {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path)

    actual_sha = compute_sha256(target_path)
    if actual_sha != expected_sha:
        raise ValueError(f"Downloaded file {target_path} SHA-256 {actual_sha} != expected {expected_sha}")
    print(f"[SUCCESS] Downloaded and verified SHA-256 for {target_path}")
    return target_path


def find_tableeg_annotation_file(repo_root: Path, annotation_subpath: str) -> Optional[Path]:
    """Locates TableEG annotation file in workspace."""
    candidates = [
        repo_root / "TableEG" / "source" / annotation_subpath,
        repo_root / "data" / "raw" / "tableeg" / "source" / annotation_subpath,
        repo_root / "data" / "raw" / "TableEG" / "source" / annotation_subpath
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_csv(filepath: Path) -> Tuple[List[str], List[List[str]]]:
    """Reads a CSV file returning header and data rows, handling utf-8-sig BOM."""
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = list(csv.reader(f))
    if not reader:
        raise ValueError(f"Empty CSV file: {filepath}")
    return reader[0], reader[1:]


def load_annotations(annotation_path: Path) -> Dict[Tuple[int, str], str]:
    """Loads TableEG annotation files (supporting .jsonl and .csv)."""
    ann_by_cell = {}
    if annotation_path.suffix == ".jsonl":
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                a = json.loads(line)
                rid = int(a.get("row_id", -1))
                col_str = str(a.get("column", "")).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
                etype = a.get("error_type", "")
                ann_by_cell[(rid, col_str)] = etype
    elif annotation_path.suffix == ".csv":
        df_ann = list(csv.DictReader(open(annotation_path, "r", encoding="utf-8-sig", errors="replace")))
        for a in df_ann:
            rid = int(a.get("row_id", -1))
            col_str = str(a.get("column", "")).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
            etype = a.get("error_type", "")
            ann_by_cell[(rid, col_str)] = etype
    return ann_by_cell


def process_table(cfg: Dict[str, Any], repo_root: Path, out_dir: Path) -> Dict[str, Any]:
    """Processes a single core table and writes cell-level ground truth."""
    tname = cfg["name"]
    clean_path = find_or_download_raha_file(repo_root, cfg["raha_subpath"], "clean.csv", cfg["expected_sha256_clean"])
    dirty_path = find_or_download_raha_file(repo_root, cfg["raha_subpath"], "dirty.csv", cfg["expected_sha256_dirty"])

    # Verify SHA-256
    clean_sha = compute_sha256(clean_path)
    dirty_sha = compute_sha256(dirty_path)
    assert clean_sha == cfg["expected_sha256_clean"], f"Clean SHA mismatch on {tname}"
    assert dirty_sha == cfg["expected_sha256_dirty"], f"Dirty SHA mismatch on {tname}"

    c_hdr, c_data = load_csv(clean_path)
    d_hdr, d_data = load_csv(dirty_path)

    n_rows = min(len(c_data), len(d_data))
    n_cols = min(len(c_hdr), len(d_hdr))

    # Exclude metadata columns by name
    ex_indices = set()
    for i in range(n_cols):
        ch = c_hdr[i].strip().lower()
        dh = d_hdr[i].strip().lower()
        if ch in EXCLUDE_COLS or dh in EXCLUDE_COLS:
            ex_indices.add(i)

    # Normalized column names for annotation mapping
    clean_col_norm = {c: c_hdr[c].strip().lower().replace(" ", "").replace("_", "").replace("-", "") for c in range(n_cols)}
    dirty_col_norm = {c: d_hdr[c].strip().lower().replace(" ", "").replace("_", "").replace("-", "") for c in range(n_cols)}

    # Load annotations
    ann_path = find_tableeg_annotation_file(repo_root, cfg["annotation_subpath"])
    ann_by_cell = load_annotations(ann_path) if ann_path and ann_path.exists() else {}

    cell_records = []
    error_count = 0
    type_counts = Counter()
    unannotated_count = 0

    for r in range(n_rows):
        cr = c_data[r]
        dr = d_data[r]
        for c in range(n_cols):
            if c in ex_indices:
                continue

            c_val = cr[c] if c < len(cr) else ""
            d_val = dr[c] if c < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            col_name = c_hdr[c]
            is_error = (norm_c != norm_d)
            error_type = ""

            if is_error:
                error_count += 1
                c_norm = clean_col_norm[c]
                d_norm = dirty_col_norm[c]

                # Map annotation
                if (r, c_norm) in ann_by_cell:
                    error_type = ann_by_cell[(r, c_norm)]
                elif (r, d_norm) in ann_by_cell:
                    error_type = ann_by_cell[(r, d_norm)]
                else:
                    unannotated_count += 1
                    # Deterministic M1 fallback rule:
                    # Missing tokens -> missing_value (e.g. Beers 1,005 'N/A' cells, Flights missing departure/arrival times)
                    norm_d_lower = norm_d.lower()
                    norm_c_lower = norm_c.lower()
                    if norm_d_lower in MISSING_TOKENS or norm_c_lower in MISSING_TOKENS:
                        error_type = "missing_value"
                    else:
                        is_numeric = False
                        try:
                            float(norm_d)
                            float(norm_c)
                            is_numeric = True
                        except ValueError:
                            pass
                        if is_numeric:
                            error_type = "outliers"
                        else:
                            error_type = "pattern_violation"

                type_counts[error_type] += 1

            cell_records.append({
                "row": r,
                "column": col_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_error,
                "error_type": error_type
            })

    # Write cell-level output CSV
    out_csv = out_dir / f"{tname}_cells.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "column", "dirty_value", "clean_value", "is_error", "error_type"])
        writer.writeheader()
        writer.writerows(cell_records)

    # Verify counts
    assert error_count == cfg["expected_errors"], (
        f"Error count mismatch for {tname}: got {error_count}, expected {cfg['expected_errors']}"
    )
    if cfg.get("expected_type_counts"):
        for etype, exp_cnt in cfg["expected_type_counts"].items():
            assert type_counts[etype] == exp_cnt, (
                f"Error type {etype} count mismatch on {tname}: got {type_counts[etype]}, expected {exp_cnt}"
            )

    return {
        "name": tname,
        "rows": n_rows,
        "data_cols": n_cols - len(ex_indices),
        "total_cells": len(cell_records),
        "error_cells": error_count,
        "unannotated_recovered": unannotated_count,
        "type_counts": dict(type_counts),
        "clean_sha256": clean_sha,
        "dirty_sha256": dirty_sha,
        "output_path": out_csv
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = repo_root / "data" / "processed" / "core"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   ContextGuard Milestone 2: Load Core Benchmark Tables & Build Ground Truth   ")
    print("=" * 80)
    print(f"Repository Root : {repo_root.name}")
    print(f"Output Directory: data/processed/core/\n")

    grand_total_errors = 0
    results = []
    for cfg in CORE_TABLE_CONFIGS:
        res = process_table(cfg, repo_root, processed_dir)
        results.append(res)
        grand_total_errors += res["error_cells"]

        print(f"[*] Processed {res['name'].capitalize()}:")
        print(f"    - Dimensions: {res['rows']:,} rows x {res['data_cols']} data columns ({res['total_cells']:,} total cells)")
        print(f"    - Clean SHA-256: {res['clean_sha256'][:16]}...")
        print(f"    - Dirty SHA-256: {res['dirty_sha256'][:16]}...")
        print(f"    - Error Cells: {res['error_cells']:,} ({res['error_cells']/res['total_cells']*100:.2f}%)")
        print(f"    - Type Mix: {res['type_counts']}")
        if res["unannotated_recovered"] > 0:
            print(f"    - Unannotated Recovered: {res['unannotated_recovered']:,} cells")
        print(f"    - Output File: data/processed/core/{res['name']}_cells.csv\n")

    # Assert grand total
    assert grand_total_errors == 18414, f"Grand total error mismatch: got {grand_total_errors}, expected 18,414"

    print("=" * 80)
    print("                      ALL CORE TABLES VERIFIED SUCCESSFULLY                    ")
    print("=" * 80)
    print("Summary of M1 Exact Matches:")
    print("  - Hospital : 509 errors")
    print("  - Flights  : 4,920 errors")
    print("  - Beers    : 4,362 errors")
    print("  - Rayyan   : 948 errors (75 missing_value, 254 pattern_violation, 619 rule_violation)")
    print("  - Movies   : 7,675 errors")
    print(f"  - Grand Total: {grand_total_errors:,} errors (EXACT CONSERVATION)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

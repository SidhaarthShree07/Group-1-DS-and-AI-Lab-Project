# Author: Jashan Tiwari
"""Standardize unseen-table evaluation data and downstream Credit data for Milestone 5.

Loads the 5 unseen evaluation tables (Restaurant, Marketing, Soccer, Billionaire, German slice)
and the downstream test table (Credit), standardizes their clean/dirty tables and cell files
into data/unseen/<table>/{clean,dirty,cells}.csv (and test aliases), and verifies that all
ground-truth error counts match the Milestone 1 audit exactly.

Strict M5 Rule:
No thresholds are tuned, no errors are inspected to design rules, and no settings are
selected on the unseen tables. Only standardized evaluation files and audit records are generated.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

csv.field_size_limit(10**9)

META_COLUMNS = {"row_id", "tuple_id", "index", "id", "id_col"}
MISSING_TOKENS = {"", "nan", "null", "none", "n/a", "na", "?", "-", "not available", "empty"}
NUM_PAT = re.compile(r"^-?\d+(\.\d+)?$")


def normalize_raha(val: Any) -> str:
    """Normalize string using Raha's standard ground-truth rule."""
    if val is None:
        return ""
    s = str(val)
    s = html.unescape(s)
    s = re.sub(r"[\t\n\r ]+", " ", s)
    return s.strip("\t\n\r ")


def normalize_lenient(val: Any) -> str:
    """Lenient normalization for numbers and missing values."""
    s = normalize_raha(val)
    if s.lower() in MISSING_TOKENS:
        return "<MISSING>"
    try:
        f = float(s)
        return f"{f:.4f}"
    except ValueError:
        return s


def is_format_artifact(a: str, b: str) -> bool:
    """Same number written differently (e.g. 45 vs 45.0) or zero prefix added/dropped."""
    if NUM_PAT.match(a) and NUM_PAT.match(b):
        try:
            return float(a) == float(b) or a.lstrip("0") == b.lstrip("0")
        except ValueError:
            return False
    return False


def load_csv_data(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read CSV file handling UTF-8-SIG BOM."""
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = list(csv.reader(f))
    if not reader:
        raise ValueError(f"Empty CSV file: {path}")
    return reader[0], reader[1:]


def process_restaurant(repo_root: Path) -> Dict[str, Any]:
    """Process Restaurant from TableEG/source/Restaurant."""
    clean_p = repo_root / "TableEG/source/Restaurant/clean.csv"
    dirty_p = repo_root / "TableEG/source/Restaurant/dirty.csv"
    ann_p = repo_root / "TableEG/source/Restaurant/Restaurant_annotation.jsonl"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    # Exclude row_id
    data_cols = [c for c in c_hdr if c.strip().lower() not in META_COLUMNS]
    col_idx = [c_hdr.index(c) for c in data_cols]

    # Load annotations
    ann_map = {}
    if ann_p.exists():
        with open(ann_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    r_id = int(item["row_id"])
                    c_name = str(item["column"]).strip().lower()
                    ann_map[(r_id, c_name)] = item.get("error_type", "pattern_violation")

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for c_name, c_i in zip(data_cols, col_idx):
            c_val = cr[c_i] if c_i < len(cr) else ""
            d_val = dr[c_i] if c_i < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                lookup_key = (r_idx, c_name.lower())
                etype = ann_map.get(lookup_key, "pattern_violation")
                if "missing" in etype:
                    type_counts["missing"] += 1
                    etype = "missing_value"
                elif "rule" in etype:
                    type_counts["rule"] += 1
                    etype = "rule_violation"
                elif "outlier" in etype:
                    type_counts["outlier"] += 1
                    etype = "outliers"
                else:
                    type_counts["pattern"] += 1
                    etype = "pattern_violation"

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    clean_data = [[r[i] for i in col_idx] for r in c_rows]
    dirty_data = [[r[i] for i in col_idx] for r in d_rows]

    return {
        "table": "restaurant",
        "role": "unseen_eval",
        "header": data_cols,
        "clean_rows": clean_data,
        "dirty_rows": dirty_data,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 633,
    }


def process_marketing(repo_root: Path) -> Dict[str, Any]:
    """Process Marketing from TableEG."""
    clean_p = repo_root / "TableEG/test_dataset/exp_1/TableEG_output/Marketing/clean_Marketing.csv"
    dirty_p = repo_root / "TableEG/source/Marketing/dirty.csv"
    ann_p = repo_root / "TableEG/source/Marketing/Marketing_annotation.csv"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    data_cols = [c for c in c_hdr if c.strip().lower() not in META_COLUMNS]
    col_idx = [c_hdr.index(c) for c in data_cols]

    # Load annotations
    ann_map = {}
    if ann_p.exists():
        with open(ann_p, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_id = int(row["row_id"])
                c_name = row["column"].strip().lower()
                ann_map[(r_id, c_name)] = row.get("error_type", "pattern_violation")

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for c_name, c_i in zip(data_cols, col_idx):
            c_val = cr[c_i] if c_i < len(cr) else ""
            d_val = dr[c_i] if c_i < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                lookup_key = (r_idx, c_name.lower())
                raw_etype = ann_map.get(lookup_key, "")
                if not raw_etype:
                    if norm_d.lower() in MISSING_TOKENS:
                        raw_etype = "missing_value"
                    else:
                        raw_etype = "pattern_violation"

                if "missing" in raw_etype:
                    type_counts["missing"] += 1
                    etype = "missing_value"
                elif "rule" in raw_etype:
                    type_counts["rule"] += 1
                    etype = "rule_violation"
                elif "outlier" in raw_etype:
                    type_counts["outlier"] += 1
                    etype = "outliers"
                else:
                    type_counts["pattern"] += 1
                    etype = "pattern_violation"

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    clean_data = [[r[i] for i in col_idx] for r in c_rows]
    dirty_data = [[r[i] for i in col_idx] for r in d_rows]

    return {
        "table": "marketing",
        "role": "unseen_eval",
        "header": data_cols,
        "clean_rows": clean_data,
        "dirty_rows": dirty_data,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 26815,
    }


def process_soccer(repo_root: Path) -> Dict[str, Any]:
    """Process Soccer from TableEG/test_dataset/exp_1/BART_output/soccer."""
    clean_p = repo_root / "TableEG/test_dataset/exp_1/BART_output/soccer/clean_soccer.csv"
    dirty_p = repo_root / "TableEG/test_dataset/exp_1/BART_output/soccer/soccer_10/dirty_data.csv"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    # Standardize column names (Soccer clean has 'birthyear (integer)', dirty has 'birthyear(long)')
    # Align by position excluding row_id (col 0)
    data_cols = [c for c in c_hdr if c.strip().lower() not in META_COLUMNS]
    col_idx = [c_hdr.index(c) for c in data_cols]

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for c_name, c_i in zip(data_cols, col_idx):
            c_val = cr[c_i] if c_i < len(cr) else ""
            d_val = dr[c_i] if c_i < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                # Dynamic typing as established in M1 (birthyear numeric differences -> outlier, text -> pattern)
                if NUM_PAT.match(norm_c) and NUM_PAT.match(norm_d):
                    type_counts["outlier"] += 1
                    etype = "outliers"
                else:
                    type_counts["pattern"] += 1
                    etype = "pattern_violation"

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    clean_data = [[r[i] for i in col_idx] for r in c_rows]
    dirty_data = [[r[i] for i in col_idx] for r in d_rows]

    return {
        "table": "soccer",
        "role": "unseen_eval",
        "header": data_cols,
        "clean_rows": clean_data,
        "dirty_rows": dirty_data,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 796,
    }


def process_billionaire(repo_root: Path) -> Dict[str, Any]:
    """Process Billionaire from ZeroED."""
    clean_p = repo_root / "data/raw/ZeroED/data/billionaire_clean.csv"
    dirty_p = repo_root / "data/raw/ZeroED/data/billionaire_error-01.csv"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    data_cols = list(c_hdr)
    cols = range(len(data_cols))
    clean_vals = [set(normalize_raha(r[j]) for r in c_rows) for j in cols]

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for j, c_name in enumerate(data_cols):
            c_val = cr[j] if j < len(cr) else ""
            d_val = dr[j] if j < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                if norm_d.lower() in MISSING_TOKENS and norm_c.lower() not in MISSING_TOKENS:
                    etype = "missing_value"
                    type_counts["missing"] += 1
                elif is_format_artifact(norm_c, norm_d):
                    etype = "format_artifact"
                elif norm_d in clean_vals[j]:
                    etype = "rule_violation"
                    type_counts["rule"] += 1
                else:
                    etype = "pattern_violation"
                    type_counts["pattern"] += 1

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    return {
        "table": "billionaire",
        "role": "unseen_eval",
        "header": data_cols,
        "clean_rows": c_rows,
        "dirty_rows": d_rows,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 5195,
    }


def process_german(repo_root: Path) -> Dict[str, Any]:
    """Process German 5,000-row slice from Clean Me If You Can."""
    clean_p = repo_root / "data/processed/germany/germany_5000_clean.csv"
    dirty_p = repo_root / "data/processed/germany/germany_5000_dirty.csv"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    # Germany has 'id' as column 0; drop 'id' for data columns
    data_cols = [c for c in c_hdr if c.strip().lower() not in META_COLUMNS]
    col_idx = [c_hdr.index(c) for c in data_cols]

    clean_vals = [set(normalize_raha(r[i]) for r in c_rows) for i in col_idx]

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for j, (c_name, c_i) in enumerate(zip(data_cols, col_idx)):
            c_val = cr[c_i] if c_i < len(cr) else ""
            d_val = dr[c_i] if c_i < len(dr) else ""

            norm_c = normalize_raha(c_val)
            norm_d = normalize_raha(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                if norm_d.lower() in MISSING_TOKENS and norm_c.lower() not in MISSING_TOKENS:
                    etype = "missing_value"
                    type_counts["missing"] += 1
                elif is_format_artifact(norm_c, norm_d):
                    etype = "format_artifact"
                elif norm_d in clean_vals[j]:
                    etype = "rule_violation"
                    type_counts["rule"] += 1
                else:
                    etype = "pattern_violation"
                    type_counts["pattern"] += 1

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    clean_data = [[r[i] for i in col_idx] for r in c_rows]
    dirty_data = [[r[i] for i in col_idx] for r in d_rows]

    return {
        "table": "german",
        "role": "unseen_eval",
        "header": data_cols,
        "clean_rows": clean_data,
        "dirty_rows": dirty_data,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 18678,
    }


def process_credit(repo_root: Path) -> Dict[str, Any]:
    """Process Credit downstream evaluation table."""
    clean_p = repo_root / "TableEG/source/Credit/clean.csv"
    dirty_p = repo_root / "TableEG/source/Credit/dirty.csv"
    ann_p = repo_root / "TableEG/source/Credit/Credit_annotation.csv"

    c_hdr, c_rows = load_csv_data(clean_p)
    d_hdr, d_rows = load_csv_data(dirty_p)

    # Drop row_id
    data_cols = [c for c in c_hdr if c.strip().lower() not in META_COLUMNS]
    col_idx = [c_hdr.index(c) for c in data_cols]

    # Load annotations
    ann_map = {}
    if ann_p.exists():
        with open(ann_p, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_id = int(row["row_id"])
                c_name = row["column"].strip().lower()
                ann_map[(r_id, c_name)] = row.get("error_type", "missing_value")

    cells = []
    error_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}

    for r_idx in range(len(c_rows)):
        cr = c_rows[r_idx]
        dr = d_rows[r_idx]
        for c_name, c_i in zip(data_cols, col_idx):
            c_val = cr[c_i] if c_i < len(cr) else ""
            d_val = dr[c_i] if c_i < len(dr) else ""

            # Lenient normalization isolates float re-saving format artifacts
            norm_c = normalize_lenient(c_val)
            norm_d = normalize_lenient(d_val)

            is_err = norm_c != norm_d
            etype = ""
            if is_err:
                error_count += 1
                lookup_key = (r_idx, c_name.lower())
                raw_type = ann_map.get(lookup_key, "")
                if not raw_type:
                    if norm_d == "<MISSING>" or norm_d.lower() in MISSING_TOKENS:
                        raw_type = "missing_value"
                    elif NUM_PAT.match(normalize_raha(c_val)) and NUM_PAT.match(normalize_raha(d_val)):
                        raw_type = "outliers"
                    else:
                        raw_type = "pattern_violation"

                if "missing" in raw_type:
                    type_counts["missing"] += 1
                    etype = "missing_value"
                elif "outlier" in raw_type:
                    type_counts["outlier"] += 1
                    etype = "outliers"
                elif "rule" in raw_type:
                    type_counts["rule"] += 1
                    etype = "rule_violation"
                else:
                    type_counts["pattern"] += 1
                    etype = "pattern_violation"
            elif normalize_raha(c_val) != normalize_raha(d_val):
                # Float format discrepancy (not a semantic error)
                etype = "format_artifact"

            cells.append({
                "row": r_idx,
                "column": c_name,
                "dirty_value": d_val,
                "clean_value": c_val,
                "is_error": is_err,
                "error_type": etype,
            })

    clean_data = [[r[i] for i in col_idx] for r in c_rows]
    dirty_data = [[r[i] for i in col_idx] for r in d_rows]

    return {
        "table": "credit",
        "role": "downstream_test",
        "header": data_cols,
        "clean_rows": clean_data,
        "dirty_rows": dirty_data,
        "cells": cells,
        "rows": len(c_rows),
        "columns": len(data_cols),
        "total_cells": len(cells),
        "error_cells": error_count,
        "type_counts": type_counts,
        "m1_errors": 38936,
    }


def write_table_files(tbl_info: Dict[str, Any], target_dir: Path) -> None:
    """Write standardized clean.csv, dirty.csv, cells.csv and split aliases."""
    t_dir = target_dir / tbl_info["table"]
    t_dir.mkdir(parents=True, exist_ok=True)

    header = tbl_info["header"]

    # Write clean.csv & test_clean.csv
    for fname in ("clean.csv", "test_clean.csv"):
        with open(t_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(tbl_info["clean_rows"])

    # Write dirty.csv & test.csv
    for fname in ("dirty.csv", "test.csv"):
        with open(t_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(tbl_info["dirty_rows"])

    # Write cells.csv & test_cells.csv
    cell_fields = ["row", "column", "dirty_value", "clean_value", "is_error", "error_type"]
    for fname in ("cells.csv", "test_cells.csv"):
        with open(t_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cell_fields)
            w.writeheader()
            w.writerows(tbl_info["cells"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--out-dir", default="data/unseen", help="Output directory for standardized files")
    parser.add_argument("--summary-table", default="docs/tables/m5_unseen_data_summary.csv", help="Summary CSV destination")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = repo_root / args.out_dir
    summary_path = repo_root / args.summary_table
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Milestone 5 Unseen & Credit Data Preparation...")
    processors = [
        ("Restaurant", process_restaurant),
        ("Marketing", process_marketing),
        ("Soccer", process_soccer),
        ("Billionaire", process_billionaire),
        ("German", process_german),
        ("Credit", process_credit),
    ]

    summary_rows = []
    all_matched = True

    for name, proc in processors:
        print(f"[*] Processing {name}...")
        info = proc(repo_root)
        print(f"    -> Rows: {info['rows']:,}, Cols: {info['columns']}, Total Cells: {info['total_cells']:,}")
        print(f"    -> Errors: {info['error_cells']:,} (M1 audit: {info['m1_errors']:,})")
        print(f"    -> Error types: {info['type_counts']}")

        match = (info["error_cells"] == info["m1_errors"])
        if not match:
            all_matched = False
            print(f"    [!] MISMATCH: {info['error_cells']} != {info['m1_errors']}")
        else:
            print("    [+] M1 audit matched perfectly.")

        write_table_files(info, out_dir)
        print(f"    -> Standardized files written to {out_dir / info['table']}")

        err_rate = round(100.0 * info["error_cells"] / info["total_cells"], 4)
        tc = info["type_counts"]
        summary_rows.append({
            "table": info["table"],
            "role": info["role"],
            "rows": info["rows"],
            "columns": info["columns"],
            "total_cells": info["total_cells"],
            "error_cells": info["error_cells"],
            "error_rate_pct": err_rate,
            "missing_errors": tc["missing"],
            "pattern_errors": tc["pattern"],
            "rule_errors": tc["rule"],
            "outlier_errors": tc["outlier"],
            "m1_audit_errors": info["m1_errors"],
            "audit_match": match,
        })

    fields = [
        "table", "role", "rows", "columns", "total_cells", "error_cells",
        "error_rate_pct", "missing_errors", "pattern_errors", "rule_errors",
        "outlier_errors", "m1_audit_errors", "audit_match"
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[+] Successfully generated {summary_path}")
    print(f"[+] All 6 datasets match M1 audit: {all_matched}")
    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())

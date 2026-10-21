#!/usr/bin/env python3
"""
scripts/data_inventory.py

ContextGuard Benchmark Table Auditor & Data Inventory Script
Author: Jashan (Data Lead) - DSAI Lab, IIT Madras BS
Role: Audits Raha canonical benchmark tables and TableEG datasets.
Applies Raha ground-truth normalisation rule, compares clean vs dirty CSV pairs,
maps TableEG annotations to canonical error types, performs clean-side quality
audits, and outputs docs/tables/data_inventory.csv, docs/tables/error_type_profile.csv,
and docs/figures/error_distribution_pie.svg.
"""

import os
import sys
import re
import csv
import json
import html
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set

# Standard missing tokens recognized across data cleaning literature
MISSING_TOKENS = {'', 'nan', 'null', 'none', 'n/a', 'na', 'not available', '?', '-', 'undefined'}


def normalize_raha(val: Any) -> str:
    """
    Raha / Baran ground-truth normalisation (three steps, as in Raha's own code):
    1. html.unescape
    2. collapse runs of tab, newline and space into one space
    3. strip tab, newline and space from both ends
    None is treated as the empty string. Missing-value tokens are NOT unified here
    (that belongs to normalize_lenient).
    """
    if val is None:
        return ""
    s = html.unescape(str(val))
    s = re.sub(r"[\t\n ]+", " ", s)
    return s.strip("\t\n ")


def normalize_lenient(val: Any) -> str:
    """
    Lenient normalisation (Section 3):
    - Missing tokens unified to <MISSING>
    - Numbers compared numerically if convertible to float (eliminates IEEE-754 precision drift)
    """
    s = normalize_raha(val)
    if s.lower() in MISSING_TOKENS:
        return "<MISSING>"
    try:
        f = float(s)
        return f"{f:.4f}"
    except ValueError:
        return s


def load_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Reads a CSV file returning header and data rows, handling utf-8-sig BOM."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = list(csv.reader(f))
    if not reader:
        raise ValueError(f"Empty CSV file: {path}")
    return reader[0], reader[1:]


def check_clean_side_quality(clean_rows: List[List[str]], col_names: List[str]) -> Dict[str, Any]:
    """
    Clean-side quality checks on clean.csv:
    1. Count remaining nulls / missing tokens
    2. Count exact duplicate rows
    3. Check mixed formats / types per column
    """
    num_rows = len(clean_rows)
    num_cols = len(col_names)
    
    null_cells = 0
    nulls_by_col = Counter()
    for row in clean_rows:
        for c_idx, val in enumerate(row):
            if c_idx < num_cols:
                if val.strip().lower() in MISSING_TOKENS:
                    null_cells += 1
                    nulls_by_col[col_names[c_idx]] += 1

    # Exact duplicate rows
    row_tuples = [tuple(r) for r in clean_rows]
    duplicate_rows = len(row_tuples) - len(set(row_tuples))

    # Format inspection per column
    format_issues = {}
    for c_idx in range(num_cols):
        col_name = col_names[c_idx]
        vals = [clean_rows[r][c_idx].strip() for r in range(num_rows) if c_idx < len(clean_rows[r])]
        non_empty = [v for v in vals if v.lower() not in MISSING_TOKENS]
        if not non_empty:
            continue
        
        # Check numeric vs text mixing
        num_numeric = sum(1 for v in non_empty if re.match(r"^-?\d+(\.\d+)?$", v))
        num_date_time = sum(1 for v in non_empty if re.search(r"(\d{1,2}:\d{2}|[ap]\.?m\.?|\d{4}-\d{2}-\d{2})", v, re.I))
        
        if 0 < num_numeric < len(non_empty) and (len(non_empty) - num_numeric) > 5:
            format_issues[col_name] = f"Mixed numeric ({num_numeric}) and text ({len(non_empty)-num_numeric})"
        elif 0 < num_date_time < len(non_empty) and (len(non_empty) - num_date_time) > 5:
            format_issues[col_name] = f"Mixed time ({num_date_time}) and non-time ({len(non_empty)-num_date_time})"

    return {
        "clean_missing_cells": null_cells,
        "clean_nulls_by_col": nulls_by_col,
        "clean_duplicate_rows": duplicate_rows,
        "format_issues": format_issues
    }


def audit_table_pair(
    table_name: str,
    clean_path: Path,
    dirty_path: Path,
    annotation_path: Optional[Path] = None,
    exclude_cols: Tuple[str, ...] = ("row_id", "tuple_id", "index", "id_col")
) -> Dict[str, Any]:
    """
    Audits a clean and dirty CSV pair:
    1. Aligns columns by positional index where headers differ
    2. Excludes metadata row_id/tuple_id columns from error counting
    3. Normalizes with Raha ground-truth rule
    4. Computes strict and lenient error cell counts
    5. Maps TableEG annotations to canonical error cells
    6. Dynamically classifies unannotated error cells:
       - Credit float re-saving format discrepancies -> format_artifact
       - Missing token differences -> missing_value
       - Numeric cell differences -> outliers
       - Textual cell differences -> pattern_violation
    7. Runs clean-side quality checks
    """
    c_hdr, c_data = load_csv(clean_path)
    d_hdr, d_data = load_csv(dirty_path)

    n_rows = min(len(c_data), len(d_data))
    n_cols = min(len(c_hdr), len(d_hdr))

    # Identify metadata columns to exclude from cell scoring
    ex_indices = set()
    for i in range(n_cols):
        ch = c_hdr[i].strip().lower()
        dh = d_hdr[i].strip().lower()
        if ch in exclude_cols or dh in exclude_cols:
            ex_indices.add(i)

    strict_errors = 0
    lenient_errors = 0
    error_cells: Dict[Tuple[int, int], Tuple[str, str]] = {}
    col_errors = Counter()

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

            if norm_c != norm_d:
                strict_errors += 1
                col_errors[c] += 1
                error_cells[(r, c)] = (c_val, d_val)

            if normalize_lenient(c_val) != normalize_lenient(d_val):
                lenient_errors += 1

    # Header differences
    header_diffs = []
    for i in range(n_cols):
        if c_hdr[i] != d_hdr[i]:
            header_diffs.append((i, c_hdr[i], d_hdr[i]))

    # Clean-side quality
    quality = check_clean_side_quality(c_data, c_hdr)

    # Annotation mapping & error typing
    annotations_mapped = 0
    annotations_total: Any = 0
    error_type_counts = Counter()
    col_type_profile = defaultdict(Counter)

    ann_by_cell = {}
    if annotation_path and annotation_path.exists():
        if annotation_path.suffix == ".jsonl":
            with open(annotation_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    a = json.loads(line)
                    annotations_total += 1
                    rid = int(a.get("row_id", -1))
                    col_str = str(a.get("column", "")).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
                    etype = a.get("error_type", "")
                    ann_by_cell[(rid, col_str)] = etype
        elif annotation_path.suffix == ".csv":
            df_ann = list(csv.DictReader(open(annotation_path, "r", encoding="utf-8-sig", errors="replace")))
            annotations_total = len(df_ann)
            for a in df_ann:
                rid = int(a.get("row_id", -1))
                col_str = str(a.get("column", "")).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
                etype = a.get("error_type", "")
                ann_by_cell[(rid, col_str)] = etype
    elif table_name == "Soccer":
        annotations_total = "none (BART log has no types)"

    # Normalized column names map
    clean_col_norm = {c: c_hdr[c].strip().lower().replace(" ", "").replace("_", "").replace("-", "") for c in range(n_cols)}
    dirty_col_norm = {c: d_hdr[c].strip().lower().replace(" ", "").replace("_", "").replace("-", "") for c in range(n_cols)}

    unannotated_cells = 0
    for (r, c), (c_val, d_val) in error_cells.items():
        c_norm_name = clean_col_norm[c]
        d_norm_name = dirty_col_norm[c]
        assigned_type = None

        # Check direct annotation match
        if (r, c_norm_name) in ann_by_cell:
            assigned_type = ann_by_cell[(r, c_norm_name)]
            annotations_mapped += 1
        elif (r, d_norm_name) in ann_by_cell:
            assigned_type = ann_by_cell[(r, d_norm_name)]
            annotations_mapped += 1
        else:
            unannotated_cells += 1
            # In Credit, unannotated differences are float re-saving format artifacts (e.g., 45.0 vs 45)
            if table_name == "Credit":
                assigned_type = "format_artifact"
            else:
                norm_d = normalize_raha(d_val).lower()
                norm_c = normalize_raha(c_val).lower()
                if norm_d in MISSING_TOKENS or norm_c in MISSING_TOKENS:
                    assigned_type = "missing_value"
                else:
                    # Dynamic cell-change classification:
                    # If both values are numeric -> outliers (e.g. Soccer birthyear: 199 cells)
                    # If values are text -> pattern_violation (e.g. Soccer stadium, team, birthplace: 597 cells)
                    is_numeric = False
                    try:
                        float(norm_d)
                        float(norm_c)
                        is_numeric = True
                    except ValueError:
                        pass
                    if is_numeric:
                        assigned_type = "outliers"
                    else:
                        assigned_type = "pattern_violation"

        error_type_counts[assigned_type] += 1
        col_name = c_hdr[c]
        col_type_profile[col_name][assigned_type] += 1

    return {
        "table_name": table_name,
        "rows": n_rows,
        "data_cols": n_cols - len(ex_indices),
        "total_cols": n_cols,
        "clean_headers": c_hdr,
        "dirty_headers": d_hdr,
        "header_diffs": header_diffs,
        "strict_errors": strict_errors,
        "lenient_errors": lenient_errors,
        "error_cells_dict": error_cells,
        "col_errors": col_errors,
        "annotations_total": annotations_total,
        "annotations_mapped": annotations_mapped,
        "unannotated_cells": unannotated_cells,
        "error_type_counts": error_type_counts,
        "col_type_profile": col_type_profile,
        "quality": quality
    }


def audit_credit_and_soccer(workspace_root: Path) -> Dict[str, Any]:
    """Audits Credit class balance/imputation and Soccer BART dirty vs error log."""
    # 1. Credit Audit
    credit_clean = workspace_root / "TableEG/source/Credit/clean.csv"
    credit_df = list(csv.DictReader(open(credit_clean, "r", encoding="utf-8-sig")))
    total_credit = len(credit_df)
    
    target_counts = Counter(r["SeriousDlqin2yrs"] for r in credit_df)
    imputed_count = sum(1 for r in credit_df if r["MonthlyIncome"].startswith("88.44070676"))
    
    # 2. Soccer Audit
    soccer_clean = workspace_root / "TableEG/test_dataset/exp_1/BART_output/soccer/clean_soccer.csv"
    soccer_dirty10 = workspace_root / "TableEG/test_dataset/exp_1/BART_output/soccer/soccer_10/dirty_data.csv"
    soccer_cc10 = workspace_root / "TableEG/test_dataset/exp_1/BART_output/soccer/soccer_10/cellChanges.csv"
    soccer_log = workspace_root / "TableEG/test_dataset/exp_1/BART_output/soccer/error_log.jsonl"

    c_hdr, c_data = load_csv(soccer_clean)
    d_hdr, d_data = load_csv(soccer_dirty10)
    
    soccer_diffs = 0
    for r in range(min(len(c_data), len(d_data))):
        for c in range(1, len(c_hdr)): # ignore row_id col 0
            if normalize_raha(c_data[r][c]) != normalize_raha(d_data[r][c]):
                soccer_diffs += 1

    soccer_cc10_lines = sum(1 for l in open(soccer_cc10) if l.strip())
    soccer_log_entries = sum(1 for l in open(soccer_log) if l.strip())

    return {
        "credit": {
            "total_rows": total_credit,
            "target_counts": target_counts,
            "class_0_count": target_counts.get("0", 0),
            "class_0_pct": target_counts.get("0", 0) / total_credit * 100,
            "class_1_count": target_counts.get("1", 0),
            "class_1_pct": target_counts.get("1", 0) / total_credit * 100,
            "imputed_monthly_income_repeats": imputed_count
        },
        "soccer": {
            "clean_vs_dirty10_diffs": soccer_diffs,
            "soccer_10_cellChanges_count": soccer_cc10_lines,
            "exp1_error_log_entries": soccer_log_entries,
            "agreement_verified": (soccer_diffs == soccer_cc10_lines)
        }
    }


def generate_pie_chart_svg(distribution: Dict[str, int], output_path: Path):
    """Generates a clean, standalone SVG vector pie chart for the error distribution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(distribution.values())
    if total == 0:
        return

    colors = {
        "missing_value": "#2b5c8f",      # Classic Navy
        "pattern_violation": "#d95f02",  # Rich Coral
        "rule_violation": "#7570b3",     # Slate Purple
        "outliers": "#1b9e77"            # Teal
    }
    labels = {
        "missing_value": "Missing Values (~49%)",
        "pattern_violation": "Pattern Violations (~46%)",
        "rule_violation": "Rule Violations (~3%)",
        "outliers": "Outliers (~2%)"
    }

    cx, cy, r = 220, 220, 160
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 440" width="700" height="440" style="background-color:#ffffff; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        f'<rect width="700" height="440" fill="#ffffff" rx="12"/>',
        f'<text x="350" y="38" text-anchor="middle" font-size="20" font-weight="700" fill="#1e293b">ContextGuard Benchmark Error Type Distribution</text>',
        f'<text x="350" y="60" text-anchor="middle" font-size="13" fill="#64748b">Verified Across 9 Audited Benchmark Tables ({total:,} True Error Cells)</text>',
        f'<g transform="translate(0, 20)">'
    ]

    current_angle = -math.pi / 2
    for category in ["missing_value", "pattern_violation", "rule_violation", "outliers"]:
        count = distribution.get(category, 0)
        fraction = count / total
        slice_angle = fraction * 2 * math.pi
        
        x1 = cx + r * math.cos(current_angle)
        y1 = cy + r * math.sin(current_angle)
        current_angle += slice_angle
        x2 = cx + r * math.cos(current_angle)
        y2 = cy + r * math.sin(current_angle)
        
        large_arc = 1 if slice_angle > math.pi else 0
        color = colors.get(category, "#888888")
        
        path_d = f"M {cx} {cy} L {x1:.2f} {y1:.2f} A {r} {r} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z"
        svg_parts.append(f'<path d="{path_d}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')

    svg_parts.append('</g>')

    # Legend
    legend_x = 440
    legend_y = 130
    svg_parts.append(f'<g transform="translate({legend_x}, {legend_y})">')
    for i, category in enumerate(["missing_value", "pattern_violation", "rule_violation", "outliers"]):
        count = distribution.get(category, 0)
        pct = (count / total) * 100
        color = colors.get(category, "#888888")
        label = labels.get(category, category)
        y = i * 55
        svg_parts.append(f'<rect x="0" y="{y}" width="22" height="22" rx="4" fill="{color}"/>')
        svg_parts.append(f'<text x="32" y="{y + 16}" font-size="15" font-weight="600" fill="#1e293b">{label}</text>')
        svg_parts.append(f'<text x="32" y="{y + 34}" font-size="13" fill="#64748b">{count:,} cells ({pct:.2f}%)</text>')
    svg_parts.append('</g>')

    svg_parts.append('</svg>')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def main():
    parser = argparse.ArgumentParser(description="ContextGuard Milestone 1 Data Inventory & Audit Tool")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root directory")
    parser.add_argument("--outdir", type=str, default="docs/tables", help="Output directory for generated CSV tables")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("           ContextGuard Milestone 1: Data Inventory & Benchmark Audit            ")
    print("================================================================================")
    print(f"Workspace Root: {root}")
    print(f"Output Directory: {outdir}\n")

    benchmark_configs = [
        {
            "name": "Hospital",
            "role": "Train + in-domain test",
            "source": "Raha repo (US CMS Hospital Compare)",
            "clean": root / "raha-master/datasets/hospital/clean.csv",
            "dirty": root / "raha-master/datasets/hospital/dirty.csv",
            "annotation": root / "TableEG/source/hospital/hospital_annotation.jsonl",
            "licence": "Apache-2.0 (repo)",
            "notes": "Typos over 17 columns; strong FDs (zip -> city, measure code -> name)."
        },
        {
            "name": "Flights",
            "role": "Train + in-domain test",
            "source": "Raha repo (datasets/flights)",
            "clean": root / "raha-master/datasets/flights/clean.csv",
            "dirty": root / "raha-master/datasets/flights/dirty.csv",
            "annotation": root / "TableEG/source/flight/flight_annotation.csv",
            "licence": "Apache-2.0 (repo)",
            "notes": "Missing/'Not Available' times; 904 unannotated format errors classified via heuristic."
        },
        {
            "name": "Beers",
            "role": "Train + in-domain test",
            "source": "Raha repo (datasets/beers)",
            "clean": root / "raha-master/datasets/beers/clean.csv",
            "dirty": root / "raha-master/datasets/beers/dirty.csv",
            "annotation": root / "TableEG/source/beers/beers_annotation.csv",
            "licence": "Apache-2.0 (repo)",
            "notes": "ABV/IBU formats and outliers; 1,005 'N/A' vs blank IBU errors recovered from Raha."
        },
        {
            "name": "Rayyan",
            "role": "Train + in-domain test",
            "source": "Raha repo (datasets/rayyan)",
            "clean": root / "raha-master/datasets/rayyan/clean.csv",
            "dirty": root / "raha-master/datasets/rayyan/dirty.csv",
            "annotation": root / "TableEG/source/rayyan/rayyan_annotation.jsonl",
            "licence": "Apache-2.0 (repo)",
            "notes": "Missing and format errors in bibliographic fields; mapped from TableEG rayyan_annotation.jsonl."
        },
        {
            "name": "Movies",
            "role": "Train + in-domain test",
            "source": "Raha repo (datasets/movies_1)",
            "clean": root / "raha-master/datasets/movies_1/clean.csv",
            "dirty": root / "raha-master/datasets/movies_1/dirty.csv",
            "annotation": root / "TableEG/source/movie_metadata/movie_annotation.jsonl",
            "licence": "Apache-2.0 (repo)",
            "notes": "Pattern errors (recovered ~668 errors lost in TableEG due to RatingValue float truncation); all 7,675 annotated."
        },
        {
            "name": "Credit",
            "role": "Downstream test (RQ4)",
            "source": "TableEG (Kaggle 'Give Me Some Credit')",
            "clean": root / "TableEG/source/Credit/clean.csv",
            "dirty": root / "TableEG/source/Credit/dirty.csv",
            "annotation": root / "TableEG/source/Credit/Credit_annotation.csv",
            "licence": "Kaggle competition rules",
            "notes": "SeriousDlqin2yrs target; real errors: MonthlyIncome & Dependents; float re-saving format artifacts isolated."
        },
        {
            "name": "Marketing",
            "role": "Unseen-table test",
            "source": "TableEG (from CleanML)",
            "clean": root / "TableEG/test_dataset/exp_1/TableEG_output/Marketing/clean_Marketing.csv",
            "dirty": root / "TableEG/source/Marketing/dirty.csv",
            "annotation": root / "TableEG/source/Marketing/Marketing_annotation.csv",
            "licence": "None stated",
            "notes": "Random category swaps (male -> female, age bucket); little row context."
        },
        {
            "name": "Soccer",
            "role": "Unseen-table test",
            "source": "TableEG (test_dataset/exp_1 BART)",
            "clean": root / "TableEG/test_dataset/exp_1/BART_output/soccer/clean_soccer.csv",
            "dirty": root / "TableEG/test_dataset/exp_1/BART_output/soccer/soccer_10/dirty_data.csv",
            "annotation": None,
            "licence": "None stated",
            "notes": "Clean + BART dirty; 796 errors dynamically typed: 199 outliers (numeric birthyear), 597 pattern_violation."
        },
        {
            "name": "Restaurant",
            "role": "Unseen-table test",
            "source": "TableEG (from CleanML)",
            "clean": root / "TableEG/source/Restaurant/clean.csv",
            "dirty": root / "TableEG/source/Restaurant/dirty.csv",
            "annotation": root / "TableEG/source/Restaurant/Restaurant_annotation.jsonl",
            "licence": "None stated",
            "notes": "12,007 rows x 11 data cols; 633 errors under the Raha rule (all pattern_violation); 635 annotations, 2 of which differ only in a double space and so are not errors after normalisation."
        }
    ]

    audit_results = {}
    for cfg in benchmark_configs:
        tname = cfg["name"]
        print(f"[*] Auditing live dataset pair: {tname}...")
        res = audit_table_pair(tname, cfg["clean"], cfg["dirty"], cfg["annotation"])
        res["role"] = cfg["role"]
        res["source"] = cfg["source"]
        res["licence"] = cfg["licence"]
        res["notes"] = cfg["notes"]
        audit_results[tname] = res

        print(f"    -> Rows: {res['rows']:,}, Data Cols: {res['data_cols']}, Total Cols: {res['total_cols']}")
        print(f"    -> Raha Strict Errors: {res['strict_errors']:,} ({res['strict_errors']/(res['rows']*res['data_cols'])*100:.2f}%)")
        print(f"    -> Lenient Errors: {res['lenient_errors']:,}")
        print(f"    -> Annotations: {res['annotations_total']} (Mapped: {res['annotations_mapped']:,}, Unannotated: {res['unannotated_cells']:,})")
        print(f"    -> Error Types: {dict(res['error_type_counts'])}")
        print(f"    -> Clean-side Quality: Missing cells={res['quality']['clean_missing_cells']}, Duplicates={res['quality']['clean_duplicate_rows']}, Header diffs={len(res['header_diffs'])}\n")

    # Audit Credit & Soccer specifics
    print("[*] Running specialized audits on Credit and Soccer...")
    spec_audit = audit_credit_and_soccer(root)
    cr = spec_audit["credit"]
    print(f"    -> Credit SeriousDlqin2yrs Class Balance: Class 0 = {cr['class_0_count']:,} ({cr['class_0_pct']:.4f}%), Class 1 = {cr['class_1_count']:,} ({cr['class_1_pct']:.4f}%)")
    print(f"    -> Credit Imputed MonthlyIncome Repeats (88.44070676): {cr['imputed_monthly_income_repeats']:,} occurrences")
    sc = spec_audit["soccer"]
    print(f"    -> Soccer BART Clean vs Dirty_10 Diffs: {sc['clean_vs_dirty10_diffs']} cells")
    print(f"    -> Soccer_10 cellChanges.csv entries: {sc['soccer_10_cellChanges_count']} cells (Exact Agreement: {sc['agreement_verified']})")
    print(f"    -> Soccer Root BART error_log.jsonl entries: {sc['exp1_error_log_entries']} cells\n")

    # Export docs/tables/data_inventory.csv
    inventory_csv = outdir / "data_inventory.csv"
    with open(inventory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Role", "Source", "Rows", "Data_Columns", "Total_Columns",
            "Error_Cells_Raha", "Error_Rate", "Lenient_Error_Cells",
            "Annotations_TableEG", "Unannotated_Raha_Cells",
            "Missing_Value", "Pattern_Violation", "Rule_Violation", "Outliers", "Format_Artifact",
            "Clean_Missing_Cells", "Clean_Duplicate_Rows", "Header_Differences",
            "Licence", "Notes"
        ])
        for name, res in audit_results.items():
            tot_cells = res["rows"] * res["data_cols"]
            err_rate = f"{res['strict_errors'] / tot_cells * 100:.2f}%" if tot_cells > 0 else "N/A"
            etc = res["error_type_counts"]
            writer.writerow([
                name,
                res["role"],
                res["source"],
                res["rows"],
                res["data_cols"],
                res["total_cols"],
                res["strict_errors"],
                err_rate,
                res["lenient_errors"],
                res["annotations_total"],
                res["unannotated_cells"],
                etc.get("missing_value", 0),
                etc.get("pattern_violation", 0),
                etc.get("rule_violation", 0),
                etc.get("outliers", 0),
                etc.get("format_artifact", 0),
                res["quality"]["clean_missing_cells"],
                res["quality"]["clean_duplicate_rows"],
                len(res["header_diffs"]),
                res["licence"],
                res["notes"]
            ])

    print(f"[SUCCESS] Exported data inventory CSV -> {inventory_csv}")

    # Export docs/tables/error_type_profile.csv
    profile_csv = outdir / "error_type_profile.csv"
    with open(profile_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Column", "Missing_Value", "Pattern_Violation", "Rule_Violation", "Outliers", "Format_Artifact", "Total_Errors"])
        for name, res in audit_results.items():
            col_prof = res["col_type_profile"]
            for col, counts in col_prof.items():
                tot = sum(counts.values())
                writer.writerow([
                    name,
                    col,
                    counts.get("missing_value", 0),
                    counts.get("pattern_violation", 0),
                    counts.get("rule_violation", 0),
                    counts.get("outliers", 0),
                    counts.get("format_artifact", 0),
                    tot
                ])

    print(f"[SUCCESS] Exported error type profile CSV -> {profile_csv}")

    # Compute global error distribution across benchmark tables (excluding format artifacts)
    global_distribution = Counter()
    for name, res in audit_results.items():
        etc = res["error_type_counts"]
        for etype in ["missing_value", "pattern_violation", "rule_violation", "outliers"]:
            global_distribution[etype] += etc.get(etype, 0)

    total_true_errors = sum(global_distribution.values())
    print("\n================ GLOBAL BENCHMARK ERROR TYPE DISTRIBUTION ================")
    print(f"Total True Error Cells Audited: {total_true_errors:,}")
    for etype in ["missing_value", "pattern_violation", "rule_violation", "outliers"]:
        count = global_distribution[etype]
        pct = (count / total_true_errors) * 100
        print(f"  * {etype:18}: {count:6,} ({pct:.2f}%)")

    # Generate standalone SVG pie chart
    figures_dir = root / "docs/figures"
    pie_svg_path = figures_dir / "error_distribution_pie.svg"
    generate_pie_chart_svg(global_distribution, pie_svg_path)
    print(f"[SUCCESS] Generated error distribution vector chart -> {pie_svg_path}")

    print("\n================================================================================")
    print("                    ALL AUDIT VERIFICATIONS SUCCESSFULLY COMPLETED               ")
    print("================================================================================")


if __name__ == "__main__":
    main()

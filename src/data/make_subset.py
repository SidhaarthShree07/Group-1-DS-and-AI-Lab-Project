# Author: Jashan Tiwari
"""
src/data/make_subset.py

ContextGuard Milestone 3: Small-Scale Validation Subset Generator
Author: Jashan Tiwari (Data Lead) - DSAI Lab, IIT Madras BS
Role: Implements Decision D17 and Section 7 Interface:
1. Reads validation split files (val.csv, val_clean.csv, val_cells.csv) from data/splits/core/<table>/
   for Hospital, Flights, and Beers. Test split and unseen tables are never touched.
2. Samples 100 validation rows per table using fixed seed=42, preserving all cells of those rows
   to ensure real natural error rates.
3. Re-indexes row indices in val_cells.csv to 0..99 to maintain exact positional alignment with
   the 100-row subset tables.
4. Writes subset files in data/subsets/m3/<table>/val{,_clean,_cells}.csv.
5. Computes and exports validation vs subset error rates and error-type breakdowns to
   docs/tables/m3_subset_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

CORE_SUBSET_TABLES = ["hospital", "flights", "beers"]
TYPE_MAP = {
    "missing_value": "missing",
    "pattern_violation": "pattern",
    "rule_violation": "rule",
    "outliers": "outlier",
    "outlier": "outlier",
    "": "none",
}


def make_subsets(
    repo_root: Path,
    tables: List[str] = CORE_SUBSET_TABLES,
    n_sample_rows: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates 100-row validation subsets and writes summary statistics."""
    splits_base = repo_root / "data" / "splits" / "core"
    subsets_base = repo_root / "data" / "subsets" / "m3"
    tables_dir = repo_root / "docs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("=" * 80)
    print("      MILESTONE 3: GENERATING 100-ROW VALIDATION SUBSETS (DECISION D17)        ")
    print(f"      Tables: {', '.join(tables)} | Seed: {seed} | Rows per Table: {n_sample_rows}")
    print("=" * 80)

    for table in tables:
        t_split_dir = splits_base / table
        t_sub_dir = subsets_base / table
        t_sub_dir.mkdir(parents=True, exist_ok=True)

        val_path = t_split_dir / "val.csv"
        val_clean_path = t_split_dir / "val_clean.csv"
        val_cells_path = t_split_dir / "val_cells.csv"

        if not val_path.exists() or not val_clean_path.exists() or not val_cells_path.exists():
            raise FileNotFoundError(
                f"Missing split files for {table} in {t_split_dir}. "
                "Ensure Milestone 2 splits have been generated."
            )

        with open(val_path, "r", encoding="utf-8") as f:
            v_rows = list(csv.reader(f))
        with open(val_clean_path, "r", encoding="utf-8") as f:
            c_rows = list(csv.reader(f))
        with open(val_cells_path, "r", encoding="utf-8") as f:
            cell_reader = csv.DictReader(f)
            cell_fieldnames = list(cell_reader.fieldnames or ["row", "column", "dirty_value", "clean_value", "is_error", "error_type"])
            val_cells = list(cell_reader)

        v_header, v_data = v_rows[0], v_rows[1:]
        c_header, c_data = c_rows[0], c_rows[1:]

        val_row_count = len(v_data)
        if val_row_count < n_sample_rows:
            raise ValueError(
                f"{table} validation split has only {val_row_count} rows, requested {n_sample_rows}."
            )

        val_total_cells = len(val_cells)
        n_cols = val_total_cells // val_row_count
        if val_total_cells != val_row_count * n_cols:
            raise ValueError(
                f"{table}: {val_total_cells} cells not divisible by {val_row_count} rows."
            )

        val_errors = sum(1 for c in val_cells if c["is_error"].strip().lower() == "true")
        val_err_rate = (val_errors / val_total_cells * 100.0) if val_total_cells > 0 else 0.0

        # Deterministic sampling with fixed seed per table
        rng = random.Random(seed)
        sampled_indices = sorted(rng.sample(range(val_row_count), n_sample_rows))

        sub_v_data = [v_data[i] for i in sampled_indices]
        sub_c_data = [c_data[i] for i in sampled_indices]

        # Extract cells and re-index row from 0 to n_sample_rows - 1
        sub_cells: List[Dict[str, Any]] = []
        for new_r, old_r in enumerate(sampled_indices):
            chunk = val_cells[old_r * n_cols : (old_r + 1) * n_cols]
            for c in chunk:
                cell_copy = dict(c)
                cell_copy["row"] = str(new_r)
                sub_cells.append(cell_copy)

        # Write subset dirty CSV
        with open(t_sub_dir / "val.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(v_header)
            w.writerows(sub_v_data)

        # Write subset clean CSV
        with open(t_sub_dir / "val_clean.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(c_header)
            w.writerows(sub_c_data)

        # Write subset cells CSV
        with open(t_sub_dir / "val_cells.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cell_fieldnames)
            w.writeheader()
            w.writerows(sub_cells)

        # Tally subset metrics
        sub_total_cells = len(sub_cells)
        sub_errors = sum(1 for c in sub_cells if c["is_error"].strip().lower() == "true")
        sub_err_rate = (sub_errors / sub_total_cells * 100.0) if sub_total_cells > 0 else 0.0

        type_counts = Counter(
            TYPE_MAP.get(c["error_type"].strip(), c["error_type"].strip())
            for c in sub_cells
            if c["is_error"].strip().lower() == "true"
        )

        delta_rate = sub_err_rate - val_err_rate

        rec = {
            "table": table,
            "val_rows": val_row_count,
            "val_cells": val_total_cells,
            "val_errors": val_errors,
            "val_error_rate_pct": round(val_err_rate, 2),
            "subset_rows": n_sample_rows,
            "subset_cells": sub_total_cells,
            "subset_errors": sub_errors,
            "subset_error_rate_pct": round(sub_err_rate, 2),
            "subset_missing": type_counts["missing"],
            "subset_pattern": type_counts["pattern"],
            "subset_rule": type_counts["rule"],
            "subset_outlier": type_counts["outlier"],
            "delta_rate_pct": round(delta_rate, 2),
        }
        summary_rows.append(rec)

        print(
            f"[*] {table.upper():<10} | "
            f"Val: {val_row_count} rows, {val_total_cells} cells, {val_errors} err ({val_err_rate:.2f}%) | "
            f"Subset: {n_sample_rows} rows, {sub_total_cells} cells, {sub_errors} err ({sub_err_rate:.2f}%, delta={delta_rate:+.2f}%) | "
            f"Types: M={type_counts['missing']}, P={type_counts['pattern']}, R={type_counts['rule']}, O={type_counts['outlier']}"
        )

    # Compute Total row across all 3 tables
    total_val_rows = sum(r["val_rows"] for r in summary_rows)
    total_val_cells = sum(r["val_cells"] for r in summary_rows)
    total_val_errors = sum(r["val_errors"] for r in summary_rows)
    total_val_rate = (total_val_errors / total_val_cells * 100.0) if total_val_cells > 0 else 0.0

    total_sub_rows = sum(r["subset_rows"] for r in summary_rows)
    total_sub_cells = sum(r["subset_cells"] for r in summary_rows)
    total_sub_errors = sum(r["subset_errors"] for r in summary_rows)
    total_sub_rate = (total_sub_errors / total_sub_cells * 100.0) if total_sub_cells > 0 else 0.0

    total_rec = {
        "table": "Total",
        "val_rows": total_val_rows,
        "val_cells": total_val_cells,
        "val_errors": total_val_errors,
        "val_error_rate_pct": round(total_val_rate, 2),
        "subset_rows": total_sub_rows,
        "subset_cells": total_sub_cells,
        "subset_errors": total_sub_errors,
        "subset_error_rate_pct": round(total_sub_rate, 2),
        "subset_missing": sum(r["subset_missing"] for r in summary_rows),
        "subset_pattern": sum(r["subset_pattern"] for r in summary_rows),
        "subset_rule": sum(r["subset_rule"] for r in summary_rows),
        "subset_outlier": sum(r["subset_outlier"] for r in summary_rows),
        "delta_rate_pct": round(total_sub_rate - total_val_rate, 2),
    }
    summary_rows.append(total_rec)

    # Write summary CSV
    summary_csv_path = tables_dir / "m3_subset_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("=" * 80)
    print(f"[SUCCESS] Exported M3 subset summary -> {summary_csv_path.relative_to(repo_root)}")
    print(
        f"          Total: {total_sub_rows} rows, {total_sub_cells} cells, "
        f"{total_sub_errors} errors ({total_sub_rate:.2f}%)"
    )
    print("=" * 80)

    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 100-row validation subsets for M3")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default 42)",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=100,
        help="Number of rows to sample per table (default 100)",
    )
    args = parser.parse_args()

    make_subsets(repo_root=args.root, n_sample_rows=args.n_rows, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())


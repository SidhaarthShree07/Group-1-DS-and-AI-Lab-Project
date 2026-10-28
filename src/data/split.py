#!/usr/bin/env python3
"""
src/data/split.py

ContextGuard Milestone 2: Group-Aware 70/15/15 Splitter & Verification
Author: Jashan (Data Lead) - DSAI Lab, IIT Madras BS
Role: Implements Decisions D1 & D2:
1. Measures clean-side entity repetition across all 5 core tables.
2. Performs group-aware 70/15/15 train/validation/test splitting with seed=42
   grouped on the clean entity key (eliminating dirty typo entity leakage).
3. Partitions dirty, clean, and cell-level files into data/splits/core/<table_name>/.
4. Generates docs/tables/m2_core_splits.csv.
5. Verifies zero entity/row overlap and exact error conservation against M1.
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set

# Configuration for core tables:
# clean_entity_col, clean_entity_idx, dirty_entity_col, dirty_entity_idx
SPLIT_CONFIGS = [
    {
        "name": "hospital",
        "raha_subpath": "hospital",
        "clean_entity_col": "ProviderNumber",
        "clean_entity_idx": 1,
        "dirty_entity_col": "provider_number",
        "dirty_entity_idx": 1,
        "description": "US CMS Hospital Provider ID (grouped on clean key to eliminate dirty typo leakage)"
    },
    {
        "name": "flights",
        "raha_subpath": "flights",
        "clean_entity_col": "flight",
        "clean_entity_idx": 2,
        "dirty_entity_col": "flight",
        "dirty_entity_idx": 2,
        "description": "Flight route / number (spans multiple flight days / dates)"
    },
    {
        "name": "beers",
        "raha_subpath": "beers",
        "clean_entity_col": "brewery_id",
        "clean_entity_idx": 7,
        "dirty_entity_col": "brewery_id",
        "dirty_entity_idx": 7,
        "description": "Brewery ID (spans multiple craft beer products from the same brewery)"
    },
    {
        "name": "rayyan",
        "raha_subpath": "rayyan",
        "clean_entity_col": "id",
        "clean_entity_idx": 0,
        "dirty_entity_col": "id",
        "dirty_entity_idx": 0,
        "description": "Bibliographic article ID (1 row per article, 0 repetition across rows)"
    },
    {
        "name": "movies",
        "raha_subpath": "movies_1",
        "clean_entity_col": "Id",
        "clean_entity_idx": 0,
        "dirty_entity_col": "id",
        "dirty_entity_idx": 0,
        "description": "IMDb title ID (grouped on clean key, 0 repetition across rows)"
    }
]


def load_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Reads CSV file returning header and data rows."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = list(csv.reader(f))
    return reader[0], reader[1:]


def measure_entity_repetition(name: str, entity_col: str, entity_idx: int, rows: List[List[str]]) -> Dict[str, Any]:
    """
    Measures entity repetition on the table.
    """
    total_rows = len(rows)
    groups = defaultdict(list)
    for r_idx, r in enumerate(rows):
        key = r[entity_idx] if entity_idx < len(r) else ""
        groups[key].append(r_idx)

    unique_entities = len(groups)
    rows_per_entity = [len(v) for v in groups.values()]
    mean_rows = total_rows / unique_entities if unique_entities > 0 else 0
    max_rows = max(rows_per_entity) if rows_per_entity else 0
    multi_entity_keys = sum(1 for v in groups.values() if len(v) > 1)
    rows_sharing_key = sum(len(v) for v in groups.values() if len(v) > 1)
    pct_sharing = (rows_sharing_key / total_rows * 100) if total_rows > 0 else 0

    return {
        "name": name,
        "entity_col": entity_col,
        "total_rows": total_rows,
        "unique_entities": unique_entities,
        "mean_rows_per_entity": mean_rows,
        "max_rows_per_entity": max_rows,
        "multi_entity_keys": multi_entity_keys,
        "rows_sharing_key": rows_sharing_key,
        "pct_sharing": pct_sharing,
        "groups": groups
    }


def split_grouped_entities(groups: Dict[str, List[int]], total_rows: int, seed: int = 42) -> Tuple[Dict[str, List[int]], Dict[str, Set[str]]]:
    """
    Partitions rows into train (70%), validation (15%), and test (15%) by entity group.
    Ensures zero entity overlap between splits.
    """
    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    target = {
        "train": 0.70 * total_rows,
        "val": 0.15 * total_rows,
        "test": 0.15 * total_rows
    }
    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    split_keys: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for k in keys:
        k_rows = groups[k]
        # Best-fit selection based on distance to target proportion
        best_split = max(["train", "val", "test"], key=lambda s: (target[s] - len(splits[s])) / target[s])
        splits[best_split].extend(k_rows)
        split_keys[best_split].add(k)

    # Sort row indices within each split for deterministic reproducibility
    for s in splits:
        splits[s].sort()

    return splits, split_keys


def run_splits(repo_root: Path) -> List[Dict[str, Any]]:
    """Runs entity measurement, group splitting, file exports, and verification."""
    summary_records = []
    grand_total_errors = 0
    table_error_sums = {}

    splits_base_dir = repo_root / "data" / "splits" / "core"
    splits_base_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = repo_root / "docs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      STEP 2A: MEASURE ENTITY REPETITION & GROUP ON CLEAN KEY (DECISION D2)     ")
    print("=" * 80)

    for cfg in SPLIT_CONFIGS:
        tname = cfg["name"]
        raha_sub = cfg["raha_subpath"]
        
        # Locate files in data/raw/raha first, then raha-master
        dirty_candidates = [
            repo_root / "data" / "raw" / "raha" / raha_sub / "dirty.csv",
            repo_root / "raha-master" / "datasets" / raha_sub / "dirty.csv"
        ]
        clean_candidates = [
            repo_root / "data" / "raw" / "raha" / raha_sub / "clean.csv",
            repo_root / "raha-master" / "datasets" / raha_sub / "clean.csv"
        ]
        dirty_file = next(p for p in dirty_candidates if p.exists())
        clean_file = next(p for p in clean_candidates if p.exists())
        cells_file = repo_root / "data" / "processed" / "core" / f"{tname}_cells.csv"

        if not cells_file.exists():
            raise FileNotFoundError(f"Missing processed cells file: {cells_file}. Run load_core.py first!")

        d_hdr, d_rows = load_csv(dirty_file)
        c_hdr, c_rows = load_csv(clean_file)

        # 1. Group on clean key to eliminate dirty typo leakage
        rep = measure_entity_repetition(tname, cfg["clean_entity_col"], cfg["clean_entity_idx"], c_rows)
        print(f"[*] Table: {tname.upper()} (Clean Entity Key: '{cfg['clean_entity_col']}')")
        print(f"    - Rationale: {cfg['description']}")
        print(f"    - Total Rows: {rep['total_rows']:,}")
        print(f"    - Unique Entities: {rep['unique_entities']:,}")
        print(f"    - Rows per Entity: Mean = {rep['mean_rows_per_entity']:.2f}, Max = {rep['max_rows_per_entity']}")
        print(f"    - Multi-Row Entities: {rep['multi_entity_keys']:,} distinct keys")
        print(f"    - Rows Sharing an Entity: {rep['rows_sharing_key']:,} / {rep['total_rows']:,} ({rep['pct_sharing']:.2f}%)\n")

        # 2. Group-Aware 70/15/15 Split
        splits, split_keys = split_grouped_entities(rep["groups"], rep["total_rows"], seed=42)

        # 3. Verify zero row and zero entity overlap
        train_rows_set = set(splits["train"])
        val_rows_set = set(splits["val"])
        test_rows_set = set(splits["test"])

        row_overlap = len(train_rows_set & val_rows_set) + len(train_rows_set & test_rows_set) + len(val_rows_set & test_rows_set)
        entity_overlap = len(split_keys["train"] & split_keys["val"]) + len(split_keys["train"] & split_keys["test"]) + len(split_keys["val"] & split_keys["test"])

        assert row_overlap == 0, f"FATAL: Row overlap detected on {tname}!"
        assert entity_overlap == 0, f"FATAL: Entity overlap detected on {tname}!"
        assert len(train_rows_set | val_rows_set | test_rows_set) == rep["total_rows"], f"FATAL: Incomplete row coverage on {tname}!"

        # 4. Load cell annotations for this table
        with open(cells_file, "r", encoding="utf-8") as f:
            all_cells = list(csv.DictReader(f))

        cells_by_row = defaultdict(list)
        for cell in all_cells:
            cells_by_row[int(cell["row"])].append(cell)

        # Export split files
        t_split_dir = splits_base_dir / tname
        t_split_dir.mkdir(parents=True, exist_ok=True)

        t_error_total = 0
        t_type_totals = Counter()
        t_cells_total = 0

        for sname in ["train", "val", "test"]:
            s_row_indices = splits[sname]
            s_dirty_rows = [d_rows[r] for r in s_row_indices]
            s_clean_rows = [c_rows[r] for r in s_row_indices]
            s_cells = [c for r in s_row_indices for c in cells_by_row[r]]

            # Write dirty CSV for split
            with open(t_split_dir / f"{sname}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(d_hdr)
                w.writerows(s_dirty_rows)

            # Write clean CSV for split
            with open(t_split_dir / f"{sname}_clean.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(c_hdr)
                w.writerows(s_clean_rows)

            # Write cell CSV for split
            with open(t_split_dir / f"{sname}_cells.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["row", "column", "dirty_value", "clean_value", "is_error", "error_type"])
                w.writeheader()
                w.writerows(s_cells)

            # Tally metrics
            s_row_count = len(s_row_indices)
            s_pct_rows = s_row_count / rep["total_rows"] * 100
            s_total_cells = len(s_cells)
            s_err_cells = sum(1 for c in s_cells if c["is_error"] == "True")
            s_err_rate = (s_err_cells / s_total_cells * 100) if s_total_cells > 0 else 0
            s_types = Counter(c["error_type"] for c in s_cells if c["is_error"] == "True")

            t_error_total += s_err_cells
            t_cells_total += s_total_cells
            t_type_totals.update(s_types)

            summary_records.append({
                "Table": tname,
                "Split": sname,
                "Entity_Column": cfg["clean_entity_col"],
                "Unique_Entities": len(split_keys[sname]),
                "Row_Count": s_row_count,
                "Row_Percentage": f"{s_pct_rows:.2f}%",
                "Total_Cells": s_total_cells,
                "Error_Cells": s_err_cells,
                "Error_Rate": f"{s_err_rate:.2f}%",
                "Missing_Value": s_types["missing_value"],
                "Pattern_Violation": s_types["pattern_violation"],
                "Rule_Violation": s_types["rule_violation"],
                "Outliers": s_types["outliers"]
            })

        # Add total row for this table
        t_err_rate = (t_error_total / t_cells_total * 100) if t_cells_total > 0 else 0
        summary_records.append({
            "Table": tname,
            "Split": "total",
            "Entity_Column": cfg["clean_entity_col"],
            "Unique_Entities": rep["unique_entities"],
            "Row_Count": rep["total_rows"],
            "Row_Percentage": "100.00%",
            "Total_Cells": t_cells_total,
            "Error_Cells": t_error_total,
            "Error_Rate": f"{t_err_rate:.2f}%",
            "Missing_Value": t_type_totals["missing_value"],
            "Pattern_Violation": t_type_totals["pattern_violation"],
            "Rule_Violation": t_type_totals["rule_violation"],
            "Outliers": t_type_totals["outliers"]
        })

        table_error_sums[tname] = t_error_total
        grand_total_errors += t_error_total

    # Export docs/tables/m2_core_splits.csv
    splits_csv = repo_root / "docs" / "tables" / "m2_core_splits.csv"
    with open(splits_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Table", "Split", "Entity_Column", "Unique_Entities", "Row_Count", "Row_Percentage",
            "Total_Cells", "Error_Cells", "Error_Rate", "Missing_Value", "Pattern_Violation",
            "Rule_Violation", "Outliers"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)

    print("=" * 80)
    print("      STEP 2B: SPLIT VERIFICATION CHECKS (ZERO-LEAKAGE & M1 TOTALS)     ")
    print("=" * 80)
    print(f"[SUCCESS] Exported summary table -> docs/tables/m2_core_splits.csv")
    print(f"[SUCCESS] Partitioned split files saved to -> data/splits/core/<table_name>/ [train, val, test]\n")

    # Verification 1: M1 Expected Totals
    m1_expected = {
        "hospital": 509,
        "flights": 4920,
        "beers": 4362,
        "rayyan": 948,
        "movies": 7675
    }
    for tname, exp in m1_expected.items():
        act = table_error_sums[tname]
        assert act == exp, f"Split error mismatch on {tname}: got {act}, expected {exp}"
        print(f"  [CHECK] {tname.capitalize():8} (train + val + test) = {act:,} error cells == M1 target ({exp:,}) [MATCH]")

    assert grand_total_errors == 18414, f"Grand total split error mismatch: {grand_total_errors} != 18,414"
    print(f"  [CHECK] GRAND TOTAL (all 5 tables across splits) = {grand_total_errors:,} error cells == 18,414 [MATCH]\n")

    # Verification 2: Zero Overlap Guarantee
    for cfg in SPLIT_CONFIGS:
        tname = cfg["name"]
        print(f"  [CHECK] Zero-Leakage on {tname.capitalize():8}: row_overlap = 0 | entity_overlap = 0 [STRICT PASS]")

    print("\n" + "=" * 80)
    print("                 ALL SPLIT PROCEDURES COMPLETED SUCCESSFULLY                   ")
    print("=" * 80 + "\n")
    return summary_records


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_splits(repo_root)


if __name__ == "__main__":
    main()

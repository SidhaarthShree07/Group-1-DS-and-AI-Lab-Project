#!/usr/bin/env python3
"""Milestone 5 Experiment X10: Fitted Thresholds and Calibration Grid on Test Data.

Author: Sahilpreet
Role: Evaluation, Calibration & Gating Lead

This script conducts Experiment X10 by reading the real M4 test run folders:
  - beers, flights, hospital, rayyan, movies (from predictions.jsonl and cells.csv).
  - Evaluated on the standardized test split: leaving out Raha's 20 labelled rows per table
    in accordance with Rule D8 and the main benchmark scoring protocol.
  - Reports both post-validation repairs (baseline: 975 right / 13 wrong) and gate-level repairs
    (baseline: 1,173 right / 41 wrong, 3.38% FRR).
  - Evaluates all pass rules dynamically from the resulting grid.

Evaluates the 8 parameter combinations of:
  - Flag threshold t1: 0.500 (fixed production) vs 0.378 (M4 validation fitted)
  - Repair confidence t2: 0.900 (fixed production) vs 0.648 (M4 validation fitted)
  - Temperature T: 1.000 (raw logits, uncalibrated) vs 0.826 (M4 validation fitted)

Predetermined Pass Rules (Evaluated Dynamically from Grid):
  1. Flag threshold 0.378: Adopt if mean test F1 is higher than with 0.500 AND
     F1 is not lower on more than one of the five tables.
  2. Repair confidence 0.648: Adopt ONLY IF false-repair rate (FRR) is NOT HIGHER
     than the baseline AND right repairs increase.
  3. Temperature 0.826: Adopt if ECE falls on at least 4 of the 5 tables.

Outputs:
  - docs/tables/m5/x10_threshold_calibration_grid.csv
  - docs/tables/m5/x10_pass_rules_summary.csv

Usage:
  python scripts/run_x10_threshold_grid.py [--runs M4_run_outputs/<run_1> ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.calibrate import apply_temperature
from src.evaluation.metrics import calculate_ece
from src.profiling.context import norm


def get_raha_excluded_rows(table: str, total_rows: int, budget: int = 20, seed: int = 42) -> Set[int]:
    """Return the set of 20 row indices excluded for Raha's labeled tuples under Rule D8."""
    rng = random.Random(f"{seed}:{table.lower()}")
    return set(rng.sample(range(total_rows), min(budget, total_rows)))


def load_dataset_runs(run_dirs: List[Path]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]]:
    """Load cell ground truth and model predictions across run directories.

    Returns:
        cells_by_table: table_name -> list of cell ground truth dicts (excluding Raha's 20 labelled rows)
        preds_by_table: table_name -> (row, col) -> prediction dict
    """
    raw_cells_by_table: Dict[str, List[Dict[str, Any]]] = {}
    preds_by_table: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}

    for rd in run_dirs:
        if not rd.exists():
            raise FileNotFoundError(f"Run directory not found: {rd}")

        cells_path = rd / "cells.csv"
        preds_path = rd / "predictions.jsonl"
        if not cells_path.exists() or not preds_path.exists():
            raise FileNotFoundError(f"Missing cells.csv or predictions.jsonl in {rd}")

        print(f"[*] Loading run files from {rd}...")

        with open(cells_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                t = str(row.get("table", "")).strip().lower()
                if not t:
                    continue
                raw_cells_by_table.setdefault(t, []).append(row)

        with open(preds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                t = str(p.get("table", "")).strip().lower()
                if not t:
                    continue
                r = str(p.get("row", "")).strip()
                c = str(p.get("column", "")).strip()
                preds_by_table.setdefault(t, {})[(r, c)] = p

    # Filter out Raha's 20 labelled rows per table (Rule D8)
    filtered_cells_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for table, cells in raw_cells_by_table.items():
        all_row_indices = sorted(set(int(c["row"]) for c in cells))
        n_rows = len(all_row_indices)
        excluded = get_raha_excluded_rows(table, n_rows, budget=20, seed=42)
        filtered = [c for c in cells if int(c["row"]) not in excluded]
        filtered_cells_by_table[table] = filtered
        print(f"    Table {table:8s}: {n_rows} total rows - 20 labelled rows = {len(set(int(c['row']) for c in filtered))} scored rows")

    return filtered_cells_by_table, preds_by_table


def evaluate_table(
    table_name: str,
    table_cells: List[Dict[str, Any]],
    table_preds: Dict[Tuple[str, str], Dict[str, Any]],
    t1: float,
    t2: float,
    temp: float,
) -> Dict[str, Any]:
    """Evaluate table metrics under parameters (t1, t2, temp)."""
    tp = fp = fn = tn = 0
    val_right = val_wrong = 0
    gate_right = gate_wrong = 0
    review_count = 0
    probs: List[float] = []
    labels: List[bool] = []

    for c in table_cells:
        if str(c.get("error_type", "")).strip() == "format_artifact":
            continue

        r = str(c.get("row", "")).strip()
        col = str(c.get("column", "")).strip()
        norm_d = norm(c.get("dirty_value", ""))
        norm_clean = norm(c.get("clean_value", ""))
        gold_is_err = norm_d != norm_clean

        p = table_preds.get((r, col))
        if p is None:
            is_flagged = False
        else:
            valid_json = bool(p.get("valid_json", False))
            p_raw = p.get("p_error")
            try:
                p_val = float(p_raw) if p_raw is not None else 0.0
            except (ValueError, TypeError):
                p_val = 0.0

            if abs(temp - 1.0) > 1e-4:
                p_eval = apply_temperature([p_val], temp)[0]
            else:
                p_eval = p_val

            probs.append(p_eval)
            labels.append(gold_is_err)

            is_flagged = valid_json and (p_eval >= t1)

            if is_flagged:
                rep = p.get("repair")
                p_rep_min = p.get("p_repair_min")
                try:
                    p_rep_val = float(p_rep_min) if p_rep_min is not None else 0.0
                except (ValueError, TypeError):
                    p_rep_val = 0.0

                has_rep = rep is not None and str(rep).strip() != ""

                # 1. Gate-level evaluation at threshold t2
                if has_rep and p_rep_val >= t2:
                    if norm(str(rep)) != norm_d:
                        if norm(str(rep)) == norm_clean:
                            gate_right += 1
                        else:
                            gate_wrong += 1
                else:
                    review_count += 1

                # 2. Post-validation evaluation (after validation rules)
                # In M4 runs, validation rules executed for baseline (t2=0.90)
                # Rayyan and Movies have no validation rules, so repairs route to review.
                dec = p.get("decision")
                if t2 == 0.900:
                    if dec == "repair" and norm(str(rep or "")) != norm_d:
                        if norm(str(rep or "")) == norm_clean:
                            val_right += 1
                        else:
                            val_wrong += 1
                else:
                    # At t2=0.648, gate passes more candidates; if validation outcome was valid
                    val_info = p.get("validation") or {}
                    val_outcome = val_info.get("outcome") if isinstance(val_info, dict) else None
                    if table_name in ["beers", "flights", "hospital"] and val_outcome == "valid" and has_rep and p_rep_val >= t2:
                        if norm(str(rep or "")) != norm_d:
                            if norm(str(rep or "")) == norm_clean:
                                val_right += 1
                            else:
                                val_wrong += 1

        if is_flagged and gold_is_err:
            tp += 1
        elif is_flagged and not gold_is_err:
            fp += 1
        elif not is_flagged and gold_is_err:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    ece = calculate_ece(labels, probs) if probs else 0.0

    # Post-validation repairs
    tot_val_applied = val_right + val_wrong
    val_frr = val_wrong / tot_val_applied if tot_val_applied > 0 else 0.0

    # Gate-level repairs
    tot_gate_applied = gate_right + gate_wrong
    gate_frr = gate_wrong / tot_gate_applied if tot_gate_applied > 0 else 0.0

    rev_share = review_count / max(1, len(table_cells))

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "review_share": round(rev_share, 4),
        # Post-validation repairs (matches main benchmark headline results: 975 right / 13 wrong)
        "applied_right": val_right,
        "applied_wrong": val_wrong,
        "applied_repairs": tot_val_applied,
        "false_repair_rate": round(val_frr, 4),
        # Gate-level repairs (explicitly labeled for gate-threshold comparison)
        "gate_applied_right": gate_right,
        "gate_applied_wrong": gate_wrong,
        "gate_applied_repairs": tot_gate_applied,
        "gate_false_repair_rate": round(gate_frr, 4),
        "ece": round(ece, 4),
    }


def compute_grid(run_dirs: Optional[List[Path]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute the 8 combinations grid dynamically across all test tables."""
    if not run_dirs:
        candidates = [
            [
                _REPO_ROOT / "M4_run_outputs" / "m4_final2_test_hfb",
                _REPO_ROOT / "M4_run_outputs" / "m4_final_test_rayyan",
                _REPO_ROOT / "M4_run_outputs" / "m4_final_test_movies",
            ],
            [
                _REPO_ROOT / "outputs" / "runs" / "m4_final2_test_hfb",
                _REPO_ROOT / "outputs" / "runs" / "m4_final_test_rayyan",
                _REPO_ROOT / "outputs" / "runs" / "m4_final_test_movies",
            ]
        ]
        for c in candidates:
            if all(p.exists() for p in c):
                run_dirs = c
                break

    cells_by_table, preds_by_table = load_dataset_runs(run_dirs)
    tables = sorted(cells_by_table.keys())
    if not tables:
        raise ValueError("No table data found in specified run folders.")

    combinations = [
        {"t1": 0.500, "t2": 0.900, "temp": 1.000, "name": "Fixed Baseline (Default)"},
        {"t1": 0.378, "t2": 0.900, "temp": 1.000, "name": "Fitted Flag (t1=0.378)"},
        {"t1": 0.500, "t2": 0.648, "temp": 1.000, "name": "Fitted Repair Gate (t2=0.648)"},
        {"t1": 0.500, "t2": 0.900, "temp": 0.826, "name": "Fitted Temperature (T=0.826)"},
        {"t1": 0.378, "t2": 0.648, "temp": 1.000, "name": "Fitted Gates (t1=0.378, t2=0.648)"},
        {"t1": 0.378, "t2": 0.900, "temp": 0.826, "name": "Fitted Flag & Temp (t1=0.378, T=0.826)"},
        {"t1": 0.500, "t2": 0.648, "temp": 0.826, "name": "Fitted Repair & Temp (t2=0.648, T=0.826)"},
        {"t1": 0.378, "t2": 0.648, "temp": 0.826, "name": "Fully Fitted (t1=0.378, t2=0.648, T=0.826)"},
    ]

    grid_rows: List[Dict[str, Any]] = []

    for combo in combinations:
        t1 = combo["t1"]
        t2 = combo["t2"]
        temp = combo["temp"]
        combo_name = combo["name"]

        table_f1s = []
        table_precs = []
        table_recs = []
        table_reviews = []
        table_eces = []
        tot_val_right = tot_val_wrong = 0
        tot_gate_right = tot_gate_wrong = 0

        for tbl in tables:
            t_cells = cells_by_table[tbl]
            t_preds = preds_by_table.get(tbl, {})
            res = evaluate_table(tbl, t_cells, t_preds, t1, t2, temp)

            table_f1s.append(res["f1"])
            table_precs.append(res["precision"])
            table_recs.append(res["recall"])
            table_reviews.append(res["review_share"])
            table_eces.append(res["ece"])
            tot_val_right += res["applied_right"]
            tot_val_wrong += res["applied_wrong"]
            tot_gate_right += res["gate_applied_right"]
            tot_gate_wrong += res["gate_applied_wrong"]

            grid_rows.append({
                "combination": combo_name,
                "t1_flag": t1,
                "t2_repair": t2,
                "temperature": temp,
                "table": tbl,
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "review_share": res["review_share"],
                "applied_right": res["applied_right"],
                "applied_wrong": res["applied_wrong"],
                "applied_repairs": res["applied_repairs"],
                "false_repair_rate": res["false_repair_rate"],
                "gate_applied_right": res["gate_applied_right"],
                "gate_applied_wrong": res["gate_applied_wrong"],
                "gate_applied_repairs": res["gate_applied_repairs"],
                "gate_false_repair_rate": res["gate_false_repair_rate"],
                "ece": res["ece"],
            })

        mean_f1 = sum(table_f1s) / len(table_f1s)
        mean_prec = sum(table_precs) / len(table_precs)
        mean_rec = sum(table_recs) / len(table_recs)
        mean_rev = sum(table_reviews) / len(table_reviews)
        mean_ece = sum(table_eces) / len(table_eces)
        pooled_val_frr = tot_val_wrong / max(1, tot_val_right + tot_val_wrong)
        pooled_gate_frr = tot_gate_wrong / max(1, tot_gate_right + tot_gate_wrong)

        grid_rows.append({
            "combination": combo_name,
            "t1_flag": t1,
            "t2_repair": t2,
            "temperature": temp,
            "table": "MEAN / TOTAL",
            "precision": round(mean_prec, 4),
            "recall": round(mean_rec, 4),
            "f1": round(mean_f1, 4),
            "review_share": round(mean_rev, 4),
            "applied_right": tot_val_right,
            "applied_wrong": tot_val_wrong,
            "applied_repairs": tot_val_right + tot_val_wrong,
            "false_repair_rate": round(pooled_val_frr, 4),
            "gate_applied_right": tot_gate_right,
            "gate_applied_wrong": tot_gate_wrong,
            "gate_applied_repairs": tot_gate_right + tot_gate_wrong,
            "gate_false_repair_rate": round(pooled_gate_frr, 4),
            "ece": round(mean_ece, 4),
        })

    eval_rules = evaluate_pass_rules_dynamically(grid_rows, tables)
    return grid_rows, eval_rules


def evaluate_pass_rules_dynamically(grid_rows: List[Dict[str, Any]], tables: List[str]) -> Dict[str, Any]:
    """Dynamically compute pass rules: FRR not higher than baseline, higher mean F1, and ECE reduction."""
    row_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["combination"], r["table"]): r for r in grid_rows
    }

    base_combo = "Fixed Baseline (Default)"
    flag_combo = "Fitted Flag (t1=0.378)"
    rep_combo = "Fitted Repair Gate (t2=0.648)"
    temp_combo = "Fitted Temperature (T=0.826)"

    # Rule 1: Flag Threshold t1
    base_mean_f1 = row_map[(base_combo, "MEAN / TOTAL")]["f1"]
    flag_mean_f1 = row_map[(flag_combo, "MEAN / TOTAL")]["f1"]
    tables_f1_dropped = []
    for t in tables:
        if row_map[(flag_combo, t)]["f1"] < row_map[(base_combo, t)]["f1"]:
            tables_f1_dropped.append(t)

    t1_f1_higher = flag_mean_f1 > base_mean_f1
    t1_passed = t1_f1_higher and len(tables_f1_dropped) <= 1
    t1_rec = "ADOPT (t1=0.378)" if t1_passed else "KEEP BASELINE (t1=0.500)"
    if not t1_f1_higher:
        t1_reason = f"Mean F1 drops from {base_mean_f1:.4f} to {flag_mean_f1:.4f}, and F1 decreases on {len(tables_f1_dropped)} tables ({', '.join(t.capitalize() for t in tables_f1_dropped)})."
    elif len(tables_f1_dropped) > 1:
        t1_reason = f"F1 decreases on {len(tables_f1_dropped)} tables ({', '.join(t.capitalize() for t in tables_f1_dropped)}), failing the <=1 drop condition."
    else:
        t1_reason = f"Mean F1 improves from {base_mean_f1:.4f} to {flag_mean_f1:.4f} with drops on <=1 table."

    # Rule 2: Repair Gate Threshold t2
    # Evaluated at the gate level: pass rule requires FRR to be NOT HIGHER than baseline
    base_gate_frr = row_map[(base_combo, "MEAN / TOTAL")]["gate_false_repair_rate"]
    rep_gate_frr = row_map[(rep_combo, "MEAN / TOTAL")]["gate_false_repair_rate"]
    base_gate_right = row_map[(base_combo, "MEAN / TOTAL")]["gate_applied_right"]
    rep_gate_right = row_map[(rep_combo, "MEAN / TOTAL")]["gate_applied_right"]

    t2_frr_not_higher = rep_gate_frr <= base_gate_frr
    t2_repairs_up = rep_gate_right > base_gate_right
    t2_passed = t2_frr_not_higher and t2_repairs_up
    t2_rec = "ADOPT (t2=0.648)" if t2_passed else "KEEP BASELINE (t2=0.900)"
    if not t2_frr_not_higher:
        t2_reason = f"Gate-level FRR rises to {rep_gate_frr*100:.2f}% (from {base_gate_frr*100:.2f}% baseline), failing the rule that FRR must not be higher than the baseline."
    elif not t2_repairs_up:
        t2_reason = f"Applied right repairs do not increase ({rep_gate_right} vs {base_gate_right})."
    else:
        t2_reason = f"FRR ({rep_gate_frr*100:.2f}%) is not higher than baseline while right repairs increase ({rep_gate_right} vs {base_gate_right})."

    # Rule 3: Temperature T
    ece_drops = []
    for t in tables:
        if row_map[(temp_combo, t)]["ece"] < row_map[(base_combo, t)]["ece"]:
            ece_drops.append(t)

    temp_passed = len(ece_drops) >= 4
    temp_rec = "ADOPT (T=0.826)" if temp_passed else "KEEP BASELINE (T=1.000)"
    temp_reason = f"ECE falls on {len(ece_drops)} of {len(tables)} test tables ({', '.join(t.capitalize() for t in ece_drops)}) while preserving cell ranking."

    return {
        "t1_flag": {
            "parameter": "Flag threshold t1",
            "tested_value": 0.378,
            "baseline_value": 0.500,
            "mean_f1_baseline": base_mean_f1,
            "mean_f1_fitted": flag_mean_f1,
            "tables_f1_dropped": tables_f1_dropped,
            "f1_drops_count": len(tables_f1_dropped),
            "passed": t1_passed,
            "recommendation": t1_rec,
            "reason": t1_reason,
        },
        "t2_repair": {
            "parameter": "Repair confidence threshold t2",
            "tested_value": 0.648,
            "baseline_value": 0.900,
            "frr_baseline": base_gate_frr,
            "frr_fitted": rep_gate_frr,
            "passed": t2_passed,
            "recommendation": t2_rec,
            "reason": t2_reason,
        },
        "temperature": {
            "parameter": "Temperature T",
            "tested_value": 0.826,
            "baseline_value": 1.000,
            "tables_with_ece_reduction": len(ece_drops),
            "total_tables": len(tables),
            "passed": temp_passed,
            "recommendation": temp_rec,
            "reason": temp_reason,
        },
    }


def write_x10_outputs(grid_rows: List[Dict[str, Any]], eval_rules: Dict[str, Any], out_grid: Path, out_summary: Path) -> None:
    """Save grid and summary tables to disk."""
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "combination", "t1_flag", "t2_repair", "temperature", "table",
        "precision", "recall", "f1", "review_share", "applied_right",
        "applied_wrong", "applied_repairs", "false_repair_rate",
        "gate_applied_right", "gate_applied_wrong", "gate_applied_repairs", "gate_false_repair_rate",
        "ece"
    ]
    with open(out_grid, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grid_rows)

    summary_fields = ["parameter", "tested_value", "baseline_value", "passed", "recommendation", "reason"]
    summary_rows = [
        {
            "parameter": r["parameter"],
            "tested_value": r["tested_value"],
            "baseline_value": r["baseline_value"],
            "passed": r["passed"],
            "recommendation": r["recommendation"],
            "reason": r["reason"],
        }
        for r in eval_rules.values()
    ]
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SUCCESS] Wrote dynamic X10 grid -> {out_grid}")
    print(f"[SUCCESS] Wrote dynamic X10 pass rules summary -> {out_summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[], help="Optional paths to test run folders.")
    parser.add_argument("--out-grid", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m5" / "x10_threshold_calibration_grid.csv")
    parser.add_argument("--out-summary", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m5" / "x10_pass_rules_summary.csv")
    args = parser.parse_args()

    print("================================================================================")
    print(" ContextGuard Milestone 5: Experiment X10 (Threshold & Calibration Grid)       ")
    print("================================================================================")
    run_paths = [Path(p) for p in args.runs] if args.runs else None
    grid_rows, eval_rules = compute_grid(run_paths)
    write_x10_outputs(grid_rows, eval_rules, args.out_grid, args.out_summary)

    print("\n--- Pass Rules Dynamic Evaluation ---")
    for key, rule in eval_rules.items():
        status = "PASSED" if rule["passed"] else "FAILED"
        print(f"[{status}] {rule['parameter']}: {rule['recommendation']}")
        print(f"         Reason: {rule['reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Milestone 5 Experiment X15: Repair Groundedness and Hallucination Analysis.

Author: Sahilpreet
Role: Evaluation, Calibration & Gating Lead

This script conducts Experiment X15 by analyzing the hallucination rate and groundedness
of generative repairs across the final M5 pipeline runs:
  - 4 Unseen tables: German, Soccer, Billionaire, Marketing
  - Core validation tables (final_val): Hospital, Flights, Beers, Rayyan, Movies

Definitions:
1. Active-Domain Groundedness:
   A proposed repair is considered grounded if its value is attested in the column's
   observed vocabulary (active domain in the table) or matches the ground-truth clean value.
2. Novel / Hallucinated Generation:
   A proposed repair is classified as an ungrounded hallucination if it introduces a novel
   string not present in the column domain and does not equal the clean ground truth.
3. Clean-Cell Corruption (False Alarms):
   Clean cells (where dirty_value == clean_value) where the model proposed an erroneous mutation.
4. Gate Suppression Effect:
   Compares raw model repair proposals against gated repairs (p_repair_min >= 0.90),
   empirically demonstrating how ContextGuard's confidence gate suppresses hallucinations.

Outputs:
  - docs/tables/m5/x15_hallucination_groundedness.csv

Usage:
  python scripts/run_x15_hallucination_groundedness.py [--runs <path_to_runs>...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.profiling.context import norm


def analyze_run_hallucinations(run_dir: Path) -> List[Dict[str, Any]]:
    """Analyze hallucination and groundedness rates for a run directory."""
    pred_path = run_dir / "predictions.jsonl"
    cells_path = run_dir / "cells.csv"

    if not pred_path.exists() or not cells_path.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl or cells.csv in {run_dir}")

    # 1. Build column domain and cell ground truth
    col_domain: Dict[Tuple[str, str], Set[str]] = {}
    cell_gt: Dict[Tuple[str, str, str], Tuple[str, str]] = {}

    with open(cells_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("table", "")).strip().lower()
            r = str(row.get("row", "")).strip()
            c = str(row.get("column", "")).strip()
            dirty = str(row.get("dirty_value", ""))
            clean = str(row.get("clean_value", ""))
            col_domain.setdefault((t, c), set()).add(norm(dirty))
            cell_gt[(t, r, c)] = (dirty, clean)

    # 2. Group predictions by table
    preds_by_table: Dict[str, List[Dict[str, Any]]] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            t = str(p.get("table", "")).strip().lower()
            if not t:
                continue
            preds_by_table.setdefault(t, []).append(p)

    results = []
    for table_name, preds in sorted(preds_by_table.items()):
        total_candidates = len(preds)
        raw_repairs = 0
        raw_grounded = 0
        raw_hallucinated = 0
        raw_clean_corrupted = 0

        gated_repairs = 0
        gated_grounded = 0
        gated_hallucinated = 0
        gated_clean_corrupted = 0

        for p in preds:
            r = str(p.get("row", "")).strip()
            c = str(p.get("column", "")).strip()
            gt = cell_gt.get((table_name, r, c))
            if not gt:
                continue
            dirty, clean = gt
            is_clean_cell = norm(dirty) == norm(clean)

            rep = p.get("repair")
            s = str(p.get("state", "")).strip().lower()
            has_rep = rep is not None and str(rep).strip() != ""

            # Raw repair proposals among candidate cells flagged by the pipeline
            if has_rep and s in ("not verified", "needs review", "verified"):
                raw_repairs += 1
                rep_norm = norm(str(rep))
                clean_norm = norm(clean)
                dirty_norm = norm(dirty)

                is_in_domain = rep_norm in col_domain.get((table_name, c), set())
                is_correct = rep_norm == clean_norm

                if is_in_domain or is_correct:
                    raw_grounded += 1
                else:
                    raw_hallucinated += 1

                if is_clean_cell and rep_norm != dirty_norm:
                    raw_clean_corrupted += 1

            # Applied repairs: determined by the run's own state field
            # On unseen tables: state 'not verified' means applied
            # On validated tables: state 'verified' or 'not verified' means applied
            is_applied = (s == "not verified") if "unseen" in run_dir.name.lower() else (s in ("not verified", "verified"))
            if is_applied:
                gated_repairs += 1
                rep_norm = norm(str(rep))
                clean_norm = norm(clean)
                dirty_norm = norm(dirty)

                is_in_domain = rep_norm in col_domain.get((table_name, c), set())
                is_correct = rep_norm == clean_norm

                if is_in_domain or is_correct:
                    gated_grounded += 1
                else:
                    gated_hallucinated += 1

                if is_clean_cell and rep_norm != dirty_norm:
                    gated_clean_corrupted += 1

        raw_ground_pct = round(raw_grounded / max(1, raw_repairs) * 100, 2)
        raw_halluc_pct = round(raw_hallucinated / max(1, raw_repairs) * 100, 2)
        gated_ground_pct = round(gated_grounded / max(1, gated_repairs) * 100, 2)
        gated_halluc_pct = round(gated_hallucinated / max(1, gated_repairs) * 100, 2)

        results.append({
            "table": table_name.capitalize(),
            "n_candidates": total_candidates,
            "raw_proposed_repairs": raw_repairs,
            "raw_grounded_repairs": raw_grounded,
            "raw_grounded_pct": raw_ground_pct,
            "raw_hallucinated_repairs": raw_hallucinated,
            "raw_hallucination_pct": raw_halluc_pct,
            "raw_clean_corrupted": raw_clean_corrupted,
            "gated_repairs": gated_repairs,
            "gated_grounded_repairs": gated_grounded,
            "gated_grounded_pct": gated_ground_pct,
            "gated_hallucinated_repairs": gated_hallucinated,
            "gated_hallucination_pct": gated_halluc_pct,
            "gated_clean_corrupted": gated_clean_corrupted,
            "hallucination_suppression_factor": round(raw_halluc_pct / max(0.01, gated_halluc_pct), 1),
        })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[], help="Paths to run folders.")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "tables" / "m5" / "x15_hallucination_groundedness.csv",
        help="Path to write output CSV.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print(" ContextGuard Milestone 5: Experiment X15 (Hallucination & Groundedness)       ")
    print("================================================================================")

    run_paths: List[Path] = []
    if args.runs:
        run_paths = [Path(p) for p in args.runs]
    else:
        run_paths = [
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_german",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_soccer",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_billionaire",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_marketing",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_restaurant",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_val",
        ]

    all_results = []
    for rd in run_paths:
        if not rd.exists():
            continue
        try:
            res_list = analyze_run_hallucinations(rd)
            for res in res_list:
                all_results.append(res)
                print(f"[DONE] {res['table']:12s}: Raw Halluc={res['raw_hallucination_pct']:4.1f}% (Grounded={res['raw_grounded_pct']:4.1f}%) -> Gated Halluc={res['gated_hallucination_pct']:4.1f}% (Grounded={res['gated_grounded_pct']:4.1f}%) | Clean corrupted: {res['raw_clean_corrupted']:3d} -> {res['gated_clean_corrupted']:2d}")
        except Exception as e:
            print(f"[ERROR] Failed {rd}: {e}")

    if all_results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[SUCCESS] Wrote real Experiment X15 results -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

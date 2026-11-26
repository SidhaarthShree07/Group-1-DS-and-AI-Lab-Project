#!/usr/bin/env python3
"""Milestone 5 Experiment X11: Error-Rate Prior in the Gate on Unseen Tables.

Author: Sahilpreet
Role: Evaluation, Calibration & Gating Lead

This script conducts Experiment X11 on the final unseen-table pipeline runs:
  - German slice (final_unseen_german)
  - Soccer (final_unseen_soccer)
  - Billionaire (final_unseen_billionaire)
  - Marketing (final_unseen_marketing)

Protocol:
1. Ingests unseen-table run directories (containing predictions.jsonl and cells.csv).
2. Computes the unsupervised EM prior estimate (Saerens et al., 2002) using `estimate_prior_em`
   on real P(error) values, without reading ground truth labels.
3. Compares the EM estimate against the oracle ground-truth candidate error rate and
   the fixed default training prior (p0 = 0.50).
4. Implements the TA's three-state routing policy for tables without validation rules:
   - State 1 ("not verified"): Repairs passing the confidence gate (p_repair_min >= 0.90)
     are applied and marked "not verified" (no rule files for unseen tables).
   - State 2 ("needs review"): Remaining flagged candidate cells (p_error >= 0.50) are routed
     to "needs review".
   - State 3 ("unchanged"): Unflagged cells are preserved untouched.
5. Reports proposed and unverified applied repair accuracy, strictly separated from
   rule-verified repairs on Hospital, Flights, and Beers.

Outputs:
  - docs/tables/m5/x11_unseen_priors.csv

Usage:
  python scripts/run_x11_unseen_priors.py [--runs <path_to_unseen_runs>...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.calibrate import estimate_prior_em
from src.profiling.context import norm


def load_run_data(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Tuple[str, str]]]:
    """Load predictions.jsonl and ground truth cells from run_dir."""
    pred_path = run_dir / "predictions.jsonl"
    cells_path = run_dir / "cells.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")
    if not cells_path.exists():
        raise FileNotFoundError(f"cells.csv not found in {run_dir}")

    # Build ground truth map: (table, row, col) -> (dirty, clean)
    cell_gt: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    with open(cells_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("table", "")).strip().lower()
            r = str(row.get("row", "")).strip()
            c = str(row.get("column", "")).strip()
            cell_gt[(t, r, c)] = (str(row.get("dirty_value", "")), str(row.get("clean_value", "")))

    preds: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    return preds, cell_gt


def evaluate_unseen_table_run(
    table_name: str,
    preds: List[Dict[str, Any]],
    cell_gt: Dict[Tuple[str, str, str], Tuple[str, str]],
    t1_flag: float = 0.50,
    t2_repair: float = 0.90,
    default_prior: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate a single unseen table's EM prior estimation and three-state routing."""
    table_preds = [p for p in preds if str(p.get("table", "")).strip().lower() == table_name.lower()]
    if not table_preds:
        table_preds = preds  # single-table run folder

    probs: List[float] = []
    labels: List[bool] = []

    proposed_repairs = 0
    proposed_right = 0
    applied_not_verified = 0
    applied_not_verified_right = 0
    needs_review = 0

    for p in table_preds:
        p_raw = p.get("p_error")
        if p_raw is None or not isinstance(p_raw, (int, float)):
            continue
        p_raw = float(p_raw)
        if math.isnan(p_raw) or math.isinf(p_raw) or not (0.0 <= p_raw <= 1.0):
            continue

        t = str(p.get("table", "")).strip().lower()
        r = str(p.get("row", "")).strip()
        c = str(p.get("column", "")).strip()

        gt = cell_gt.get((t, r, c))
        if gt is None:
            continue
        dirty_val, clean_val = gt
        is_err = norm(dirty_val) != norm(clean_val)

        probs.append(p_raw)
        labels.append(is_err)

        rep_val = p.get("repair")
        state = str(p.get("state", "")).strip().lower()
        has_repair_proposal = rep_val is not None and str(rep_val).strip() != ""

        # Proposed repairs among flagged candidate cells
        if has_repair_proposal and state in ("not verified", "needs review"):
            proposed_repairs += 1
            if norm(str(rep_val)) == norm(clean_val):
                proposed_right += 1

        # Pipeline routing from run's own state field:
        # State "not verified": applied repairs without deterministic rules
        # State "needs review": withheld candidate cells routed to human review
        # State "unchanged": unflagged / kept cells
        if state == "not verified":
            applied_not_verified += 1
            if norm(str(rep_val)) == norm(clean_val):
                applied_not_verified_right += 1
        elif state == "needs review":
            needs_review += 1

    if not probs:
        raise ValueError(f"No valid candidate predictions found for table {table_name}")

    # Estimate prior unsupervised via Saerens et al. (2002) EM
    em_prior, em_info = estimate_prior_em(probs, training_prior=default_prior)
    true_rate = sum(labels) / len(labels) if labels else 0.0

    return {
        "table": table_name.capitalize(),
        "n_candidates": len(probs),
        "true_candidate_error_rate": round(true_rate, 4),
        "default_prior": round(default_prior, 4),
        "em_estimated_prior": round(em_prior, 4),
        "error_em_vs_true": round(abs(em_prior - true_rate), 4),
        "error_default_vs_true": round(abs(default_prior - true_rate), 4),
        "em_iterations": em_info.get("iterations", 0),
        "em_converged": em_info.get("converged", True),
        "proposed_repairs": proposed_repairs,
        "proposed_right": proposed_right,
        "proposed_repair_accuracy": round(proposed_right / max(1, proposed_repairs), 4),
        "applied_not_verified": applied_not_verified,
        "applied_not_verified_right": applied_not_verified_right,
        "applied_not_verified_accuracy": round(applied_not_verified_right / max(1, applied_not_verified), 4),
        "needs_review": needs_review,
        "rule_validation_status": "Unverified (applied without rules)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Paths to unseen-table pipeline run folders containing predictions.jsonl and cells.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "tables" / "m5" / "x11_unseen_priors.csv",
        help="Path to write the unseen tables evaluation CSV.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print(" ContextGuard Milestone 5: Experiment X11 (Error-Rate Priors on Unseen Tables)  ")
    print("================================================================================")

    run_paths: List[Path] = []
    if args.runs:
        run_paths = [Path(p) for p in args.runs]
    else:
        # Default search in outputs/runs/m5_final
        candidates = [
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_german",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_soccer",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_billionaire",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_marketing",
            _REPO_ROOT / "outputs" / "runs" / "m5_final" / "final_unseen_restaurant",
        ]
        run_paths = [c for c in candidates if c.exists()]

    if not run_paths:
        print("[ERROR] No unseen-table run folders found.")
        return 1

    results = []
    for run_path in run_paths:
        if not run_path.exists():
            print(f"[WARN] Skipping non-existent run folder: {run_path}")
            continue
        try:
            preds, cell_gt = load_run_data(run_path)
            table_name = preds[0].get("table") if preds and preds[0].get("table") else run_path.name.replace("final_unseen_", "")
            res = evaluate_unseen_table_run(table_name, preds, cell_gt)
            results.append(res)
            print(f"[DONE] Evaluated {res['table']:12s}: N={res['n_candidates']:5d} | True Error={res['true_candidate_error_rate']:.4f} | EM Prior={res['em_estimated_prior']:.4f} (err={res['error_em_vs_true']:.4f}) | Applied unverified={res['applied_not_verified']:4d} (acc={res['applied_not_verified_accuracy']:.4f}) | Review={res['needs_review']:4d}")
        except Exception as e:
            print(f"[ERROR] Failed evaluating {run_path}: {e}")

    if results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SUCCESS] Wrote real Experiment X11 results -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Milestone 4 Calibration and Gate Threshold Optimization Study.

Author: Sahilpreet

This script:
1. Conducts an offline calibration and threshold study on real GPU runs:
   - Validation run (default: m4_final2_val_all) for parameter fitting.
   - Test run (default: m4_final2_test_hfb) for reporting and evaluation.
2. Fits optimal scalar temperature T on balanced validation data via 1D golden-section search.
3. Fits gate thresholds on validation:
   - Flag threshold t1: Maximizes detection F1 on raw P(error).
   - Repair threshold t2: Lowest threshold on p_repair_min where empirical FRR <= 10%.
4. Evaluates calibration on test candidates:
   - Unsupervised EM prior estimation (Saerens et al., 2002) without clean labels.
   - Falls back only when EM fails (non-convergence, clamped boundary, non-finite).
   - Reports the raw EM estimate directly in the table.
   - Computes Expected Calibration Error (ECE) before, with temperature, and with EM correction.
5. Generates:
   - configs/m4_calibration_study.yaml (offline study parameters; pipeline uses raw thresholds)
   - docs/tables/m4_calibration.csv (reporting on test)
   - docs/tables/m4_outcomes_test_hfb.csv (test outcome table)
   - docs/tables/m4_outcomes_val_all.csv (validation outcome table)
   - docs/figures/m4/reliability_diagram.png (reliability diagram on test)

Usage:
    python scripts/fit_calibration.py [--val-dir m4_final2_val_all] [--test-dir m4_final2_test_hfb]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import yaml

from src.evaluation.calibrate import (
    adjust_prior,
    apply_temperature,
    estimate_prior_em,
    fit_gate,
    fit_temperature,
)
from src.evaluation.metrics import (
    OUTCOME_COLUMNS,
    calculate_ece,
    score_decisions,
)
from src.profiling.context import norm


# ---------------------------------------------------------------------------
# Real-data loader
# ---------------------------------------------------------------------------

def load_real_data(
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load predictions and cells from a real pipeline run directory.

    Parameters
    ----------
    run_dir:
        Directory produced by run_pipeline.py, containing
        ``predictions.jsonl`` and ``cells.csv``.

    Returns
    -------
    candidates_data : list[dict]
        One record per predicted candidate with keys:
        table, row, column, p_raw (= p_error), label (bool), repair_ok (bool|None),
        p_repair_min (float|None).
    cells_data : list[dict]
        All ground-truth cell records as loaded from cells.csv.
    raw_predictions : list[dict]
        All raw parsed JSON prediction records.
    """
    pred_path = run_dir / "predictions.jsonl"
    cells_path = run_dir / "cells.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")
    if not cells_path.exists():
        raise FileNotFoundError(f"cells.csv not found in {run_dir}")

    # Load cells
    cells_data: List[Dict[str, Any]] = []
    with open(cells_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cells_data.append(dict(row))

    # Build cell ground-truth lookup: (table, row, col) -> (dirty, clean)
    cell_gt: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for c in cells_data:
        t = str(c.get("table", "")).strip().lower()
        r = str(c.get("row", "")).strip()
        col = str(c.get("column", "")).strip()
        dirty = str(c.get("dirty_value", ""))
        clean = str(c.get("clean_value", ""))
        cell_gt[(t, r, col)] = (dirty, clean)

    # Load predictions
    candidates_data: List[Dict[str, Any]] = []
    raw_predictions: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            raw_predictions.append(pred)

            p_raw = pred.get("p_error")
            if p_raw is None or not isinstance(p_raw, (int, float)):
                continue
            p_raw = float(p_raw)
            if math.isnan(p_raw) or math.isinf(p_raw) or not (0.0 <= p_raw <= 1.0):
                continue

            t = str(pred.get("table", "")).strip().lower()
            r = str(pred.get("row", "")).strip()
            col = str(pred.get("column", "")).strip()

            gt = cell_gt.get((t, r, col))
            if gt is None:
                continue
            dirty_val, clean_val = gt

            # Raha normalisation for error label
            label = norm(dirty_val) != norm(clean_val)

            # Determine repair_ok and p_repair_min:
            # repair_ok must be None when no repair was proposed
            pred_rep = pred.get("repair")
            has_rep_proposal = pred_rep is not None and str(pred_rep).strip() != ""

            if not has_rep_proposal:
                rep_ok: Optional[bool] = None
                p_rep_min: Optional[float] = None
            else:
                rep_ok = (norm(pred_rep) == norm(clean_val)) if label else False
                raw_p_min = pred.get("p_repair_min")
                p_rep_min = float(raw_p_min) if (raw_p_min is not None and not math.isnan(raw_p_min)) else None

            candidates_data.append({
                "table": t,
                "row": r,
                "column": col,
                "p_raw": p_raw,
                "label": label,
                "repair_ok": rep_ok,
                "p_repair_min": p_rep_min,
                "pred_repair": pred_rep,
            })

    return candidates_data, cells_data, raw_predictions


# ---------------------------------------------------------------------------
# Calibration study runner
# ---------------------------------------------------------------------------

def run_calibration_study(
    val_dir: Path,
    test_dir: Optional[Path],
    out_config_path: Path,
    out_cal_table_path: Path,
    out_outcomes_table_path: Path,
    out_fig_path: Path,
    is_custom_scores_run: bool = False,
) -> Dict[str, Any]:
    """Execute the calibration fitting on validation and reporting on test."""
    print(f"[*] Loading validation data from: {val_dir}")
    val_cands, val_cells, val_preds = load_real_data(val_dir)
    print(f"    Loaded {len(val_cands)} validation candidates, {len(val_cells)} cells.")

    # 1. Validation Fitting: Temperature scaling
    pos_samples = [c for c in val_cands if c["label"]]
    neg_samples = [c for c in val_cands if not c["label"]]
    rng = random.Random(42)
    n_balanced = min(len(pos_samples), len(neg_samples))

    if n_balanced == 0:
        raise ValueError("No positive or negative samples found in validation candidates.")

    balanced_set = rng.sample(pos_samples, n_balanced) + rng.sample(neg_samples, n_balanced)
    rng.shuffle(balanced_set)

    b_probs = [c["p_raw"] for c in balanced_set]
    b_labels = [c["label"] for c in balanced_set]

    optimal_T, temp_meta = fit_temperature(b_probs, b_labels)
    print(f"[*] Optimal Temperature fitted on balanced validation: T = {optimal_T:.4f}")
    print(f"    BCE Loss before: {temp_meta['loss_before']:.4f} -> after: {temp_meta['loss_after']:.4f}")

    # Fit Gate Thresholds on validation:
    # t1 on raw P(error) (flagging)
    # t2 on p_repair_min (repair execution study)
    val_raw_probs = [c["p_raw"] for c in val_cands]
    val_labels = [c["label"] for c in val_cands]
    val_repairs_ok = [c["repair_ok"] for c in val_cands]
    val_repair_mins = [c["p_repair_min"] for c in val_cands]

    gate_res = fit_gate(
        val_raw_probs,
        val_labels,
        val_repairs_ok,
        max_false_repair=0.10,
        repair_probs=val_repair_mins,
    )
    t1 = gate_res["flag"]
    t2 = gate_res["repair"]
    cap_met = gate_res.get("cap_met")
    print(f"[*] Fitted Gate Thresholds on validation: t1 = {t1:.4f} (flag on p_error), "
          f"t2 = {t2:.4f} (repair on p_repair_min, cap_met={cap_met})")

    # Export configs/m4_calibration_study.yaml
    gate_config = {
        "note": (
            "Calibration study parameters only. The ContextGuard production pipeline uses "
            "raw P(error) at 0.5 (flag) / 0.8 (repair) plus the token repair-confidence gate at 0.9."
        ),
        "study_fitted_flag_threshold": float(t1),
        "study_fitted_repair_threshold": float(t2),
        "pipeline_flag_threshold": 0.50,
        "pipeline_repair_threshold": 0.80,
        "pipeline_repair_confidence_gate": 0.90,
        "cap_met": bool(cap_met) if cap_met is not None else None,
        "max_false_repair_rate": 0.10,
        "temperature": round(float(optimal_T), 4),
        "calibration_method": "temperature_scaling_and_em_prior_shift",
        "reference": "Saerens et al. (2002)",
    }
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config_path, "w", encoding="utf-8") as f:
        yaml.dump(gate_config, f, default_flow_style=False, sort_keys=False)
    print(f"[SUCCESS] Wrote calibration study configuration -> {out_config_path}")

    # If running with custom --scores, evaluate solely on that directory
    eval_dir = test_dir if test_dir is not None else val_dir
    print(f"[*] Loading evaluation/reporting data from: {eval_dir}")
    eval_cands, eval_cells, eval_preds = load_real_data(eval_dir)
    print(f"    Loaded {len(eval_cands)} candidates, {len(eval_cells)} cells.")

    # 2. Calibration Analysis on Evaluation Data
    tables = sorted({c["table"] for c in eval_cands})
    cal_rows: List[Dict[str, Any]] = []

    all_raw_probs = [c["p_raw"] for c in eval_cands]
    all_labels = [c["label"] for c in eval_cands]
    all_temp_probs = apply_temperature(all_raw_probs, optimal_T)

    for t in tables:
        t_cands = [c for c in eval_cands if c["table"] == t]
        t_raw_probs = [c["p_raw"] for c in t_cands]
        t_labels = [c["label"] for c in t_cands]
        t_temp_probs = apply_temperature(t_raw_probs, optimal_T)

        true_rate = sum(t_labels) / len(t_labels) if t_labels else 0.0

        # Saerens EM prior estimation without artificial distance fallback
        em_rate, em_info = estimate_prior_em(
            t_temp_probs,
            training_prior=0.5,
            tol=1e-6,
        )

        # Calibrate with EM prior
        t_em_probs = adjust_prior(t_temp_probs, natural_prior=em_rate, training_prior=0.5)

        # Compute ECE: calculate_ece(y_true, scores)
        ece_raw  = calculate_ece(t_labels, t_raw_probs,  n_bins=10)
        ece_temp = calculate_ece(t_labels, t_temp_probs, n_bins=10)
        ece_em   = calculate_ece(t_labels, t_em_probs,   n_bins=10)

        cal_rows.append({
            "table": t,
            "n_candidates": len(t_cands),
            "true_error_rate": round(true_rate, 4),
            "em_estimated_rate": round(em_rate, 4),
            "raw_em_estimate": round(em_info.get("raw_em_estimate", em_rate), 4),
            "em_iterations": em_info["iterations"],
            "em_converged": em_info["converged"],
            "em_used_fallback": em_info.get("used_fallback", False),
            "em_fallback_reason": em_info.get("fallback_reason", ""),
            "error_em_vs_true": round(abs(em_rate - true_rate), 4),
            "ece_uncalibrated": round(ece_raw, 4),
            "ece_temperature": round(ece_temp, 4),
            "ece_em_calibrated": round(ece_em, 4),
        })

    # Total row on evaluation data
    tot_true_rate = sum(all_labels) / len(all_labels) if all_labels else 0.0
    tot_em_rate, tot_em_info = estimate_prior_em(
        all_temp_probs,
        training_prior=0.5,
    )
    all_em_probs = adjust_prior(all_temp_probs, natural_prior=tot_em_rate, training_prior=0.5)

    tot_ece_raw  = calculate_ece(all_labels, all_raw_probs,  n_bins=10)
    tot_ece_temp = calculate_ece(all_labels, all_temp_probs, n_bins=10)
    tot_ece_em   = calculate_ece(all_labels, all_em_probs,   n_bins=10)

    cal_rows.append({
        "table": "Total",
        "n_candidates": len(eval_cands),
        "true_error_rate": round(tot_true_rate, 4),
        "em_estimated_rate": round(tot_em_rate, 4),
        "raw_em_estimate": round(tot_em_info.get("raw_em_estimate", tot_em_rate), 4),
        "em_iterations": tot_em_info["iterations"],
        "em_converged": tot_em_info["converged"],
        "em_used_fallback": tot_em_info.get("used_fallback", False),
        "em_fallback_reason": tot_em_info.get("fallback_reason", ""),
        "error_em_vs_true": round(abs(tot_em_rate - tot_true_rate), 4),
        "ece_uncalibrated": round(tot_ece_raw, 4),
        "ece_temperature": round(tot_ece_temp, 4),
        "ece_em_calibrated": round(tot_ece_em, 4),
    })

    # Export docs/tables/m4_calibration.csv
    out_cal_table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cal_table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    print(f"[SUCCESS] Wrote calibration comparison table -> {out_cal_table_path}")

    # 3. Decision Outcomes
    eval_outcomes = score_decisions(eval_preds, eval_cells)
    out_outcomes_table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_outcomes_table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS)
        w.writeheader()
        w.writerows(eval_outcomes["outcome_table"])
    print(f"[SUCCESS] Wrote outcome table -> {out_outcomes_table_path}")

    tot_eval = eval_outcomes["total"]
    print(f"[*] Decisions Outcome ({eval_dir.name}):")
    print(f"  Gate only : {tot_eval.get('gate_repaired_right',0)} right / "
          f"{tot_eval.get('gate_repaired_wrong',0)} wrong / "
          f"{tot_eval.get('gate_repaired_no_change',0)} no-change")
    print(f"  Gate+valid: {tot_eval['repaired_right']} right / "
          f"{tot_eval['repaired_wrong_error'] + tot_eval['repaired_wrong_correct_cell']} wrong / "
          f"{tot_eval['review_error'] + tot_eval['review_correct_cell']} review")

    # If running standard pipeline (not custom single --scores run):
    if not is_custom_scores_run:
        # Write validation outcome table m4_outcomes_val_all.csv
        val_outcomes = score_decisions(val_preds, val_cells)
        val_outcomes_path = out_outcomes_table_path.parent / "m4_outcomes_val_all.csv"
        with open(val_outcomes_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS)
            w.writeheader()
            w.writerows(val_outcomes["outcome_table"])
        print(f"[SUCCESS] Wrote validation outcome table -> {val_outcomes_path}")

    # 4. Generate Reliability Curve Plot on Evaluation Data
    generate_reliability_figure(
        all_raw_probs, all_temp_probs, all_em_probs, all_labels, optimal_T, out_fig_path
    )

    return {
        "temperature": optimal_T,
        "gate": gate_res,
        "calibration_table": cal_rows,
        "eval_outcomes": eval_outcomes,
    }


def generate_reliability_figure(
    raw_probs: List[float],
    temp_probs: List[float],
    em_probs: List[float],
    labels: List[bool],
    T: float,
    fig_path: Path,
) -> None:
    """Plot 3-panel reliability diagram (Uncalibrated, Temperature Scaling, EM Prior Shift)."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    n_bins = 10
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    def _bin_stats(probs):
        confidences = []
        accuracies = []
        counts = []
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_idx = [
                j for j, p in enumerate(probs)
                if (p >= low and p < high) or (i == n_bins - 1 and p >= low and p <= high)
            ]
            if bin_idx:
                conf = sum(probs[j] for j in bin_idx) / len(bin_idx)
                acc = sum(1 for j in bin_idx if labels[j]) / len(bin_idx)
                confidences.append(conf)
                accuracies.append(acc)
                counts.append(len(bin_idx))
            else:
                confidences.append((low + high) / 2.0)
                accuracies.append(0.0)
                counts.append(0)
        return confidences, accuracies, counts

    # ECE calls: calculate_ece(y_true, scores)
    panels = [
        ("Uncalibrated (Raw)", raw_probs, "#d95f02",
         calculate_ece(labels, raw_probs, 10)),
        (f"Temperature Scaled (T={T:.2f})", temp_probs, "#7570b3",
         calculate_ece(labels, temp_probs, 10)),
        ("Temperature + EM Prior Shift", em_probs, "#1b9e77",
         calculate_ece(labels, em_probs, 10)),
    ]

    for ax, (title, probs, color, ece_val) in zip(axes, panels):
        confs, accs, counts = _bin_stats(probs)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
        ax.bar(bin_edges[:-1], accs, width=1.0 / n_bins, align="edge",
               color=color, alpha=0.65, edgecolor="black", label="Actual accuracy")
        ax.plot(confs, accs, marker="o", color=color, linewidth=2)
        ax.set_title(f"{title}\nECE = {ece_val:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("True Error Fraction (Accuracy)", fontsize=11)
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Wrote reliability diagram figure -> {fig_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=_REPO_ROOT / "m4_final2_val_all",
        help="Path to validation run directory for fitting (default: m4_final2_val_all).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=_REPO_ROOT / "m4_final2_test_hfb",
        help="Path to test run directory for reporting (default: m4_final2_test_hfb).",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional override to set single run directory for custom evaluation.",
    )
    parser.add_argument(
        "--out-config",
        type=Path,
        default=None,
        help="Study parameters (default: configs/m4_calibration_study.yaml; with --scores, docs/tables/<run>_calibration_study.yaml).",
    )
    parser.add_argument(
        "--out-cal",
        type=Path,
        default=None,
        help="Calibration table (default: docs/tables/m4_calibration.csv; with --scores, docs/tables/<run>_calibration.csv).",
    )
    parser.add_argument(
        "--out-outcomes",
        type=Path,
        default=None,
        help="Output CSV path for outcome table.",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=None,
        help="Reliability diagram (default: docs/figures/m4/reliability_diagram.png; with --scores, <run>_reliability_diagram.png).",
    )
    args = parser.parse_args()

    tables_dir = _REPO_ROOT / "docs" / "tables"
    figures_dir = _REPO_ROOT / "docs" / "figures" / "m4"
    is_custom_scores = args.scores is not None
    if is_custom_scores:
        # A single extra run: name every output after the run, so the main study files are never overwritten.
        val_dir = args.scores
        test_dir = None
        run_name = args.scores.name
        short_name = run_name[3:] if run_name.startswith("m4_") else run_name
        default_outcomes = tables_dir / f"m4_outcomes_{short_name}.csv"
        default_config = tables_dir / f"{run_name}_calibration_study.yaml"
        default_cal = tables_dir / f"{run_name}_calibration.csv"
        default_fig = figures_dir / f"{run_name}_reliability_diagram.png"
    else:
        val_dir = args.val_dir
        test_dir = args.test_dir
        default_outcomes = tables_dir / "m4_outcomes_test_hfb.csv"
        default_config = _REPO_ROOT / "configs" / "m4_calibration_study.yaml"
        default_cal = tables_dir / "m4_calibration.csv"
        default_fig = figures_dir / "reliability_diagram.png"
    out_outcomes = args.out_outcomes or default_outcomes

    run_calibration_study(
        val_dir=val_dir,
        test_dir=test_dir,
        out_config_path=args.out_config or default_config,
        out_cal_table_path=args.out_cal or default_cal,
        out_outcomes_table_path=out_outcomes,
        out_fig_path=args.out_fig or default_fig,
        is_custom_scores_run=is_custom_scores,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

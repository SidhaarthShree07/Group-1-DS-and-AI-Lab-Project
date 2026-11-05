#!/usr/bin/env python3
"""Evaluation metrics engine for ContextGuard.

Author: Sahilpreet

Computes all required Milestone 3 metrics for cell-level error detection,
typing, repair, safety, and calibration:
- Error detection: TP, FP, TN, FN, Precision, Recall, F1 (per-table and macro-averaged)
- Error-type macro-F1 on correctly detected errors across missing, pattern, rule, outlier
- Repair accuracy on all true error cells
- False-repair rate on all executed repairs
- False-alarm rate and false alarms per 1,000 cells
- PR-AUC (Average Precision) using strictly valid p_error confidence scores
- Precision-coverage curve across valid confidence thresholds
- Expected Calibration Error (ECE) across 10 equal-width bins on valid scores
- Candidate recall and pass-through rate with candidate deduplication
- Empirical bootstrap confidence intervals (B=1,000, 95% CI)
- Mandatory D18 reporting grid (table x error_type) export to CSV

Fixed Integration Contract:
score(predictions, cells, candidates=None, output_csv_path=None, ...) -> Dict[str, Any]
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.parse import normalise_repair
from src.profiling.context import norm

CANONICAL_ERROR_TYPES = ("missing", "pattern", "rule", "outlier")

TYPE_MAP = {
    "missing": "missing",
    "missing_value": "missing",
    "pattern": "pattern",
    "pattern_violation": "pattern",
    "rule": "rule",
    "rule_violation": "rule",
    "outlier": "outlier",
    "outliers": "outlier",
    "none": "none",
    "": "none",
}

REPORTING_GRID_COLUMNS = [
    "table",
    "error_type",
    "n_errors",
    "is_low_sample",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "successful_repairs",
    "repair_accuracy",
    "executed_repairs",
    "incorrect_repairs",
    "false_repair_rate",
]


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division returning default when denominator is zero."""
    return numerator / denominator if denominator != 0.0 else default


def _canonicalise_type(raw_type: Any) -> str:
    """Map raw or legacy error type strings to canonical categories."""
    if raw_type is None:
        return "none"
    s = str(raw_type).strip().lower()
    return TYPE_MAP.get(s, s)


def _load_records(
    data: Union[Sequence[Dict[str, Any]], str, Path, Any]
) -> List[Dict[str, Any]]:
    """Load records from in-memory sequence of dicts, pandas DataFrame, or file path."""
    if hasattr(data, "to_dict"):
        # pandas DataFrame
        return data.to_dict(orient="records")

    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() == ".jsonl":
            records = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif p.suffix.lower() in (".csv", ".tsv"):
            delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")

    if isinstance(data, Iterable):
        return list(data)

    raise TypeError(f"Cannot load records from type {type(data)}")


def _make_cell_key(r: Dict[str, Any], include_table: bool) -> Tuple[Any, ...]:
    """Create a stable matching tuple for a cell record."""
    row = int(r["row"])
    col = str(r["column"])
    if include_table and "table" in r and r["table"] is not None:
        return (str(r["table"]).strip().lower(), row, col)
    return (row, col)


def calculate_detection_prf(
    tp: int, fp: int, tn: int, fn: int
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 with safe division."""
    p = _safe_div(tp, tp + fp, default=0.0)
    r = _safe_div(tp, tp + fn, default=0.0)
    f1 = _safe_div(2.0 * p * r, p + r, default=0.0)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def calculate_error_type_f1(
    tp_items: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate error-type macro-F1 among correctly detected errors (TPs).

    Macro-F1 convention:
    - per_type: computes precision, recall, F1, and support for all 4 canonical types.
    - macro_f1: unweighted average across categories that have positive support in TPs.
      If no true error of a category was detected, that category is not penalized
      as 0.0 unless the model produced false positives for it.
    - macro_f1_strict_4: unweighted average strictly across all 4 categories (empty=0.0).
    """
    per_type = {}
    evaluated_f1s = []

    for etype in CANONICAL_ERROR_TYPES:
        tp_t = sum(
            1
            for item in tp_items
            if item["gold_type"] == etype and item["pred_type"] == etype
        )
        fp_t = sum(
            1
            for item in tp_items
            if item["gold_type"] != etype and item["pred_type"] == etype
        )
        fn_t = sum(
            1
            for item in tp_items
            if item["gold_type"] == etype and item["pred_type"] != etype
        )
        support = sum(1 for item in tp_items if item["gold_type"] == etype)

        p = _safe_div(tp_t, tp_t + fp_t, default=0.0)
        r = _safe_div(tp_t, tp_t + fn_t, default=0.0)
        f1 = _safe_div(2.0 * p * r, p + r, default=0.0)

        per_type[etype] = {
            "tp": tp_t,
            "fp": fp_t,
            "fn": fn_t,
            "support": support,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        # Include in macro average if present in ground-truth TPs or if model made false predictions
        if support > 0 or fp_t > 0:
            evaluated_f1s.append(f1)

    macro_f1 = (sum(evaluated_f1s) / len(evaluated_f1s)) if evaluated_f1s else 0.0
    macro_f1_strict_4 = sum(per_type[e]["f1"] for e in CANONICAL_ERROR_TYPES) / 4.0

    return {
        "macro_f1": macro_f1,
        "macro_f1_strict_4": macro_f1_strict_4,
        "per_type": per_type,
    }


def calculate_pr_auc(
    y_true: Sequence[bool], scores: Sequence[float]
) -> float:
    """Calculate Area Under the Precision-Recall Curve (Average Precision).

    Evaluated strictly over cells with valid confidence scores.
    Returns 0.0 if population has no positive ground-truth or no negative ground-truth.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    n_pos = sum(1 for y in y_true if y)
    if n_pos == 0 or n_pos == n:
        return 0.0

    paired = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    i = 0
    while i < n:
        curr_score = paired[i][0]
        while i < n and paired[i][0] == curr_score:
            if paired[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1

        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall

    return ap


def calculate_precision_coverage(
    y_true: Sequence[bool], scores: Sequence[float]
) -> List[Dict[str, float]]:
    """Compute precision-coverage points across descending valid confidence thresholds."""
    n = len(y_true)
    if n == 0:
        return []

    paired = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    unique_thresholds = sorted(set(scores), reverse=True)

    curve = []
    tp = 0
    count = 0
    idx = 0

    for tau in unique_thresholds:
        while idx < n and paired[idx][0] >= tau:
            count += 1
            if paired[idx][1]:
                tp += 1
            idx += 1

        coverage = count / n
        precision = (tp / count) if count > 0 else 1.0
        curve.append({
            "threshold": float(tau),
            "coverage": float(coverage),
            "precision": float(precision),
        })

    return curve


def calculate_ece(
    y_true: Sequence[bool], scores: Sequence[float], n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error across equal-width confidence bins in [0, 1].

    Evaluated strictly over cells with valid confidence scores.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    ece = 0.0
    bin_step = 1.0 / n_bins

    for b in range(n_bins):
        low = b * bin_step
        high = (b + 1) * bin_step

        if b == 0:
            bin_indices = [i for i, s in enumerate(scores) if low <= s <= high]
        else:
            bin_indices = [i for i, s in enumerate(scores) if low < s <= high]

        k = len(bin_indices)
        if k > 0:
            mean_conf = sum(scores[i] for i in bin_indices) / k
            acc = sum(1 for i in bin_indices if y_true[i]) / k
            ece += (k / n) * abs(mean_conf - acc)

    return ece


def generate_reporting_grid(
    eval_items: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate the D18 reporting grid: table x error_type with N and low-sample markers.

    Deterministic ordering: table alphabetical, then error_type in canonical order.
    """
    if not eval_items:
        return []

    tables = sorted({x["table"] for x in eval_items})
    grid = []

    for t in tables:
        t_items = [x for x in eval_items if x["table"] == t]
        for etype in CANONICAL_ERROR_TYPES:
            # Ground-truth errors of this error type in this table
            n_errors = sum(
                1 for x in t_items if x["gold_is_error"] and x["gold_type"] == etype
            )
            is_low_sample = n_errors < 30

            # True Positives: true error of this type, flagged, and typed as this type
            tp = sum(
                1
                for x in t_items
                if x["gold_is_error"]
                and x["gold_type"] == etype
                and x["pred_flag"]
                and x["pred_type"] == etype
            )
            # False Negatives: true error of this type, but not flagged or typed differently
            fn = sum(
                1
                for x in t_items
                if x["gold_is_error"]
                and x["gold_type"] == etype
                and not (x["pred_flag"] and x["pred_type"] == etype)
            )
            # False Positives: flagged as this error type, but not actually this error type
            fp = sum(
                1
                for x in t_items
                if x["pred_flag"]
                and x["pred_type"] == etype
                and not (x["gold_is_error"] and x["gold_type"] == etype)
            )

            p = _safe_div(tp, tp + fp, default=0.0)
            r = _safe_div(tp, tp + fn, default=0.0)
            f1 = _safe_div(2.0 * p * r, p + r, default=0.0)

            # Repair metrics for this specific table x error type
            successful_repairs = sum(
                1
                for x in t_items
                if x["gold_is_error"]
                and x["gold_type"] == etype
                and x["pred_flag"]
                and x["pred_repair"] is not None
                and norm(x["pred_repair"]) == norm(x["clean_value"])
            )
            repair_accuracy = _safe_div(successful_repairs, n_errors, default=0.0)

            # Executed repairs on cells predicted as this error type
            executed_repairs = sum(
                1
                for x in t_items
                if x["pred_flag"]
                and x["pred_type"] == etype
                and x["pred_repair"] is not None
            )
            incorrect_repairs = sum(
                1
                for x in t_items
                if x["pred_flag"]
                and x["pred_type"] == etype
                and x["pred_repair"] is not None
                and norm(x["pred_repair"]) != norm(x["clean_value"])
            )
            false_repair_rate = _safe_div(
                incorrect_repairs, executed_repairs, default=0.0
            )

            grid.append({
                "table": t,
                "error_type": etype,
                "n_errors": n_errors,
                "is_low_sample": is_low_sample,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "successful_repairs": successful_repairs,
                "repair_accuracy": round(repair_accuracy, 4),
                "executed_repairs": executed_repairs,
                "incorrect_repairs": incorrect_repairs,
                "false_repair_rate": round(false_repair_rate, 4),
            })

    return grid


def export_reporting_grid(
    grid_rows: Sequence[Dict[str, Any]], output_path: Union[str, Path]
) -> Path:
    """Write the reporting grid to CSV according to the D18 specification."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORTING_GRID_COLUMNS)
        writer.writeheader()
        for row in grid_rows:
            writer.writerow(row)

    return p


def calculate_metrics(
    eval_items: Sequence[Dict[str, Any]],
    candidate_map: Optional[Dict[Tuple[Any, ...], bool]] = None,
) -> Dict[str, Any]:
    """Calculate the complete set of core metrics over evaluated cell items."""
    n_total = len(eval_items)
    if n_total == 0:
        return {
            "n_cells": 0,
            "detection": calculate_detection_prf(0, 0, 0, 0),
            "macro_f1_tables": 0.0,
            "per_table": {},
            "error_types": calculate_error_type_f1([]),
            "repair": {"accuracy": 0.0, "successful_repairs": 0, "total_errors": 0},
            "false_repair": {"rate": 0.0, "incorrect_repairs": 0, "executed_repairs": 0},
            "false_alarm": {
                "rate": 0.0,
                "per_1000_clean_cells": 0.0,
                "per_1000_total_cells": 0.0,
                "fp": 0,
                "clean_cells": 0,
            },
            "confidence_eval": {"valid_cells": 0, "total_cells": 0, "coverage_pct": 0.0},
            "pr_auc": 0.0,
            "precision_coverage": [],
            "ece": 0.0,
            "candidates": None,
            "reporting_grid": [],
        }

    # 1. Detection counts
    tp = sum(1 for x in eval_items if x["gold_is_error"] and x["pred_flag"])
    fp = sum(1 for x in eval_items if not x["gold_is_error"] and x["pred_flag"])
    tn = sum(1 for x in eval_items if not x["gold_is_error"] and not x["pred_flag"])
    fn = sum(1 for x in eval_items if x["gold_is_error"] and not x["pred_flag"])
    det = calculate_detection_prf(tp, fp, tn, fn)

    # 2. Per-table detection and Macro-F1
    tables = sorted({x["table"] for x in eval_items})
    per_table = {}
    f1s = []

    for t in tables:
        t_items = [x for x in eval_items if x["table"] == t]
        t_tp = sum(1 for x in t_items if x["gold_is_error"] and x["pred_flag"])
        t_fp = sum(1 for x in t_items if not x["gold_is_error"] and x["pred_flag"])
        t_tn = sum(1 for x in t_items if not x["gold_is_error"] and not x["pred_flag"])
        t_fn = sum(1 for x in t_items if x["gold_is_error"] and not x["pred_flag"])

        # Table with no positive ground-truth errors
        if t_tp + t_fn == 0:
            # If table is completely clean:
            # No false alarms -> perfect detection (F1=1.0)
            # One or more false alarms -> F1=0.0
            t_f1 = 1.0 if t_fp == 0 else 0.0
            t_p = 1.0 if t_fp == 0 else 0.0
            t_r = 1.0
            t_prf = {
                "tp": t_tp,
                "fp": t_fp,
                "tn": t_tn,
                "fn": t_fn,
                "precision": t_p,
                "recall": t_r,
                "f1": t_f1,
            }
        else:
            t_prf = calculate_detection_prf(t_tp, t_fp, t_tn, t_fn)

        per_table[t] = t_prf
        f1s.append(t_prf["f1"])

    macro_f1 = (sum(f1s) / len(f1s)) if f1s else 0.0

    # 3. Error-type classification on TPs (M3 specification: correctly detected errors)
    tp_items = [x for x in eval_items if x["gold_is_error"] and x["pred_flag"]]
    error_types = calculate_error_type_f1(tp_items)

    # Also compute end-to-end error typing across all cells for full-table visibility
    all_cells_et = calculate_error_type_f1(eval_items)
    error_types["all_cells_macro_f1"] = all_cells_et["macro_f1"]
    error_types["all_cells_per_type"] = all_cells_et["per_type"]

    # 4. Repair accuracy (denominator = all true error cells)
    total_errors = sum(1 for x in eval_items if x["gold_is_error"])
    successful_repairs = sum(
        1
        for x in eval_items
        if x["gold_is_error"]
        and x["pred_flag"]
        and x["pred_repair"] is not None
        and norm(x["pred_repair"]) == norm(x["clean_value"])
    )
    repair_acc = _safe_div(successful_repairs, total_errors, default=0.0)

    # 5. False-repair rate (denominator = all executed repairs)
    executed_repairs = sum(
        1 for x in eval_items if x["pred_flag"] and x["pred_repair"] is not None
    )
    incorrect_repairs = sum(
        1
        for x in eval_items
        if x["pred_flag"]
        and x["pred_repair"] is not None
        and norm(x["pred_repair"]) != norm(x["clean_value"])
    )
    false_repair_rate = _safe_div(incorrect_repairs, executed_repairs, default=0.0)

    # 6. False-alarm rate (clean cells)
    clean_cells = tn + fp
    far = _safe_div(fp, clean_cells, default=0.0)
    far_1000_clean = far * 1000.0
    far_1000_total = (fp / n_total) * 1000.0

    # 7. PR-AUC, Precision-Coverage, and ECE
    # Strictly evaluate on cells with valid confidence scores; never infer confidence from is_error
    valid_conf_items = [x for x in eval_items if x["has_valid_p"]]
    n_valid_conf = len(valid_conf_items)
    conf_coverage_pct = _safe_div(n_valid_conf, n_total, default=0.0) * 100.0

    confidence_eval = {
        "valid_cells": n_valid_conf,
        "total_cells": n_total,
        "coverage_pct": round(conf_coverage_pct, 2),
    }

    if n_valid_conf > 0:
        y_valid = [x["gold_is_error"] for x in valid_conf_items]
        s_valid = [x["p_error"] for x in valid_conf_items]
        pr_auc = calculate_pr_auc(y_valid, s_valid)
        prec_cov = calculate_precision_coverage(y_valid, s_valid)
        ece = calculate_ece(y_valid, s_valid, n_bins=10)
    else:
        pr_auc = 0.0
        prec_cov = []
        ece = 0.0

    # 8. Candidate stage metrics (with deduplication)
    cand_metrics = None
    if candidate_map is not None:
        cand_flags = [candidate_map.get(x["cell_key"], False) for x in eval_items]
        cand_recall_num = sum(
            1 for x, c in zip(eval_items, cand_flags) if x["gold_is_error"] and c
        )
        cand_recall = _safe_div(cand_recall_num, total_errors, default=0.0)
        pass_through = sum(1 for c in cand_flags if c) / n_total
        cand_metrics = {
            "recall": cand_recall,
            "pass_through": pass_through,
            "candidates_count": sum(1 for c in cand_flags if c),
            "true_errors_in_candidates": cand_recall_num,
        }

    # 9. Reporting grid
    reporting_grid = generate_reporting_grid(eval_items)

    return {
        "n_cells": n_total,
        "detection": det,
        "macro_f1_tables": macro_f1,
        "per_table": per_table,
        "error_types": error_types,
        "repair": {
            "accuracy": repair_acc,
            "successful_repairs": successful_repairs,
            "total_errors": total_errors,
        },
        "false_repair": {
            "rate": false_repair_rate,
            "incorrect_repairs": incorrect_repairs,
            "executed_repairs": executed_repairs,
        },
        "false_alarm": {
            "rate": far,
            "per_1000_clean_cells": far_1000_clean,
            "per_1000_total_cells": far_1000_total,
            "fp": fp,
            "clean_cells": clean_cells,
        },
        "confidence_eval": confidence_eval,
        "pr_auc": pr_auc,
        "precision_coverage": prec_cov,
        "ece": ece,
        "candidates": cand_metrics,
        "reporting_grid": reporting_grid,
    }


def calculate_bootstrap_ci(
    eval_items: Sequence[Dict[str, Any]],
    candidate_map: Optional[Dict[Tuple[Any, ...], bool]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals over cell samples."""
    n = len(eval_items)
    if n == 0 or n_bootstrap <= 0:
        return {}

    rng = random.Random(seed)
    sampled_metrics: Dict[str, List[float]] = {
        "detection_precision": [],
        "detection_recall": [],
        "detection_f1": [],
        "macro_f1_tables": [],
        "repair_accuracy": [],
        "false_repair_rate": [],
        "false_alarm_rate": [],
        "pr_auc": [],
        "ece": [],
    }

    for _ in range(n_bootstrap):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        sample_items = [eval_items[i] for i in sample_indices]
        m = calculate_metrics(sample_items, candidate_map=candidate_map)

        sampled_metrics["detection_precision"].append(m["detection"]["precision"])
        sampled_metrics["detection_recall"].append(m["detection"]["recall"])
        sampled_metrics["detection_f1"].append(m["detection"]["f1"])
        sampled_metrics["macro_f1_tables"].append(m["macro_f1_tables"])
        sampled_metrics["repair_accuracy"].append(m["repair"]["accuracy"])
        sampled_metrics["false_repair_rate"].append(m["false_repair"]["rate"])
        sampled_metrics["false_alarm_rate"].append(m["false_alarm"]["rate"])
        sampled_metrics["pr_auc"].append(m["pr_auc"])
        sampled_metrics["ece"].append(m["ece"])

    ci_results = {}
    low_idx = int(math.floor((alpha / 2.0) * n_bootstrap))
    high_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1
    high_idx = min(high_idx, n_bootstrap - 1)

    for k, vals in sampled_metrics.items():
        vals.sort()
        ci_results[k] = (float(vals[low_idx]), float(vals[high_idx]))

    return ci_results


def score(
    predictions: Union[Sequence[Dict[str, Any]], str, Path, Any],
    cells: Union[Sequence[Dict[str, Any]], str, Path, Any],
    candidates: Optional[Union[Sequence[Dict[str, Any]], str, Path, Any]] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    n_bootstrap: int = 1000,
    compute_bootstrap: bool = True,
) -> Dict[str, Any]:
    """Score model predictions against ground-truth cells according to M3 contracts.

    Args:
        predictions: Iterable of prediction dicts or path to predictions.jsonl.
        cells: Iterable of ground-truth cell dicts or path to cells.csv.
        candidates: Optional iterable of candidate dicts or path to candidates.csv.
        output_csv_path: Optional path to export metrics_by_table_type.csv.
        seed: Random seed for bootstrap intervals.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        compute_bootstrap: Whether to compute bootstrap intervals.

    Returns:
        A structured summary dictionary containing overall detection,
        per-table metrics, error-type macro-F1, repair accuracy,
        false-repair rate, false-alarm rate, PR-AUC, precision-coverage
        curve, ECE, candidate metrics, reporting grid, and bootstrap intervals.
    """
    raw_cells = _load_records(cells)
    raw_preds = _load_records(predictions)

    has_table_in_cells = any("table" in c and c["table"] is not None for c in raw_cells)
    has_table_in_preds = any("table" in p and p["table"] is not None for p in raw_preds)
    include_table = has_table_in_cells and has_table_in_preds
    # Split cell files have no `table` column. Scoring several tables against such files would match cells of
    # different tables that share (row, column); refuse instead of mixing them up silently.
    pred_tables = {str(p["table"]).strip().lower() for p in raw_preds if p.get("table") is not None}
    if not has_table_in_cells and len(pred_tables) > 1:
        raise ValueError("predictions cover several tables but the cells have no `table` field; add it or score "
                         "one table at a time")

    pred_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for p in raw_preds:
        k = _make_cell_key(p, include_table)
        pred_map[k] = p

    # Candidate deduplication: union candidate flags across duplicate entries
    cand_map = None
    if candidates is not None:
        raw_cands = _load_records(candidates)
        cand_map = {}
        for c in raw_cands:
            k = _make_cell_key(c, include_table)
            flag = c.get("is_candidate", False)
            if isinstance(flag, str):
                flag_bool = flag.strip().lower() in ("true", "1", "yes")
            else:
                flag_bool = bool(flag)
            cand_map[k] = cand_map.get(k, False) or flag_bool

    eval_items: List[Dict[str, Any]] = []
    inconsistencies = 0
    format_artifacts_skipped = 0

    for c in raw_cells:
        # Format-only differences (3 against 3.0) are not labels (Milestone 1 decision, M2 Section 6.2): they are
        # neither errors nor correct cells and are left out of every metric.
        if str(c.get("error_type", "")).strip() == "format_artifact":
            format_artifacts_skipped += 1
            continue
        k = _make_cell_key(c, include_table)
        d_val = c.get("dirty_value", "")
        c_val = c.get("clean_value", "")

        # Canonical Raha normalisation decides ground-truth error status
        norm_d = norm(d_val)
        norm_c = norm(c_val)
        gold_is_error = norm_d != norm_c

        # Audit check against supplied is_error column
        supplied_is_error_raw = c.get("is_error")
        if supplied_is_error_raw is not None:
            if isinstance(supplied_is_error_raw, str):
                supplied_bool = supplied_is_error_raw.strip().lower() == "true"
            else:
                supplied_bool = bool(supplied_is_error_raw)
            if supplied_bool != gold_is_error:
                inconsistencies += 1

        gold_type = _canonicalise_type(c.get("error_type", "none")) if gold_is_error else "none"
        table_name = str(c.get("table", "default")).strip()

        pred = pred_map.get(k)
        if pred is None:
            # Missing prediction counts as not flagged
            pred_flag = False
            pred_type = "none"
            pred_repair = None
            valid_json = False
            has_valid_p = False
            p_error = None
        else:
            valid_json = bool(pred.get("valid_json", False))
            is_err = bool(pred.get("is_error", False))
            pred_flag = valid_json and is_err
            pred_type = _canonicalise_type(pred.get("error_type", "none")) if pred_flag else "none"
            pred_repair = pred.get("repair") if pred_flag else None

            # Strictly validate confidence score p_error; NEVER infer from is_error
            p_err_raw = pred.get("p_error")
            has_valid_p = False
            p_error = None

            if p_err_raw is not None and p_err_raw != "":
                try:
                    p_val = float(p_err_raw)
                    # Must be finite and within valid probability interval [0, 1]
                    if not (math.isnan(p_val) or math.isinf(p_val)) and 0.0 <= p_val <= 1.0:
                        p_error = p_val
                        has_valid_p = True
                except (ValueError, TypeError):
                    pass

        eval_items.append({
            "cell_key": k,
            "table": table_name,
            "dirty_value": d_val,
            "clean_value": c_val,
            "gold_is_error": gold_is_error,
            "gold_type": gold_type,
            "pred_flag": pred_flag,
            "pred_type": pred_type,
            "pred_repair": pred_repair,
            "valid_json": valid_json,
            "has_valid_p": has_valid_p,
            "p_error": p_error,
        })

    metrics = calculate_metrics(eval_items, candidate_map=cand_map)
    metrics["ground_truth_label_inconsistencies"] = inconsistencies
    metrics["format_artifacts_skipped"] = format_artifacts_skipped

    # Export reporting grid if requested
    if output_csv_path is not None:
        out_p = export_reporting_grid(metrics["reporting_grid"], output_csv_path)
        metrics["reporting_grid_path"] = str(out_p)

    if compute_bootstrap and len(eval_items) > 0 and n_bootstrap > 0:
        ci = calculate_bootstrap_ci(
            eval_items,
            candidate_map=cand_map,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        metrics["bootstrap_95ci"] = ci
    else:
        metrics["bootstrap_95ci"] = None

    return metrics

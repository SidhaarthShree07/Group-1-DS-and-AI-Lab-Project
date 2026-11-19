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


OUTCOME_COLUMNS = [
    "table",
    "gate_repaired_right",
    "gate_repaired_wrong",
    "gate_repaired_no_change",
    "repaired_right",
    "repaired_wrong_error",
    "repaired_wrong_correct_cell",
    "review_error",
    "review_correct_cell",
    "kept_missed_error",
    "kept_correct",
    "total_cells",
    "total_errors",
    "precision",
    "recall",
    "f1",
    "repair_accuracy",
    "false_repair_rate",
    "false_repair_rate_gate",
    "false_alarms_per_1000",
    "review_share",
]


def _parse_decision(raw_decision: str) -> str:
    """Parse a raw decision string to exactly one of 'repair', 'review', 'keep'.

    Decision taxonomy (order matters: check for 'review' and 'flag' before 'repair'
    so that 'review (repair rejected by validation)' and
    'review (repair cannot be validated)' are classified as 'review', not 'repair').

    - 'repair'   : raw_decision == 'repair' exactly (after strip/lower)
    - 'review'   : raw_decision starts with 'review' or contains 'flag'
    - 'keep'     : everything else
    """
    d = raw_decision.strip().lower()
    if d.startswith("review") or "flag" in d:
        return "review"
    if d == "repair":
        return "repair"
    return "keep"


def score_decisions(
    predictions: Union[Sequence[Dict[str, Any]], str, Path, Any],
    cells: Union[Sequence[Dict[str, Any]], str, Path, Any],
    output_csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Score gate-level decisions against ground-truth cells (Milestone 4).

    Produces two sets of decision columns:
      * ``gate_decision`` — the M4 gate output *before* validation
        (uses the ``gate_decision`` field when present, falls back to inferring
        from the model's raw ``decision`` field treating any non-keep/non-review
        value as the gate's raw repair signal).
      * ``decision``      — the pipeline's final decision *after* validation
        (uses the ``decision`` field, which may be
        'repair', 'keep', 'flag for review',
        'review (repair rejected by validation)', or
        'review (repair cannot be validated)').

    Decision parsing rules (applied to both fields independently):
    - 'repair'  : decision string is exactly 'repair' (strip/lower).
    - 'review'  : starts with 'review' OR contains 'flag' — this correctly
                  classifies 'flag for review', 'review (repair rejected …)',
                  and 'review (repair cannot be validated)' as review, not repair.
    - 'keep'    : everything else.

    No-Op Rule: A proposed repair equal to the dirty value (under norm) counts
    as 'not an error' / kept rather than a repair — applied independently to
    both gate_decision and decision columns.

    7 outcome buckets (gate + validation combined ``decision``):
        1. repaired right (true error correctly repaired)
        2. repaired wrong (error) (true error changed to incorrect value)
        3. repaired wrong (correct cell) (correct cell erroneously changed)
        4. review (error) (true error routed to human review)
        5. review (correct cell) (correct cell routed to human review)
        6. kept (missed error) (true error unflagged / kept)
        7. kept (correct) (correct cell kept untouched)

    Gate-only columns (gate_repaired_right, gate_repaired_wrong, gate_repaired_no_change)
    are computed separately from the ``gate_decision`` field, as a cross-check.

    Args:
        predictions: Predictions list/DataFrame/path with 'decision' and optional
            'gate_decision', 'repair' fields.
        cells: Ground-truth cells list/DataFrame/path.
        output_csv_path: Optional path to export outcomes.csv.

    Returns:
        Dictionary with outcome_table (list of dicts), per-table metrics, total metrics,
        and model_proposals metrics.
    """
    raw_cells = _load_records(cells)
    raw_preds = _load_records(predictions)

    has_table_in_cells = any("table" in c and c["table"] is not None for c in raw_cells)
    has_table_in_preds = any("table" in p and p["table"] is not None for p in raw_preds)
    include_table = has_table_in_cells and has_table_in_preds

    pred_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for p in raw_preds:
        k = _make_cell_key(p, include_table)
        pred_map[k] = p

    # Per-table counters
    table_names: List[str] = sorted(
        list({str(c.get("table", "default")).strip() for c in raw_cells})
    )
    counts: Dict[str, Dict[str, int]] = {
        t: {
            "repaired_right": 0,
            "repaired_wrong_error": 0,
            "repaired_wrong_correct_cell": 0,
            "review_error": 0,
            "review_correct_cell": 0,
            "kept_missed_error": 0,
            "kept_correct": 0,
            "total_cells": 0,
            "total_errors": 0,
            # Gate-only counters (before validation layer)
            "gate_repaired_right": 0,
            "gate_repaired_wrong": 0,
            "gate_repaired_no_change": 0,
        }
        for t in table_names
    }

    # Model proposal counters (model level, before gate)
    model_props: Dict[str, Dict[str, int]] = {
        t: {
            "proposals_total": 0,
            "proposals_repairs": 0,
            "proposals_no_op": 0,
            "successful_repairs": 0,
            "false_repairs": 0,
        }
        for t in table_names
    }

    format_artifacts_skipped = 0

    for c in raw_cells:
        if str(c.get("error_type", "")).strip() == "format_artifact":
            format_artifacts_skipped += 1
            continue

        k = _make_cell_key(c, include_table)
        d_val = c.get("dirty_value", "")
        c_val = c.get("clean_value", "")
        t_name = str(c.get("table", "default")).strip()
        cnt = counts[t_name]
        m_cnt = model_props[t_name]

        norm_d = norm(d_val)
        norm_c = norm(c_val)
        gold_is_error = norm_d != norm_c

        cnt["total_cells"] += 1
        if gold_is_error:
            cnt["total_errors"] += 1

        pred = pred_map.get(k)
        decision = "keep"
        gate_decision_resolved = "keep"
        pred_repair = None
        has_repair_proposal = False

        if pred is not None:
            # Model-level proposal tally
            is_err_pred = bool(pred.get("is_error", False)) and bool(pred.get("valid_json", False))
            raw_rep = pred.get("repair")
            if is_err_pred:
                m_cnt["proposals_total"] += 1
                if raw_rep is not None and str(raw_rep).strip() != "":
                    if norm(raw_rep) == norm_d:
                        m_cnt["proposals_no_op"] += 1
                    else:
                        m_cnt["proposals_repairs"] += 1
                        if norm(raw_rep) == norm_c and gold_is_error:
                            m_cnt["successful_repairs"] += 1
                        else:
                            m_cnt["false_repairs"] += 1

            raw_decision = str(pred.get("decision", "keep")).strip().lower()
            pred_repair = pred.get("repair")

            # --- Gate decision (before validation) ---
            # Prefer the explicit gate_decision field written by run_pipeline.py.
            # Fall back to inferring from raw model decision: treat anything that
            # the validation layer could have changed ('review (repair …)') back to
            # 'repair' as the gate having originally said 'repair'.
            raw_gate_str = pred.get("gate_decision")
            if raw_gate_str is not None:
                gate_raw = str(raw_gate_str).strip().lower()
            else:
                # Infer gate decision: validation converts 'repair' → 'review (repair …)'
                # so any review-of-repair string was originally a gate repair.
                gate_raw = raw_decision
                if gate_raw.startswith("review (repair"):
                    gate_raw = "repair"

            # Apply no-op rule to gate decision
            if _parse_decision(gate_raw) == "repair":
                if pred_repair is not None and norm(pred_repair) == norm_d:
                    gate_decision_resolved = "keep"  # no-op → keep
                    cnt["gate_repaired_no_change"] += 1
                else:
                    gate_decision_resolved = "repair"
            elif _parse_decision(gate_raw) == "review":
                gate_decision_resolved = "review"
            else:
                gate_decision_resolved = "keep"

            # --- Final decision (gate + validation) ---
            # Apply _parse_decision which correctly classifies:
            #   'repair'                                 → 'repair'
            #   'review (repair rejected by validation)' → 'review'
            #   'review (repair cannot be validated)'    → 'review'
            #   'flag for review'                        → 'review'
            #   'keep'                                   → 'keep'
            if _parse_decision(raw_decision) == "repair":
                if pred_repair is not None and norm(pred_repair) == norm_d:
                    decision = "keep"  # no-op → keep
                else:
                    decision = "repair"
            elif _parse_decision(raw_decision) == "review":
                decision = "review"
            else:
                decision = "keep"

        # Categorize cell by final combined decision (gate + validation) → 7 bins
        if decision == "repair":
            if not gold_is_error:
                cnt["repaired_wrong_correct_cell"] += 1
            else:
                if pred_repair is not None and norm(pred_repair) == norm_c:
                    cnt["repaired_right"] += 1
                else:
                    cnt["repaired_wrong_error"] += 1
        elif decision == "review":
            if gold_is_error:
                cnt["review_error"] += 1
            else:
                cnt["review_correct_cell"] += 1
        else:  # decision == "keep"
            if gold_is_error:
                cnt["kept_missed_error"] += 1
            else:
                cnt["kept_correct"] += 1

        # Gate-only 3-bin tally (before validation layer)
        if gate_decision_resolved == "repair":
            if gold_is_error:
                if pred_repair is not None and norm(pred_repair) == norm_c:
                    cnt["gate_repaired_right"] += 1
                else:
                    cnt["gate_repaired_wrong"] += 1
            else:
                cnt["gate_repaired_wrong"] += 1  # false alarm on clean cell
        # (review and keep at gate level are not separately tallied here)

    # Tally metrics per table and total
    def _compute_metrics_row(t_name: str, c: Dict[str, int]) -> Dict[str, Any]:
        rr = c["repaired_right"]
        rwe = c["repaired_wrong_error"]
        rwcc = c["repaired_wrong_correct_cell"]
        re = c["review_error"]
        rcc = c["review_correct_cell"]
        kme = c["kept_missed_error"]
        kc = c["kept_correct"]
        tot_cells = c["total_cells"]
        tot_errors = c["total_errors"]
        tot_correct = tot_cells - tot_errors
        g_rr = c.get("gate_repaired_right", 0)
        g_rw = c.get("gate_repaired_wrong", 0)
        g_nc = c.get("gate_repaired_no_change", 0)

        # Sanity check conservation
        sum_7 = rr + rwe + rwcc + re + rcc + kme + kc
        if sum_7 != tot_cells:
            raise ValueError(f"Outcome conservation failed for {t_name}: {sum_7} != {tot_cells}")

        tp = rr + rwe + re
        fp = rwcc + rcc
        fn = kme
        executed_repairs = rr + rwe + rwcc
        total_review = re + rcc

        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * prec * rec, prec + rec)
        repair_acc = _safe_div(rr, tot_errors)
        false_repair_rate = _safe_div(rwe + rwcc, executed_repairs)
        # Gate-only false-repair rate (before validation)
        gate_exec = g_rr + g_rw
        false_repair_rate_gate = _safe_div(g_rw, gate_exec)
        false_alarms_per_1000 = _safe_div(fp, tot_correct) * 1000.0
        review_share = _safe_div(total_review, tot_cells)

        return {
            "table": t_name,
            "gate_repaired_right": g_rr,
            "gate_repaired_wrong": g_rw,
            "gate_repaired_no_change": g_nc,
            "repaired_right": rr,
            "repaired_wrong_error": rwe,
            "repaired_wrong_correct_cell": rwcc,
            "review_error": re,
            "review_correct_cell": rcc,
            "kept_missed_error": kme,
            "kept_correct": kc,
            "total_cells": tot_cells,
            "total_errors": tot_errors,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "repair_accuracy": round(repair_acc, 4),
            "false_repair_rate": round(false_repair_rate, 4),
            "false_repair_rate_gate": round(false_repair_rate_gate, 4),
            "false_alarms_per_1000": round(false_alarms_per_1000, 2),
            "review_share": round(review_share, 4),
        }

    outcome_rows: List[Dict[str, Any]] = []
    for t in table_names:
        outcome_rows.append(_compute_metrics_row(t, counts[t]))

    # Total row
    total_counts = {
        k: sum(counts[t][k] for t in table_names)
        for k in counts[table_names[0]].keys()
    }
    total_row = _compute_metrics_row("Total", total_counts)
    outcome_rows.append(total_row)

    # Model proposals summary row
    model_proposals_summary = {
        "proposals_total": sum(model_props[t]["proposals_total"] for t in table_names),
        "proposals_repairs": sum(model_props[t]["proposals_repairs"] for t in table_names),
        "proposals_no_op": sum(model_props[t]["proposals_no_op"] for t in table_names),
        "successful_repairs": sum(model_props[t]["successful_repairs"] for t in table_names),
        "false_repairs": sum(model_props[t]["false_repairs"] for t in table_names),
        "by_table": model_props,
    }

    # Export to CSV if requested
    if output_csv_path is not None:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS)
            w.writeheader()
            for r in outcome_rows:
                w.writerow(r)

    return {
        "outcome_table": outcome_rows,
        "by_table": {r["table"]: r for r in outcome_rows if r["table"] != "Total"},
        "total": total_row,
        "model_proposals": model_proposals_summary,
        "format_artifacts_skipped": format_artifacts_skipped,
    }


def score_flags(
    flagged_cells: Union[Iterable[Tuple[Any, ...]], Iterable[Dict[str, Any]], Any],
    cells: Union[Sequence[Dict[str, Any]], str, Path, Any],
    exclude_rows: Optional[Union[Sequence[int], Sequence[Tuple[str, int]]]] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Score baseline error flags with standardized normalisation and exclusions (Milestone 4).

    Ensures that baselines (HoloClean, Raha, Baran, dBoost) use the identical evaluation
    protocol as ContextGuard:
        - Raha canonical normalization: norm(dirty) != norm(clean)
        - Format artifacts excluded from scoring
        - Optional exclude_rows (e.g. 20 labeled tuples for semi-supervised baselines)

    Args:
        flagged_cells: Sequence/set of flagged cells. Can be tuples (row, col) or
            (table, row, col), or dicts containing 'row', 'column', and optional 'table'.
        cells: Ground-truth cells list/DataFrame/path.
        exclude_rows: Set/sequence of row indices (or (table, row) tuples) to exclude.
        output_csv_path: Optional path to export summary CSV.

    Returns:
        Dictionary shaped like score()["detection"] with keys:
        n_cells, n_errors, tp, fp, tn, fn, precision, recall, f1,
        false_alarms_per_1000, per_table, excluded_rows_count.
    """
    raw_cells = _load_records(cells)
    has_table_in_cells = any("table" in c and c["table"] is not None for c in raw_cells)
    # Determine distinct table names for per-table scoped exclusions
    _all_table_names = sorted(list({str(c.get("table", "default")).strip() for c in raw_cells}))
    _multi_table = len(_all_table_names) > 1

    # Process exclude_rows
    # An integer-only entry must be scoped per-table when multiple tables are present
    # to avoid silently dropping that row in EVERY table.
    excluded_int_rows: set = set()   # bare ints → only effective when single-table
    excluded_table_rows: set = set() # (table, row) tuples → always effective
    if exclude_rows is not None:
        for ex in exclude_rows:
            if isinstance(ex, tuple) and len(ex) == 2:
                excluded_table_rows.add((str(ex[0]).strip().lower(), int(ex[1])))
            else:
                excluded_int_rows.add(int(ex))

    if _multi_table and excluded_int_rows:
        raise ValueError(
            f"Bare integer in exclude_rows ({sorted(excluded_int_rows)}) is ambiguous when multiple tables "
            f"are present ({_all_table_names}). Pass (table_name, row_idx) tuples instead."
        )

    def _is_excluded(t_name: str, row_int: int) -> bool:
        """Return True if this (table, row) should be excluded from scoring."""
        if (t_name.lower(), row_int) in excluded_table_rows:
            return True
        if not _multi_table and row_int in excluded_int_rows:
            return True
        return False

    # Process flagged cells into set of lookup keys
    flagged_set = set()
    if hasattr(flagged_cells, "to_dict"):
        flagged_cells = flagged_cells.to_dict(orient="records")

    for fc in flagged_cells:
        if isinstance(fc, dict):
            r = int(fc["row"])
            c = str(fc["column"])
            t = str(fc.get("table", "")).strip().lower() if has_table_in_cells and "table" in fc else None
            flagged_set.add((t, r, c) if t is not None else (r, c))
        elif isinstance(fc, (list, tuple)):
            if len(fc) == 3:
                flagged_set.add((str(fc[0]).strip().lower(), int(fc[1]), str(fc[2])))
            elif len(fc) == 2:
                flagged_set.add((int(fc[0]), str(fc[1])))
            else:
                raise ValueError(f"Invalid flagged cell tuple: {fc}")
        else:
            raise TypeError(f"Unsupported flagged cell type: {type(fc)}")

    table_names = sorted(list({str(c.get("table", "default")).strip() for c in raw_cells}))
    per_table_stats: Dict[str, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_cells": 0, "n_errors": 0}
        for t in table_names
    }

    excluded_count = 0
    format_artifacts_skipped = 0

    for c in raw_cells:
        if str(c.get("error_type", "")).strip() == "format_artifact":
            format_artifacts_skipped += 1
            continue

        r = int(c["row"])
        col = str(c["column"])
        t_name = str(c.get("table", "default")).strip()

        # Check exclusion
        if _is_excluded(t_name, r):
            excluded_count += 1
            continue

        norm_d = norm(c.get("dirty_value", ""))
        norm_c = norm(c.get("clean_value", ""))
        gold_is_error = norm_d != norm_c

        # Check if flagged
        is_flagged = (r, col) in flagged_set or (t_name.lower(), r, col) in flagged_set

        st = per_table_stats[t_name]
        st["n_cells"] += 1
        if gold_is_error:
            st["n_errors"] += 1

        if is_flagged and gold_is_error:
            st["tp"] += 1
        elif is_flagged and not gold_is_error:
            st["fp"] += 1
        elif not is_flagged and not gold_is_error:
            st["tn"] += 1
        elif not is_flagged and gold_is_error:
            st["fn"] += 1

    per_table_results: Dict[str, Dict[str, Any]] = {}
    total_tp = sum(s["tp"] for s in per_table_stats.values())
    total_fp = sum(s["fp"] for s in per_table_stats.values())
    total_tn = sum(s["tn"] for s in per_table_stats.values())
    total_fn = sum(s["fn"] for s in per_table_stats.values())
    total_cells = sum(s["n_cells"] for s in per_table_stats.values())
    total_errors = sum(s["n_errors"] for s in per_table_stats.values())
    total_correct = total_cells - total_errors

    for t, s in per_table_stats.items():
        prec = _safe_div(s["tp"], s["tp"] + s["fp"])
        rec = _safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = _safe_div(2.0 * prec * rec, prec + rec)
        fa1k = _safe_div(s["fp"], s["n_cells"] - s["n_errors"]) * 1000.0
        per_table_results[t] = {
            "tp": s["tp"],
            "fp": s["fp"],
            "tn": s["tn"],
            "fn": s["fn"],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_1000": round(fa1k, 2),
            "n_cells": s["n_cells"],
            "n_errors": s["n_errors"],
        }

    overall_prec = _safe_div(total_tp, total_tp + total_fp)
    overall_rec = _safe_div(total_tp, total_tp + total_fn)
    overall_f1 = _safe_div(2.0 * overall_prec * overall_rec, overall_prec + overall_rec)
    overall_fa1k = _safe_div(total_fp, total_correct) * 1000.0
    macro_f1 = sum(r["f1"] for r in per_table_results.values()) / len(per_table_results) if per_table_results else 0.0

    result = {
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "precision": round(overall_prec, 4),
        "recall": round(overall_rec, 4),
        "f1": round(overall_f1, 4),
        "macro_f1_tables": round(macro_f1, 4),
        "false_alarms_per_1000": round(overall_fa1k, 2),
        "n_cells": total_cells,
        "n_errors": total_errors,
        "per_table": per_table_results,
        "excluded_rows_count": excluded_count,
        "format_artifacts_skipped": format_artifacts_skipped,
    }

    if output_csv_path is not None:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"table": t, **res} for t, res in per_table_results.items()]
        rows.append({"table": "Total", **{k: v for k, v in result.items() if k not in ("per_table",)}})
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return result


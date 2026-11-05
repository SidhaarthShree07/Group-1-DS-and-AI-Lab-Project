"""Unit tests for the evaluation metrics engine (src/evaluation/metrics.py).

Author: Sahilpreet

Comprehensive tests verifying all Milestone 3 requirements:
1. Perfect error detection
2. All predictions unflagged
3. False positives and false negatives
4. Invalid JSON counts as not flagged
5. Missing prediction counts as not flagged
6. Macro-F1 across multiple tables (including clean tables with/without FPs)
7. Error-type classification and macro-F1 (with zero-support & incorrect predictions)
8. Repair accuracy with correct and incorrect repairs
9. Repair accuracy when an error has no repair
10. False-repair rate on erroneous cells
11. False-repair rate when a clean cell is corrupted
12. False-alarm rate
13. False alarms per 1,000 cells
14. PR-AUC with known confidence scores
15. PR-AUC edge cases
16. ECE with hand-calculable probabilities
17. Missing, non-finite, out-of-bounds confidence scores (no inferring from is_error)
18. Candidate recall
19. Candidate pass-through rate and candidate deduplication
20. Bootstrap reproducibility with a fixed seed
21. Empty input and zero-denominator behaviour
22. D18 reporting grid (multiple tables, error types, N<30, N>=30, CSV export, ordering)
"""
from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Gracefully mock optional heavy dependencies if run in an environment without them
for _mod in ("numpy", "pandas"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from src.evaluation.metrics import (
    REPORTING_GRID_COLUMNS,
    calculate_detection_prf,
    calculate_ece,
    calculate_error_type_f1,
    calculate_pr_auc,
    calculate_precision_coverage,
    export_reporting_grid,
    generate_reporting_grid,
    score,
)


class TestMetricsEngine(unittest.TestCase):
    """Test suite for evaluation metrics engine."""

    def test_01_perfect_error_detection(self):
        """Test case 1: Perfect detection (all errors flagged, no false alarms)."""
        cells = [
            {"row": 0, "column": "a", "dirty_value": "x", "clean_value": "y"},
            {"row": 1, "column": "a", "dirty_value": "b", "clean_value": "b"},
        ]
        preds = [
            {"row": 0, "column": "a", "valid_json": True, "is_error": True},
            {"row": 1, "column": "a", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        det = res["detection"]
        self.assertEqual(det["tp"], 1)
        self.assertEqual(det["fp"], 0)
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["fn"], 0)
        self.assertEqual(det["precision"], 1.0)
        self.assertEqual(det["recall"], 1.0)
        self.assertEqual(det["f1"], 1.0)

    def test_02_all_predictions_unflagged(self):
        """Test case 2: Model flags nothing (P=0.0, R=0.0, F1=0.0)."""
        cells = [
            {"row": 0, "column": "a", "dirty_value": "x", "clean_value": "y"},
            {"row": 1, "column": "a", "dirty_value": "b", "clean_value": "b"},
        ]
        preds = [
            {"row": 0, "column": "a", "valid_json": True, "is_error": False},
            {"row": 1, "column": "a", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        det = res["detection"]
        self.assertEqual(det["tp"], 0)
        self.assertEqual(det["fp"], 0)
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["fn"], 1)
        self.assertEqual(det["precision"], 0.0)
        self.assertEqual(det["recall"], 0.0)
        self.assertEqual(det["f1"], 0.0)

    def test_03_false_positives_and_false_negatives(self):
        """Test case 3: Hand-calculable mix of TP, FP, TN, FN."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "bad1", "clean_value": "good1"},  # TP
            {"row": 1, "column": "c", "dirty_value": "bad2", "clean_value": "good2"},  # FN
            {"row": 2, "column": "c", "dirty_value": "ok1", "clean_value": "ok1"},    # FP
            {"row": 3, "column": "c", "dirty_value": "ok2", "clean_value": "ok2"},    # TN
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True},
            {"row": 1, "column": "c", "valid_json": True, "is_error": False},
            {"row": 2, "column": "c", "valid_json": True, "is_error": True},
            {"row": 3, "column": "c", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        det = res["detection"]
        self.assertEqual(det["tp"], 1)
        self.assertEqual(det["fp"], 1)
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["fn"], 1)
        self.assertAlmostEqual(det["precision"], 0.5)
        self.assertAlmostEqual(det["recall"], 0.5)
        self.assertAlmostEqual(det["f1"], 0.5)

    def test_04_invalid_json_counts_as_not_flagged(self):
        """Test case 4: valid_json=False must count strictly as not flagged."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "bad", "clean_value": "good"},
            {"row": 1, "column": "c", "dirty_value": "clean", "clean_value": "clean"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": False, "is_error": True},
            {"row": 1, "column": "c", "valid_json": False, "is_error": True},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        det = res["detection"]
        self.assertEqual(det["tp"], 0)
        self.assertEqual(det["fn"], 1)
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["fp"], 0)

    def test_05_missing_prediction_counts_as_not_flagged(self):
        """Test case 5: Missing prediction in predictions list counts as not flagged."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "bad", "clean_value": "good"},
            {"row": 1, "column": "c", "dirty_value": "ok", "clean_value": "ok"},
        ]
        preds = [
            {"row": 1, "column": "c", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        det = res["detection"]
        self.assertEqual(det["tp"], 0)
        self.assertEqual(det["fn"], 1)
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["fp"], 0)

    def test_06_macro_f1_across_multiple_tables(self):
        """Test case 6: Macro-F1 across tables (Table A: F1=1.0, Table B: F1=0.0 -> Macro=0.5)."""
        cells = [
            {"table": "table_a", "row": 0, "column": "col", "dirty_value": "e1", "clean_value": "c1"},
            {"table": "table_b", "row": 0, "column": "col", "dirty_value": "e2", "clean_value": "c2"},
        ]
        preds = [
            {"table": "table_a", "row": 0, "column": "col", "valid_json": True, "is_error": True},
            {"table": "table_b", "row": 0, "column": "col", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["per_table"]["table_a"]["f1"], 1.0)
        self.assertEqual(res["per_table"]["table_b"]["f1"], 0.0)
        self.assertEqual(res["macro_f1_tables"], 0.5)

    def test_06b_macro_f1_clean_tables(self):
        """Test case 6b: Tables with no true errors (clean tables with and without false alarms)."""
        cells = [
            # clean_table_1: clean, no false alarm -> F1 = 1.0
            {"table": "clean_1", "row": 0, "column": "col", "dirty_value": "v", "clean_value": "v"},
            # clean_table_2: clean, with false alarm -> F1 = 0.0
            {"table": "clean_2", "row": 0, "column": "col", "dirty_value": "v", "clean_value": "v"},
        ]
        preds = [
            {"table": "clean_1", "row": 0, "column": "col", "valid_json": True, "is_error": False},
            {"table": "clean_2", "row": 0, "column": "col", "valid_json": True, "is_error": True},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["per_table"]["clean_1"]["f1"], 1.0)
        self.assertEqual(res["per_table"]["clean_2"]["f1"], 0.0)
        self.assertEqual(res["macro_f1_tables"], 0.5)

    def test_07_error_type_classification_and_macro_f1(self):
        """Test case 7: Error-type macro-F1 among correctly detected errors (TPs)."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "e1", "clean_value": "c1", "error_type": "pattern"},
            {"row": 1, "column": "c", "dirty_value": "e2", "clean_value": "c2", "error_type": "rule"},
            {"row": 2, "column": "c", "dirty_value": "e3", "clean_value": "c3", "error_type": "outlier"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "error_type": "pattern"},
            {"row": 1, "column": "c", "valid_json": True, "is_error": True, "error_type": "rule"},
            {"row": 2, "column": "c", "valid_json": True, "is_error": True, "error_type": "pattern"},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        et = res["error_types"]
        self.assertAlmostEqual(et["per_type"]["pattern"]["f1"], 2.0 / 3.0)
        self.assertEqual(et["per_type"]["rule"]["f1"], 1.0)
        self.assertEqual(et["per_type"]["outlier"]["f1"], 0.0)
        self.assertAlmostEqual(et["macro_f1"], (2.0 / 3.0 + 1.0 + 0.0) / 3.0)
        # Category missing has 0 support and 0 FP -> support=0, f1=0.0
        self.assertEqual(et["per_type"]["missing"]["support"], 0)
        self.assertEqual(et["per_type"]["missing"]["f1"], 0.0)

    def test_07b_error_type_zero_support_and_incorrect_prediction(self):
        """Test case 7b: Category with zero ground-truth support but incorrect model predictions."""
        # 1 TP error of type pattern; model mistakenly predicts type "outlier" (zero true outliers)
        tp_items = [
            {"gold_type": "pattern", "pred_type": "outlier"},
        ]
        res = calculate_error_type_f1(tp_items)
        # Outlier: TP=0, FP=1, FN=0 -> P=0, R=0, F1=0.0
        self.assertEqual(res["per_type"]["outlier"]["support"], 0)
        self.assertEqual(res["per_type"]["outlier"]["fp"], 1)
        self.assertEqual(res["per_type"]["outlier"]["f1"], 0.0)
        # Pattern: TP=0, FP=0, FN=1 -> F1=0.0
        self.assertEqual(res["per_type"]["pattern"]["support"], 1)
        self.assertEqual(res["per_type"]["pattern"]["f1"], 0.0)

    def test_07c_error_type_comprehensive_audit(self):
        """Test case 7c: Comprehensive audit of error-type macro-F1 covering all 6 required conditions.

        1. Correct detection and correct error type (cell 0: missing -> missing)
        2. Correct detection but incorrect error type (cell 1: missing -> pattern)
        3. Incorrect detection / not flagged (cell 2: missing -> unflagged)
        4. False positive on clean cell (cell 3: clean -> flagged pattern)
        5. Zero-support error categories (rule has support = 0)
        6. Mixture of canonical error types (missing, pattern, outlier, clean)
        """
        cells = [
            # 1. Correct detection & correct type
            {"table": "test_tab", "row": 0, "column": "col", "dirty_value": "m1_d", "clean_value": "m1_c", "error_type": "missing"},
            # 2. Correct detection & wrong type
            {"table": "test_tab", "row": 1, "column": "col", "dirty_value": "m2_d", "clean_value": "m2_c", "error_type": "missing"},
            # 3. Missed error (not flagged)
            {"table": "test_tab", "row": 2, "column": "col", "dirty_value": "m3_d", "clean_value": "m3_c", "error_type": "missing"},
            # 4. Clean cell with false positive
            {"table": "test_tab", "row": 3, "column": "col", "dirty_value": "c1", "clean_value": "c1", "error_type": "none"},
            # 6. Mixture: Outlier (correct detection & correct type)
            {"table": "test_tab", "row": 4, "column": "col", "dirty_value": "999", "clean_value": "10", "error_type": "outlier"},
            # 6. Mixture: Pattern (correct detection & correct type)
            {"table": "test_tab", "row": 5, "column": "col", "dirty_value": "p_d", "clean_value": "p_c", "error_type": "pattern"},
        ]
        preds = [
            {"table": "test_tab", "row": 0, "column": "col", "valid_json": True, "is_error": True, "error_type": "missing"},
            {"table": "test_tab", "row": 1, "column": "col", "valid_json": True, "is_error": True, "error_type": "pattern"},
            {"table": "test_tab", "row": 2, "column": "col", "valid_json": True, "is_error": False, "error_type": "none"},
            {"table": "test_tab", "row": 3, "column": "col", "valid_json": True, "is_error": True, "error_type": "pattern"},
            {"table": "test_tab", "row": 4, "column": "col", "valid_json": True, "is_error": True, "error_type": "outlier"},
            {"table": "test_tab", "row": 5, "column": "col", "valid_json": True, "is_error": True, "error_type": "pattern"},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        et = res["error_types"]

        # Scope 1: On correctly detected errors (cells 0, 1, 4, 5):
        # missing: TP=1 (cell 0), FP=0, FN=1 (cell 1) -> P=1.0, R=0.5, F1=2/3
        self.assertEqual(et["per_type"]["missing"]["tp"], 1)
        self.assertEqual(et["per_type"]["missing"]["fn"], 1)
        self.assertEqual(et["per_type"]["missing"]["fp"], 0)
        self.assertEqual(et["per_type"]["missing"]["support"], 2)
        self.assertAlmostEqual(et["per_type"]["missing"]["f1"], 2.0 / 3.0)

        # pattern: TP=1 (cell 5), FP=1 (cell 1 which was gold missing), FN=0 -> P=0.5, R=1.0, F1=2/3
        self.assertEqual(et["per_type"]["pattern"]["tp"], 1)
        self.assertEqual(et["per_type"]["pattern"]["fp"], 1)
        self.assertEqual(et["per_type"]["pattern"]["fn"], 0)
        self.assertEqual(et["per_type"]["pattern"]["support"], 1)
        self.assertAlmostEqual(et["per_type"]["pattern"]["f1"], 2.0 / 3.0)

        # outlier: TP=1 (cell 4), FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        self.assertEqual(et["per_type"]["outlier"]["tp"], 1)
        self.assertEqual(et["per_type"]["outlier"]["f1"], 1.0)

        # rule: zero support (support=0, tp=0, fp=0, fn=0, f1=0.0)
        self.assertEqual(et["per_type"]["rule"]["support"], 0)
        self.assertEqual(et["per_type"]["rule"]["f1"], 0.0)

        # Macro-F1 across active categories (missing, pattern, outlier): (2/3 + 2/3 + 1.0) / 3 = 7/9
        self.assertAlmostEqual(et["macro_f1"], 7.0 / 9.0)
        # Strict 4-category macro-F1: (2/3 + 2/3 + 0 + 1.0) / 4 = 7/12
        self.assertAlmostEqual(et["macro_f1_strict_4"], 7.0 / 12.0)

        # Scope 2: In Reporting Grid (evaluating all cells in table):
        # Table test_tab x missing: N=3 (cells 0, 1, 2), TP=1, FN=2, FP=0 -> P=1.0, R=1/3, F1=0.5
        grid = res["reporting_grid"]
        row_missing = next(r for r in grid if r["error_type"] == "missing")
        self.assertEqual(row_missing["n_errors"], 3)
        self.assertEqual(row_missing["tp"], 1)
        self.assertEqual(row_missing["fn"], 2)
        self.assertEqual(row_missing["fp"], 0)
        self.assertEqual(row_missing["f1"], 0.5)

        # Table test_tab x pattern: N=1 (cell 5), TP=1, FN=0, FP=2 (cell 1 true missing, cell 3 clean) -> P=1/3, R=1.0, F1=0.5
        row_pattern = next(r for r in grid if r["error_type"] == "pattern")
        self.assertEqual(row_pattern["n_errors"], 1)
        self.assertEqual(row_pattern["tp"], 1)
        self.assertEqual(row_pattern["fn"], 0)
        self.assertEqual(row_pattern["fp"], 2)
        self.assertAlmostEqual(row_pattern["precision"], 1.0 / 3.0, places=4)
        self.assertEqual(row_pattern["recall"], 1.0)
        self.assertEqual(row_pattern["f1"], 0.5)

    def test_08_repair_accuracy_with_correct_and_incorrect_repairs(self):
        """Test case 8: 2 true errors; 1 repaired correctly, 1 repaired incorrectly -> Accuracy = 0.5."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "dirty1", "clean_value": "clean1"},
            {"row": 1, "column": "c", "dirty_value": "dirty2", "clean_value": "clean2"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "repair": "  clean1\t"},
            {"row": 1, "column": "c", "valid_json": True, "is_error": True, "repair": "wrong"},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["repair"]["successful_repairs"], 1)
        self.assertEqual(res["repair"]["total_errors"], 2)
        self.assertEqual(res["repair"]["accuracy"], 0.5)

    def test_09_repair_accuracy_when_error_has_no_repair(self):
        """Test case 9: Detected error with repair=None counts as unsuccessful repair."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "dirty", "clean_value": "clean"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "repair": None},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["repair"]["successful_repairs"], 0)
        self.assertEqual(res["repair"]["total_errors"], 1)
        self.assertEqual(res["repair"]["accuracy"], 0.0)

    def test_10_false_repair_rate_on_erroneous_cells(self):
        """Test case 10: 2 executed repairs on errors: 1 correct, 1 wrong -> False-repair rate = 0.5."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "dirty1", "clean_value": "clean1"},
            {"row": 1, "column": "c", "dirty_value": "dirty2", "clean_value": "clean2"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "repair": "clean1"},
            {"row": 1, "column": "c", "valid_json": True, "is_error": True, "repair": "wrong"},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["false_repair"]["executed_repairs"], 2)
        self.assertEqual(res["false_repair"]["incorrect_repairs"], 1)
        self.assertEqual(res["false_repair"]["rate"], 0.5)

    def test_11_false_repair_rate_when_clean_cell_is_corrupted(self):
        """Test case 11: Clean cell flagged and repaired to a wrong value -> counted as incorrect repair."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "clean_val", "clean_value": "clean_val"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "repair": "corrupted"},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["false_repair"]["executed_repairs"], 1)
        self.assertEqual(res["false_repair"]["incorrect_repairs"], 1)
        self.assertEqual(res["false_repair"]["rate"], 1.0)

    def test_12_false_alarm_rate(self):
        """Test case 12: 4 clean cells; 1 false positive -> FAR = 1/4 = 0.25."""
        cells = [
            {"row": i, "column": "c", "dirty_value": f"v{i}", "clean_value": f"v{i}"}
            for i in range(4)
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True},
            {"row": 1, "column": "c", "valid_json": True, "is_error": False},
            {"row": 2, "column": "c", "valid_json": True, "is_error": False},
            {"row": 3, "column": "c", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["false_alarm"]["fp"], 1)
        self.assertEqual(res["false_alarm"]["clean_cells"], 4)
        self.assertEqual(res["false_alarm"]["rate"], 0.25)

    def test_13_false_alarms_per_1000_cells(self):
        """Test case 13: 1 FP among 4 clean cells -> 250.0 false alarms per 1,000 clean cells."""
        cells = [
            {"row": i, "column": "c", "dirty_value": f"v{i}", "clean_value": f"v{i}"}
            for i in range(4)
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True},
            {"row": 1, "column": "c", "valid_json": True, "is_error": False},
            {"row": 2, "column": "c", "valid_json": True, "is_error": False},
            {"row": 3, "column": "c", "valid_json": True, "is_error": False},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(res["false_alarm"]["per_1000_clean_cells"], 250.0)

    def test_14_pr_auc_with_known_confidence_scores(self):
        """Test case 14: Step-wise Average Precision calculation with known scores."""
        y_true = [True, False, True, False]
        scores = [0.9, 0.8, 0.7, 0.6]
        ap = calculate_pr_auc(y_true, scores)
        self.assertAlmostEqual(ap, 5.0 / 6.0)

    def test_15_pr_auc_edge_cases(self):
        """Test case 15: PR-AUC edge cases (no positive, all positive, empty)."""
        self.assertEqual(calculate_pr_auc([], []), 0.0)
        self.assertEqual(calculate_pr_auc([False, False], [0.8, 0.2]), 0.0)
        self.assertEqual(calculate_pr_auc([True, True], [0.9, 0.4]), 0.0)

    def test_16_ece_with_hand_calculable_probabilities(self):
        """Test case 16: Expected Calibration Error hand calculation."""
        y_true = [True, False]
        scores = [0.75, 0.75]
        ece = calculate_ece(y_true, scores, n_bins=10)
        self.assertAlmostEqual(ece, 0.25)

        y_perf = [True] * 8 + [False] * 2
        scores_perf = [0.8] * 10
        ece_perf = calculate_ece(y_perf, scores_perf, n_bins=10)
        self.assertAlmostEqual(ece_perf, 0.0)

    def test_17_no_confidence_manufactured_from_is_error(self):
        """Test case 17: Changing is_error must NOT manufacture a confidence score for missing p_error."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "bad", "clean_value": "clean"},
            {"row": 1, "column": "c", "dirty_value": "bad", "clean_value": "clean"},
        ]
        # Pred 1 has is_error=True with NO p_error
        # Pred 2 has is_error=False with NO p_error
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "p_error": None},
            {"row": 1, "column": "c", "valid_json": True, "is_error": False, "p_error": ""},
        ]
        res = score(preds, cells, compute_bootstrap=False)
        conf_eval = res["confidence_eval"]
        # Valid cells must be 0; no confidence score is manufactured!
        self.assertEqual(conf_eval["valid_cells"], 0)
        self.assertEqual(conf_eval["coverage_pct"], 0.0)
        self.assertEqual(res["pr_auc"], 0.0)
        self.assertEqual(res["ece"], 0.0)
        self.assertEqual(len(res["precision_coverage"]), 0)

    def test_17b_rejection_of_invalid_confidence_values(self):
        """Test case 17b: Non-finite (NaN, inf) or out-of-bounds (<0 or >1) p_error values rejected."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "e", "clean_value": "c"},
            {"row": 1, "column": "c", "dirty_value": "e", "clean_value": "c"},
            {"row": 2, "column": "c", "dirty_value": "e", "clean_value": "c"},
            {"row": 3, "column": "c", "dirty_value": "e", "clean_value": "c"},
            {"row": 4, "column": "c", "dirty_value": "e", "clean_value": "c"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "p_error": 1.5},    # > 1.0 (invalid)
            {"row": 1, "column": "c", "valid_json": True, "is_error": True, "p_error": -0.1},   # < 0.0 (invalid)
            {"row": 2, "column": "c", "valid_json": True, "is_error": True, "p_error": "nan"},  # NaN (invalid)
            {"row": 3, "column": "c", "valid_json": True, "is_error": True, "p_error": "inf"},  # Inf (invalid)
            {"row": 4, "column": "c", "valid_json": True, "is_error": True, "p_error": 0.85},   # Valid!
        ]
        res = score(preds, cells, compute_bootstrap=False)
        conf_eval = res["confidence_eval"]
        self.assertEqual(conf_eval["valid_cells"], 1)
        self.assertEqual(conf_eval["total_cells"], 5)
        self.assertEqual(conf_eval["coverage_pct"], 20.0)

    def test_18_candidate_recall(self):
        """Test case 18: Candidate recall (proportion of true errors in candidate set)."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "e1", "clean_value": "c1"},
            {"row": 1, "column": "c", "dirty_value": "e2", "clean_value": "c2"},
            {"row": 2, "column": "c", "dirty_value": "ok", "clean_value": "ok"},
        ]
        candidates = [
            {"row": 0, "column": "c", "is_candidate": True},
            {"row": 1, "column": "c", "is_candidate": False},
            {"row": 2, "column": "c", "is_candidate": True},
        ]
        res = score([], cells, candidates=candidates, compute_bootstrap=False)
        cand = res["candidates"]
        self.assertIsNotNone(cand)
        self.assertEqual(cand["true_errors_in_candidates"], 1)
        self.assertEqual(cand["recall"], 0.5)

    def test_19_candidate_pass_through_and_deduplication(self):
        """Test case 19: Candidate pass-through and duplicate candidate deduplication."""
        cells = [
            {"row": i, "column": "c", "dirty_value": "val", "clean_value": "val"}
            for i in range(4)
        ]
        # Row 0 appears multiple times (e.g. from multiple detectors d1, d2):
        candidates = [
            {"row": 0, "column": "c", "is_candidate": False},
            {"row": 0, "column": "c", "is_candidate": True},
            {"row": 0, "column": "c", "is_candidate": True},
            {"row": 1, "column": "c", "is_candidate": False},
        ]
        res = score([], cells, candidates=candidates, compute_bootstrap=False)
        cand = res["candidates"]
        # Total evaluated population is strictly 4 cells; row 0 is counted once
        self.assertEqual(res["n_cells"], 4)
        self.assertEqual(cand["candidates_count"], 1)
        self.assertEqual(cand["pass_through"], 0.25)

    def test_20_bootstrap_reproducibility_with_fixed_seed(self):
        """Test case 20: Bootstrap confidence intervals are deterministic for a fixed seed."""
        cells = [
            {"row": 0, "column": "c", "dirty_value": "bad1", "clean_value": "clean1"},
            {"row": 1, "column": "c", "dirty_value": "bad2", "clean_value": "clean2"},
            {"row": 2, "column": "c", "dirty_value": "ok1", "clean_value": "ok1"},
            {"row": 3, "column": "c", "dirty_value": "ok2", "clean_value": "ok2"},
        ]
        preds = [
            {"row": 0, "column": "c", "valid_json": True, "is_error": True, "repair": "clean1", "p_error": 0.9},
            {"row": 1, "column": "c", "valid_json": True, "is_error": False, "p_error": 0.2},
            {"row": 2, "column": "c", "valid_json": True, "is_error": True, "repair": "wrong", "p_error": 0.8},
            {"row": 3, "column": "c", "valid_json": True, "is_error": False, "p_error": 0.1},
        ]
        res1 = score(preds, cells, seed=42, n_bootstrap=100, compute_bootstrap=True)
        res2 = score(preds, cells, seed=42, n_bootstrap=100, compute_bootstrap=True)

        ci1 = res1["bootstrap_95ci"]
        ci2 = res2["bootstrap_95ci"]

        self.assertIsNotNone(ci1)
        self.assertEqual(ci1, ci2)
        self.assertIn("detection_f1", ci1)

    def test_21_empty_input_and_zero_denominator_behaviour(self):
        """Test case 21: Completely empty input lists and zero denominators."""
        res = score([], [], candidates=None, compute_bootstrap=False)
        self.assertEqual(res["n_cells"], 0)
        self.assertEqual(res["detection"]["precision"], 0.0)
        self.assertEqual(res["detection"]["recall"], 0.0)
        self.assertEqual(res["detection"]["f1"], 0.0)
        self.assertEqual(res["macro_f1_tables"], 0.0)
        self.assertEqual(res["repair"]["accuracy"], 0.0)
        self.assertEqual(res["false_repair"]["rate"], 0.0)
        self.assertEqual(res["false_alarm"]["rate"], 0.0)
        self.assertEqual(res["pr_auc"], 0.0)
        self.assertEqual(res["ece"], 0.0)
        self.assertIsNone(res["candidates"])
        self.assertEqual(res["reporting_grid"], [])

    def test_22_d18_reporting_grid_multi_table_and_types(self):
        """Test case 22: Mandatory D18 reporting grid (table x error_type) export and low-sample markers."""
        cells = []
        preds = []

        # Table 1: Hospital
        # 35 pattern errors (N >= 30, not low-sample)
        for r in range(35):
            cells.append({"table": "hospital", "row": r, "column": "state", "dirty_value": f"d_{r}", "clean_value": f"c_{r}", "error_type": "pattern"})
            preds.append({"table": "hospital", "row": r, "column": "state", "valid_json": True, "is_error": True, "error_type": "pattern", "repair": f"c_{r}"})

        # 5 missing errors (N < 30, is_low_sample = True)
        for r in range(35, 40):
            cells.append({"table": "hospital", "row": r, "column": "zip", "dirty_value": "", "clean_value": "35233", "error_type": "missing"})
            preds.append({"table": "hospital", "row": r, "column": "zip", "valid_json": True, "is_error": True, "error_type": "missing", "repair": "35233"})

        # Table 2: Flights
        # 0 rule errors (N = 0, is_low_sample = True)
        # 2 outlier errors (N = 2, is_low_sample = True)
        cells.append({"table": "flights", "row": 0, "column": "speed", "dirty_value": "9999", "clean_value": "500", "error_type": "outlier"})
        cells.append({"table": "flights", "row": 1, "column": "speed", "dirty_value": "8888", "clean_value": "450", "error_type": "outlier"})
        preds.append({"table": "flights", "row": 0, "column": "speed", "valid_json": True, "is_error": True, "error_type": "outlier", "repair": "500"})
        preds.append({"table": "flights", "row": 1, "column": "speed", "valid_json": True, "is_error": True, "error_type": "outlier", "repair": "wrong"})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "metrics_by_table_type.csv"
            res = score(preds, cells, output_csv_path=out_csv, compute_bootstrap=False)

            # Check that file was created
            self.assertTrue(out_csv.exists())
            self.assertEqual(res["reporting_grid_path"], str(out_csv))

            # Verify contents of CSV
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Columns must match D18 specification
            self.assertEqual(reader.fieldnames, REPORTING_GRID_COLUMNS)

            # Deterministic ordering: flights first, then hospital; each has 4 canonical types
            # Total rows: 2 tables x 4 types = 8 rows
            self.assertEqual(len(rows), 8)
            table_type_pairs = [(r["table"], r["error_type"]) for r in rows]
            expected_pairs = [
                ("flights", "missing"),
                ("flights", "pattern"),
                ("flights", "rule"),
                ("flights", "outlier"),
                ("hospital", "missing"),
                ("hospital", "pattern"),
                ("hospital", "rule"),
                ("hospital", "outlier"),
            ]
            self.assertEqual(table_type_pairs, expected_pairs)

            # Check hospital x pattern: N=35 >= 30 -> is_low_sample is False
            hosp_pattern = next(r for r in rows if r["table"] == "hospital" and r["error_type"] == "pattern")
            self.assertEqual(int(hosp_pattern["n_errors"]), 35)
            self.assertEqual(hosp_pattern["is_low_sample"], "False")
            self.assertEqual(float(hosp_pattern["f1"]), 1.0)
            self.assertEqual(float(hosp_pattern["repair_accuracy"]), 1.0)

            # Check hospital x missing: N=5 < 30 -> is_low_sample is True
            hosp_missing = next(r for r in rows if r["table"] == "hospital" and r["error_type"] == "missing")
            self.assertEqual(int(hosp_missing["n_errors"]), 5)
            self.assertEqual(hosp_missing["is_low_sample"], "True")

            # Check flights x rule: N=0 -> is_low_sample is True, metrics safe/0.0
            flights_rule = next(r for r in rows if r["table"] == "flights" and r["error_type"] == "rule")
            self.assertEqual(int(flights_rule["n_errors"]), 0)
            self.assertEqual(flights_rule["is_low_sample"], "True")
            self.assertEqual(float(flights_rule["precision"]), 0.0)
            self.assertEqual(float(flights_rule["recall"]), 0.0)
            self.assertEqual(float(flights_rule["f1"]), 0.0)

            # Check flights x outlier: N=2, 1 correct repair, 1 wrong repair
            flights_outlier = next(r for r in rows if r["table"] == "flights" and r["error_type"] == "outlier")
            self.assertEqual(int(flights_outlier["n_errors"]), 2)
            self.assertEqual(int(flights_outlier["successful_repairs"]), 1)
            self.assertEqual(float(flights_outlier["repair_accuracy"]), 0.5)
            self.assertEqual(float(flights_outlier["false_repair_rate"]), 0.5)


if __name__ == "__main__":
    unittest.main()


class TestRevisionRules(unittest.TestCase):
    """Added in review: format-only cells are left out; several tables need a table field in the cells."""

    def test_format_artifacts_are_skipped(self):
        from src.evaluation.metrics import score
        cells = [
            {"row": 0, "column": "rate", "dirty_value": "3.0", "clean_value": "3", "is_error": "True", "error_type": "format_artifact"},
            {"row": 1, "column": "rate", "dirty_value": "x", "clean_value": "4", "is_error": "True", "error_type": "pattern_violation"},
        ]
        preds = [{"row": 1, "column": "rate", "is_error": True, "error_type": "pattern", "repair": "4", "valid_json": True, "p_error": 0.9}]
        m = score(preds, cells, compute_bootstrap=False)
        self.assertEqual(m["format_artifacts_skipped"], 1)
        self.assertEqual(m["n_cells"], 1)
        self.assertEqual(m["detection"]["recall"], 1.0)

    def test_several_tables_without_table_field_is_refused(self):
        from src.evaluation.metrics import score
        cells = [{"row": 0, "column": "a", "dirty_value": "x", "clean_value": "y", "is_error": "True", "error_type": "pattern_violation"}]
        preds = [{"table": "t1", "row": 0, "column": "a", "is_error": True, "error_type": "pattern", "repair": "y", "valid_json": True},
                 {"table": "t2", "row": 0, "column": "a", "is_error": False, "error_type": "none", "repair": None, "valid_json": True}]
        with self.assertRaises(ValueError):
            score(preds, cells, compute_bootstrap=False)

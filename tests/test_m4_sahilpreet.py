#!/usr/bin/env python3
"""Unit tests for Milestone 4 Sahilpreet tasks: EM prior calibration, gate fitting,

decision scoring (7-bin outcome table), and baseline scoring.

Author: Sahilpreet
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import pytest

from src.evaluation.calibrate import (
    estimate_prior_em,
    fit_gate,
)
from src.evaluation.metrics import score_decisions, score_flags


class TestEMPriorEstimation:
    """Tests for Saerens et al. (2002) EM prior estimation."""

    def test_estimate_prior_em_hand_computed(self):
        """Hand-computed 2-sample verification of Saerens EM prior update step.

        Suppose N = 2 samples with initial balanced model probabilities:
            p1 = 0.8, p2 = 0.4
        training_prior p0 = 0.5.
        Initial estimate pi^(0) = (0.8 + 0.4) / 2 = 0.6.
        Step 1:
            c_pos = 0.6 / 0.5 = 1.2
            c_neg = (1 - 0.6) / (1 - 0.5) = 0.8
            Sample 1:
                num1 = 1.2 * 0.8 = 0.96
                den1 = 0.96 + 0.8 * 0.2 = 0.96 + 0.16 = 1.12
                post1 = 0.96 / 1.12 = 6 / 7 ~= 0.857142857
            Sample 2:
                num2 = 1.2 * 0.4 = 0.48
                den2 = 0.48 + 0.8 * 0.6 = 0.48 + 0.48 = 0.96
                post2 = 0.48 / 0.96 = 0.5
            pi^(1) = (6/7 + 0.5) / 2 = (1.357142857) / 2 ~= 0.67857142857
        """
        probs = [0.8, 0.4]
        expected_pi_1 = (6.0 / 7.0 + 0.5) / 2.0
        rate, info = estimate_prior_em(probs, training_prior=0.5, max_iter=1, tol=1.0)
        assert math.isclose(rate, expected_pi_1, rel_tol=1e-5)
        assert info["iterations"] == 1
        assert math.isclose(info["initial_rate"], 0.6, abs_tol=1e-6)

    @pytest.mark.parametrize("true_rate", [0.05, 0.20, 0.50])
    def test_estimate_prior_em_recovery_known_rates(self, true_rate: float):
        """Verify EM prior estimate recovers known synthetic prevalence within +/- 2%.

        Uses class-conditional Gaussians x|y=1 ~ N(1.5, 1) and x|y=0 ~ N(-1.5, 1).
        Under balanced prior p0=0.5, logit(P0(y=1|x)) = 3*x.
        """
        rng = random.Random(42)
        n_samples = 10000
        n_pos = int(n_samples * true_rate)
        n_neg = n_samples - n_pos

        x_pos = [rng.gauss(1.5, 1.0) for _ in range(n_pos)]
        x_neg = [rng.gauss(-1.5, 1.0) for _ in range(n_neg)]
        xs = x_pos + x_neg
        probs = [1.0 / (1.0 + math.exp(-3.0 * x)) for x in xs]

        rate, info = estimate_prior_em(probs, training_prior=0.5, tol=1e-6, max_iter=100)
        assert info["converged"] is True
        assert abs(rate - true_rate) < 0.02, f"Expected ~{true_rate}, got {rate}"

    def test_estimate_prior_em_validation_error_cases(self):
        """Test input validation for estimate_prior_em."""
        with pytest.raises(ValueError, match="empty sequence"):
            estimate_prior_em([])

        with pytest.raises(ValueError, match="training_prior"):
            estimate_prior_em([0.5], training_prior=0.0)

        with pytest.raises(ValueError, match="training_prior"):
            estimate_prior_em([0.5], training_prior=1.0)

        with pytest.raises(ValueError, match="range"):
            estimate_prior_em([1.5], training_prior=0.5)

    def test_estimate_prior_em_fallback_on_no_convergence(self):
        """Verify that the fallback fires unconditionally when EM does not converge."""
        probs = [0.8, 0.4]
        fallback = 0.4321
        rate, info = estimate_prior_em(probs, training_prior=0.5, max_iter=1, tol=1e-12, fallback_rate=fallback)
        assert info["converged"] is False
        assert info["used_fallback"] is True
        assert math.isclose(rate, fallback, abs_tol=1e-6)
        assert "fallback_reason" in info
        assert "did not converge" in info["fallback_reason"].lower()

    def test_estimate_prior_em_fallback_on_clamp(self):
        """Verify fallback when estimate hits boundary clamp (<= 1e-4 or >= 1 - 1e-4)."""
        probs = [0.9999999] * 50
        fallback = 0.50
        rate, info = estimate_prior_em(probs, training_prior=0.5, fallback_rate=fallback)
        assert info["used_fallback"] is True
        assert math.isclose(rate, fallback, abs_tol=1e-6)
        assert "boundary" in info["fallback_reason"].lower() or "clamped" in info["fallback_reason"].lower()

    def test_estimate_prior_em_fallback_on_far_from_default(self):
        """Verify fallback when estimate deviates beyond max_deviation_from_default."""
        probs = [0.10] * 30 + [0.80] * 70
        fallback = 0.20
        rate, info = estimate_prior_em(
            probs,
            training_prior=0.5,
            fallback_rate=fallback,
            max_deviation_from_default=0.15,
        )
        assert info["used_fallback"] is True
        assert math.isclose(rate, fallback, abs_tol=1e-6)
        assert "deviates" in info["fallback_reason"].lower()

    def test_estimate_prior_em_cap_met_returned_by_fit_gate(self):
        """Verify fit_gate returns cap_met=True when the FRR constraint is satisfiable."""
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        is_error = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        repair_ok = [None, None, None, None, False, False, False, True, True, True]
        res = fit_gate(probs, is_error, repair_ok, max_false_repair=0.10)
        assert "cap_met" in res
        assert res["cap_met"] is True

    def test_fit_gate_cap_not_met_when_all_repairs_wrong(self):
        """Verify fit_gate returns cap_met=False when FRR can never reach <= max_false_repair."""
        probs = [0.9, 0.95, 0.99]
        is_error = [1, 1, 1]
        repair_ok = [False, False, False]
        res = fit_gate(probs, is_error, repair_ok, max_false_repair=0.10)
        assert "cap_met" in res
        assert res["cap_met"] is False

    def test_fit_gate_cap_met_none_when_no_repair_data(self):
        """Verify fit_gate returns cap_met=None when no repair data is provided."""
        probs = [0.2, 0.8]
        is_error = [0, 1]
        res1 = fit_gate(probs, is_error, repair_ok=None)
        assert res1["cap_met"] is None

        res2 = fit_gate(probs, is_error, repair_ok=[None, None])
        assert res2["cap_met"] is None


class TestGateFitting:
    """Tests for fit_gate threshold optimization."""

    def test_fit_gate_hand_computed(self):
        """Hand-computed test for flag and repair threshold selection.

        Construct 10 validation cells:
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        labels = [0,   0,   0,   0,   1,   1,   1,   1,   1,   1] (6 errors)
        repairs:
            cells 0..3: None
            cells 4..6: repair_ok = False (wrong repairs, conf 0.5, 0.6, 0.7)
            cells 7..9: repair_ok = True  (correct repairs, conf 0.8, 0.9, 0.95)
        Thresholds:
            At t = 0.5:
                Flag: TP=6, FP=0, FN=0 -> Precision=1.0, Recall=1.0, F1=1.0!
                Best flag threshold t1 = 0.50.
            Repair:
                At t = 0.5: Executed=6, wrong=3 -> FRR = 3/6 = 50% (> 10%)
                At t = 0.7: Executed=4, wrong=1 (conf 0.7) -> FRR = 1/4 = 25% (> 10%)
                At t = 0.8: Executed=3, wrong=0 -> FRR = 0/3 = 0% (<= 10%)!
                Lowest threshold with FRR <= 10% is t2 = 0.80.
        """
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        is_error = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        repair_ok = [None, None, None, None, False, False, False, True, True, True]

        res = fit_gate(probs, is_error, repair_ok, max_false_repair=0.10)
        assert res["flag"] == 0.50
        assert res["repair"] == 0.80
        assert res["repair"] >= res["flag"]

    def test_fit_gate_edge_cases(self):
        """Test fit_gate with empty inputs and mismatched lengths."""
        with pytest.raises(ValueError):
            fit_gate([], [])
        with pytest.raises(ValueError):
            fit_gate([0.5], [1], repair_ok=[True, False])


class TestGateLevelScoring:
    """Tests for score_decisions and the 7-bin outcome table."""

    def test_score_decisions_hand_computed_and_conservation(self, tmp_path: Path):
        """Hand-computed test verifying all 7 outcome buckets and strict cell count conservation.

        Table 1 (sample_table, 7 cells):
        Cell 0: true error (dirty='a', clean='b'), decision='repair', repair='b' -> repaired right
        Cell 1: true error (dirty='c', clean='d'), decision='repair', repair='e' -> repaired wrong (error)
        Cell 2: correct cell (dirty='f', clean='f'), decision='repair', repair='g' -> repaired wrong (correct cell)
        Cell 3: true error (dirty='h', clean='i'), decision='review' -> review (error)
        Cell 4: correct cell (dirty='j', clean='j'), decision='review' -> review (correct cell)
        Cell 5: true error (dirty='k', clean='l'), decision='keep' -> kept (missed error)
        Cell 6: correct cell (dirty='m', clean='m'), decision='keep' -> kept (correct)

        Total cells = 7.
        Total errors = 4 (cells 0, 1, 3, 5).
        Total correct = 3 (cells 2, 4, 6).
        """
        cells = [
            {"table": "T1", "row": "0", "column": "col1", "dirty_value": "a", "clean_value": "b"},
            {"table": "T1", "row": "1", "column": "col1", "dirty_value": "c", "clean_value": "d"},
            {"table": "T1", "row": "2", "column": "col1", "dirty_value": "f", "clean_value": "f"},
            {"table": "T1", "row": "3", "column": "col1", "dirty_value": "h", "clean_value": "i"},
            {"table": "T1", "row": "4", "column": "col1", "dirty_value": "j", "clean_value": "j"},
            {"table": "T1", "row": "5", "column": "col1", "dirty_value": "k", "clean_value": "l"},
            {"table": "T1", "row": "6", "column": "col1", "dirty_value": "m", "clean_value": "m"},
        ]
        predictions = [
            {"table": "T1", "row": 0, "column": "col1", "decision": "repair", "repair": "b", "valid_json": True, "is_error": True},
            {"table": "T1", "row": 1, "column": "col1", "decision": "repair", "repair": "e", "valid_json": True, "is_error": True},
            {"table": "T1", "row": 2, "column": "col1", "decision": "repair", "repair": "g", "valid_json": True, "is_error": True},
            {"table": "T1", "row": 3, "column": "col1", "decision": "flag for review", "repair": None, "valid_json": True, "is_error": True},
            {"table": "T1", "row": 4, "column": "col1", "decision": "review", "repair": None, "valid_json": True, "is_error": True},
            # Cells 5 and 6 have no prediction (filtered by candidates) -> default to 'keep'
        ]

        csv_path = tmp_path / "outcomes.csv"
        res = score_decisions(predictions, cells, output_csv_path=csv_path)

        t1_stats = res["by_table"]["T1"]
        assert t1_stats["repaired_right"] == 1
        assert t1_stats["repaired_wrong_error"] == 1
        assert t1_stats["repaired_wrong_correct_cell"] == 1
        assert t1_stats["review_error"] == 1
        assert t1_stats["review_correct_cell"] == 1
        assert t1_stats["kept_missed_error"] == 1
        assert t1_stats["kept_correct"] == 1
        assert t1_stats["total_cells"] == 7
        assert t1_stats["total_errors"] == 4

        # Conservation check: sum of 7 bins == 7
        sum_bins = (
            t1_stats["repaired_right"]
            + t1_stats["repaired_wrong_error"]
            + t1_stats["repaired_wrong_correct_cell"]
            + t1_stats["review_error"]
            + t1_stats["review_correct_cell"]
            + t1_stats["kept_missed_error"]
            + t1_stats["kept_correct"]
        )
        assert sum_bins == 7

        # Metric checks
        # TP = repaired right (1) + repaired wrong error (1) + review error (1) = 3
        # FP = repaired wrong correct cell (1) + review correct cell (1) = 2
        # FN = kept missed error (1)
        # Precision = 3 / 5 = 0.60
        # Recall = 3 / (3 + 1) = 0.75
        # F1 = 2 * 0.6 * 0.75 / 1.35 = 0.9 / 1.35 = 2/3 ~= 0.6667
        assert t1_stats["precision"] == 0.60
        assert t1_stats["recall"] == 0.75
        assert math.isclose(t1_stats["f1"], 0.6667, abs_tol=1e-4)

        # Repair accuracy: 1 / 4 = 0.25
        assert t1_stats["repair_accuracy"] == 0.25
        # False-repair rate: (1 wrong error + 1 wrong correct) / 3 executed = 2 / 3 ~= 0.6667
        assert math.isclose(t1_stats["false_repair_rate"], 0.6667, abs_tol=1e-4)
        # False alarms per 1000 correct cells: 2 / 3 * 1000 = 666.67
        assert math.isclose(t1_stats["false_alarms_per_1000"], 666.67, abs_tol=1e-2)
        # Review share: 2 / 7 ~= 0.2857
        assert math.isclose(t1_stats["review_share"], 0.2857, abs_tol=1e-4)

        # CSV was written
        assert csv_path.exists()

    def test_score_decisions_no_op_rule(self):
        """Verify the no-op rule: a repair proposing the old value counts as kept / not an error."""
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "BOSTON", "clean_value": "BOSTON"},
        ]
        # Model proposes to 'repair' to 'BOSTON' (same as dirty)
        preds = [
            {"table": "T1", "row": 0, "column": "c1", "decision": "repair", "repair": "BOSTON", "valid_json": True, "is_error": True}
        ]
        res = score_decisions(preds, cells)
        t_res = res["by_table"]["T1"]
        # Must be treated as kept (correct) rather than a false repair on a correct cell!
        assert t_res["repaired_wrong_correct_cell"] == 0
        assert t_res["kept_correct"] == 1
        assert t_res["total_cells"] == 1

    def test_score_decisions_review_repair_rejected_classified_as_review(self):
        """'review (repair rejected by validation)' must be classified as review, NOT repair.

        Before the fix, 'repair' in raw_decision would match this string, incorrectly
        counting validation-rejected repairs as applied repairs.
        """
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "err", "clean_value": "clean"},
        ]
        preds = [
            {
                "table": "T1", "row": 0, "column": "c1",
                "decision": "review (repair rejected by validation)",
                "gate_decision": "repair",
                "repair": "clean",
                "valid_json": True, "is_error": True,
            }
        ]
        res = score_decisions(preds, cells)
        t = res["by_table"]["T1"]
        # With validation: the cell goes to review, NOT repair
        assert t["repaired_right"] == 0, "Rejected repair must not count as repaired_right"
        assert t["review_error"] == 1, "Rejected repair on error cell must count as review_error"

    def test_score_decisions_review_cannot_be_validated_classified_as_review(self):
        """'review (repair cannot be validated)' must be classified as review."""
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "err", "clean_value": "clean"},
        ]
        preds = [
            {
                "table": "T1", "row": 0, "column": "c1",
                "decision": "review (repair cannot be validated)",
                "gate_decision": "repair",
                "repair": "clean",
                "valid_json": True, "is_error": True,
            }
        ]
        res = score_decisions(preds, cells)
        t = res["by_table"]["T1"]
        assert t["repaired_right"] == 0
        assert t["review_error"] == 1

    def test_score_decisions_gate_decision_column(self):
        """gate_decision column is tracked separately from the final decision.

        A cell with gate_decision='repair' but final decision='review (repair rejected ...)'
        should count as repaired in the gate column but reviewed in the final column.
        """
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "err", "clean_value": "clean"},
        ]
        preds = [
            {
                "table": "T1", "row": 0, "column": "c1",
                "decision": "review (repair rejected by validation)",
                "gate_decision": "repair",
                "repair": "clean",
                "valid_json": True, "is_error": True,
            }
        ]
        res = score_decisions(preds, cells)
        t = res["by_table"]["T1"]
        # Final (gate+validation): review
        assert t["review_error"] == 1
        # Gate only: repair and it was correct
        assert t.get("gate_repaired_right", 0) == 1

    def test_score_decisions_multiple_tables(self, tmp_path):
        """Test score_decisions with two tables to verify per-table isolation."""
        cells = [
            {"table": "A", "row": "0", "column": "c1", "dirty_value": "err", "clean_value": "clean"},
            {"table": "A", "row": "1", "column": "c1", "dirty_value": "ok", "clean_value": "ok"},
            {"table": "B", "row": "0", "column": "c1", "dirty_value": "bad", "clean_value": "good"},
            {"table": "B", "row": "1", "column": "c1", "dirty_value": "fine", "clean_value": "fine"},
        ]
        preds = [
            {"table": "A", "row": 0, "column": "c1", "decision": "repair", "repair": "clean", "valid_json": True, "is_error": True},
            {"table": "B", "row": 0, "column": "c1", "decision": "flag for review", "repair": None, "valid_json": True, "is_error": True},
        ]
        res = score_decisions(preds, cells, output_csv_path=tmp_path / "out.csv")

        a = res["by_table"]["A"]
        b = res["by_table"]["B"]
        assert a["repaired_right"] == 1
        assert a["total_cells"] == 2
        assert b["review_error"] == 1
        assert b["total_cells"] == 2
        assert res["total"]["total_cells"] == 4

    def test_score_decisions_format_artifact_excluded(self):
        """format_artifact cells must be excluded from scoring."""
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "3", "clean_value": "3.0", "error_type": "format_artifact"},
            {"table": "T1", "row": "1", "column": "c1", "dirty_value": "err", "clean_value": "clean"},
        ]
        preds = [
            {"table": "T1", "row": 1, "column": "c1", "decision": "repair", "repair": "clean", "valid_json": True, "is_error": True},
        ]
        res = score_decisions(preds, cells)
        t = res["by_table"]["T1"]
        assert t["total_cells"] == 1  # format_artifact excluded
        assert t["repaired_right"] == 1
        assert res["format_artifacts_skipped"] == 1


class TestBaselineScorer:
    """Tests for score_flags standardized baseline evaluation."""

    def test_score_flags_standardization_and_exclusions(self, tmp_path: Path):
        """Test score_flags with Raha normalization and exclude_rows."""
        cells = [
            {"table": "T1", "row": "0", "column": "col1", "dirty_value": "apple", "clean_value": "orange"},  # row 0 error
            {"table": "T1", "row": "1", "column": "col1", "dirty_value": "banana", "clean_value": "banana"},  # row 1 clean
            {"table": "T1", "row": "2", "column": "col1", "dirty_value": "grape", "clean_value": "cherry"},   # row 2 error
            {"table": "T1", "row": "3", "column": "col1", "dirty_value": "peach", "clean_value": "peach"},   # row 3 clean
        ]
        # Baseline flags cells (0, col1) and (1, col1)
        flagged = [(0, "col1"), (1, "col1")]

        # Case 1: No exclusions
        res1 = score_flags(flagged, cells)
        assert res1["tp"] == 1  # (0, col1) is error
        assert res1["fp"] == 1  # (1, col1) is clean
        assert res1["fn"] == 1  # (2, col1) missed
        assert res1["tn"] == 1  # (3, col1) correct rejection
        assert res1["precision"] == 0.5
        assert res1["recall"] == 0.5
        assert res1["f1"] == 0.5

        # Case 2: Exclude row 0 (e.g. labeled tuple)
        res2 = score_flags(flagged, cells, exclude_rows=[0])
        assert res2["excluded_rows_count"] == 1
        assert res2["tp"] == 0
        assert res2["fp"] == 1  # row 1
        assert res2["fn"] == 1  # row 2
        assert res2["tn"] == 1  # row 3
        assert res2["n_cells"] == 3

    def test_score_flags_exclude_rows_per_table(self):
        """Integer exclude_rows must NOT drop the same row from every table (Fix 6).

        Before the fix, exclude_rows=[0] would exclude row 0 from BOTH tables.
        After the fix, row 0 must only be excluded in the table it belongs to
        when specified as a (table, row) tuple, or only in single-table mode
        when given as a bare integer.
        """
        cells = [
            {"table": "A", "row": "0", "column": "c1", "dirty_value": "err", "clean_value": "clean"},  # A row 0: error
            {"table": "A", "row": "1", "column": "c1", "dirty_value": "ok",  "clean_value": "ok"},     # A row 1: clean
            {"table": "B", "row": "0", "column": "c1", "dirty_value": "bad", "clean_value": "good"},   # B row 0: error
            {"table": "B", "row": "1", "column": "c1", "dirty_value": "fin", "clean_value": "fin"},    # B row 1: clean
        ]
        flagged = [(0, "c1")]  # flags row 0 in both tables

        # Exclude only table A row 0 via (table, row) tuple
        res = score_flags(flagged, cells, exclude_rows=[("A", 0)])
        assert res["excluded_rows_count"] == 1
        # B row 0 must still be evaluated (it's an error that was flagged → TP)
        assert res["tp"] == 1  # B:row0 is error and was flagged
        assert res["fn"] == 0

    def test_score_flags_raises_for_bare_integer_on_multi_table(self):
        """score_flags must raise an error for a bare integer in exclude_rows when there is more than one table."""
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "a", "clean_value": "a"},
            {"table": "T2", "row": "0", "column": "c1", "dirty_value": "b", "clean_value": "b"},
        ]
        with pytest.raises(ValueError, match="Bare integer in exclude_rows"):
            score_flags([], cells, exclude_rows=[0])

    def test_score_decisions_no_change_repair(self):
        """A repair proposal equal to the dirty value under norm counts as no-change and routes to keep."""
        cells = [
            {"table": "T1", "row": "0", "column": "c1", "dirty_value": "12.0 oz", "clean_value": "12", "is_error": True},
        ]
        # Gate proposes '12.0 oz', matching dirty_value
        preds = [
            {"table": "T1", "row": "0", "column": "c1", "decision": "repair", "repair": "12.0 oz", "gate_decision": "repair"},
        ]
        res = score_decisions(preds, cells)
        t = res["total"]
        assert t["gate_repaired_no_change"] == 1
        assert t["gate_repaired_right"] == 0
        assert t["gate_repaired_wrong"] == 0
        assert t["repaired_right"] == 0
        assert t["kept_missed_error"] == 1  # routed under keep as missed error (dirty != clean)


class TestFinalM4Runs:
    """Verifications on the real final runs m4_final2_test_hfb and m4_final2_val_all."""

    def test_score_decisions_on_m4_final2_test_hfb(self):
        """Verify leader check on m4_final2_test_hfb:

        - Gate only: 1,047 right, 16 wrong; 0 no-change
        - With validation: 1,043 right, 14 wrong (10 wrong error + 4 wrong clean)
        - 5 rejected and 1 'cannot be validated' go to review
        """
        run_dir = _REPO_ROOT / "m4_final2_test_hfb"
        if not (run_dir / "predictions.jsonl").exists():
            pytest.skip("m4_final2_test_hfb not present in workspace")

        cells_df = pd.read_csv(run_dir / "cells.csv", dtype=str, keep_default_na=False)
        preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]

        res = score_decisions(preds, cells_df)
        tot = res["total"]

        # Gate only check
        assert tot["gate_repaired_right"] == 1047
        assert tot["gate_repaired_wrong"] == 16
        assert tot["gate_repaired_no_change"] == 0

        # Post-validation check
        assert tot["repaired_right"] == 1043
        assert (tot["repaired_wrong_error"] + tot["repaired_wrong_correct_cell"]) == 14
        assert tot["repaired_wrong_error"] == 10
        assert tot["repaired_wrong_correct_cell"] == 4

        # Review reasons check
        rejected = [p for p in preds if "rejected" in str(p.get("decision", ""))]
        cannot = [p for p in preds if "cannot be validated" in str(p.get("decision", ""))]
        assert len(rejected) == 5
        assert len(cannot) == 1

    def test_score_decisions_on_m4_final2_val_all(self):
        """Verify m4_final2_val_all:

        - Gate only: 1,025 right, 10 wrong; 0 no-change
        - With validation: 1,021 right, 4 wrong (validation removes 6 wrong repairs)
        """
        run_dir = _REPO_ROOT / "m4_final2_val_all"
        if not (run_dir / "predictions.jsonl").exists():
            pytest.skip("m4_final2_val_all not present in workspace")

        cells_df = pd.read_csv(run_dir / "cells.csv", dtype=str, keep_default_na=False)
        preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]

        res = score_decisions(preds, cells_df)
        tot = res["total"]

        assert tot["gate_repaired_right"] == 1025
        assert tot["gate_repaired_wrong"] == 10
        assert tot["gate_repaired_no_change"] == 0

        assert tot["repaired_right"] == 1021
        assert (tot["repaired_wrong_error"] + tot["repaired_wrong_correct_cell"]) == 4
        # Validation removed exactly 6 wrong repairs (10 gate wrong -> 4 post-val wrong)
        assert (tot["gate_repaired_wrong"] - (tot["repaired_wrong_error"] + tot["repaired_wrong_correct_cell"])) == 6


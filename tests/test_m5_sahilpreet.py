"""Milestone 5 Automated Verification Suite for Sahilpreet's Deliverables.

Tests:
1. Experiment X10 Threshold Grid and Dynamic Pass Rules (leaving out 20 labelled rows per table).
   - Baseline mean F1 = 0.7445
   - Baseline post-validation repairs = 975 right / 13 wrong
   - Gate-level FRR rule: not higher than baseline (fails at t2=0.648)
   - Temperature scaling ECE falls on all 5 tables (passes at T=0.826)
2. Experiment X11 Unseen Tables Dynamic Evaluator and Three-State Routing.
3. Conserved 7-Bin Decision Scoring and Metric Bounds.
4. Report Artifacts, Figures (PR curve among candidate cells), and Environment Documentation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_x10_threshold_grid import compute_grid
from scripts.run_x11_unseen_priors import evaluate_unseen_table_run
from src.evaluation.calibrate import apply_temperature, estimate_prior_em
from src.evaluation.metrics import calculate_ece, calculate_metrics


class TestExperimentX10:
    """Verifies Experiment X10 threshold grid and predetermined pass rules."""

    def test_pass_rules_logic(self):
        if not (_REPO_ROOT / "M4_run_outputs").exists():
            pytest.skip("the M4 test run folders (M4_run_outputs) are not in the repository")
        grid_rows, eval_rules = compute_grid()

        # Rule 1: Flag threshold t1=0.378 must fail on test data (mean F1 drops from 0.7445 to 0.7357)
        t1_rule = eval_rules["t1_flag"]
        assert t1_rule["passed"] is False
        assert t1_rule["mean_f1_baseline"] == 0.7445
        assert t1_rule["mean_f1_fitted"] < t1_rule["mean_f1_baseline"]
        assert t1_rule["f1_drops_count"] >= 2
        assert "hospital" in t1_rule["tables_f1_dropped"]
        assert t1_rule["recommendation"].startswith("KEEP BASELINE")

        # Rule 2: Repair threshold t2=0.648 must fail because FRR is higher than baseline
        t2_rule = eval_rules["t2_repair"]
        assert t2_rule["passed"] is False
        assert t2_rule["frr_fitted"] > t2_rule["frr_baseline"]
        assert t2_rule["recommendation"].startswith("KEEP BASELINE")

        # Rule 3: Temperature T=0.826 must pass because ECE falls on all 5 tables
        temp_rule = eval_rules["temperature"]
        assert temp_rule["passed"] is True
        assert temp_rule["tables_with_ece_reduction"] == 5
        assert temp_rule["recommendation"].startswith("ADOPT")

    def test_grid_evaluation_structure_and_conservation(self):
        if not (_REPO_ROOT / "M4_run_outputs").exists():
            pytest.skip("the M4 test run folders (M4_run_outputs) are not in the repository")
        grid_rows, eval_rules = compute_grid()
        # 8 combinations * (5 tables + 1 mean row) = 48 rows
        assert len(grid_rows) == 48

        # Check baseline MEAN / TOTAL row
        baseline_mean = [r for r in grid_rows if r["combination"] == "Fixed Baseline (Default)" and r["table"] == "MEAN / TOTAL"][0]
        assert baseline_mean["f1"] == 0.7445
        assert baseline_mean["applied_right"] == 975
        assert baseline_mean["applied_wrong"] == 13
        assert baseline_mean["gate_applied_wrong"] > 0

        for row in grid_rows:
            assert 0.0 <= row["precision"] <= 1.0
            assert 0.0 <= row["recall"] <= 1.0
            assert 0.0 <= row["f1"] <= 1.0
            assert 0.0 <= row["false_repair_rate"] <= 1.0
            assert 0.0 <= row["gate_false_repair_rate"] <= 1.0
            assert 0.0 <= row["ece"] <= 1.0
            assert row["applied_repairs"] == row["applied_right"] + row["applied_wrong"]
            assert row["gate_applied_repairs"] == row["gate_applied_right"] + row["gate_applied_wrong"]

    def test_grid_csv_file_on_disk(self):
        csv_path = _REPO_ROOT / "docs" / "tables" / "m5" / "x10_threshold_calibration_grid.csv"
        assert csv_path.exists(), "x10_threshold_calibration_grid.csv must exist"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 48

    def test_pass_rules_summary_csv_on_disk(self):
        csv_path = _REPO_ROOT / "docs" / "tables" / "m5" / "x10_pass_rules_summary.csv"
        assert csv_path.exists(), "x10_pass_rules_summary.csv must exist"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        # t1 failed, t2 failed, temp passed
        assert rows[0]["passed"] == "False"
        assert rows[1]["passed"] == "False"
        assert rows[2]["passed"] == "True"


class TestExperimentX11:
    """Verifies Experiment X11 dynamic evaluation and TA three-state routing."""

    def test_unseen_tables_three_state_routing(self):
        # Sample synthetic candidate predictions
        mock_preds = [
            {"table": "unseen_test", "row": "0", "column": "col1", "p_error": 0.85, "repair": "fixed", "state": "not verified"},
            {"table": "unseen_test", "row": "1", "column": "col1", "p_error": 0.75, "repair": "maybe", "state": "needs review"},
            {"table": "unseen_test", "row": "2", "column": "col1", "p_error": 0.10, "repair": None, "state": "unchanged"},
        ]
        mock_gt = {
            ("unseen_test", "0", "col1"): ("broken", "fixed"),
            ("unseen_test", "1", "col1"): ("dirty", "clean"),
            ("unseen_test", "2", "col1"): ("clean", "clean"),
        }

        res = evaluate_unseen_table_run("unseen_test", mock_preds, mock_gt, t1_flag=0.50, t2_repair=0.90)

        # Candidate 0: state == "not verified" -> applied as "not verified"
        assert res["applied_not_verified"] == 1
        assert res["applied_not_verified_right"] == 1
        # Candidate 1: state == "needs review" -> routed to "needs review"
        assert res["needs_review"] == 1
        assert "unverified" in res["rule_validation_status"].lower()

    def test_x11_unseen_priors_csv_on_disk(self):
        csv_path = _REPO_ROOT / "docs" / "tables" / "m5" / "x11_unseen_priors.csv"
        assert csv_path.exists(), "x11_unseen_priors.csv must exist"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        table_map = {r["table"].lower(): r for r in rows}
        assert "german" in table_map
        assert "soccer" in table_map
        assert "billionaire" in table_map
        assert "marketing" in table_map
        assert "restaurant" in table_map

        # Exact run numbers matching change_log.csv
        assert int(table_map["german"]["applied_not_verified"]) == 5476
        assert float(table_map["german"]["applied_not_verified_accuracy"]) == 0.9850

        assert int(table_map["soccer"]["applied_not_verified"]) == 217
        assert float(table_map["soccer"]["applied_not_verified_accuracy"]) == 0.9401

        assert int(table_map["billionaire"]["applied_not_verified"]) == 1435
        assert float(table_map["billionaire"]["applied_not_verified_accuracy"]) == 0.9533

        assert int(table_map["marketing"]["applied_not_verified"]) == 320
        assert float(table_map["marketing"]["applied_not_verified_accuracy"]) == 0.8656

        assert int(table_map["restaurant"]["applied_not_verified"]) == 59
        assert float(table_map["restaurant"]["applied_not_verified_accuracy"]) == 0.0169
        assert float(table_map["restaurant"]["em_estimated_prior"]) == 0.0494

        for r in rows:
            em_prior = float(r["em_estimated_prior"])
            assert 0.0 < em_prior < 1.0
            assert r["em_converged"] == "True"
            assert int(r["applied_not_verified"]) <= int(r["proposed_repairs"])
            assert int(r["applied_not_verified_right"]) <= int(r["applied_not_verified"])
            assert int(r["needs_review"]) > 0


class TestExperimentX15:
    """Verifies Experiment X15 groundedness and hallucination suppression."""

    def test_x15_hallucination_csv_on_disk(self):
        csv_path = _REPO_ROOT / "docs" / "tables" / "m5" / "x15_hallucination_groundedness.csv"
        assert csv_path.exists(), "x15_hallucination_groundedness.csv must exist"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 4  # 4 unseen + core val tables

        tbl_map = {r["table"].lower(): r for r in rows}
        # Verify agreement with X11 applied repairs on unseen tables
        assert int(tbl_map["german"]["gated_repairs"]) == 5476
        assert int(tbl_map["soccer"]["gated_repairs"]) == 217
        assert int(tbl_map["billionaire"]["gated_repairs"]) == 1435
        assert int(tbl_map["marketing"]["gated_repairs"]) == 320
        assert int(tbl_map["restaurant"]["gated_repairs"]) == 59

        for r in rows:
            raw_h = float(r["raw_hallucination_pct"])
            gated_h = float(r["gated_hallucination_pct"])
            assert 0.0 <= raw_h <= 100.0
            assert 0.0 <= gated_h <= 100.0
            assert gated_h <= raw_h + 1e-5  # Gating suppresses or equals hallucination rate
            assert int(r["gated_clean_corrupted"]) <= int(r["raw_clean_corrupted"])


class TestDocumentationAndDeliverables:
    """Verifies environment documentation, figures, and draft files."""

    def test_figures_exist_on_disk(self):
        fig_dir = _REPO_ROOT / "docs" / "figures" / "m5"
        for fname in [
            "confusion_matrices_test.png",
            "precision_recall_curves.png",
            "reliability_diagrams_m5.png",
        ]:
            p = fig_dir / fname
            assert p.exists() and p.stat().st_size > 1000, f"Figure {fname} must exist and be non-empty"

    def test_environment_spec_exists(self):
        env_path = _REPO_ROOT / "docs" / "evaluation_environment_m5.md"
        assert env_path.exists()

        env_content = env_path.read_text()
        # Ensure no forbidden hardware details
        assert "A100" not in env_content
        assert "RTX" not in env_content
        assert "VRAM" not in env_content
        # Ensure pinned versions match requirements.txt
        assert "torch == 2.14.1" in env_content
        assert "transformers == 5.20.1" in env_content

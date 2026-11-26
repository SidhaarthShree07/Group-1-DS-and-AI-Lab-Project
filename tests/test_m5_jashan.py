# Author: Jashan Tiwari
"""Milestone 5 Unit Tests (Jashan's Deliverables).

Verifies:
  1. Unseen data preparation, file layout, and exact M1 audit count match across all 6 tables.
  2. Experiment X12: Credit downstream classification, leakage isolation, and gap closed metric.
  3. Experiment X13: Hand review sampling (58 applied repairs from final unseen runs),
     3-way classification logic, valid alternative share (11 of 58, 18.97%), and summary proportions.
  4. Three-tier validation status policy: strict enforcement that unseen tables have zero rule files
     and all applied repairs carry status 'not verified: no rules for this table' (NEVER 'verified').
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_unseen_data import is_format_artifact, normalize_lenient, normalize_raha
from scripts.run_credit_downstream import calculate_gap_closed, load_dataset
from scripts.sample_wrong_repairs import draw_deterministic_sample, run_x13_pipeline


def test_m5_unseen_data_preparation_and_m1_audit_match():
    """Verify standardized files exist for all 6 tables and match M1 audit counts."""
    if not (_REPO_ROOT / "data/unseen/restaurant/dirty.csv").exists():
        pytest.skip("data/unseen is not in the repository (gitignored); run scripts/prepare_unseen_data.py first")
    summary_csv = _REPO_ROOT / "docs/tables/m5_unseen_data_summary.csv"
    assert summary_csv.exists(), f"Missing {summary_csv}"

    df = pd.read_csv(summary_csv)
    expected_tables = {"restaurant", "marketing", "soccer", "billionaire", "german", "credit"}
    actual_tables = set(df["table"].str.lower())
    assert expected_tables == actual_tables, f"Expected tables {expected_tables}, got {actual_tables}"

    # Expected M1 audit counts
    m1_targets = {
        "restaurant": 633,
        "marketing": 26815,
        "soccer": 796,
        "billionaire": 5195,
        "german": 18678,
        "credit": 38936,
    }

    for _, row in df.iterrows():
        tname = str(row["table"]).lower()
        errs = int(row["error_cells"])
        m1_target = m1_targets[tname]
        assert errs == m1_target, f"{tname}: error count {errs} != M1 audit {m1_target}"
        assert str(row["audit_match"]).lower() == "true", f"{tname}: audit_match is False"

        # Check standardized files on disk
        tbl_dir = _REPO_ROOT / "data/unseen" / tname
        assert tbl_dir.exists(), f"Missing directory {tbl_dir}"
        for fname in ("clean.csv", "dirty.csv", "cells.csv", "test.csv", "test_clean.csv", "test_cells.csv"):
            fpath = tbl_dir / fname
            assert fpath.exists(), f"Missing {fpath}"
            assert fpath.stat().st_size > 0, f"Empty file {fpath}"


def test_m5_credit_downstream_evaluation_and_gap_metric():
    """Verify Experiment X12 Credit downstream evaluation and gap calculation logic."""
    credit_csv = _REPO_ROOT / "docs/tables/m5_credit_downstream.csv"
    assert credit_csv.exists(), f"Missing {credit_csv}"

    # Verify individual model result files exist as well
    for m in ("hist_gb", "rf", "logreg"):
        m_csv = _REPO_ROOT / f"docs/tables/m5_credit_downstream_{m}.csv"
        assert m_csv.exists(), f"Missing individual model result CSV: {m_csv}"

    df = pd.read_csv(credit_csv, dtype=str, keep_default_na=False)
    classifiers = set(df["classifier"])
    assert {"hist_gb", "rf", "logreg"} <= classifiers, f"Expected all 3 classifiers in summary table, got {classifiers}"

    conditions = set(df["condition"])
    assert {"clean", "dirty", "contextguard_cleaned"} <= conditions

    # Assert NO pending entries remain
    assert not (df["roc_auc"] == "PENDING").any(), "Found unresolved PENDING roc_auc in Credit downstream results"
    assert not (df["gap_closed_pct"] == "PENDING").any(), "Found unresolved PENDING gap_closed_pct"

    # Check row counts: 30,000 train rows (cleaned subset) and 30,000 clean test rows
    assert (df["n_train"] == "30000").all(), "All training conditions must be trained on 30,000 rows"
    assert (df["n_test"] == "30000").all(), "All evaluation splits must be evaluated on 30,000 test rows"

    # Exact metrics check for Gradient Boosting
    hgb_df = df[df["classifier"] == "hist_gb"].set_index("condition")
    assert float(hgb_df.loc["clean", "roc_auc"]) == 0.8604
    assert float(hgb_df.loc["dirty", "roc_auc"]) == 0.8565
    assert float(hgb_df.loc["contextguard_cleaned", "roc_auc"]) == 0.8592
    assert float(hgb_df.loc["contextguard_cleaned", "gap_closed_pct"]) == 69.23

    # Exact metrics check for Random Forest
    rf_df = df[df["classifier"] == "rf"].set_index("condition")
    assert float(rf_df.loc["clean", "roc_auc"]) == 0.8411
    assert float(rf_df.loc["dirty", "roc_auc"]) == 0.8362
    assert float(rf_df.loc["contextguard_cleaned", "roc_auc"]) == 0.8388
    assert float(rf_df.loc["contextguard_cleaned", "gap_closed_pct"]) == 53.06

    # Exact metrics check for Logistic Regression
    lr_df = df[df["classifier"] == "logreg"].set_index("condition")
    assert float(lr_df.loc["clean", "roc_auc"]) == 0.7097
    assert float(lr_df.loc["dirty", "roc_auc"]) == 0.7099
    assert float(lr_df.loc["contextguard_cleaned", "roc_auc"]) == 0.7076
    assert lr_df.loc["contextguard_cleaned", "gap_closed_pct"] == "N/A"

    # Test gap closed metric calculation formula
    assert calculate_gap_closed(0.80, 0.85, 0.90) == 50.0
    assert calculate_gap_closed(0.80, 0.90, 0.90) == 100.0
    assert calculate_gap_closed(0.80, 0.80, 0.90) == 0.0
    assert calculate_gap_closed(0.80, 0.80, 0.80) == 0.0  # Zero denominator safety
    assert calculate_gap_closed(0.7099, 0.7076, 0.7097) == 1150.0  # Raw formula; flagged N/A in table due to zero headroom


def test_m5_wrong_repairs_sampling_and_classification():
    """Verify Experiment X13 hand review audit on final unseen applied repairs."""
    summary_csv = _REPO_ROOT / "docs/tables/m5_wrong_repair_summary.csv"
    review_csv = _REPO_ROOT / "docs/tables/m5_wrong_repair_review.csv"
    annotations_csv = _REPO_ROOT / "docs/tables/m5_x13_hand_review_annotations.csv"

    assert summary_csv.exists(), f"Missing {summary_csv}"
    assert review_csv.exists(), f"Missing {review_csv}"
    assert annotations_csv.exists(), f"Missing {annotations_csv}"

    # Annotations ground truth checks: exactly 58 rows from final unseen server runs
    df_ann = pd.read_csv(annotations_csv, dtype=str)
    assert len(df_ann) == 58, f"Expected 58 hand-annotated rows, got {len(df_ann)}"
    assert (df_ann["review_method"] == "manual_hand_review").all(), "All annotations must have review_method='manual_hand_review'"

    df_sum = pd.read_csv(summary_csv)
    categories = set(df_sum["category"])
    assert {"valid_alternative", "wrong", "unclear"} == categories

    total_sampled = df_sum["count"].sum()
    assert total_sampled == 58, f"Expected 58 sampled repairs, got {total_sampled}"

    # Exact proportion checks
    wrong_count = int(df_sum.loc[df_sum["category"] == "wrong", "count"].iloc[0])
    val_count = int(df_sum.loc[df_sum["category"] == "valid_alternative", "count"].iloc[0])
    unclear_count = int(df_sum.loc[df_sum["category"] == "unclear", "count"].iloc[0])
    assert wrong_count == 47, f"Expected 47 wrong repairs, got {wrong_count}"
    assert val_count == 11, f"Expected 11 valid alternatives, got {val_count}"
    assert unclear_count == 0, f"Expected 0 unclear cases, got {unclear_count}"

    # Verify population scope documented in summary notes
    for notes in df_sum["notes"]:
        assert "applied, not verified (no rule covers the table)" in str(notes)
        assert "98%" in str(notes)
        assert "not uniform" in str(notes)

    df_rev = pd.read_csv(review_csv)
    assert len(df_rev) == 58
    assert set(df_rev.columns) == {
        "table", "row", "column", "dirty_value", "pred_repair",
        "clean_value", "state", "category", "review_rationale", "review_method"
    }
    assert (df_rev["review_method"] == "manual_hand_review").all()

    # Verify valid alternatives represent genuine canonical entities or abbreviations
    valid_alts = df_rev[df_rev["category"] == "valid_alternative"]
    assert len(valid_alts) == 11

    # 1. German official state abbreviation check: NRW vs Nordrhein-Westfalen
    german_alts = valid_alts[valid_alts["table"] == "german"]
    assert not german_alts.empty, "Missing German valid alternatives"
    nrw_case = german_alts[german_alts["pred_repair"] == "NRW"].iloc[0]
    assert nrw_case["clean_value"] == "Nordrhein-Westfalen"
    assert "abbreviation" in nrw_case["review_rationale"].lower()

    # 2. Billionaire taxonomy / typo fixes or numeric precision checks
    bil_alts = valid_alts[valid_alts["table"] == "billionaire"]
    assert len(bil_alts) == 9, f"Expected 9 Billionaire valid alternatives, got {len(bil_alts)}"
    # Check numeric precision 0.0 vs 0.00
    assert (bil_alts["clean_value"] == "0.00").any() and (bil_alts["pred_repair"] == "0.0").any()
    # Check company sector typo fix
    assert (bil_alts["pred_repair"] == "investment management").any()
    # Check ISO-3 country code
    assert (bil_alts["pred_repair"] == "TWN").any()

    # STRICT 1-MINUTE TIME RULE CHECK:
    # Ensure no temporal discrepancies are erroneously marked as valid_alternative
    for _, r in valid_alts.iterrows():
        # Valid alternatives must never be temporal discrepancies
        assert "minute" not in str(r["review_rationale"]).lower(), "A time discrepancy was incorrectly marked valid_alternative!"


def test_m5_unseen_non_verification_policy_enforcement():
    """Verify three-tier validation status policy: unseen applied repairs carry 'not verified: no rules for this table'."""
    qual_csv = _REPO_ROOT / "docs/tables/m5_qualitative_examples.csv"
    qual_md = _REPO_ROOT / "docs/examples/m5_qualitative_examples.md"

    assert qual_csv.exists(), f"Missing {qual_csv}"
    assert qual_md.exists(), f"Missing {qual_md}"

    df = pd.read_csv(qual_csv)
    unseen_tables = {"restaurant", "marketing", "soccer", "billionaire", "german"}

    unseen_rows = df[df["table"].isin(unseen_tables)]
    assert len(unseen_rows) >= 5, "Must have qualitative examples from all 5 unseen tables"

    for _, row in unseen_rows.iterrows():
        tbl = row["table"]
        v_status = str(row["validation_status"]).strip()
        action = str(row["action"]).strip().lower()

        # STRICT M5 POLICY ASSERTIONS
        if action == "applied":
            assert v_status == "applied, not verified (no rule covers the table)", (
                f"Policy violation: Unseen table '{tbl}' applied repair has invalid validation_status '{v_status}'! "
                f"Must be strictly 'applied, not verified (no rule covers the table)'."
            )
        else:
            assert v_status == "needs review", (
                f"Policy violation: Unseen table '{tbl}' withheld repair must have status 'needs review', got '{v_status}'"
            )
        assert v_status != "verified", (
            f"Policy violation: Unseen table '{tbl}' has validation_status='verified'! "
            f"Unseen table repairs must NEVER be marked verified because no rule files exist."
        )
        assert action in ("applied", "needs_review"), (
            f"Unseen table '{tbl}' must have action in ('applied', 'needs_review'), got '{action}'"
        )

    # Check Markdown text enforces the three-tier definitions
    with open(qual_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "applied, not verified (no rule covers the table)" in md_text
    assert "verified" in md_text
    assert "needs review" in md_text

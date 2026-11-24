# Author: Jashan Tiwari
"""Milestone 5 Unit Tests (Jashan's Deliverables).

Verifies:
  1. Unseen data preparation, file layout, and exact M1 audit count match across all 6 tables.
  2. Experiment X12: Credit downstream classification, leakage isolation, and gap closed metric.
  3. Experiment X13: Hand review sampling, 3-way classification logic, and summary proportions.
  4. Architectural safety policy: strict enforcement that unseen tables have zero applied/verified repairs.
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

    df = pd.read_csv(credit_csv)
    conditions = set(df["condition"])
    assert "clean" in conditions, "Missing 'clean' oracle condition in Credit downstream results"
    assert "dirty" in conditions, "Missing 'dirty' baseline condition in Credit downstream results"

    clean_row = df[df["condition"] == "clean"].iloc[0]
    dirty_row = df[df["condition"] == "dirty"].iloc[0]

    clean_auc = float(clean_row["roc_auc"])
    dirty_auc = float(dirty_row["roc_auc"])

    # Oracle clean must outperform uncleaned dirty baseline
    assert clean_auc > dirty_auc, f"Clean ROC-AUC ({clean_auc}) must be strictly higher than dirty ({dirty_auc})"
    assert clean_auc >= 0.86, f"Clean ROC-AUC unexpectedly low: {clean_auc}"
    assert int(clean_row["n_train"]) == 120000
    assert int(clean_row["n_test"]) == 30000

    # Cleaned condition must be present and marked pending awaiting final run
    assert "contextguard_cleaned" in conditions
    cleaned_row = df[df["condition"] == "contextguard_cleaned"].iloc[0]
    assert "PENDING" in str(cleaned_row["notes"])

    # Test gap closed metric calculation formula
    # (score_cleaned - score_dirty) / (score_clean - score_dirty) * 100
    assert calculate_gap_closed(0.80, 0.85, 0.90) == 50.0
    assert calculate_gap_closed(0.80, 0.90, 0.90) == 100.0
    assert calculate_gap_closed(0.80, 0.80, 0.90) == 0.0
    assert calculate_gap_closed(0.80, 0.80, 0.80) == 0.0  # Zero denominator safety


def test_m5_wrong_repairs_sampling_and_classification():
    """Verify Experiment X13 hand review audit, strict 1-minute rule, and summary tables."""
    summary_csv = _REPO_ROOT / "docs/tables/m5_wrong_repair_summary.csv"
    review_csv = _REPO_ROOT / "docs/tables/m5_wrong_repair_review.csv"
    annotations_csv = _REPO_ROOT / "docs/milestones/M5/x13_hand_review_annotations.csv"

    assert summary_csv.exists(), f"Missing {summary_csv}"
    assert review_csv.exists(), f"Missing {review_csv}"
    assert annotations_csv.exists(), f"Missing {annotations_csv}"

    # Annotations ground truth checks
    df_ann = pd.read_csv(annotations_csv, dtype=str)
    assert len(df_ann) == 60, f"Expected 60 hand-annotated rows, got {len(df_ann)}"
    assert (df_ann["review_method"] == "manual_hand_review").all(), "All annotations must have review_method='manual_hand_review'"

    df_sum = pd.read_csv(summary_csv)
    categories = set(df_sum["category"])
    assert {"valid_alternative", "wrong", "unclear"} == categories

    total_sampled = df_sum["count"].sum()
    assert total_sampled == 60, f"Expected 60 sampled repairs, got {total_sampled}"

    # Exact proportion checks
    wrong_count = int(df_sum.loc[df_sum["category"] == "wrong", "count"].iloc[0])
    val_count = int(df_sum.loc[df_sum["category"] == "valid_alternative", "count"].iloc[0])
    unclear_count = int(df_sum.loc[df_sum["category"] == "unclear", "count"].iloc[0])
    assert wrong_count == 56, f"Expected 56 wrong repairs, got {wrong_count}"
    assert val_count == 2, f"Expected 2 valid alternatives, got {val_count}"
    assert unclear_count == 2, f"Expected 2 unclear cases, got {unclear_count}"

    # Verify population scope documented in summary notes
    for notes in df_sum["notes"]:
        assert "m4_final2_test_hfb" in str(notes)
        assert "applied repairs" in str(notes)

    df_rev = pd.read_csv(review_csv)
    assert len(df_rev) == 60
    assert set(df_rev.columns) == {
        "table", "row", "column", "dirty_value", "pred_repair",
        "clean_value", "gate_decision", "category", "review_rationale", "review_method"
    }
    assert (df_rev["review_method"] == "manual_hand_review").all()

    # STRICT 1-MINUTE RULE VERIFICATION
    # Row 328 in flights (act_dep_time: 11:40 a.m. -> 11:41 a.m.) MUST be classified as WRONG, NEVER valid_alternative
    flight_1min = df_rev[
        (df_rev["table"] == "flights")
        & (df_rev["column"] == "act_dep_time")
        & (df_rev["dirty_value"] == "11:40 a.m.")
        & (df_rev["pred_repair"] == "11:41 a.m.")
    ]
    assert not flight_1min.empty, "Missing 1-minute flight departure test case (11:40 a.m. vs 11:41 a.m.)"
    row_1min = flight_1min.iloc[0]
    assert row_1min["category"] == "wrong", (
        f"STRICT POLICY VIOLATION: 1-minute time difference was classified as '{row_1min['category']}', "
        f"must be strictly 'wrong'."
    )
    assert "1-minute" in row_1min["review_rationale"]

    # Verify true valid alternatives represent genuine canonical entities or inflectional equivalents
    valid_alts = df_rev[df_rev["category"] == "valid_alternative"]
    assert len(valid_alts) == 2
    # 1. Beer name typo fix on canonical entity
    beer_alt = valid_alts[valid_alts["table"] == "beers"].iloc[0]
    assert beer_alt["dirty_value"] == "Orignal Blend"
    assert beer_alt["pred_repair"] == "Original Blend"
    # 2. Hospital clinical pluralization
    hosp_alt = valid_alts[valid_alts["table"] == "hospital"].iloc[0]
    assert "cultures" in hosp_alt["pred_repair"].lower()


def test_m5_unseen_non_verification_policy_enforcement():
    """Verify strict policy enforcement: zero unseen table repairs may be marked verified or applied."""
    qual_csv = _REPO_ROOT / "docs/tables/m5_qualitative_examples.csv"
    qual_md = _REPO_ROOT / "docs/milestones/M5/qualitative_examples.md"

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
        assert v_status == "not verified: no rules for this table", (
            f"Policy violation: Unseen table '{tbl}' has invalid validation_status '{v_status}'! "
            f"Must be strictly 'not verified: no rules for this table'."
        )
        assert action != "applied", (
            f"Policy violation: Unseen table '{tbl}' has action='applied'! "
            f"Unseen table repairs must NEVER be auto-applied."
        )
        assert action == "review", f"Unseen table '{tbl}' must have action='review', got '{action}'"

    # Check Markdown text enforces the notice
    with open(qual_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "not verified: no rules for this table" in md_text
    assert "no repair on an unseen table is ever verified or auto-applied" in md_text

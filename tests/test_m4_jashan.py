# Author: Jashan Tiwari
# ContextGuard Milestone 4 Unit Tests (Jashan's Deliverables)
# Tests M4 YAML configs, plot_training parser, D7 rule validity, and sample output selector.

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_rule_validity import check_rule_validity, evaluate_rule_on_clean, evaluate_rule_on_dirty, parse_rule_entry
from scripts.pick_examples import categorize_cells, select_qualitative_sample, write_sample_outputs_csv, write_sample_outputs_markdown
from scripts.plot_training import collect_run_records, parse_train_log, write_hparam_table, write_seed_spread_table, write_training_artifacts_table


def test_m4_yaml_configs_valid():
    """Verify that all M4 YAML configs load cleanly, have required keys, and match real GPU runs."""
    configs_dir = _REPO_ROOT / "configs"
    m4_configs = list(configs_dir.glob("m4_*.yaml"))
    assert len(m4_configs) >= 13, f"Expected at least 13 M4 configs, found {len(m4_configs)}"

    run_ids = set()
    expected_runs = {
        "m4_f1_full", "m4_h0_ref", "m4_h1_lr1e-4", "m4_h2_lr4e-4", "m4_h3_r8", "m4_h4_r32",
        "m4_h5_batch16", "m4_h6_constant", "m4_h7_wd0.01", "m4_h8_dropout0.1",
        "m4_s1_seed7", "m4_s2_seed123", "m4_e5_no_synthetic",
    }

    for cfg_path in m4_configs:
        if cfg_path.name == "m4_final_pipeline.yaml":
            # Pipeline config has stages structure
            with open(cfg_path) as f:
                pcfg = yaml.safe_load(f)
            assert "stage_1" in pcfg
            assert "stage_3" in pcfg
            assert "stage_4" in pcfg
            continue
        if cfg_path.name == "m4_calibration_study.yaml":
            # Calibration study output (not a training run)
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        assert "run_id" in cfg, f"{cfg_path.name} missing run_id"
        assert "seed" in cfg, f"{cfg_path.name} missing seed"
        assert "paths" in cfg, f"{cfg_path.name} missing paths"
        assert "model" in cfg, f"{cfg_path.name} missing model"
        assert "lora" in cfg, f"{cfg_path.name} missing lora"
        assert "training" in cfg, f"{cfg_path.name} missing training"

        run_id = cfg["run_id"]
        assert run_id not in run_ids, f"Duplicate run_id {run_id} in {cfg_path.name}"
        run_ids.add(run_id)

        # Check model defaults
        assert cfg["model"]["name"] == "google/gemma-4-E4B-it"
        assert cfg["model"]["max_len"] == 1536
        assert cfg["model"]["gradient_checkpointing"] is True

        # Check LoRA defaults
        assert "q_proj" in cfg["lora"]["target_modules"]
        assert "v_proj" in cfg["lora"]["target_modules"]

        # Check training defaults
        assert cfg["training"]["optimizer"] == "adamw_torch"
        assert cfg["training"]["loss_mask_prompt"] is True
        assert cfg["training"]["ignore_index"] == -100

    assert expected_runs.issubset(run_ids), f"Missing runs: {expected_runs - run_ids}"

    # Specific real GPU run checks
    h0 = yaml.safe_load((configs_dir / "m4_h0_ref.yaml").read_text())
    assert h0["training"]["learning_rate"] == 2.0e-4
    assert h0["lora"]["rank"] == 16
    assert h0["training"]["effective_batch_size"] == 32

    h1 = yaml.safe_load((configs_dir / "m4_h1_lr1e-4.yaml").read_text())
    assert h1["training"]["learning_rate"] == 1.0e-4

    h2 = yaml.safe_load((configs_dir / "m4_h2_lr4e-4.yaml").read_text())
    assert h2["training"]["learning_rate"] == 4.0e-4

    h3 = yaml.safe_load((configs_dir / "m4_h3_r8.yaml").read_text())
    assert h3["lora"]["rank"] == 8
    assert h3["lora"]["alpha"] == 16

    h4 = yaml.safe_load((configs_dir / "m4_h4_r32.yaml").read_text())
    assert h4["lora"]["rank"] == 32
    assert h4["lora"]["alpha"] == 64

    h5 = yaml.safe_load((configs_dir / "m4_h5_batch16.yaml").read_text())
    assert h5["training"]["effective_batch_size"] == 16
    assert h5["training"]["max_steps"] == 400

    s1 = yaml.safe_load((configs_dir / "m4_s1_seed7.yaml").read_text())
    assert s1["seed"] == 7
    assert s1["training"]["learning_rate"] == 2.0e-4

    s2 = yaml.safe_load((configs_dir / "m4_s2_seed123.yaml").read_text())
    assert s2["seed"] == 123

    f1 = yaml.safe_load((configs_dir / "m4_f1_full.yaml").read_text())
    assert f1["training"]["max_steps"] == 1722
    assert f1["training"]["eval_steps"] == 200
    assert f1["training"]["n_val"] == 512
    assert f1["training"]["early_stopping_patience"] == 3

    e5 = yaml.safe_load((configs_dir / "m4_e5_no_synthetic.yaml").read_text())
    assert e5["training"]["exclude_synthetic"] is True


def test_plot_training_parses_logs_and_writes_table(tmp_path: Path):
    """Test plot_training.py log parsing, early stopping detection, and output tables."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create run A (full 100 steps)
    run_a = runs_dir / "run_a"
    run_a.mkdir()
    log_a = [
        ["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"],
        [0, 0, "", 1.20, 0, 0],
        [25, 800, 0.40, 0.15, "2.00e-04", 300],
        [50, 1600, 0.10, 0.08, "1.50e-04", 600],
        [75, 2400, 0.05, 0.05, "5.00e-05", 900],
        [100, 3200, 0.04, 0.06, "0.00e+00", 1200],
    ]
    with open(run_a / "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerows(log_a)
    (run_a / "train_summary.json").write_text(json.dumps({
        "config": {"lr": 2.0e-4, "rank": 16, "seed": 42, "max_steps": 100, "micro_batch": 4, "accum": 8},
        "seconds": 1200.0,
        "stop_reason": "max steps reached",
        "best_val_loss": 0.05,
        "best_step": 75,
    }))

    # Create run B (early stopped at step 50)
    run_b = runs_dir / "run_b"
    run_b.mkdir()
    log_b = [
        ["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"],
        [0, 0, "", 1.20, 0, 0],
        [25, 800, 0.35, 0.07, "2.00e-04", 300],
        [50, 1600, 0.08, 0.06, "1.50e-04", 600],
    ]
    with open(run_b / "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerows(log_b)
    (run_b / "train_summary.json").write_text(json.dumps({
        "config": {"lr": 2.0e-4, "rank": 16, "seed": 7, "max_steps": 100, "micro_batch": 4, "accum": 8},
        "seconds": 600.0,
        "stop_reason": "early stop: 3 evaluations without improvement",
        "best_val_loss": 0.06,
        "best_step": 50,
        "stopped_early": True,
    }))

    fixed_map = {"run_a": 0.051, "run_b": 0.049}
    records = collect_run_records(runs_dir, fixed_val_map=fixed_map)
    assert len(records) == 2

    # Check run A
    rec_a = next(r for r in records if r["run"] == "run_a")
    assert rec_a["learning_rate"] == 2.0e-4
    assert rec_a["rank"] == 16
    assert rec_a["best_val_loss"] == 0.05
    assert rec_a["best_step"] == 75
    assert rec_a["fixed_val_loss"] == 0.051
    assert rec_a["stopped_early"] is False
    assert rec_a["minutes"] == 20.0

    # Check run B
    rec_b = next(r for r in records if r["run"] == "run_b")
    assert rec_b["best_val_loss"] == 0.06
    assert rec_b["best_step"] == 50
    assert rec_b["fixed_val_loss"] == 0.049
    assert rec_b["stopped_early"] is True
    assert rec_b["minutes"] == 10.0

    # Test hparam CSV output
    hparam_csv = tmp_path / "m4_hparam_runs.csv"
    hparam_df = write_hparam_table(records, hparam_csv)
    expected_cols = [
        "run", "learning_rate", "rank", "seed", "batch", "schedule",
        "weight_decay", "dropout", "synthetic", "best_val_loss",
        "best_step", "fixed_val_loss", "stopped_early", "stop_reason", "minutes"
    ]
    assert list(hparam_df.columns) == expected_cols
    assert len(hparam_df) == 2

    # Test seed spread CSV output
    seed_csv = tmp_path / "m4_seed_spread.csv"
    seed_df = write_seed_spread_table(records, seed_csv)
    assert list(seed_df.columns) == ["metric", "runs", "mean", "std", "min", "max", "range"]


def test_check_rule_validity_catches_typo_rule(tmp_path: Path):
    """Test that check_rule_validity distinguishes valid domain rules, handles case mapping, and reads cells.csv."""
    tbl = "toy_table"
    tbl_dir = tmp_path / tbl
    tbl_dir.mkdir(parents=True)

    # Clean table: city is always 'Albany' or 'Boston'
    clean_df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "state": ["NY", "NY", "MA", "MA"],
        "city": ["Albany", "Albany", "Boston", "Boston"],
    })
    clean_df.to_csv(tbl_dir / "val_clean.csv", index=False)

    # Dirty table: row 1 has corrupted city 'Albxny'
    dirty_df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "state": ["NY", "NY", "MA", "MA"],
        "city": ["Albany", "Albxny", "Boston", "Boston"],
    })
    dirty_df.to_csv(tbl_dir / "val.csv", index=False)

    # Ground truth cells with capitalized column names to test FIX 1 column-name case normalization
    cells_rows = [
        {"table": tbl, "row": 0, "column": "City", "dirty_value": "Albany", "clean_value": "Albany", "is_error": False, "error_type": "none"},
        {"table": tbl, "row": 1, "column": "City", "dirty_value": "Albxny", "clean_value": "Albany", "is_error": True, "error_type": "typo"},
        {"table": tbl, "row": 2, "column": "City", "dirty_value": "Boston", "clean_value": "Boston", "is_error": False, "error_type": "none"},
        {"table": tbl, "row": 3, "column": "City", "dirty_value": "Boston", "clean_value": "Boston", "is_error": False, "error_type": "none"},
    ]
    pd.DataFrame(cells_rows).to_csv(tbl_dir / "cells.csv", index=False)

    rules = {
        tbl: {
            "rules": [
                {"kind": "pattern", "column": "state", "regex": "^[A-Z]{2}$", "kept": True, "reason": "holds"},
                {"kind": "allowed_values", "column": "city", "values": ["Albany", "Boston"], "kept": True, "reason": "holds"},
                {"kind": "pattern", "column": "city", "regex": "^Albxny$", "kept": True, "reason": "dirty typo rule"},
            ]
        }
    }
    rules_json_path = tmp_path / "rules.json"
    rules_json_path.write_text(json.dumps(rules, indent=2))

    out_csv = tmp_path / "m4_rule_validity.csv"
    res_df = check_rule_validity(rules_json_path, tmp_path, cells_csv=tbl_dir / "cells.csv", out_csv=out_csv)

    assert list(res_df.columns) == ["table", "rule", "clean_support", "valid", "flagged", "true_errors", "precision", "new_errors"]
    assert len(res_df) == 3

    # Rule 1: state pattern -> valid
    r1 = res_df[res_df["rule"].str.contains("state pattern")].iloc[0]
    assert r1["clean_support"] == 1.0
    assert bool(r1["valid"]) is True

    # Rule 2: clean city allowed_values -> valid, flags dirty row 1, matches true error despite case difference
    r2 = res_df[res_df["rule"].str.contains("city allowed_values")].iloc[0]
    assert r2["clean_support"] == 1.0
    assert bool(r2["valid"]) is True
    assert r2["flagged"] == 1
    assert r2["true_errors"] == 1
    assert r2["precision"] == 1.0

    # Rule 3: typo regex (^Albxny$) -> clean_support is 0.0 -> valid MUST BE False!
    r3 = res_df[res_df["rule"].str.contains("Albxny")].iloc[0]
    assert r3["clean_support"] == 0.0
    assert bool(r3["valid"]) is False


def test_pick_examples_categorization():
    """Test cell categorization across all 4 outcome categories, including rejected repairs not in Cat 2."""
    cells_data = [
        # Cat 1: Correct repair
        {"table": "t1", "row": 0, "column": "c1", "dirty_value": "err1", "clean_value": "fix1", "is_error": "True"},
        # Cat 2: Applied Wrong repair
        {"table": "t1", "row": 1, "column": "c1", "dirty_value": "err2", "clean_value": "fix2", "is_error": "True"},
        # Cat 3: Clean cell falsely flagged
        {"table": "t1", "row": 2, "column": "c1", "dirty_value": "clean3", "clean_value": "clean3", "is_error": "False"},
        # Cat 4a: Missed error (non-candidate)
        {"table": "t1", "row": 3, "column": "c1", "dirty_value": "err4", "clean_value": "fix4", "is_error": "True"},
        # Cat 4b: Missed error (candidate kept)
        {"table": "t1", "row": 4, "column": "c1", "dirty_value": "err5", "clean_value": "fix5", "is_error": "True"},
        # Cat 4c: Validation rejected repair (e.g. Hospital row 103) - must NOT be in Cat 2!
        {"table": "t1", "row": 5, "column": "c1", "dirty_value": "err6", "clean_value": "fix6", "is_error": "True"},
    ]
    cells_df = pd.DataFrame(cells_data)

    cands_map = {
        ("t1", 0, "c1"): {"is_candidate": True},
        ("t1", 1, "c1"): {"is_candidate": True},
        ("t1", 2, "c1"): {"is_candidate": True},
        ("t1", 3, "c1"): {"is_candidate": False},  # Non-candidate
        ("t1", 4, "c1"): {"is_candidate": True},
        ("t1", 5, "c1"): {"is_candidate": True},
    }

    preds_map = {
        ("t1", 0, "c1"): {"table": "t1", "row": 0, "column": "c1", "is_error": True, "repair": "fix1", "p_error": 0.95, "gate_decision": "repair", "decision": "repair"},
        ("t1", 1, "c1"): {"table": "t1", "row": 1, "column": "c1", "is_error": True, "repair": "wrong", "p_error": 0.85, "gate_decision": "repair", "decision": "repair"},
        ("t1", 2, "c1"): {"table": "t1", "row": 2, "column": "c1", "is_error": True, "repair": "fix3", "p_error": 0.75, "gate_decision": "flag for review", "decision": "flag for review"},
        ("t1", 4, "c1"): {"table": "t1", "row": 4, "column": "c1", "is_error": False, "repair": None, "p_error": 0.15, "gate_decision": "keep", "decision": "keep"},
        ("t1", 5, "c1"): {"table": "t1", "row": 5, "column": "c1", "is_error": True, "repair": "wrong_trunc", "p_error": 0.98, "gate_decision": "repair", "decision": "review (repair rejected by validation)"},
    }

    val_log_map = {
        ("t1", 5, "c1"): {"table": "t1", "row": 5, "column": "c1", "outcome": "rejected", "reason": "fd violated"},
    }

    categorized = categorize_cells(cells_df, cands_map, preds_map, val_log_map)
    assert len(categorized["Category 1: Correct Repairs"]) == 1
    assert len(categorized["Category 2: Wrong Repairs"]) == 1  # only row 1, NOT row 5!
    assert len(categorized["Category 3: Correct Cells Falsely Flagged / Repaired"]) == 1
    assert len(categorized["Category 4: Missed Errors"]) == 3  # 1 non-cand + 1 cand-kept + 1 rejected to review

    main_ex, val_ex = select_qualitative_sample(categorized, val_log_map, seed=42)
    assert len(main_ex) == 5  # 1 from Cat 1, 1 from Cat 2, 1 from Cat 3, 2 from Cat 4 (k=2 cap per table)


def test_training_artifacts_table_schema(tmp_path: Path):
    """Test training artifacts table columns and verify zero private/machine paths."""
    out_csv = tmp_path / "m4_training_artifacts.csv"
    df = write_training_artifacts_table(_REPO_ROOT, out_csv)

    expected_cols = ["Artifact", "Category", "Format", "Size", "Producer_Script", "Tracked_In_Git", "Description"]
    assert list(df.columns) == expected_cols
    assert len(df) >= 15

    # Verify zero absolute paths or machine-specific details
    csv_text = out_csv.read_text()
    assert "/Users" + "/" not in csv_text
    assert "/home/" not in csv_text
    assert "darwin" not in csv_text.lower()
    assert "apple" not in csv_text.lower()

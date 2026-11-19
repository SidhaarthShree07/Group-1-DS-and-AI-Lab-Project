# Author: Jashan Tiwari
# ContextGuard Milestone 4: Training Curve Plotter, Hyper-parameter Summary,
# Seed Spread Analysis, and Training Artifacts Generator.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_train_log(log_path: Path) -> pd.DataFrame:
    """Parse a train_log.csv into a clean pandas DataFrame."""
    df = pd.read_csv(log_path)
    required_cols = ["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {log_path}")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    return df


def load_fixed_val_eval(csv_path: Optional[Path]) -> Dict[str, float]:
    """Load m4_fixed_val_eval.csv mapping run_id to fixed validation loss."""
    fixed_map: Dict[str, float] = {}
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                r_name = str(row["run"]).strip()
                v_loss = float(row["val_loss"])
                fixed_map[r_name] = v_loss
        except Exception as e:
            print(f"[plot_training] Warning: failed to parse fixed val eval {csv_path}: {e}", file=sys.stderr)
    return fixed_map


def load_run_metadata(run_dir: Path, configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Extract hyper-parameters from train_summary.json, run.json, or configs/."""
    meta: Dict[str, Any] = {}

    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                s = json.load(f)
                meta.update(s)
                if "config" in s and isinstance(s["config"], dict):
                    meta.update(s["config"])
        except Exception:
            pass

    run_json = run_dir / "run.json"
    if run_json.exists():
        try:
            with open(run_json) as f:
                r = json.load(f)
                meta.update(r)
                if "args" in r and isinstance(r["args"], dict):
                    meta.update(r["args"])
        except Exception:
            pass

    run_id = run_dir.name
    if configs_dir and configs_dir.exists():
        yaml_candidates = [
            configs_dir / f"{run_id}.yaml",
            configs_dir / f"m4_{run_id}.yaml",
            configs_dir / f"m3_{run_id}.yaml",
        ]
        for yc in yaml_candidates:
            if yc.exists():
                try:
                    with open(yc) as f:
                        yd = yaml.safe_load(f)
                        if "training" in yd:
                            meta.setdefault("learning_rate", yd["training"].get("learning_rate"))
                            meta.setdefault("lr", yd["training"].get("learning_rate"))
                            meta.setdefault("max_steps", yd["training"].get("max_steps"))
                            meta.setdefault("early_stopping_patience", yd["training"].get("early_stopping_patience"))
                            meta.setdefault("lr_scheduler_type", yd["training"].get("lr_scheduler_type"))
                            meta.setdefault("weight_decay", yd["training"].get("weight_decay"))
                            meta.setdefault("per_device_train_batch_size", yd["training"].get("per_device_train_batch_size"))
                            meta.setdefault("gradient_accumulation_steps", yd["training"].get("gradient_accumulation_steps"))
                            meta.setdefault("exclude_synthetic", yd["training"].get("exclude_synthetic"))
                        if "lora" in yd:
                            meta.setdefault("rank", yd["lora"].get("rank"))
                            meta.setdefault("alpha", yd["lora"].get("alpha"))
                            meta.setdefault("dropout", yd["lora"].get("dropout"))
                        if "seed" in yd:
                            meta.setdefault("seed", yd.get("seed"))
                except Exception:
                    pass

    return meta


def collect_run_records(
    runs_dir: Path,
    configs_dir: Optional[Path] = None,
    fixed_val_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Scan runs_dir and return parsed run summaries."""
    records = []
    if not runs_dir.exists():
        return records

    fixed_val_map = fixed_val_map or {}

    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        log_file = sub / "train_log.csv"
        if not log_file.exists():
            continue
        try:
            df = parse_train_log(log_file)
        except Exception as e:
            print(f"[plot_training] Skipping {sub.name}: {e}", file=sys.stderr)
            continue

        meta = load_run_metadata(sub, configs_dir)

        # Best validation loss and step from log
        val_rows = df[df["val_loss"].notna() & (df["val_loss"] > 0)]
        if not val_rows.empty:
            min_idx = val_rows["val_loss"].idxmin()
            best_val_loss = float(val_rows.loc[min_idx, "val_loss"])
            best_step = int(val_rows.loc[min_idx, "step"])
        else:
            best_val_loss = float("nan")
            best_step = -1

        # Use summary values if present
        if "best_val_loss" in meta and isinstance(meta["best_val_loss"], (int, float)):
            best_val_loss = float(meta["best_val_loss"])
        if "best_step" in meta and isinstance(meta["best_step"], (int, float)):
            best_step = int(meta["best_step"])

        # Stop reason and stopped early
        stop_reason = str(meta.get("stop_reason", "max steps reached"))
        if "stopped_early" in meta:
            stopped_early = bool(meta["stopped_early"])
        else:
            stopped_early = stop_reason.lower().startswith("early stop")

        # Elapsed minutes
        if "seconds" in meta and isinstance(meta["seconds"], (int, float)):
            minutes = round(float(meta["seconds"]) / 60.0, 2)
        elif not df.empty and "seconds" in df.columns:
            minutes = round(float(df["seconds"].max()) / 60.0, 2)
        else:
            minutes = 0.0

        # Learning rate
        lr = meta.get("lr") or meta.get("learning_rate")
        if lr is None and not df.empty and "lr" in df.columns:
            for v in df["lr"].dropna():
                try:
                    f = float(v)
                    if f > 0:
                        lr = f
                        break
                except Exception:
                    pass
        if lr is not None:
            lr = float(lr)

        # Rank, seed, batch, schedule, weight decay, dropout, synthetic
        rank_val = meta.get("rank") or meta.get("r") or 16
        rank = int(rank_val)

        seed_val = meta.get("seed")
        seed = int(seed_val) if seed_val is not None else 42

        micro_b_val = meta.get("micro_batch") or meta.get("per_device_train_batch_size") or 4
        micro_b = int(micro_b_val)
        accum_val = meta.get("accum") or meta.get("gradient_accumulation_steps") or 8
        accum = int(accum_val)
        batch = micro_b * accum

        schedule_val = meta.get("schedule") or meta.get("lr_scheduler_type") or "cosine"
        schedule = str(schedule_val)

        wd_val = meta.get("weight_decay")
        weight_decay = float(wd_val) if wd_val is not None else 0.0

        drop_val = meta.get("dropout")
        dropout = float(drop_val) if drop_val is not None else 0.05

        synth_val = meta.get("exclude_synthetic")
        synthetic = not bool(synth_val) if synth_val is not None else True

        # Fixed validation loss (from m4_fixed_val_eval.csv)
        fixed_val_loss = fixed_val_map.get(sub.name, float("nan"))

        records.append({
            "run": sub.name,
            "learning_rate": lr,
            "rank": rank,
            "seed": seed,
            "batch": batch,
            "schedule": schedule,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "synthetic": synthetic,
            "best_val_loss": round(best_val_loss, 4),
            "best_step": best_step,
            "fixed_val_loss": round(fixed_val_loss, 4) if not math.isnan(fixed_val_loss) else float("nan"),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "minutes": minutes,
            "df": df,
            "meta": meta,
        })

    return records


def write_hparam_table(records: List[Dict[str, Any]], out_path: Path) -> pd.DataFrame:
    """Generate docs/tables/m4_hparam_runs.csv with all hyper-parameter fields (13 M4 runs)."""
    rows = []
    for r in records:
        if r["run"] == "m3_smoke":
            continue
        rows.append({
            "run": r["run"],
            "learning_rate": f"{r['learning_rate']:.1e}" if r["learning_rate"] is not None else "",
            "rank": r["rank"],
            "seed": r["seed"],
            "batch": r["batch"],
            "schedule": r["schedule"],
            "weight_decay": r["weight_decay"],
            "dropout": r["dropout"],
            "synthetic": r["synthetic"],
            "best_val_loss": r["best_val_loss"],
            "best_step": r["best_step"],
            "fixed_val_loss": r["fixed_val_loss"] if not math.isnan(r["fixed_val_loss"]) else "",
            "stopped_early": r["stopped_early"],
            "stop_reason": r["stop_reason"],
            "minutes": r["minutes"],
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_seed_spread_table(records: List[Dict[str, Any]], out_path: Path) -> pd.DataFrame:
    """Generate docs/tables/m4_seed_spread.csv on m4_h0_ref, m4_s1_seed7, and m4_s2_seed123."""
    # Preferred real seed group
    exact_names = {"m4_h0_ref", "m4_s1_seed7", "m4_s2_seed123"}
    seed_runs = [r for r in records if r["run"] in exact_names]

    # Fallback to older naming if exact not present
    if not seed_runs:
        seed_runs = [r for r in records if r["run"] in ("m4_h2", "m4_s1", "m4_s2") or "s1" in r["run"] or "s2" in r["run"]]
    if not seed_runs:
        seed_runs = [r for r in records if r.get("seed") in (42, 7, 123)]
    if not seed_runs:
        seed_runs = records

    run_names = ", ".join(sorted(r["run"] for r in seed_runs))
    rows = []

    # 1. Fixed val loss spread (if available)
    fixed_losses = [r["fixed_val_loss"] for r in seed_runs if not math.isnan(r["fixed_val_loss"])]
    if fixed_losses:
        arr_f = np.array(fixed_losses)
        rows.append({
            "metric": "fixed_val_loss_1024",
            "runs": run_names,
            "mean": round(float(np.mean(arr_f)), 4),
            "std": round(float(np.std(arr_f, ddof=1)) if len(arr_f) > 1 else 0.0, 4),
            "min": round(float(np.min(arr_f)), 4),
            "max": round(float(np.max(arr_f)), 4),
            "range": round(float(np.max(arr_f) - np.min(arr_f)), 4),
        })

    # 2. In-run best val loss spread
    val_losses = [r["best_val_loss"] for r in seed_runs if not math.isnan(r["best_val_loss"])]
    if val_losses:
        arr = np.array(val_losses)
        rows.append({
            "metric": "best_val_loss",
            "runs": run_names,
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "range": round(float(np.max(arr) - np.min(arr)), 4),
        })

    # 3. Best step
    best_steps = [r["best_step"] for r in seed_runs if r["best_step"] >= 0]
    if best_steps:
        arr_s = np.array(best_steps)
        rows.append({
            "metric": "best_step",
            "runs": run_names,
            "mean": round(float(np.mean(arr_s)), 1),
            "std": round(float(np.std(arr_s, ddof=1)) if len(arr_s) > 1 else 0.0, 1),
            "min": int(np.min(arr_s)),
            "max": int(np.max(arr_s)),
            "range": int(np.max(arr_s) - np.min(arr_s)),
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def plot_loss_curves(records: List[Dict[str, Any]], out_svg: Path):
    """Publication-grade SVG plot of training loss and validation loss vs. step."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Color palette
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(20)]

    for i, r in enumerate(records):
        df = r["df"]
        is_f1 = "f1" in r["run"].lower()
        color = "#d9381e" if is_f1 else colors[i % len(colors)]
        lw = 2.4 if is_f1 else 1.2
        alpha = 0.95 if is_f1 else 0.70
        label = f"{r['run']} (F1 full)" if is_f1 else r["run"]

        # Training loss
        tr_df = df[df["train_loss"].notna() & (df["train_loss"] > 0)]
        if not tr_df.empty:
            ax1.plot(tr_df["step"], tr_df["train_loss"], color=color, alpha=alpha, linewidth=lw, label=label)

        # Validation loss
        va_df = df[df["val_loss"].notna() & (df["val_loss"] > 0)]
        if not va_df.empty:
            marker = "D" if is_f1 else "o"
            ms = 5 if is_f1 else 3
            ax2.plot(va_df["step"], va_df["val_loss"], color=color, marker=marker, markersize=ms,
                     linewidth=lw, alpha=alpha, label=label)
            # Mark best step
            min_idx = va_df["val_loss"].idxmin()
            ax2.scatter(va_df.loc[min_idx, "step"], va_df.loc[min_idx, "val_loss"],
                        color=color, s=70 if is_f1 else 40, edgecolors="black", zorder=5)

    ax1.set_title("Training Loss vs. Optimiser Step", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Optimiser Step", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss (Answer Tokens Only)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)

    ax2.set_title("Validation Loss vs. Optimiser Step", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Optimiser Step", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_hparam_comparison(records: List[Dict[str, Any]], out_svg: Path):
    """Comparison SVG plot highlighting screening runs, seed spread, and ablation."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Bar plot: validation loss across runs (prefer fixed_val_loss if available, else best_val_loss)
    screening_runs = [r for r in records if "m4_h" in r["run"] or "m4_s" in r["run"] or "m4_e5" in r["run"] or "m4_f1" in r["run"]]
    if not screening_runs:
        screening_runs = records

    labels = [r["run"].replace("m4_", "") for r in screening_runs]
    losses = [r["fixed_val_loss"] if not math.isnan(r["fixed_val_loss"]) else r["best_val_loss"] for r in screening_runs]
    metric_label = "Fixed Val Loss (1,024 ex)" if any(not math.isnan(r["fixed_val_loss"]) for r in screening_runs) else "Best Val Loss"

    x = np.arange(len(labels))
    bar_colors = ["#1f77b4" if "f1" not in l and "e5" not in l else ("#2ca02c" if "f1" in l else "#ff7f0e") for l in labels]

    bars = ax1.bar(x, losses, color=bar_colors, width=0.6, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax1.set_ylabel(metric_label, fontsize=11)
    ax1.set_title(f"Validation Loss Across M4 Configurations ({metric_label})", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if not math.isnan(h) and h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.001, f"{h:.4f}", ha="center", va="bottom", fontsize=7.5)

    # 2. Learning rate comparison trajectories (H1: 1e-4, H0: 2e-4, H2: 4e-4)
    lr_names = ["m4_h1_lr1e-4", "m4_h0_ref", "m4_h2_lr4e-4"]
    lr_runs = [r for r in records if r["run"] in lr_names]
    if len(lr_runs) >= 2:
        for r in lr_runs:
            df = r["df"]
            va_df = df[df["val_loss"].notna() & (df["val_loss"] > 0)]
            if not va_df.empty:
                ax2.plot(va_df["step"], va_df["val_loss"], marker="s", markersize=4, linewidth=1.8,
                         label=f"{r['run']} (lr={r['learning_rate']:.1e})")
        ax2.set_title("Validation Trajectory by Learning Rate (Rank 16, Cosine)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Optimiser Step", fontsize=11)
        ax2.set_ylabel("In-Run Validation Loss", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9, loc="upper right")
    else:
        # Fallback trajectory of available runs
        for r in screening_runs[:4]:
            df = r["df"]
            va_df = df[df["val_loss"].notna() & (df["val_loss"] > 0)]
            if not va_df.empty:
                ax2.plot(va_df["step"], va_df["val_loss"], marker="o", markersize=4, label=r["run"])
        ax2.set_title("Validation Loss Trajectories", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_training_artifacts_table(repo_root: Path, out_path: Path) -> pd.DataFrame:
    """Generate docs/tables/m4_training_artifacts.csv for Report Section 5 with real sizes."""
    artifacts = [
        {
            "Artifact": "Best LoRA Adapter Weights (adapter_best/)",
            "Category": "Model Weights",
            "Format": "safetensors / bin",
            "Size": "~140 MB",
            "Producer_Script": "src/llm/train_lora.py",
            "Tracked_In_Git": False,
            "Description": "Fine-tuned rank-16 LoRA adapter weights saved at optimal validation loss checkpoint (step 1,200 of 1,722 in m4_f1_full).",
        },
        {
            "Artifact": "Final Step LoRA Adapter (adapter_last/)",
            "Category": "Model Weights",
            "Format": "safetensors / bin",
            "Size": "~140 MB",
            "Producer_Script": "src/llm/train_lora.py",
            "Tracked_In_Git": False,
            "Description": "LoRA adapter state saved at final step before early stopping check (step 1,722 in m4_f1_full).",
        },
        {
            "Artifact": "LoRA Adapter Config (adapter_config.json)",
            "Category": "Model Config",
            "Format": "JSON",
            "Size": "850 B",
            "Producer_Script": "src/llm/train_lora.py",
            "Tracked_In_Git": False,
            "Description": "PEFT configuration defining rank=16, alpha=32, dropout=0.05, target modules (q,k,v,o,gate,up,down), and task type CAUSAL_LM.",
        },
        {
            "Artifact": "Training Run Logs (train_log.csv)",
            "Category": "Logs",
            "Format": "CSV",
            "Size": "3.5 KB to 28 KB per run",
            "Producer_Script": "src/llm/train_lora.py",
            "Tracked_In_Git": False,
            "Description": "Step-by-step training log with step, examples_seen, train_loss, val_loss, lr, and elapsed seconds.",
        },
        {
            "Artifact": "Training Summary Records (train_summary.json, run.json)",
            "Category": "Run Metadata",
            "Format": "JSON",
            "Size": "1.2 KB to 2.4 KB per run",
            "Producer_Script": "src/llm/train_lora.py, scripts/run_pipeline.py",
            "Tracked_In_Git": False,
            "Description": "Environment metadata, parameter counts (34.9M trainable LoRA params / 7.94B total), reload parity check, and peak GPU memory.",
        },
        {
            "Artifact": "Fixed Validation Evaluation (docs/tables/m4_fixed_val_eval.csv)",
            "Category": "Evaluation Summary",
            "Format": "CSV",
            "Size": "397 B",
            "Producer_Script": "scripts/eval_adapters.py",
            "Tracked_In_Git": True,
            "Description": "Validation loss evaluated on the exact same 1,024 validation examples across all 13 adapter checkpoints and base model.",
        },
        {
            "Artifact": "Test Benchmark vs Baselines (docs/tables/m4_test_vs_baselines.csv)",
            "Category": "Evaluation Summary",
            "Format": "CSV",
            "Size": "857 B",
            "Producer_Script": "scripts/score_like_baselines.py",
            "Tracked_In_Git": True,
            "Description": "Final test split evaluation across 5 core tables vs Raha (20 labels), dBoost, and constraint check (mean F1: 0.741 vs Raha 0.515).",
        },
        {
            "Artifact": "Final Pipeline Configuration (configs/m4_final_pipeline.yaml)",
            "Category": "Configuration",
            "Format": "YAML",
            "Size": "2.2 KB",
            "Producer_Script": "configs/m4_final_pipeline.yaml",
            "Tracked_In_Git": True,
            "Description": "Frozen pipeline specification: m4_f1_full adapter + D1-D6 + D7 format mix + d5c entity consensus + repair gate 0.9 + validation layer.",
        },
        {
            "Artifact": "Training Run Configurations (configs/m4_*.yaml)",
            "Category": "Configuration",
            "Format": "YAML",
            "Size": "1.0 KB to 1.3 KB each",
            "Producer_Script": "scripts/generate_configs.py",
            "Tracked_In_Git": True,
            "Description": "Reproducible YAML configs for 13 training runs (f1_full, h0_ref..h8_dropout0.1, s1_seed7, s2_seed123, e5_no_synthetic).",
        },
        {
            "Artifact": "Stage 0 Discovered Rules (rules.json)",
            "Category": "Intermediate Rules",
            "Format": "JSON",
            "Size": "4.6 KB",
            "Producer_Script": "src/detectors/llm_rules.py",
            "Tracked_In_Git": False,
            "Description": "Stage 0 proposed rules evaluated by scripts/check_rule_validity.py on clean tables (dropped from final pipeline after E6).",
        },
        {
            "Artifact": "Model Pipeline Predictions (predictions.jsonl)",
            "Category": "Predictions",
            "Format": "JSON Lines",
            "Size": "1.8 MB to 18.2 MB per run",
            "Producer_Script": "scripts/run_pipeline.py",
            "Tracked_In_Git": False,
            "Description": "Cell-level outputs: raw_text, valid_json, is_error, error_type, evidence, repair, p_error, gate_decision, decision, validation.",
        },
        {
            "Artifact": "Validation Audit Log (validation_log.jsonl)",
            "Category": "Audit Log",
            "Format": "JSON Lines",
            "Size": "948 KB",
            "Producer_Script": "src/validation/repair_validator.py",
            "Tracked_In_Git": False,
            "Description": "Detailed log of every post-repair validation check: outcome (validated, rejected, unvalidated), violated rules before and after.",
        },
        {
            "Artifact": "Validated Pipeline Change Log (change_log.csv, change_log_validated.csv)",
            "Category": "Audit Log",
            "Format": "CSV",
            "Size": "191 KB",
            "Producer_Script": "scripts/run_pipeline.py",
            "Tracked_In_Git": False,
            "Description": "Record of all accepted cell repairs with table, row, column, old value, repair value, model score, and validation status.",
        },
        {
            "Artifact": "Review Queue List (review_list_validated.csv)",
            "Category": "Audit Log",
            "Format": "CSV",
            "Size": "33 KB to 80 KB",
            "Producer_Script": "scripts/run_pipeline.py",
            "Tracked_In_Git": False,
            "Description": "Flagged cells routed to human review (either rejected by validation layer or below repair threshold).",
        },
        {
            "Artifact": "Validation Layer Comparison (validation_comparison.csv)",
            "Category": "Evaluation Summary",
            "Format": "CSV",
            "Size": "611 B",
            "Producer_Script": "scripts/validate_repairs.py",
            "Tracked_In_Git": False,
            "Description": "Ablation comparing repair accuracy, false repair rate, and precision with and without independent rule validation.",
        },
        {
            "Artifact": "Final Test Run Folders (outputs/runs/m4_final*)",
            "Category": "Run Outputs",
            "Format": "Directory",
            "Size": "3.5 MB to 12 MB each",
            "Producer_Script": "scripts/run_m4_final_eval.sh",
            "Tracked_In_Git": False,
            "Description": "Outputs for final test evaluations: m4_final2_test_hfb, m4_final_test_rayyan, m4_final_test_movies, m4_final2_val_all.",
        },
        {
            "Artifact": "PyTorch Dataset & Loss Masking (src/llm/dataset.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "14.9 KB",
            "Producer_Script": "src/llm/dataset.py",
            "Tracked_In_Git": True,
            "Description": "Gemma 4 chat templating, prompt token loss masking to -100 (D14), and dynamic micro-batch collation (D13).",
        },
        {
            "Artifact": "LoRA Fine-Tuning Script (src/llm/train_lora.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "9.5 KB",
            "Producer_Script": "src/llm/train_lora.py",
            "Tracked_In_Git": True,
            "Description": "LoRA training loop with AdamW, cosine schedule, early stopping (patience 3), and adapter parity verification.",
        },
        {
            "Artifact": "Training Curves & Run Analysis (scripts/plot_training.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "30.5 KB",
            "Producer_Script": "scripts/plot_training.py",
            "Tracked_In_Git": True,
            "Description": "Automated log discovery, fixed validation loss extraction, seed spread computation, and SVG curve rendering.",
        },
        {
            "Artifact": "Rule Validity Checker (scripts/check_rule_validity.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "15.5 KB",
            "Producer_Script": "scripts/check_rule_validity.py",
            "Tracked_In_Git": True,
            "Description": "Evaluates Stage 0 rules on clean validation ground truth for >=99% support, flags, precision, and new error detection.",
        },
        {
            "Artifact": "Qualitative Example Selector (scripts/pick_examples.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "16.6 KB",
            "Producer_Script": "scripts/pick_examples.py",
            "Tracked_In_Git": True,
            "Description": "Deterministic seeded selection (seed=42) of model predictions across 4 outcome categories plus validation layer cases.",
        },
        {
            "Artifact": "Training Config Generator (scripts/generate_configs.py)",
            "Category": "Code",
            "Format": "Python",
            "Size": "3.6 KB",
            "Producer_Script": "scripts/generate_configs.py",
            "Tracked_In_Git": True,
            "Description": "Generates reproducible YAML configuration files for all 13 training runs from run metadata.",
        },
    ]
    df = pd.DataFrame(artifacts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / "outputs" / "runs", help="Directory containing run logs")
    parser.add_argument("--configs-dir", type=Path, default=_REPO_ROOT / "configs", help="Directory containing run configs")
    parser.add_argument("--tables-dir", type=Path, default=_REPO_ROOT / "docs" / "tables", help="Directory for CSV outputs")
    parser.add_argument("--figures-dir", type=Path, default=_REPO_ROOT / "docs" / "figures" / "m4", help="Directory for SVG outputs")
    parser.add_argument("--fixed-val-eval", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m4_fixed_val_eval.csv",
                        help="Path to m4_fixed_val_eval.csv")
    args = parser.parse_args()

    print("[plot_training] Discovering runs in", args.runs_dir)
    fixed_val_map = load_fixed_val_eval(args.fixed_val_eval)
    records = collect_run_records(args.runs_dir, args.configs_dir, fixed_val_map)

    if not records:
        print("[plot_training] No training records found. Creating directory and exiting.", file=sys.stderr)
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        return 0

    print(f"[plot_training] Processed {len(records)} runs: {[r['run'] for r in records]}")

    # 1. Write hparam runs table
    hparam_csv = args.tables_dir / "m4_hparam_runs.csv"
    hparam_df = write_hparam_table(records, hparam_csv)
    print(f"[plot_training] Wrote {hparam_csv} ({len(hparam_df)} rows)")

    # 2. Write seed spread table
    seed_csv = args.tables_dir / "m4_seed_spread.csv"
    seed_df = write_seed_spread_table(records, seed_csv)
    print(f"[plot_training] Wrote {seed_csv} ({len(seed_df)} rows)")

    # 3. Generate SVG figures
    curves_svg = args.figures_dir / "training_loss_curves.svg"
    plot_loss_curves(records, curves_svg)
    print(f"[plot_training] Generated {curves_svg}")

    comparison_svg = args.figures_dir / "hparam_comparison.svg"
    plot_hparam_comparison(records, comparison_svg)
    print(f"[plot_training] Generated {comparison_svg}")

    # 4. Generate training artifacts table
    artifacts_csv = args.tables_dir / "m4_training_artifacts.csv"
    artifacts_df = write_training_artifacts_table(_REPO_ROOT, artifacts_csv)
    print(f"[plot_training] Wrote {artifacts_csv} ({len(artifacts_df)} artifacts)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Author: Jashan Tiwari
# ContextGuard Milestone 4: Training Run Configuration Generator
# Generates reproducible YAML configuration files from real run metadata.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RUNS = [
    "m4_f1_full", "m4_h0_ref", "m4_h1_lr1e-4", "m4_h2_lr4e-4", "m4_h3_r8", "m4_h4_r32",
    "m4_h5_batch16", "m4_h6_constant", "m4_h7_wd0.01", "m4_h8_dropout0.1",
    "m4_s1_seed7", "m4_s2_seed123", "m4_e5_no_synthetic"
]


def generate_config_for_run(run_id: str, runs_dir: Path, out_dir: Path):
    """Generate YAML configuration for a specific training run."""
    summary_path = runs_dir / run_id / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with open(summary_path) as f:
        data = json.load(f)
    c = data.get("config", {})

    doc = {
        "run_id": run_id,
        "seed": c.get("seed", 42),
        "output_dir": f"outputs/runs/{run_id}",
        "paths": {
            "train_file": "data/processed/examples/train.jsonl",
            "val_file": "data/processed/examples/val.jsonl"
        },
        "model": {
            "name": "google/gemma-4-E4B-it",
            "max_len": c.get("max_len", 1536),
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
            "memory_fraction": c.get("memory_fraction", 0.45)
        },
        "lora": {
            "rank": c.get("rank", 16),
            "alpha": c.get("alpha", 32),
            "dropout": c.get("dropout", 0.05),
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training": {
            "optimizer": "adamw_torch",
            "learning_rate": float(c.get("lr", 2e-4)),
            "lr_scheduler_type": c.get("schedule", "cosine"),
            "warmup_ratio": c.get("warmup", 0.03),
            "weight_decay": float(c.get("weight_decay", 0.0)),
            "per_device_train_batch_size": c.get("micro_batch", 4),
            "gradient_accumulation_steps": c.get("accum", 8),
            "effective_batch_size": c.get("micro_batch", 4) * c.get("accum", 8),
            "max_steps": c.get("max_steps", 200),
            "epochs": float(c.get("epochs", 0.0)),
            "n_train": c.get("n_train", 6400),
            "n_val": c.get("n_val", 256),
            "eval_steps": c.get("eval_steps", 50),
            "save_steps": c.get("save_steps", 1000),
            "early_stopping_patience": c.get("patience", 0),
            "exclude_synthetic": c.get("exclude_synthetic", False),
            "loss_mask_prompt": True,
            "ignore_index": -100
        },
        "inference": {
            "do_sample": False,
            "max_new_tokens": 160,
            "confidence_prefix": '{"is_error":',
            "temperature": 1.0
        }
    }

    out_file = out_dir / f"{run_id}.yaml"
    header = f"# Author: Jashan Tiwari\n# ContextGuard M4 Training Run: {run_id}\n"
    yaml_str = yaml.dump(doc, sort_keys=False, default_flow_style=False)
    out_file.write_text(header + yaml_str)
    return out_file


def main():
    runs_dir = _REPO_ROOT / "outputs" / "runs"
    configs_dir = _REPO_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for r in RUNS:
        out_f = generate_config_for_run(r, runs_dir, configs_dir)
        print(f"Generated {out_f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validation loss of several saved LoRA adapters, all on the same fixed validation sample.

Author: Sidhaarth Shree

The screening runs drew their validation sample from the same random stream as their training sample, so runs with
another seed or without the synthetic examples were scored on other validation examples. This script scores every
adapter (and the base model without an adapter) on the same seeded sample, so the runs can be compared fairly.

Usage: python scripts/eval_adapters.py --model <base> --runs <run dir> ... [--n-val 1024] [--seed 0] --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.dataset import ExampleDataset  # noqa: E402
from src.llm.train_lora import answer_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with an `adapter` subfolder")
    ap.add_argument("--val-file", default="data/processed/examples/val.jsonl")
    ap.add_argument("--n-val", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--memory-fraction", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    val = ExampleDataset(args.val_file, tokenizer=tok, max_len=args.max_len)
    idx = random.Random(args.seed).sample(range(len(val)), min(args.n_val, len(val)))
    base = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        try:
            base = cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[eval] {name} failed: {type(e).__name__}", flush=True)
    rows = []
    t = time.time()
    rows.append({"run": "base model (no adapter)", "val_loss": round(answer_loss(base, val, idx, args.micro_batch, device, torch), 4),
                 "seconds": round(time.time() - t, 1)})
    print(rows[-1], flush=True)
    for run in args.runs:
        t = time.time()
        model = PeftModel.from_pretrained(base, Path(run) / "adapter")
        loss = answer_loss(model, val, idx, args.micro_batch, device, torch)
        base = model.unload()
        rows.append({"run": Path(run).name, "val_loss": round(loss, 4), "seconds": round(time.time() - t, 1)})
        print(rows[-1], flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "val_loss", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"DONE n_val={len(idx)} seed={args.seed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

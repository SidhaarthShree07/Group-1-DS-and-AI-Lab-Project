#!/usr/bin/env python3
"""Merge the LoRA adapter into the base model and save one bfloat16 model folder (M5, experiment X1).

Author: Sidhaarth Shree

The merged model gives the same answers as base + adapter, without the extra adapter work in every layer, and it is
the model that X1 quantises to 8-bit and 4-bit (quantise after merging, never merge into a quantised base). The
tokenizer, chat template and processor files are copied from the base folder.

Usage:
  python scripts/merge_adapter.py --model <base dir> --adapter <adapter dir> --out <merged dir>
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def merge_adapter(model_path: str, adapter_path: str, out_dir: str) -> dict:
    import torch
    import transformers
    from peft import PeftModel

    t0 = time.time()
    model = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):  # same order as src/llm/infer.py
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_path, dtype=torch.bfloat16,
                                        device_map="cuda" if torch.cuda.is_available() else "cpu")
            break
        except Exception as e:
            print(f"[merge] {name} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
    if model is None:
        raise RuntimeError(f"could not load {model_path}")
    merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    copied = []
    for f in Path(model_path).iterdir():  # tokenizer, chat template, processor and generation files
        if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    size_gib = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    return {"out": str(out), "seconds": round(time.time() - t0, 1), "weights_gib": round(size_gib, 2), "copied": copied}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(merge_adapter(args.model, args.adapter, args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

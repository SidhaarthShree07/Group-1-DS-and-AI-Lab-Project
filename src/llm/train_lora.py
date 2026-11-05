#!/usr/bin/env python3
"""LoRA fine-tuning of Gemma 4 E4B-it on the instruction examples (decisions D12 to D14).

Author: Sidhaarth Shree

Milestone 3 uses this script for a SMOKE run only: a small number of optimiser steps on a slice of train.jsonl, to
show that the loss falls, that the validation loss is computed, and that the saved adapter loads again and gives the
same loss. Real training (epochs, early stopping, hyper-parameters) is Milestone 4; the same script runs it with other
arguments.

Training set-up (D12): base weights frozen; LoRA rank 16, alpha 32, dropout 0.05 on the attention and MLP projections
of the language model only (no vision or audio modules); bf16; gradient checkpointing; AdamW, learning rate 2e-4,
cosine schedule with 3% warm-up; 4 examples per micro-batch x 8 accumulation steps = 32 per optimiser step.
Loss (D14): token cross-entropy on the answer tokens only (prompt and padding have label -100, src/llm/dataset.py).
Checkpointing: the adapter is saved every --save-steps and at the end; the best validation loss is recorded.

Usage: python src/llm/train_lora.py --model <path> --out <run dir> [--max-steps 100] [--n-train 3200] [--n-val 256]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.dataset import ExampleDataset, collate  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_modules(model) -> list:
    """Full names of the language-model projections (vision and audio towers are left out)."""
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] in PROJ]
    keep = [n for n in names if not any(x in n for x in ("vision", "audio", "embed_"))]
    if not keep:
        raise RuntimeError("no LoRA target modules found")
    return keep


def batches(ds, idx, size, rng, shuffle):
    idx = list(idx)
    if shuffle:
        rng.shuffle(idx)
    for b in range(0, len(idx), size):
        yield collate([ds[i] for i in idx[b: b + size]], pad_token_id=ds.tokenizer.pad_token_id or 0)


def mean_loss(model, ds, idx, size, device, torch):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in batches(ds, idx, size, None, False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            k = int((batch["labels"][:, 1:] != -100).sum())  # answer tokens in the batch (labels shift by one)
            tot, n = tot + float(out.loss) * k, n + k
    model.train()
    return tot / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path or Hub name of google/gemma-4-E4B-it")
    ap.add_argument("--train-file", default="data/processed/examples/train.jsonl")
    ap.add_argument("--val-file", default="data/processed/examples/val.jsonl")
    ap.add_argument("--out", required=True, help="run folder; the adapter goes to <out>/adapter")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=3200, help="training examples used (seeded sample)")
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-fraction", type=float, default=0.6)
    args = ap.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    train = ExampleDataset(args.train_file, tokenizer=tok, max_len=args.max_len)
    val = ExampleDataset(args.val_file, tokenizer=tok, max_len=args.max_len)
    tr_idx = rng.sample(range(len(train)), min(args.n_train, len(train)))
    va_idx = rng.sample(range(len(val)), min(args.n_val, len(val)))

    model = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
            break
        except Exception as e:
            print(f"[train] {name} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
    if model is None:
        raise RuntimeError("could not load the model")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    total_params = sum(p.numel() for p in model.parameters())
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                     target_modules=target_modules(model), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: total {total_params / 1e9:.2f}B, trainable {trainable / 1e6:.1f}M "
          f"({100 * trainable / total_params:.2f}%), LoRA modules {len(cfg.target_modules)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warm = max(1, int(args.warmup * args.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.max_steps - warm))))
    log = open(out / "train_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"])
    val0 = mean_loss(model, val, va_idx, args.micro_batch, device, torch)
    w.writerow([0, 0, "", round(val0, 4), 0, 0])
    print(f"step 0 val_loss {val0:.4f}", flush=True)

    model.train()
    step, seen, t0, run_loss, run_n, best = 0, 0, time.time(), 0.0, 0, (val0, 0)
    stream = batches(train, tr_idx, args.micro_batch, rng, True)
    while step < args.max_steps:
        for _ in range(args.accum):
            try:
                batch = next(stream)
            except StopIteration:
                stream = batches(train, tr_idx, args.micro_batch, rng, True)
                batch = next(stream)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            run_loss, run_n, seen = run_loss + float(loss) * args.accum, run_n + 1, seen + batch["input_ids"].shape[0]
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        vl = ""
        if step % args.eval_steps == 0 or step == args.max_steps:
            vl = mean_loss(model, val, va_idx, args.micro_batch, device, torch)
            if vl < best[0]:
                best = (vl, step)
            vl = round(vl, 4)
        w.writerow([step, seen, round(run_loss / max(1, run_n), 4), vl, f"{sched.get_last_lr()[0]:.2e}", round(time.time() - t0, 1)])
        log.flush()
        print(f"step {step} train_loss {run_loss / max(1, run_n):.4f} val_loss {vl} {time.time() - t0:.0f}s", flush=True)
        run_loss, run_n = 0.0, 0
        if step % args.save_steps == 0 or step == args.max_steps:
            model.save_pretrained(out / "adapter")
    log.close()

    # the saved adapter loads again and gives the same validation loss on a few examples
    check = va_idx[:16]
    before = mean_loss(model, val, check, args.micro_batch, device, torch)
    base = model.unload() if hasattr(model, "unload") else model.get_base_model()
    reloaded = PeftModel.from_pretrained(base, out / "adapter")
    after = mean_loss(reloaded, val, check, args.micro_batch, device, torch)
    summary = {"total_params": total_params, "trainable_params": trainable, "lora_modules": len(cfg.target_modules),
               "steps": step, "examples_seen": seen, "val_loss_start": round(val0, 4), "best_val_loss": round(best[0], 4),
               "best_step": best[1], "reload_val_loss_before": round(before, 4), "reload_val_loss_after": round(after, 4),
               "reload_matches": abs(before - after) < 1e-3, "seconds": round(time.time() - t0, 1),
               "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
               "config": vars(args), "torch": torch.__version__, "transformers": transformers.__version__}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

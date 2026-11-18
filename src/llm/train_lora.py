#!/usr/bin/env python3
"""LoRA fine-tuning of Gemma 4 E4B-it on the instruction examples (decisions D12 to D14).

Author: Sidhaarth Shree

Milestone 3 used it for a smoke run (100 steps). Milestone 4 uses the same script for the hyper-parameter screening
runs and the full training run.

Training set-up (D12): base weights frozen; LoRA rank 16, alpha 32, dropout 0.05 on the attention and MLP projections
of the language model only (no vision or audio modules); bf16; gradient checkpointing; AdamW, learning rate 2e-4,
cosine schedule with 3% warm-up; 4 examples per micro-batch x 8 accumulation steps = 32 per optimiser step;
gradient clipping at 1.0.
Loss (D14): token cross-entropy on the answer tokens only (prompt and padding have label -100, src/llm/dataset.py).
Checkpointing and early stopping (M4): the validation loss is computed every --eval-steps; the adapter with the best
validation loss is kept in <out>/adapter and the last one in <out>/adapter_last; with --patience N training stops
after N evaluations in a row without improvement.

Usage:
  smoke / screening: python src/llm/train_lora.py --model <path> --out <run dir> --max-steps 200 --n-train 6400
  full training:     python src/llm/train_lora.py --model <path> --out <run dir> --n-train 0 --epochs 1 --patience 3
                     --eval-steps 200 --save-steps 200
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


class EarlyStopper:
    """Stop after `patience` evaluations in a row without an improvement larger than `min_delta` (0 = never stop)."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_step, self.bad = math.inf, None, 0

    def update(self, value: float, step: int) -> bool:
        """Record one validation loss; return True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best, self.best_step, self.bad = value, step, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad >= self.patience


def total_steps(max_steps: int, epochs: float, n_examples: int, per_step: int) -> int:
    """Optimiser steps: epochs x examples / examples per step (rounded up) if --epochs is given, else --max-steps."""
    if epochs > 0:
        return max(1, math.ceil(epochs * n_examples / per_step))
    if max_steps <= 0:
        raise ValueError("give --max-steps or --epochs")
    return max_steps


def lr_multiplier(step: int, warmup_steps: int, total: int, schedule: str = "cosine") -> float:
    """Linear warm-up, then cosine decay to 0 (or constant)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total - warmup_steps)))


def iterate_batches(dataset, ids, batch_size, rng, shuffle):
    """Padded micro-batches of the examples `ids` (one pass over them)."""
    ids = list(ids)
    if shuffle:
        rng.shuffle(ids)
    pad_id = dataset.tokenizer.pad_token_id or 0
    for start in range(0, len(ids), batch_size):
        yield collate([dataset[i] for i in ids[start: start + batch_size]], pad_token_id=pad_id)


def answer_loss(model, dataset, ids, batch_size, device, torch):
    """Mean cross-entropy per answer token over the examples `ids` (the validation loss)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in iterate_batches(dataset, ids, batch_size, None, False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n_tokens = int((batch["labels"][:, 1:] != -100).sum())  # labels are shifted by one inside the model
            total_loss += float(out.loss) * n_tokens
            total_tokens += n_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path or Hub name of google/gemma-4-E4B-it")
    ap.add_argument("--train-file", default="data/processed/examples/train.jsonl")
    ap.add_argument("--val-file", default="data/processed/examples/val.jsonl")
    ap.add_argument("--out", required=True, help="run folder; the adapter goes to <out>/adapter")
    ap.add_argument("--max-steps", type=int, default=100, help="optimiser steps (ignored when --epochs is given)")
    ap.add_argument("--epochs", type=float, default=0.0, help="epochs over the training sample; overrides --max-steps")
    ap.add_argument("--n-train", type=int, default=3200, help="training examples used (seeded sample); 0 = all")
    ap.add_argument("--exclude-synthetic", action="store_true", help="drop the synthetic examples (experiment E5)")
    ap.add_argument("--patience", type=int, default=0, help="early stopping: evaluations without improvement; 0 = off")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop after this many hours (0 = no limit)")
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--val-seed", type=int, default=None,
                    help="seed of a separate validation sample, the same for every run (M4 runs before this option drew "
                         "it after the training sample, so it depended on --seed and --exclude-synthetic)")
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
    pool = [i for i, r in enumerate(train.records) if not (args.exclude_synthetic and r.get("source") == "synthetic")]
    n_train = len(pool) if args.n_train <= 0 else min(args.n_train, len(pool))
    train_ids = rng.sample(pool, n_train)
    val_rng = rng if args.val_seed is None else random.Random(args.val_seed)
    val_ids = val_rng.sample(range(len(val)), min(args.n_val, len(val)))
    args.max_steps = total_steps(args.max_steps, args.epochs, len(train_ids), args.micro_batch * args.accum)
    print(f"training examples {len(train_ids)} (synthetic {'excluded' if args.exclude_synthetic else 'included'}), "
          f"validation examples {len(val_ids)}, optimiser steps {args.max_steps}", flush=True)

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
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                             target_modules=target_modules(model), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: total {total_params / 1e9:.2f}B, trainable {trainable / 1e6:.1f}M "
          f"({100 * trainable / total_params:.2f}%), LoRA modules {len(lora_config.target_modules)}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(args.warmup * args.max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_multiplier(s, warmup_steps, args.max_steps, args.schedule))
    log = open(out / "train_log.csv", "w", newline="")
    log_writer = csv.writer(log)
    log_writer.writerow(["step", "examples_seen", "train_loss", "val_loss", "lr", "seconds"])
    start_val_loss = answer_loss(model, val, val_ids, args.micro_batch, device, torch)
    log_writer.writerow([0, 0, "", round(start_val_loss, 4), 0, 0])
    print(f"step 0 val_loss {start_val_loss:.4f}", flush=True)

    model.train()
    step, seen, t0, run_loss, run_n = 0, 0, time.time(), 0.0, 0
    stopper = EarlyStopper(args.patience, args.min_delta)
    stopper.update(start_val_loss, 0)
    model.save_pretrained(out / "adapter")  # step 0 is the best until a later evaluation improves on it
    stop_reason = "max steps reached"
    stream = iterate_batches(train, train_ids, args.micro_batch, rng, True)
    while step < args.max_steps:
        for _ in range(args.accum):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iterate_batches(train, train_ids, args.micro_batch, rng, True)
                batch = next(stream)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            run_loss, run_n, seen = run_loss + float(loss.detach()) * args.accum, run_n + 1, seen + batch["input_ids"].shape[0]
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        val_loss = ""
        if step % args.eval_steps == 0 or step == args.max_steps:
            val_loss = answer_loss(model, val, val_ids, args.micro_batch, device, torch)
            if stopper.update(val_loss, step):
                model.save_pretrained(out / "adapter")  # best so far
            val_loss = round(val_loss, 4)
        train_loss = run_loss / max(1, run_n)
        log_writer.writerow([step, seen, round(train_loss, 4), val_loss, f"{scheduler.get_last_lr()[0]:.2e}",
                             round(time.time() - t0, 1)])
        log.flush()
        print(f"step {step} train_loss {train_loss:.4f} val_loss {val_loss} {time.time() - t0:.0f}s", flush=True)
        run_loss, run_n = 0.0, 0
        if step % args.save_steps == 0 or step == args.max_steps:
            model.save_pretrained(out / "adapter_last")
        if stopper.should_stop:
            stop_reason = f"early stop: {args.patience} evaluations without improvement"
            break
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            stop_reason = f"time limit of {args.max_hours} h"
            break
    model.save_pretrained(out / "adapter_last")
    log.close()

    # the saved best adapter loads again and gives the best validation loss recorded during training
    base = model.unload() if hasattr(model, "unload") else model.get_base_model()
    reloaded = PeftModel.from_pretrained(base, out / "adapter")
    after = answer_loss(reloaded, val, val_ids, args.micro_batch, device, torch)
    summary = {"total_params": total_params, "trainable_params": trainable,
               "lora_modules": len(lora_config.target_modules),
               "steps": step, "examples_seen": seen, "n_train": len(train_ids), "stop_reason": stop_reason,
               "val_loss_start": round(start_val_loss, 4), "best_val_loss": round(stopper.best, 4),
               "best_step": stopper.best_step, "reload_val_loss": round(after, 4),
               "reload_matches": abs(stopper.best - after) < 1e-3, "seconds": round(time.time() - t0, 1),
               "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
               "config": vars(args), "torch": torch.__version__, "transformers": transformers.__version__}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

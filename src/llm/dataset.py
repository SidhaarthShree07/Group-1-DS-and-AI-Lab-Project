# Author: Jashan Tiwari
"""
src/llm/dataset.py

ContextGuard Milestone 3: Gemma Chat-Template Training Data Loader & Batch Collation
Author: Jashan Tiwari (Data Lead) - DSAI Lab, IIT Madras BS
Role: Implements Decisions D13, D14 and Section 7 Interface:
1. ExampleDataset:
   - Reads instruction-response examples from a .jsonl file (e.g. data/processed/examples/train.jsonl).
   - Formats user prompt and model response with the tokenizer's own chat template
     (tokenizer.apply_chat_template). For Gemma 4 E4B-it this gives
     <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n{response}<turn|>\n
     so training sees exactly the text that inference sends.
   - Tokenises sequence and builds 1D int64 tensors:
     * input_ids: full sequence token IDs
     * attention_mask: 1 for active tokens
     * labels: prompt tokens and assistant turn header masked to -100 (loss computed on response tokens only)
   - Enforces max_len constraint (D13, default 1536 tokens).
2. collate(batch, pad_token_id=0):
   - Right-pads batch to [batch, max_batch_length] with pad_token_id for input_ids, 0 for attention_mask,
     and -100 for labels. Returns int64 2D tensors.
3. CLI Verification:
   - Runs validation over train.jsonl and val.jsonl, computes sequence length distributions (min, median,
     p95, max), and saves docs/tables/m3_loader_stats.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_TOKENIZER = "google/gemma-4-E4B-it"
# Used only with the offline test tokenizer; real tokenizers use their own chat template (Gemma 4 E4B-it format).
GEMMA_USER_PREFIX = "<bos><|turn>user\n"
GEMMA_USER_SUFFIX = "<turn|>\n"
GEMMA_ASSISTANT_PREFIX = "<|turn>model\n"
GEMMA_ASSISTANT_SUFFIX = "<turn|>\n"


class FastGemmaFallbackTokenizer:
    """Lightweight, deterministic fallback tokenizer for testing without external models."""

    def __init__(self, pad_token_id: int = 0, eos_token_id: int = 1, bos_token_id: int = 2):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.special_tokens = {
            "<pad>": pad_token_id,
            "<eos>": eos_token_id,
            "<bos>": bos_token_id,
            "<|turn>": 105,
            "<turn|>": 106,
        }
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.next_id = 1000

    def _get_token_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.inv_vocab[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # Simple delimiter-preserving split for deterministic token representation
        import re
        parts = re.split(r"(<bos>|<\|turn>|<turn\|>|\s+|[^\w\s])", text)
        for p in parts:
            if not p:
                continue
            tokens.append(self._get_token_id(p))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        res = []
        for tid in token_ids:
            if tid in self.inv_vocab:
                tok_str = self.inv_vocab[tid]
                if tok_str not in ("<pad>", "<bos>", "<eos>", "<|turn>", "<turn|>"):
                    res.append(tok_str)
        return "".join(res)

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def resolve_tokenizer(tokenizer_or_name: Any = None) -> Any:
    """Return a tokenizer object: the one given, the Hugging Face tokenizer named (default: Gemma 4 E4B-it),
    or the offline test tokenizer when "fallback" is asked for explicitly. A missing tokenizer is an error:
    token counts from a stand-in tokenizer must never reach the report."""
    if tokenizer_or_name is not None and not isinstance(tokenizer_or_name, str):
        return tokenizer_or_name
    if tokenizer_or_name == "fallback":
        return FastGemmaFallbackTokenizer()
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(tokenizer_or_name or DEFAULT_TOKENIZER)


class ExampleDataset(Dataset):
    """
    Dataset of instruction-tuning examples formatted for Gemma.
    Returns input_ids, attention_mask, and labels with prompt tokens masked to -100.
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: Any = None,
        max_len: int = 1536,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Example dataset file not found: {self.path}")

        self.tokenizer = resolve_tokenizer(tokenizer)
        self.max_len = max_len

        # Read JSONL records
        self.records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.records.append(data)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Corrupted JSON on line {line_no} in {self.path}: {err}") from err

    def __len__(self) -> int:
        return len(self.records)

    def _extract_prompt_response(self, record: Dict[str, Any]) -> Tuple[str, str]:
        prompt = record.get("prompt") or record.get("instruction")
        response = record.get("response") or record.get("answer")
        if prompt is None:
            raise KeyError(f"Record missing 'prompt' or 'instruction': {record}")
        if response is None:
            raise KeyError(f"Record missing 'response' or 'answer': {record}")
        return str(prompt), str(response)

    def _format_chat(self, prompt: str, response: str) -> Tuple[str, str]:
        """Prompt text (up to and including the model-turn header) and response text, from the tokenizer's chat
        template. The prompt text is exactly what inference sends with add_generation_prompt=True."""
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            user = [{"role": "user", "content": prompt}]
            prompt_text = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full_text = tok.apply_chat_template(user + [{"role": "assistant", "content": response}], tokenize=False)
            if not full_text.startswith(prompt_text):
                raise ValueError("chat template: the full conversation does not start with the generation prompt")
            return prompt_text, full_text[len(prompt_text):]
        prompt_text = f"{GEMMA_USER_PREFIX}{prompt}{GEMMA_USER_SUFFIX}{GEMMA_ASSISTANT_PREFIX}"
        response_text = f"{response}{GEMMA_ASSISTANT_SUFFIX}"
        return prompt_text, response_text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        prompt, response = self._extract_prompt_response(record)
        prompt_text, response_text = self._format_chat(prompt, response)

        # Tokenise prompt and response separately to guarantee exact unmasked boundary
        p_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        r_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

        full_ids = p_ids + r_ids
        prompt_len = len(p_ids)

        # Enforce max_len constraint (D13)
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Label masking (D14): prompt tokens masked to -100
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        if prompt_len < len(input_ids):
            labels[prompt_len:] = input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function that right-pads batch of examples into 2D int64 tensors:
    - input_ids padded with pad_token_id
    - attention_mask padded with 0
    - labels padded with -100
    Shape: [batch_size, max_seq_len]
    """
    if not batch:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
            "labels": torch.empty((0, 0), dtype=torch.long),
        }

    max_len = max(ex["input_ids"].size(0) for ex in batch)
    b_input_ids: List[torch.Tensor] = []
    b_attention_mask: List[torch.Tensor] = []
    b_labels: List[torch.Tensor] = []

    for ex in batch:
        cur_len = ex["input_ids"].size(0)
        pad_len = max_len - cur_len
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((pad_len,), dtype=torch.long)
            pad_lab = torch.full((pad_len,), -100, dtype=torch.long)

            b_input_ids.append(torch.cat([ex["input_ids"], pad_ids]))
            b_attention_mask.append(torch.cat([ex["attention_mask"], pad_mask]))
            b_labels.append(torch.cat([ex["labels"], pad_lab]))
        else:
            b_input_ids.append(ex["input_ids"])
            b_attention_mask.append(ex["attention_mask"])
            b_labels.append(ex["labels"])

    return {
        "input_ids": torch.stack(b_input_ids, dim=0),
        "attention_mask": torch.stack(b_attention_mask, dim=0),
        "labels": torch.stack(b_labels, dim=0),
    }


def compute_loader_statistics(
    dataset: ExampleDataset,
    split_name: str = "train",
) -> Dict[str, Any]:
    """Computes exact sequence, prompt, and answer token length distributions."""
    seq_lengths: List[int] = []
    prompt_lengths: List[int] = []
    answer_lengths: List[int] = []
    exceeded_count = 0

    tok = dataset.tokenizer

    for i in range(len(dataset)):
        rec = dataset.records[i]
        prompt, response = dataset._extract_prompt_response(rec)
        prompt_text, response_text = dataset._format_chat(prompt, response)

        p_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        r_len = len(tok(response_text, add_special_tokens=False)["input_ids"])
        seq_len = p_len + r_len

        if seq_len > dataset.max_len:
            exceeded_count += 1

        prompt_lengths.append(p_len)
        answer_lengths.append(r_len)
        seq_lengths.append(seq_len)

    seq_arr = np.array(seq_lengths)
    p_arr = np.array(prompt_lengths)
    r_arr = np.array(answer_lengths)

    stats = {
        "split": split_name,
        "total_examples": len(dataset),
        "min_seq_len": int(np.min(seq_arr)),
        "median_seq_len": int(np.median(seq_arr)),
        "p95_seq_len": int(np.percentile(seq_arr, 95)),
        "max_seq_len": int(np.max(seq_arr)),
        "max_prompt_len": int(np.max(p_arr)),
        "max_answer_len": int(np.max(r_arr)),
        "exceeded_max_len": exceeded_count,
        "max_len_limit": dataset.max_len,
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ExampleDataset and compute token statistics")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/examples/train.jsonl",
        help="Path to training jsonl file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/examples/val.jsonl",
        help="Path to validation jsonl file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name or path (the real Gemma 4 tokenizer; the statistics go into the report)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1536,
        help="Maximum sequence length (default 1536)",
    )
    args = parser.parse_args()

    repo_root = args.root
    tables_dir = repo_root / "docs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_path = repo_root / args.train_file
    val_path = repo_root / args.val_file

    print("=" * 80)
    print("      MILESTONE 3: VERIFYING GEMMA TRAINING DATA LOADER (DECISIONS D13, D14)   ")
    print(f"      Tokenizer: {args.tokenizer} | Max Length Budget: {args.max_len}")
    print("=" * 80)

    stats_list = []

    for name, path in [("train", train_path), ("val", val_path)]:
        if not path.exists():
            print(f"[WARN] {path} not found; skipping {name} statistics.")
            continue

        print(f"[*] Loading and verifying {name} dataset from {path.relative_to(repo_root)}...")
        ds = ExampleDataset(path=path, tokenizer=args.tokenizer, max_len=args.max_len)
        print(f"    Loaded {len(ds)} records. Running full token length audit...")

        stats = compute_loader_statistics(ds, split_name=name)
        stats_list.append(stats)

        print(
            f"[*] {name.upper():<6} | Examples: {stats['total_examples']:,} | "
            f"SeqLen: min={stats['min_seq_len']}, median={stats['median_seq_len']}, "
            f"p95={stats['p95_seq_len']}, max={stats['max_seq_len']} | "
            f"MaxPrompt={stats['max_prompt_len']}, MaxAnswer={stats['max_answer_len']} | "
            f"Over Budget ({args.max_len}): {stats['exceeded_max_len']}"
        )

    # Export statistics to CSV
    if stats_list:
        out_csv = tables_dir / "m3_loader_stats.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
            w.writeheader()
            w.writerows(stats_list)
        print("=" * 80)
        print(f"[SUCCESS] Exported token statistics -> {out_csv.relative_to(repo_root)}")
        print("=" * 80)

    # Test small batch collation
    if train_path.exists():
        ds = ExampleDataset(path=train_path, tokenizer=args.tokenizer, max_len=args.max_len)
        sample_batch = [ds[i] for i in range(min(4, len(ds)))]
        collated = collate(sample_batch, pad_token_id=0)
        assert collated["input_ids"].shape[0] == len(sample_batch)
        assert collated["labels"].dtype == torch.int64
        print(f"[PASS] Batch collation test passed: shape {collated['input_ids'].shape}, dtype {collated['input_ids'].dtype}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


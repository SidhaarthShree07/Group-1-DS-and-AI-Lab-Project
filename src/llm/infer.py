#!/usr/bin/env python3
"""Inference with Gemma 4 E4B-it (base weights, or base weights plus a LoRA adapter).

Author: Sidhaarth Shree

Stage 3 of the pipeline: one prompt per candidate cell, greedy decoding, the answer starts with the fixed prefix
`{"is_error":` so that the first generated token is the decision. P(is_error) is read from that token (softmax mass on
"true" against "false"), as in the model comparison (scripts/model_bakeoff/run_model.py). The prompt is wrapped with
the tokenizer's own chat template, exactly as src/llm/dataset.py does for training.

The same Generator also serves stage 0 (src/detectors/llm_rules.py) through `complete()`, without the prefix.

Output (`predictions.jsonl`, plan Section 7): id, table, split, row, column, raw_text, valid_json, is_error, error_type,
evidence, repair, p_error, seconds.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.parse import parse_prediction  # noqa: E402

PREFIX = '{"is_error":'
REPAIR_VALUE = re.compile(r'"repair"\s*:\s*"((?:[^"\\]|\\.)*)"')


def repair_confidence(tok, token_ids: List[int], token_probs: List[float], text: str) -> Optional[Dict[str, float]]:
    """How sure the model is of the repair value it wrote.

    P(error) says whether the old value is wrong; this says whether the model is sure of the new one. We take the
    probabilities of the generated tokens that cover the characters of the "repair" string in `text` and return
    their minimum and geometric mean, or None when the answer has no repair string."""
    match = REPAIR_VALUE.search(text)
    if not match or match.end(1) == match.start(1):
        return None
    value_start, value_end = match.start(1), match.end(1)
    value_probs, chars_before = [], 0
    for k in range(len(token_ids)):
        chars_after = len(tok.decode(token_ids[: k + 1], skip_special_tokens=True))
        if chars_after > value_start and chars_before < value_end:
            value_probs.append(token_probs[k])
        chars_before = chars_after
        if chars_before >= value_end:
            break
    if not value_probs:
        return None
    log_mean = sum(math.log(max(p, 1e-12)) for p in value_probs) / len(value_probs)
    return {"min": float(min(value_probs)), "gmean": float(math.exp(log_mean))}


def chat_text(tokenizer, prompt: str) -> str:
    """The generation prompt for one user message (the same call as the training loader)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def first_token_ids(tokenizer, words: Iterable[str]) -> List[int]:
    ids = set()
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


class Generator:
    """Loads the model once; answers batches of prompts."""

    def __init__(self, model_path: str, adapter: Optional[str] = None, batch_size: int = 12, max_new_tokens: int = 160,
                 memory_fraction: float = 0.5):
        import torch
        import transformers
        from transformers import AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(memory_fraction)  # leave room for other users of the GPU
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"  # generation pads on the left (D13)
        model = None
        for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu")
                break
            except Exception as e:  # try the next model class
                print(f"[infer] {name} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if model is None:
            raise RuntimeError(f"could not load {model_path}")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.true_ids = first_token_ids(self.tok, [" true", "true"])
        self.false_ids = first_token_ids(self.tok, [" false", "false"])

    def _generate(self, texts: List[str], max_new_tokens: int, want_p: bool):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_logits=want_p,
                                      return_dict_in_generate=True, pad_token_id=self.tok.pad_token_id)
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        out = self.tok.batch_decode(new, skip_special_tokens=True)
        p, repair_confs = None, None
        if want_p:
            lp = torch.log_softmax(gen.logits[0].float(), dim=-1)  # first generated token
            pt = torch.logsumexp(lp[:, self.true_ids], dim=-1)
            pf = torch.logsumexp(lp[:, self.false_ids], dim=-1)
            p = torch.sigmoid(pt - pf).cpu().tolist()
            # probability of each generated token, used for the confidence of the repair value
            step_probs = []
            for step, logits in enumerate(gen.logits):
                step_probs.append(torch.softmax(logits.float(), dim=-1).gather(1, new[:, step: step + 1]).squeeze(1).cpu())
            token_probs = torch.stack(step_probs, dim=1).tolist()
            repair_confs = [repair_confidence(self.tok, new[i].tolist(), token_probs[i], out[i]) for i in range(len(out))]
        return out, p, repair_confs

    def answer(self, prompts: List[str]) -> List[Dict]:
        """Stage 3: the JSON answer and P(is_error) for each prompt, in the input order."""
        texts = [chat_text(self.tok, p) + PREFIX for p in prompts]
        order = sorted(range(len(texts)), key=lambda k: len(texts[k]))  # similar lengths in one batch
        res: List[Optional[Dict]] = [None] * len(texts)
        for b in range(0, len(order), self.batch_size):
            ids = order[b: b + self.batch_size]
            t0 = time.time()
            out, p, repair_confs = self._generate([texts[k] for k in ids], self.max_new_tokens, True)
            dt = (time.time() - t0) / len(ids)
            for k, o, pk, conf in zip(ids, out, p, repair_confs):
                res[k] = {"raw_text": PREFIX + o, "p_error": float(pk), "seconds": round(dt, 4),
                          "p_repair_min": None if conf is None else conf["min"],
                          "p_repair_gmean": None if conf is None else conf["gmean"]}
        return res  # type: ignore[return-value]

    def complete(self, prompts: List[str], max_new_tokens: int = 600) -> List[str]:
        """Free generation (stage 0 rules)."""
        texts = [chat_text(self.tok, p) for p in prompts]
        out: List[str] = []
        for b in range(0, len(texts), self.batch_size):
            o, _, _ = self._generate(texts[b: b + self.batch_size], max_new_tokens, False)
            out += o
        return out


def to_prediction(meta: Dict, answer: Dict) -> Dict:
    """One `predictions.jsonl` record: cell metadata, the raw answer, the parsed fields and the confidence."""
    parsed = parse_prediction(answer["raw_text"])
    return {**{k: meta[k] for k in ("id", "table", "split", "row", "column")}, "raw_text": answer["raw_text"],
            "valid_json": parsed["valid_json"], "is_error": parsed["is_error"], "error_type": parsed["error_type"],
            "evidence": parsed["evidence"], "repair": parsed["repair"], "p_error": answer["p_error"],
            "p_repair_min": answer.get("p_repair_min"), "p_repair_gmean": answer.get("p_repair_gmean"),
            "seconds": answer["seconds"]}


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

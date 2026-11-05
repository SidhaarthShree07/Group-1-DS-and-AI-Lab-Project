# Author: Jashan Tiwari
"""
tests/test_m3_jashan.py

ContextGuard Milestone 3: Unit Tests for Dataset Loader, Collation, and Validation Subsets
Author: Jashan Tiwari (Data Lead) - DSAI Lab, IIT Madras BS
Role: Validates Milestone 3 Decisions D13, D14, D17, and Section 7 Fixed Interfaces:
- Test 1 (Answer-Only Unmasked Labels - D14): All prompt & padding tokens masked to -100.
- Test 2 (Max Length Constraint & Batch Collation - D13): Tensors 2D int64 of shape [batch, length].
- Test 3 (Exact Decoded Response Match): Decoding unmasked labels equals ground-truth response.
- Test 4 (Subset Integrity - D17): 100 rows, zero test leakage, natural error rates within variance.
- Test 5 (Sample JSONL Loading): Full pipeline test on data/sample/m2_examples_sample.jsonl.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from src.data.make_subset import CORE_SUBSET_TABLES, make_subsets
from src.llm.dataset import (
    ExampleDataset,
    FastGemmaFallbackTokenizer,
    collate,
    resolve_tokenizer,
)


@pytest.fixture
def dummy_jsonl(tmp_path: Path) -> Path:
    """Creates a temporary JSONL file with sample records."""
    file_path = tmp_path / "dummy.jsonl"
    records = [
        {
            "id": "h:train:0:state",
            "table": "hospital",
            "prompt": "Table: hospital | Column: state | Cell value: ax\nRow: 10018 callahan ax",
            "response": '{"is_error": true, "error_type": "rule", "repair": "al"}',
        },
        {
            "id": "h:train:1:city",
            "table": "hospital",
            "prompt": "Table: hospital | Column: city | Cell value: birminghxm",
            "response": '{"is_error": true, "error_type": "pattern", "repair": "birmingham"}',
        },
        {
            "id": "h:train:2:zip",
            "table": "hospital",
            "prompt": "Table: hospital | Column: zip | Cell value: 35233",
            "response": '{"is_error": false, "error_type": "none", "repair": null}',
        },
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return file_path


def test_answer_only_unmasked_labels(dummy_jsonl: Path):
    """Test 1 (D14): Verifies prompt tokens are masked to -100 and response tokens are unmasked."""
    tokenizer = resolve_tokenizer("fallback")
    dataset = ExampleDataset(dummy_jsonl, tokenizer=tokenizer, max_len=1536)

    for i in range(len(dataset)):
        ex = dataset[i]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        attention_mask = ex["attention_mask"]

        assert input_ids.dtype == torch.int64
        assert labels.dtype == torch.int64
        assert attention_mask.dtype == torch.int64

        # Labels must have prompt tokens masked to -100
        unmasked_mask = labels != -100
        assert unmasked_mask.any(), f"Example {i} has no unmasked labels!"

        # The first token MUST be masked to -100 (part of prompt/turn header)
        assert labels[0].item() == -100, f"Example {i} first token is not masked!"

        # All masked tokens must come before unmasked tokens (contiguous split)
        first_unmasked = int(torch.where(unmasked_mask)[0][0].item())
        assert (labels[:first_unmasked] == -100).all(), "Prompt contains unmasked tokens!"
        assert (labels[first_unmasked:] == input_ids[first_unmasked:]).all(), "Response labels mismatch input_ids!"


def test_max_length_constraint_and_collate(dummy_jsonl: Path):
    """Test 2 (D13): Verifies max_len truncation and batch collation into 2D int64 tensors."""
    tokenizer = resolve_tokenizer("fallback")
    max_len = 25  # Force short max_len for testing truncation
    dataset = ExampleDataset(dummy_jsonl, tokenizer=tokenizer, max_len=max_len)

    for i in range(len(dataset)):
        ex = dataset[i]
        assert ex["input_ids"].size(0) <= max_len
        assert ex["labels"].size(0) <= max_len
        assert ex["attention_mask"].size(0) <= max_len

    # Test batch collation
    batch = [dataset[0], dataset[1], dataset[2]]
    collated = collate(batch, pad_token_id=0)

    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "labels" in collated

    assert collated["input_ids"].dim() == 2
    assert collated["input_ids"].shape[0] == 3
    assert collated["input_ids"].dtype == torch.int64
    assert collated["labels"].dtype == torch.int64
    assert collated["attention_mask"].dtype == torch.int64

    # Verify right-padding: if padding occurred, padded labels must be -100 and mask must be 0
    max_seq_len = collated["input_ids"].shape[1]
    for b_idx in range(len(batch)):
        orig_len = batch[b_idx]["input_ids"].size(0)
        if orig_len < max_seq_len:
            assert (collated["attention_mask"][b_idx, orig_len:] == 0).all()
            assert (collated["labels"][b_idx, orig_len:] == -100).all()
            assert (collated["input_ids"][b_idx, orig_len:] == 0).all()


def test_exact_decoded_response_match(dummy_jsonl: Path):
    """Test 3: Verifies decoding labels[labels != -100] matches the record response."""
    tokenizer = resolve_tokenizer("fallback")
    dataset = ExampleDataset(dummy_jsonl, tokenizer=tokenizer, max_len=1536)

    for i in range(len(dataset)):
        ex = dataset[i]
        labels = ex["labels"]
        unmasked = labels[labels != -100].tolist()

        decoded = tokenizer.decode(unmasked)
        # Strip trailing Gemma turn end tokens if present
        clean_decoded = decoded.replace("<turn|>", "").strip()

        ground_truth_response = dataset.records[i]["response"].strip()
        assert clean_decoded == ground_truth_response, (
            f"Decoded response mismatch at index {i}:\n"
            f"Decoded : {clean_decoded}\n"
            f"Expected: {ground_truth_response}"
        )


def test_subset_integrity_and_zero_leakage():
    """Test 4 (D17): Verifies 100-row validation subsets, zero test leakage, and valid error rates."""
    subsets_dir = REPO_ROOT / "data" / "subsets" / "m3"
    splits_dir = REPO_ROOT / "data" / "splits" / "core"

    if not (splits_dir / "hospital" / "val.csv").exists():
        pytest.skip("split files not built (run scripts/build_all.sh first)")
    # Ensure subsets exist or generate them
    if not (subsets_dir / "hospital" / "val.csv").exists():
        make_subsets(REPO_ROOT, tables=CORE_SUBSET_TABLES, n_sample_rows=100, seed=42)

    for table in CORE_SUBSET_TABLES:
        t_sub_dir = subsets_dir / table
        t_split_dir = splits_dir / table

        val_file = t_sub_dir / "val.csv"
        val_clean_file = t_sub_dir / "val_clean.csv"
        val_cells_file = t_sub_dir / "val_cells.csv"

        assert val_file.exists(), f"Missing {val_file}"
        assert val_clean_file.exists(), f"Missing {val_clean_file}"
        assert val_cells_file.exists(), f"Missing {val_cells_file}"

        with open(val_file, "r", encoding="utf-8") as f:
            v_rows = list(csv.reader(f))[1:]
        with open(val_clean_file, "r", encoding="utf-8") as f:
            c_rows = list(csv.reader(f))[1:]
        with open(val_cells_file, "r", encoding="utf-8") as f:
            cells = list(csv.DictReader(f))

        # Check exact 100 rows
        assert len(v_rows) == 100, f"{table}: subset rows = {len(v_rows)}, expected 100"
        assert len(c_rows) == 100, f"{table}: clean rows = {len(c_rows)}, expected 100"

        # Check row indices re-indexed 0..99
        unique_cell_rows = sorted(int(c["row"]) for c in cells)
        assert set(unique_cell_rows) == set(range(100)), f"{table}: cell row indices are not 0..99"

        # Check zero leakage with test split
        test_file = t_split_dir / "test.csv"
        assert test_file.exists(), f"Missing test split for {table}"
        with open(test_file, "r", encoding="utf-8") as f:
            test_rows = list(csv.reader(f))[1:]

        # Row-level check: no identical rows between subset and test
        sub_tuples = {tuple(r[1:]) for r in v_rows}  # exclude index/tuple_id
        test_tuples = {tuple(r[1:]) for r in test_rows}
        overlap = sub_tuples.intersection(test_tuples)
        assert len(overlap) == 0, f"{table}: leakage detected! {len(overlap)} rows overlap with test split."


def test_example_dataset_on_sample_file():
    """Test 5: Verifies ExampleDataset runs on the repository sample file."""
    sample_file = REPO_ROOT / "data" / "sample" / "m2_examples_sample.jsonl"
    if not sample_file.exists():
        pytest.skip("Sample file data/sample/m2_examples_sample.jsonl not present.")

    tokenizer = resolve_tokenizer("fallback")
    ds = ExampleDataset(sample_file, tokenizer=tokenizer, max_len=1536)
    assert len(ds) == 20

    batch = [ds[i] for i in range(4)]
    collated = collate(batch, pad_token_id=0)
    assert collated["input_ids"].shape[0] == 4
    assert collated["labels"].shape[0] == 4
    assert collated["attention_mask"].shape[0] == 4
    assert collated["input_ids"].dtype == torch.int64


def test_real_gemma_template_matches_inference():
    """Test 6: with the real Gemma 4 tokenizer (if it is available offline), the prompt part equals the generation
    prompt that inference sends, and the unmasked labels decode to the response plus the end-of-turn marker."""
    try:
        tok = resolve_tokenizer()
    except Exception:
        pytest.skip("Gemma 4 tokenizer not available")
    ds_dir = Path(tempfile.mkdtemp())
    path = ds_dir / "one.jsonl"
    path.write_text(json.dumps({"prompt": "Table: t | Column: c | Cell value: 'x'", "response": '{"is_error": false}'}) + "\n")
    ds = ExampleDataset(path, tokenizer=tok, max_len=1536)
    prompt_text, response_text = ds._format_chat("Table: t | Column: c | Cell value: 'x'", '{"is_error": false}')
    gen = tok.apply_chat_template([{"role": "user", "content": "Table: t | Column: c | Cell value: 'x'"}],
                                  tokenize=False, add_generation_prompt=True)
    assert prompt_text == gen
    ex = ds[0]
    assert ex["input_ids"][0].item() == tok.bos_token_id
    decoded = tok.decode(ex["labels"][ex["labels"] != -100].tolist())
    assert decoded.startswith('{"is_error": false}')

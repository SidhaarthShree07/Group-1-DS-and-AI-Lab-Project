# Milestone 2 hand-over check

**Goal set by the course team:** another team must be able to take our prepared datasets and start training a model immediately, without further data preparation.

## Check 1: clean-tree rebuild (run on 28 Oct 2026)

A copy of the repository without any data was filled by `python scripts/download_data.py` and then built with `bash scripts/build_all.sh --skip-download`. TableEG's two zip files were copied into `data/raw/tableeg/` first (they exist only on Google Drive).

| Step | Result |
|---|---|
| Download of Raha, BClean, ZeroED, Germany slice, UCI files; SHA-256 against `data/download_log.md` | all verified |
| Core tables: 18,414 error cells, type counts as in Milestone 1 | match |
| Entity-aware splits, error counts conserved, zero row and entity overlap | pass |
| External tables: Inpatient 4,465, Facilities 8,318, Billionaire 5,195 (5,177 true), Germany sample 18,678 (18,674 true), Tax sample 12,120 (302 true) | match Milestone 1 |
| Synthetic tables (40,132 corrupted cells) | built, seed 42 |
| Fine-tuning examples: 55,091 train, 9,341 validation, 9,547 test | built |
| Leakage checks | 29 of 29 pass |
| Tests | 18 pass (re-run on 29 Oct 2026 on the merged main) |
| Determinism: a second independent build gave byte-identical `train.jsonl`, `val.jsonl`, `test.jsonl` and all `docs/tables/m2_*.csv` | identical |

## Check 2: dry pass of a data loader (to be run by Aaradhya on his own machine)

Run on a machine that did not write the scripts, from a fresh clone:

1. `bash scripts/build_all.sh` (after copying the TableEG zip files as described in the README).
2. Read `data/processed/examples/train.jsonl` line by line with `json.loads`; confirm that every record has the keys `id`, `table`, `split`, `row`, `column`, `source`, `is_error`, `error_type`, `prompt`, `response`, and that `response` parses as JSON with the keys `is_error`, `error_type`, `evidence`, `repair`.
3. Measure the token length of prompt plus response with the Gemma tokenizer (`python src/llm/build_examples.py --tokenizer google/gemma-4-E4B-it`) and confirm that no record exceeds 1,500 tokens.
4. Write down anything that was unclear or failed.

Run by Aaradhya on a fresh clone of the merged `main` (commit e020f1f).

| Item | Result | Date |
|---|---|---|
| Step 1: clean rebuild and tests (`pytest tests`) | PASS, 18 of 18 tests passed | 29 Oct 2026 |
| Step 2: `train.jsonl` parse with `json.loads` | PASS, 55,091 records, 0 bad; all 10 keys present; every `response` parses with `is_error`, `error_type`, `evidence`, `repair` | 29 Oct 2026 |
| Step 3: token budget with the Gemma tokenizer (`gemma-4-E4B-it`) | PASS. Aaradhya's first report of 958 tokens had been read from `m2_token_lengths.csv`, which is the characters ÷ 3.2 estimate; the step was then run with the real tokenizer (Sidhaarth), and Aaradhya checked it again and confirmed the same result: longest prompt plus answer 1,009 tokens (Hospital, train), limit 1,500; 0 over budget in all 29 splits (`docs/tables/m2_token_lengths_gemma.csv`) | 29 Oct 2026 |
| Notes | No local paths found in the report text (sections 2.3, 5.3, 9.2, 12.4). Nothing unclear or failing. | 29 Oct 2026 |

Check 2 is closed: the hand-over check passed on a machine that did not write the scripts.

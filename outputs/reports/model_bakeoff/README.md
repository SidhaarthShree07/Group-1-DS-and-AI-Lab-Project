# Base-model bake-off (zero-shot), 19 Oct 2026

**Question:** which base model should we fine-tune for ContextGuard?
**Candidates:** Qwen3-4B-Instruct-2507, Qwen3.5-9B, Gemma 4 E4B-it (all bf16, same prompts, greedy decoding, thinking off).
**Result:** **Gemma 4 E4B-it** scores highest and is the model we carry forward.

## Test set
1,500 cells from 5 Raha benchmark tables (Hospital, Flights, Beers, Rayyan, Movies): per table 100 erroneous cells and 200 correct cells,
all taken from columns that contain errors, seed 42. Ground truth = Raha rule (normalised dirty ≠ clean). Each prompt has the row, a column profile,
dependency evidence and up to 3 similar rows, computed from the dirty table only. The model answers
`{"is_error", "error_type", "evidence", "repair"}`; P(is_error) is read from the first answer token.

## Score (fixed before running)
`Score = 0.4 × detection macro-F1 over the 5 tables + 0.3 × repair accuracy on true errors + 0.3 × AUROC of P(is_error)`, valid if ≥ 90% of answers are valid JSON.

| Model | Params (loaded) | JSON valid | Precision | Recall | Macro-F1 | AUROC | Repair acc. | False-repair rate | Type acc. | s/cell | Peak VRAM | **Score** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 4.0B | 99.9% | 0.50 | 0.62 | 0.580 | 0.767 | 0.274 | 0.67 | 0.64 | 0.21 | 13.8 GiB | 0.544 |
| Qwen3.5-9B | 9.0B | 99.3% | 0.55 | 0.75 | 0.643 | 0.843 | 0.328 | 0.69 | 0.70 | 0.29 | 23.3 GiB | 0.608 |
| **Gemma 4 E4B-it** | 7.9B (4.5B effective) | 100% | 0.67 | 0.73 | 0.692 | 0.860 | 0.340 | **0.36** | 0.76 | 0.24 | 19.0 GiB | **0.637** |

Per-table detection F1 (Qwen3-4B / Qwen3.5-9B / Gemma 4 E4B): Hospital 0.93 / 0.90 / 0.86, Flights 0.29 / 0.36 / **0.66**, Beers 0.53 / 0.67 / **0.69**,
Rayyan 0.57 / **0.70** / 0.68, Movies **0.59** / 0.58 / 0.56.

## Notes and caveats
- Gemma's advantage comes mainly from Flights (+0.30 F1) and from far fewer wrong repairs (36% vs 67–69%). Qwen3.5-9B has slightly higher Rayyan F1; Qwen3-4B is best on Hospital and Movies.
- This is a **zero-shot** comparison on one prompt. The first run used one sample of cells (seed 42); the multi-seed run below uses three fresh samples. Fine-tuning may still change the order.
- The first run (80-token output cap, parser cut at the first `}`) wrongly gave the Qwen models 82–89% valid JSON. Both issues were fixed and all three models were re-run; the winner did not change.
- Error types in the prompt are only annotated for cells covered by TableEG's annotation files (`n_typed`).
- Resource use: the models were run one at a time in bf16; the largest needed 23.3 GiB of GPU memory at peak.

## Reproduce
`scripts/model_bakeoff/prep_prompts.py` builds the prompts, `run_model.py <name>` runs one model, `score.py` computes the table. Set the environment variable `E` to a working folder that holds `data/`, `results/` and `weights/` (one folder per model).

## Sampling uncertainty (bootstrap, 2,000 resamples of the 1,500 cells, stratified by table and error/correct)
Greedy decoding has no randomness, so the uncertainty here comes from *which cells* were sampled. See `bootstrap_results.md` and `scripts/model_bakeoff/bootstrap.py`.

| Model | Score (mean ± std) | 95% interval |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.544 ± 0.010 | 0.523 – 0.564 |
| Qwen3.5-9B | 0.609 ± 0.010 | 0.588 – 0.627 |
| Gemma 4 E4B-it | 0.637 ± 0.011 | 0.616 – 0.659 |

Gemma minus Qwen3.5-9B: +0.028 (95% interval +0.003 to +0.053), ahead in 98.5% of resamples; most of the gain is macro-F1 (+0.048), while repair accuracy is statistically tied (+0.012, interval −0.020 to +0.046).
Gemma minus Qwen3-4B: +0.093 (+0.067 to +0.117), ahead in 100% of resamples.
Not covered by this bootstrap: prompt-wording variance and a different random draw of the test cells (the multi-seed run below covers the draw).

## Multi-seed run (3 fresh samples of 1,500 cells: seeds 1, 2, 3)
Greedy decoding is deterministic, so a seed only changes *which cells* are tested. Mean ± std over the 3 seeds (full table in `multiseed_results.md`):

| Model | Macro-F1 | AUROC | Repair acc. | False-repair rate | **Score** |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.585 ± 0.017 | 0.776 ± 0.013 | 0.271 ± 0.004 | 0.677 ± 0.010 | 0.548 ± 0.009 |
| Qwen3.5-9B | 0.642 ± 0.008 | 0.838 ± 0.001 | 0.317 ± 0.008 | 0.701 ± 0.003 | 0.603 ± 0.005 |
| **Gemma 4 E4B-it** | **0.698 ± 0.011** | **0.864 ± 0.004** | **0.357 ± 0.014** | **0.393 ± 0.022** | **0.645 ± 0.007** |

Gemma won on **all 3 seeds** (scores 0.652 / 0.646 / 0.637 vs Qwen3.5-9B 0.601 / 0.609 / 0.599). Gemma minus Qwen3.5-9B: +0.042 ± 0.007 (per seed +0.051, +0.037, +0.038); Gemma minus Qwen3-4B: +0.097 ± 0.016.
Per-table F1 means: Flights Gemma 0.68 vs 0.36 / 0.28, Beers 0.66 vs 0.60 / 0.46, Movies 0.67 vs 0.65 / 0.62; Rayyan Qwen3.5-9B 0.70 vs Gemma 0.62; Hospital Qwen3-4B 0.91, Qwen3.5-9B 0.89, Gemma 0.86.
Seconds per cell in this run are not comparable with the table above, because the three models shared the GPU at the same time (use the sequential run for speed).
Still not covered: prompt-wording variance. Scripts: `prep_prompts.py` (env `SEED`), `run_model_seeds.py`, `score_seeds.py`.

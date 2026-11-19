# Milestone 4 worklog

Each member fills in their own section: what they did, hours, and links to the files or pull requests they produced. Links are relative to this file.

## Sidhaarth — Training runs, pipeline experiments, detectors, validation layer, assembly (GPU)
**Dates:** 9–19 Nov 2026 · **Hours:** 18

Wrote the M4 plan and task split and mapped it onto the course team's 12-section outline.

**Training.** Added early stopping, epoch and validation-sampling options to the LoRA training script. Ran the full training run (55,091 examples, one epoch, best step 1,200) and the eleven screening and seed runs. Re-scored every adapter on one fixed validation sample, because each run had drawn its own.

**Pipeline experiments.** Ran E1, E2, E3 and E6 on the validation subset. From the missed errors, built two generic detectors:
- the format detector (D7);
- the entity-consensus extension of D5.

Then added entity evidence to the prompt and a repair-confidence gate.

**Validation layer.** Built the repair validation layer requested by the course team, with independent rule files.

**Final runs and scoring.**
- Ran the final configuration on the validation and test splits.
- Wrote the scorer that uses the baselines' rows, and the test-against-baselines table.

**Reviews.** Reviewed every member's work against the real runs and fixed the smaller issues before hand-back:
- Raha choosing its own labelled rows, and runs that did not repeat;
- rejected repairs counted as applied;
- the Hospital column names in the rule check;
- the E6 rule-validity value.

Assembled the report and this worklog.

**Files:** [`src/llm/train_lora.py`](../../../src/llm/train_lora.py), [`src/llm/infer.py`](../../../src/llm/infer.py), [`src/profiling/context.py`](../../../src/profiling/context.py), [`src/detectors/d7_format_mix.py`](../../../src/detectors/d7_format_mix.py), [`src/detectors/d5_entity_consensus.py`](../../../src/detectors/d5_entity_consensus.py), [`src/detectors/llm_rules.py`](../../../src/detectors/llm_rules.py), [`src/validation/repair_validator.py`](../../../src/validation/repair_validator.py), [`configs/validation/`](../../../configs/validation/), [`configs/m4_final_pipeline.yaml`](../../../configs/m4_final_pipeline.yaml), [`scripts/run_pipeline.py`](../../../scripts/run_pipeline.py), [`scripts/eval_adapters.py`](../../../scripts/eval_adapters.py), [`scripts/validate_repairs.py`](../../../scripts/validate_repairs.py), [`scripts/score_like_baselines.py`](../../../scripts/score_like_baselines.py), [`scripts/build_test_vs_baselines.py`](../../../scripts/build_test_vs_baselines.py), `scripts/run_m4_*.sh`, [`tests/test_m4_detectors.py`](../../../tests/test_m4_detectors.py), [`tests/test_m4_context_and_confidence.py`](../../../tests/test_m4_context_and_confidence.py), [`tests/test_repair_validator.py`](../../../tests/test_repair_validator.py), [`tests/test_train_lora.py`](../../../tests/test_train_lora.py), `docs/tables/m4_training_runs.csv`, `m4_f1_train_log.csv`, `m4_fixed_val_eval.csv`, `m4_pipeline_runs.csv`, `m4_inference_variants.csv`, `m4_validation_comparison.csv`, `m4_test_vs_baselines.csv`, [`Milestone4_Report.md`](Milestone4_Report.md) (Sections 1, 3, 5, 7, 10.1, 10.3–10.5, 11–13, appendices)

## Jashan — Training records, hyper-parameter analysis, seed spread, rule validity, sample outputs, artifacts (laptop)
**Dates:** 10–18 Nov 2026 · **Hours:** 14

Generated one configuration file per real training run from each run's `train_summary.json` (`scripts/generate_configs.py`). Extended `scripts/plot_training.py` to read all 13 runs: it builds the runs table with batch, schedule, weight decay, dropout and synthetic columns and the fixed-sample loss, plus the seed spread (0.0023 on the shared 1,024 examples) and the loss and comparison figures.

Fixed `scripts/check_rule_validity.py` to read the truth from the run's cell file, and ran it on the M3 and the M4 stage 0 rules. Found that the Hospital "state is al" rule holds on validation only because that split has no Alaska rows.

Rewrote `scripts/pick_examples.py` to cover missed errors, the gate decision, the validation outcome and the final decision. Selected the sample outputs from the final test run. Built the training artifacts inventory, wrote 5 tests and drafted report Sections 2, 4 (versions), 6 and 9.

**Files:** [`scripts/generate_configs.py`](../../../scripts/generate_configs.py), [`scripts/plot_training.py`](../../../scripts/plot_training.py), [`scripts/check_rule_validity.py`](../../../scripts/check_rule_validity.py), [`scripts/pick_examples.py`](../../../scripts/pick_examples.py), `configs/m4_f1_full.yaml`, `configs/m4_h0_ref.yaml` to `m4_h8_dropout0.1.yaml`, `m4_s1_seed7.yaml`, `m4_s2_seed123.yaml`, `m4_e5_no_synthetic.yaml`, `m4_train.yaml`, [`tests/test_m4_jashan.py`](../../../tests/test_m4_jashan.py), `docs/tables/m4_hparam_runs.csv`, `m4_seed_spread.csv`, `m4_rule_validity.csv`, `m4_rule_validity_e6.csv`, `m4_sample_outputs.csv`, `m4_training_artifacts.csv`, [`docs/figures/m4/`](../../figures/m4/), [`sample_outputs.md`](sample_outputs.md)

## Sahilpreet — Decision scoring, prior estimation, calibration study, outcome tables (laptop)
**Dates:** 10–16 Nov 2026 (revisions to 19 Nov) · **Hours:** 14.5

**Decision scoring.** Implemented `score_decisions`. It counts every cell once (repaired right or wrong, review, kept) and scores the gate decision and the final decision after validation separately, with no-change repairs counted on their own. It reproduces the validation layer's counts on every real run.

**Prior estimation.** Implemented the EM prior estimate of Saerens et al. (2002) in `estimate_prior_em`. It falls back to a default only when EM fails, and the raw estimate is always reported. Implemented `fit_gate`, which reports whether the false-repair cap was met.

**Calibration study.** Built `scripts/fit_calibration.py` on the real runs: temperature 0.826, EM estimates per table, thresholds fitted on validation and the reliability diagram. It also produces the outcome tables for the base, fine-tuned, all-cells and final runs.

**M3 re-score.** Re-scored the M3 run from the real base-model run (`scripts/rescore_m3_gate.py`): 62 right, 57 wrong, 9 no-change at the gate, and 59 right, 1 wrong after validation.

Made `score_flags` refuse ambiguous row exclusions across tables. Wrote 26 tests and drafted report Sections 5 (loss), 8 (prior shift), 10.2 and 10.6.

**Files:** [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py), [`src/evaluation/calibrate.py`](../../../src/evaluation/calibrate.py), [`src/evaluation/__init__.py`](../../../src/evaluation/__init__.py), [`scripts/fit_calibration.py`](../../../scripts/fit_calibration.py), [`scripts/rescore_m3_gate.py`](../../../scripts/rescore_m3_gate.py), [`scripts/score_run.py`](../../../scripts/score_run.py), [`configs/m4_calibration_study.yaml`](../../../configs/m4_calibration_study.yaml), [`tests/test_m4_sahilpreet.py`](../../../tests/test_m4_sahilpreet.py), `docs/tables/m4_calibration.csv`, `m4_outcomes_test_hfb.csv`, `m4_outcomes_val_all.csv`, `m4_outcomes_e1_ft.csv`, `m4_outcomes_e3_all_ft.csv`, `m3_rescored_gate.csv`, `docs/figures/m4/reliability_diagram.png`

## Aaradhya — Baselines on the test splits, candidate recall, E3 and E6 tables, missed errors (GPU and CPU)
**Dates:** 16–18 Nov 2026 · **Hours:** 15

**Baselines.** Ran the three baselines on the test splits, leaving the 20 labelled rows out of scoring. Fixed the earlier problems:
- the constraint check no longer "repairs" a cell to its own value;
- dBoost failures are reported, not hidden;
- Raha is imported only when B3 runs.

The M3 script is kept as `run_baselines_m3.py`.

**E3 and E6.** Built the E3 and E6 tables from the real runs (`scripts/build_m4_experiments.py`). E3: widen on Beers and Hospital, not on Flights. E6: stage 0 fails the recall criterion and is dropped.

**Detectors.** Removed the table-specific parts of D2. Measured candidate recall with the final detector set on the validation splits (`scripts/evaluate_m4_candidates.py`), and analysed the missed errors by table, column, type and nearest detector (`scripts/analyze_missed_errors.py`). Drafted report Sections 10.3 (E3, E6), and the baselines.

**Files:** [`scripts/run_baselines_full.py`](../../../scripts/run_baselines_full.py), [`scripts/run_baselines_m3.py`](../../../scripts/run_baselines_m3.py), [`scripts/run_baselines.py`](../../../scripts/run_baselines.py), [`scripts/build_m4_experiments.py`](../../../scripts/build_m4_experiments.py), [`scripts/evaluate_m4_candidates.py`](../../../scripts/evaluate_m4_candidates.py), [`scripts/analyze_missed_errors.py`](../../../scripts/analyze_missed_errors.py), [`src/detectors/d2_pattern.py`](../../../src/detectors/d2_pattern.py), [`tests/test_detectors.py`](../../../tests/test_detectors.py), `docs/tables/m4_baselines_full.csv`, `m4_candidate_recall.csv`, `m4_e3.csv`, `m4_e6.csv`, `m4_missed_errors.csv`

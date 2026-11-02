# Changelog

All notable changes to this project are recorded here, one section per milestone.

## [Unreleased] — Milestone 2 (dataset preparation)
- Added `docs/milestones/M2/Milestone2_Report.md` (13-section report with a team sign-off page), `Milestone2_Worklog.md`, `Handover_Check.md` (hand-over check, passed), the PDF of the report, and the review deck (`ContextGuard_M2_Slides.pptx` and PDF) in `docs/milestones/M2/slides/`.
- Core tables: `src/data/load_core.py` (pinned sources, SHA-256, Raha rule, error types), `src/data/split.py` (70 / 15 / 15 split by entity, seed 42, grouped on the clean key), `scripts/generate_m2_outputs.py` (quality table, data dictionary, figures; Figure 1 now shows all five tables, Figure 2 labels no longer clipped), `docs/tables/m2_core_*.csv`, `docs/figures/m2/`.
- External tables and synthetic data: `src/data/load_external.py` (Tax sample, Inpatient, Facilities, Billionaire, Germany slice: checks, counts, cell files, splits, dictionary), `src/data/synthetic.py` (four error recipes on clean UCI tables, dependencies discovered from the clean table), `docs/tables/m2_external_*.csv`, `m2_synthetic_summary.csv`, `m2_germany_sample_metadata.json`.
- Analysis and baselines: `scripts/eda_analysis.py`, `src/profiling/dependencies.py` (g3 dependency scan with key, support and gain filters), `src/data/export_baselines.py` (inputs of B1 to B3 from the test split, Rule D8), `docs/tables/m2_outliers.csv`, `m2_dependencies.csv`, `m2_competitor_data_practice.csv`, `docs/figures/m2/dependency_heatmap.png`.
- Fine-tuning data: `src/profiling/context.py` (row, column profile, dependency evidence and similar rows for one cell, from the dirty table only), `src/llm/build_examples.py` (selection rule, token budget, 20% synthetic cap), `src/data/check_leakage.py` (29 checks, `docs/tables/m2_leakage_checks.csv`), `docs/tables/m2_final_dataset_summary.csv`, `m2_token_lengths.csv` (length estimate), `m2_token_lengths_gemma.csv` (counted with the Gemma tokenizer; longest example 1,009 of 1,500), `data/sample/m2_examples_sample.jsonl`.
- Reproducibility: `scripts/download_data.py` (every source at a pinned version, checked by SHA-256), `scripts/build_all.sh` (one command for the whole build), `tests/` (18 tests).
- Updated `README.md`, `data/README.md`, `THIRD_PARTY.md` (licence review and attribution), `requirements.txt` and `.gitignore` (splits, local clones).
- Revision after the course team's review (November 2026): report Section 14 answers the five review points (label audit and handling of uncertain labels, limits of synthetic errors and the real-data tests including a required E5 for outliers, evaluation at real error rates per table and error type, example quality, reproducibility).
  - New `scripts/audit_labels.py` (checks the rule-assigned error types of Tax, Inpatient and Facilities and draws three frozen samples for hand review: 80 cells, 80 fresh cells and a random sample of 150 cells after the correction, none with a wrong type; reviewer verdicts are kept across rebuilds), new `scripts/check_examples.py` (all 73,979 examples checked against the split files), new `docs/tables/m2_label_audit_summary.csv`, `m2_label_audit_sample.csv`, `m2_label_audit_holdout.csv`, `m2_label_audit_random.csv` (with the reviewers' verdicts), `m2_example_checks.csv`; `scripts/build_all.sh` runs both scripts before the tests.
  - **Data change:** the rule that types the errors of the external tables (`src/data/load_external.py::classify`) is extended after the audit sample showed swapped and substituted values typed `pattern`: a value that is the clean value of another column of the same row, or a well-formed value that is a different value, is now a `rule` violation. 2,230 error cells changed their type (1,117 in the training tables: 1,113 `pattern` to `rule` in Tax 36, Inpatient 477 and Facilities 600, and 4 `rule` to `pattern`; 1,113 in the unseen tables Billionaire and Germany). Two edge cases found in the hand review were corrected too (a zip whose clean value lost its leading zero; a value that is the clean value with something added or dropped). Error status, repairs, splits, prompts and example counts are unchanged; `train.jsonl`, `val.jsonl` and `test.jsonl` changed (new SHA-256 prefixes `99e3cdf9`, `e8f75de3`, `16c3dee4`).
  - Updated for the new types: `docs/tables/m2_external_splits.csv`, `m2_external_sources.csv`, `m2_final_dataset_summary.csv`, `m2_token_lengths.csv`, `m2_token_lengths_gemma.csv`, the counts in report Sections 3.2, 5.2 and 10, `data/download_log.md` and `data/README.md`.
  - `docs/milestones/M2/Milestone2_Report.pdf` regenerated from the revised report (32 pages).
  - `README.md` gives the TableEG Drive link, file names, target folder and the checks for a finished build; `Handover_Check.md` records Check 3 (clean rebuild on 2 Nov 2026).

## [Unreleased] — Milestone 1 (problem definition and literature review)
- Repository created with the course folder structure.
- Added README, requirements and data notes; model comparison scripts and results (`scripts/model_bakeoff/`, `outputs/reports/model_bakeoff/`).
- Added the benchmark-table audit (`scripts/data_inventory.py`, `docs/tables/`, `docs/figures/`) and the Section 7 text.
- Added the external-table audit (`scripts/external_inventory.py`, `docs/tables/external_*.csv`) and the download log (`data/download_log.md`).
- Added the Raha smoke-test script and output (`outputs/logs/raha_hospital_smoke_test.txt`) and the merged `Milestone1_Report.md`.
- Added `docs/milestones/M1/slides/M1_slide_text.md`, the text of the 10-slide review deck.
- Added `docs/Project_Proposal.pdf`, the problem statement as submitted to the course team.
- Added the Milestone 1 review deck (`docs/milestones/M1/slides/ContextGuard_M1_Slides.pptx`) and a PDF copy.
- Defined baseline B1 as HoloClean with constraints mined by TANE (one pipeline for all tables) in the report and the slide text record.
- Revised `docs/milestones/M1/Milestone1_Report.md` and its PDF after the Milestone 1 review meeting:
  - Section 5.7: what existing work has established, the research gaps that remain, and how the project addresses each one.
  - Section 6.1: why the approach is expected to work, the assumptions behind it, and the main research goal against the supporting benefits.
  - Section 7.8: what the dirty-versus-clean labels establish, their known limits and how they are handled.
  - Sections 8.5 to 8.7: experiments that isolate each component, what the generalisation tests establish, and how mixed results will be judged.
  - Section 4 (evidence paragraph) and Section 9 (plan and risks) updated to refer to the new sections.

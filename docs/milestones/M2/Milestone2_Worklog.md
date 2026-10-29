# Milestone 2 worklog

Each member fills in their own section: what they did, hours, and links to the files or pull requests they produced. Links are relative to this file.

## Jashan — Data (benchmark tables, splits, description)
**Dates:** 24–28 Oct 2026 · **Hours:** 13.0

Wrote the loader for the five core benchmark tables (pinned Raha commit, SHA-256 checks, positional column alignment, the Raha comparison rule, TableEG error types with a documented rule for unannotated cells) and verified all counts against Milestone 1 (18,414 error cells). Measured entity repetition and implemented the 70 / 15 / 15 entity-aware split with seed 42 (Decisions D1 and D2); after the leakage check showed that the dirty Hospital key split 12 of 45 hospitals across splits, regrouped on the clean key. Produced the quality table (clean-file gaps, placeholders, duplicates, format-only cells), the 63-column data dictionary and three figures, and wrote the report sections on identification, description, governance, EDA, preprocessing and splitting of the core tables.

**Files:** [`src/data/load_core.py`](../../../src/data/load_core.py), [`src/data/split.py`](../../../src/data/split.py), [`scripts/generate_m2_outputs.py`](../../../scripts/generate_m2_outputs.py), [`docs/tables/m2_core_splits.csv`](../../tables/m2_core_splits.csv), [`docs/tables/m2_core_quality.csv`](../../tables/m2_core_quality.csv), [`docs/tables/m2_core_dictionary.csv`](../../tables/m2_core_dictionary.csv), [`docs/figures/m2/`](../../figures/m2/); report Sections 2.1, 3.1, 4.2, 5.1, 6.1, 9.1 and 12.2 in [`Milestone2_Report.md`](Milestone2_Report.md)

## Sahilpreet — Data (external tables, licences, synthetic data)
**Dates:** 27–28 Oct 2026 · **Hours:** 11.0

Built the loader and audit for Tax (20,000-row sample), Inpatient, Facilities, Billionaire and the Germany slice (SHA-256 checks, Raha-rule counts matching Milestone 1, typed cells, entity-aware splits for Inpatient by clinician and Facilities by facility, data dictionary). Reproduced the Germany sample with the Milestone 1 method. Wrote the synthetic generator on clean UCI tables (four recipes, dependencies discovered from the clean table, 40,132 corrupted cells in five tables) and the 20% cap. Compiled the licence and status table for all sources and the privacy assessment per table, and wrote the report sections for the external data.

**Files:** [`src/data/load_external.py`](../../../src/data/load_external.py), [`src/data/synthetic.py`](../../../src/data/synthetic.py), [`docs/tables/m2_external_sources.csv`](../../tables/m2_external_sources.csv), [`docs/tables/m2_external_dictionary.csv`](../../tables/m2_external_dictionary.csv), [`docs/tables/m2_external_splits.csv`](../../tables/m2_external_splits.csv), [`docs/tables/m2_synthetic_summary.csv`](../../tables/m2_synthetic_summary.csv), [`docs/tables/m2_germany_sample_metadata.json`](../../tables/m2_germany_sample_metadata.json); report Sections 2.2, 3.2, 4.1, 5.2, 8 and 12.3 in [`Milestone2_Report.md`](Milestone2_Report.md)

## Aaradhya — Outlier and dependency analysis, baselines
**Dates:** 27–28 Oct 2026 · **Hours:** 7.5

Analysed outliers (robust z-score and Isolation Forest) on the numeric measure columns of the eight training tables, with format-only cells left out, and scanned approximate column dependencies (g3 measure, key, support and gain filters; 365 dependencies). Reviewed how Raha, Baran, HoloClean, TableEG, Jellyfish and ZeroED prepare and split their data against the papers. Wrote the exporter for the baseline inputs from the test split (HoloClean constraints in its syntax, dBoost grid, Raha and Baran with 20 labelled tuples excluded from scoring, Rule D8 verified) and the hand-over check note.

**Files:** [`scripts/eda_analysis.py`](../../../scripts/eda_analysis.py), [`src/profiling/dependencies.py`](../../../src/profiling/dependencies.py), [`src/data/export_baselines.py`](../../../src/data/export_baselines.py), [`docs/tables/m2_outliers.csv`](../../tables/m2_outliers.csv), [`docs/tables/m2_dependencies.csv`](../../tables/m2_dependencies.csv), [`docs/tables/m2_competitor_data_practice.csv`](../../tables/m2_competitor_data_practice.csv), [`docs/figures/m2/dependency_heatmap.png`](../../figures/m2/dependency_heatmap.png), [`Handover_Check.md`](Handover_Check.md); report Sections 2.3, 5.3, 9.2 and 12.4 in [`Milestone2_Report.md`](Milestone2_Report.md)

## Sidhaarth — Examples, leakage checks, integration
**Dates:** 24–28 Oct 2026 · **Hours:** 14.0

Wrote the plan and task split for the milestone and reviewed the work of all three members against the original files, running their scripts and checking every count; found the Hospital split leak, the unverified masking policy, the Germany sampling difference and the trivial dependencies, and prepared the fixes with the owners. Built the prompt-context module, the fine-tuning example builder (selection rule, token budget, synthetic cap), the leakage checker (29 checks), the download script and the one-command build, and the tests. Fixed the decisions on statistics, selection and token budget (D3 to D5), assembled the report and the sign-off page, and updated the README, data README, attributions, changelog and requirements.

**Files:** [`src/profiling/context.py`](../../../src/profiling/context.py), [`src/llm/build_examples.py`](../../../src/llm/build_examples.py), [`src/data/check_leakage.py`](../../../src/data/check_leakage.py), [`scripts/download_data.py`](../../../scripts/download_data.py), [`scripts/build_all.sh`](../../../scripts/build_all.sh), [`tests/`](../../../tests/), [`Milestone2_Report.md`](Milestone2_Report.md) (Sections 1, 4.3, 6.2, 7, 9.3, 10 to 13), [`README.md`](../../../README.md), [`data/README.md`](../../../data/README.md), [`THIRD_PARTY.md`](../../../THIRD_PARTY.md)

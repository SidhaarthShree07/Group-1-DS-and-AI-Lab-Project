# Milestone 1 worklog

Each member fills in their own section: what they did, hours, and links to the files or pull requests they produced. Links are relative to this file.

## Jashan — Data (benchmark tables)
**Dates:** 18–20 Oct 2026 · **Hours:** 9.5

Audited the nine benchmark tables (Hospital, Flights, Beers, Rayyan, Movies, Credit, Marketing, Soccer, Restaurant) with `scripts/data_inventory.py` under Raha's comparison rule. Found and resolved the errors lost in TableEG's copies of Beers (1,005) and Movies (a net 668, from `RatingValue` `7.0` against `7`) by using the Raha originals. Mapped TableEG's annotations (Rayyan: 75 missing, 254 pattern, 619 rule), typed Soccer's changes (597 pattern, 199 outlier) and Restaurant's errors (633 pattern) from the files, and kept Credit's 315,466 float re-saving differences apart from the true errors. Audited the clean side of every table (blanks, duplicates, mixed formats), the header mismatches, Credit's class balance and its 17,435 imputed income values, and Soccer's BART log. Wrote Section 7 (benchmark tables) and generated the summary tables, the error-type chart and two review slides.

**Files:** [`scripts/data_inventory.py`](../../../scripts/data_inventory.py), [`docs/tables/data_inventory.csv`](../../tables/data_inventory.csv), [`docs/tables/error_type_profile.csv`](../../tables/error_type_profile.csv), [`docs/figures/error_distribution_pie.svg`](../../figures/error_distribution_pie.svg); report Section 7 (benchmark tables) in [`Milestone1_Report.md`](Milestone1_Report.md)

## Sahilpreet — Data (external datasets)
**Dates:** 20 Oct 2026 · **Hours:** 8.5

Downloaded and audited the external clean/dirty pairs: Tax (Raha), Billionaire (ZeroED), Inpatient and Facilities (BClean, 10% variants). Recorded the source repositories and commits, file sizes and the UCI clean tables with their citations and CC BY 4.0 terms. Checked row counts and column alignment for all four pairs, including BClean's extra row-index column, and compared the BClean variants (5%, 10%, 15%, 20%). Wrote an inventory script and generated the first inventory and per-column difference tables. Drafted two review slides and a first version of the download log. After review, confirmed the Billionaire count of 5,195 under Raha's rule and traced his BClean copy to the paper's own repository (commit `bb8e09e…`).

**Files:** [`data/download_log.md`](../../../data/download_log.md), [`scripts/external_inventory.py`](../../../scripts/external_inventory.py), [`docs/tables/external_inventory.csv`](../../tables/external_inventory.csv), [`docs/tables/external_column_differences.csv`](../../tables/external_column_differences.csv); report Section 7.7 in [`Milestone1_Report.md`](Milestone1_Report.md)

## Aaradhya — Detectors and baselines
**Dates:** 18–20 Oct 2026 · **Hours:** 14

Reviewed the eight classical and learning-based systems from their papers and corrected the literature table. Wrote the detector references for D3 to D6 (including the g3 measure), re-checked the numbers to beat against the original papers (Raha Table 5, Baran Tables 3 and 7, HoloClean Table 2, ZeroED Table III), defined baselines B1 to B5 (HoloClean for B1, Gemma 4 E4B-it for B4 and B5) and the exact Raha evaluation protocol, and reviewed ten open-source and commercial tools from their documentation. Installed Raha, patched a Windows file-locking problem in its dBoost runner, and ran the Hospital smoke test (F1 0.6589). Drafted two review slides.

**Files:** [`scripts/run_raha_baseline.py`](../../../scripts/run_raha_baseline.py), [`outputs/logs/raha_hospital_smoke_test.txt`](../../../outputs/logs/raha_hospital_smoke_test.txt); report Sections 5.1, 5.2, 5.4, 8.2 to 8.4 and Appendices A and B in [`Milestone1_Report.md`](Milestone1_Report.md)

## Sidhaarth — LLM, evaluation, integration
**Dates:** 18–21 Oct 2026 · **Hours:** 15

Chose the base model: built a fixed 1,500-cell zero-shot test set from five Raha tables, ran Qwen3-4B-Instruct-2507, Qwen3.5-9B and Gemma 4 E4B-it fixed two scoring bugs (answer cap and JSON parser) and re-ran, then repeated the comparison on three sampling seeds and a bootstrap (Gemma 4 E4B-it 0.645 ± 0.007 against 0.603 ± 0.005 and 0.548 ± 0.009). Audited the data work of the team against the original files (error counts, Beers and Movies losses, TableEG leakage). Decided the dataset list and set up the repository (structure, licence, contribution rules, third-party list, README). Wrote Milestone 1 sections 1–4, 5.3, 5.5, 6, 8 (metrics) and 9, and the two review slides, and built the 10-slide review deck from the team's slide text. Checked the teammates' parts against the papers and the data, and merged all parts into `Milestone1_Report.md` with one reference list. Completed the external-table part of Section 7 from Sahilpreet's audit: Raha-rule error counts and error types, the Germany slice (found in the Zenodo archive and sampled with seed 42), overlap checks, licences, checksums and the corrected download log.

**Files:** [`Milestone1_Report.md`](Milestone1_Report.md) (Sections 1 to 4, 5.3, 5.5, 5.6, 6, 8.1, 9), [`slides/ContextGuard_M1_Slides.pptx`](slides/ContextGuard_M1_Slides.pptx), [`slides/M1_slide_text.md`](slides/M1_slide_text.md), [`scripts/model_bakeoff/`](../../../scripts/model_bakeoff/), [`outputs/reports/model_bakeoff/README.md`](../../../outputs/reports/model_bakeoff/README.md), [`README.md`](../../../README.md)

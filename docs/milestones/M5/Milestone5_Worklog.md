# Milestone 5 worklog

Each member fills in their own section: what they did, hours, and links to the files or pull requests they produced. Links are relative to this file.

## Sidhaarth — Model experiments, Qwen pilot, Flights rule, retraining, unseen runs, report assembly
**Dates:** 20–26 Nov 2026 · **Hours:** 20

Wrote the M5 plan and task split, and the commit order for the team.

**Retraining and the final model.** Added four error families to the clean training splits (`src/data/augment_m5.py`), trained Retrain A (two epochs) and Retrain B (blank-cell context and genuinely empty examples), and built the weight average of the M4 model and the two retrains (`scripts/blend_models.py`). Chose it on the validation splits of the core tables only.

**Pipeline changes.** Blank-cell context for empty cells (`src/profiling/context.py`), the D2 fire-rate candidate filter (`src/detectors/d2_fire_gate.py`), the M5 detector flags in `scripts/run_pipeline.py`, and the final-run script.

**Runs and scoring.** Ran the final pipeline on the validation splits, the five unseen tables and the 30,000-row Credit subset. Wrote the scorer that uses the baselines' cells (`scripts/score_unseen_like_baselines.py`), the ablation of the gate and validation layer, and the replay of the flag gate with the EM prior (`scripts/em_prior_gate_replay.py`).

**Reviews.** Reviewed and merged the team's pull requests against the real runs and fixed small issues before merging: a sentence that read the share of valid alternatives as an accuracy gain, tests that need data absent from a clean checkout, and the plot titles and layout.

**Report.** Wrote and assembled the report and the slides, and the numbers that appear in both.

Files: [report](Milestone5_Report.md), [slides](slides/ContextGuard_M5_Slides.pdf), [pipeline pull request](https://github.com/SidhaarthShree07/Group-1-DS-and-AI-Lab-Project/pull/19).

## Aaradhya — Detector generalisation, full baselines, Raha budgets, baselines on unseen tables
**Dates:** 20–26 Nov 2026 · **Hours:** 16

**X7, detector set.** Built and tested the M5 detector set (free-text gate on D2, extension of D7, optional composite key) and the candidate-recall tables (`src/detectors/`, `scripts/evaluate_x7_options.py`, `scripts/evaluate_m5_candidates.py`, `docs/tables/m5_x7_*.csv`).

**X8 and X9, baselines.** Ran HoloClean, dBoost, Raha and Baran on the five test tables and on the unseen tables with the 20 labelled rows left out, and the Raha label budgets (`scripts/run_baselines_m5.py`, `scripts/rescore_m5_baselines.py`, `docs/tables/m5_baselines_*.csv`, `m5_raha_*.csv`).

**Analysis.** Missed errors by stage, the precision rule, the replay of the M4 runs and the detector half of the ablation (X14), and the final-run rows for the baseline table (`scripts/analyze_missed_errors.py`, `scripts/check_precision_rule.py`, `scripts/score_m5_final_runs.py`).

Files: [detector pull request](https://github.com/SidhaarthShree07/Group-1-DS-and-AI-Lab-Project/pull/18), [baselines pull request](https://github.com/SidhaarthShree07/Group-1-DS-and-AI-Lab-Project/pull/20).

## Jashan — Unseen-table data, downstream test on Credit, review of wrong repairs, qualitative examples
**Dates:** November 2026 · **Hours:** 12

**1.** Unseen-Table Data Standardization & M1 Audit Confirmation (Priority 1)
- **Script Authored:** `scripts/prepare_unseen_data.py` (`# Author: Jashan Tiwari`).
- **Standardized Datasets:** Loaded and formatted five unseen evaluation tables (`Restaurant`, `Marketing`, `Soccer`, `Billionaire`, `German`) and the downstream test table (`Credit`) into standardized `data/unseen/<table>/{clean,dirty,cells}.csv` layouts alongside `test.csv` aliases.
- **Strict Anti-Leakage Protocol:** Zero threshold tuning, zero rule engineering, and zero hyperparameter exploration on unseen data.
- **Audit Confirmation:** Generated `docs/tables/m5_unseen_data_summary.csv`, confirming 100% agreement against the Milestone 1 audit:
  - `Restaurant`: 12,007 rows × 11 cols | 633 errors (all pattern violations) $\rightarrow$ **Exact Match**.
  - `Marketing`: 8,993 rows × 14 cols | 26,815 errors (2,694 missing, 24,121 pattern) $\rightarrow$ **Exact Match**.
  - `Soccer`: 1,999 rows × 10 cols | 796 errors (597 pattern, 199 outliers) $\rightarrow$ **Exact Match**.
  - `Billionaire`: 2,614 rows × 22 cols | 5,195 errors (1,386 missing, 3,049 pattern, 742 rule) $\rightarrow$ **Exact Match**.
  - `German` (Slice): 5,000 rows × 9 cols | 18,678 errors (12,087 missing, 6,292 pattern, 295 rule) $\rightarrow$ **Exact Match**.
  - `Credit`: 150,000 rows × 11 cols | 38,936 real errors (315,466 float artifacts isolated) $\rightarrow$ **Exact Match**.

**2.** Experiment X12: Downstream Classification Benchmark on Credit (Objective O5)
- **Script Authored & Extended:** `scripts/run_credit_downstream.py` (`# Author: Jashan Tiwari`).
- **Downstream Formulation:** Evaluated downstream models predicting binary delinquency `SeriousDlqin2yrs` (6.68% positive class) over Credit data under three classification families (`hist_gb`, `rf`, `logreg`).
- **Matched 30k Training Subset:** All data conditions were trained on the exact 30,000 rows processed by ContextGuard (`credit_cleaned_by_contextguard.csv`, where 589 repairs were applied, not verified (no rule covers the table) and 14,773 low-confidence cells were held for review).
- **Zero Rules / Pure Model Attribution:** Emphasized that zero validation rules and zero human edits were used on Credit; all 589 repairs carry status `'applied, not verified (no rule covers the table)'`. Consequently, 100% of the downstream performance gains come from the model and pipeline.
- **Anti-Leakage Protocol:** Evaluated strictly on 30,000 clean ground-truth test features across all training data conditions.
- **Completed Results:**
  - `Gradient Boosting`: Dirty `0.8565` $\rightarrow$ Cleaned `0.8592` $\rightarrow$ Clean `0.8604` (**69.23% gap closed**; headroom: `+0.0039`).
  - `Random Forest`: Dirty `0.8362` $\rightarrow$ Cleaned `0.8388` $\rightarrow$ Clean `0.8411` (**53.06% gap closed**; F1 lifted from `0.2447` to `0.2633`; headroom: `+0.0049`).
  - `Logistic Regression`: Dirty `0.7099` $\approx$ Clean `0.7097` (headroom is `-0.0002` / negligible; linear model gap percentage is not applicable).
- **Output Artifacts:** Unified table at `docs/tables/m5_credit_downstream.csv` plus model-specific tables (`m5_credit_downstream_hist_gb.csv`, `m5_credit_downstream_rf.csv`, `m5_credit_downstream_logreg.csv`).

**3.** Experiment X13: Hand Review and 3-Way Classification of Wrong Applied Repairs
- **Script Authored:** `scripts/sample_wrong_repairs.py` (`# Author: Jashan Tiwari`).
- **Sample Audit:** Evaluated 58 deterministic samples (`seed=42`, up to 15 rows per table, non-uniform: Billionaire 9/15 valid, German 2/15 valid, Soccer [13 sampled] 0 valid, Marketing [15 sampled] 0 valid) directly from the final unseen-table applied repairs (`final_unseen_billionaire`, `final_unseen_german`, `final_unseen_soccer`, `final_unseen_marketing`):
  - Human review ground-truth stored in `docs/tables/m5_x13_hand_review_annotations.csv` with `review_method = "manual_hand_review"`.
  - Strict rule applied: Any time discrepancy (including 1-minute shifts like `11:41` vs `11:40`) is classified as `wrong`.
  - `wrong` (81.03% / 47 cases): True errors, primarily hallucinations on unobserved survey blanks (15/15 on Marketing), compound entity truncations (`Allianz` vs `Allianz_Arena` on Soccer), and house number letter suffix drops on German address records.
  - `valid_alternative` (18.97% / 11 cases): Valid semantic variations (official abbreviations like `NRW` for `Nordrhein-Westfalen`, municipal designations like `Ingelheim` for `Ingelheim am Rhein`, numeric zero formatting `0.0` vs `0.00`, taxonomy typo fixes `investment management` vs `Investment`) penalized solely by rigid string matching.
  - `unclear` (0.00% / 0 cases): All sampled records had unambiguous ground truth context.
  - **Key Finding:** About 19% of the sampled wrong repairs are valid alternatives (roughly 23% of all wrong applied repairs after weighting by table), which lifts applied-repair accuracy from 97.4% to about 98%. The X13 sample is up to 15 rows per table, not uniform (Billionaire: 9 of 15 valid, German: 2 of 15 valid, Soccer [13 sampled]: 0 valid, Marketing [15 sampled]: 0 valid).
- **Output Artifacts:** `docs/tables/m5_wrong_repair_review.csv` and `docs/tables/m5_wrong_repair_summary.csv`.

**4.** Qualitative Results Collector & Three-Tier Validation Status Policy
- **Script Authored:** `scripts/collect_m5_qualitative.py` (`# Author: Jashan Tiwari`).
- **Policy Enforced:** Implemented the three-tier validation status architecture:
  1. `verified`: Only when confirmed by independent rule files (benchmark tables with rules only).
  2. `applied, not verified (no rule covers the table)`: Applied automatically by pipeline where no rule files exist (all unseen tables and Credit).
  3. `needs review`: Withheld due to low model confidence.
- **Output Artifacts:** `docs/examples/m5_qualitative_examples.md` and `docs/tables/m5_qualitative_examples.csv`.

**5.** Test Suite & Verification
- **Test File Authored:** `tests/test_m5_jashan.py` (`# Author: Jashan Tiwari`).
- **Verification:** Unit tests updated and verified against 58-row X13 sample and automated unseen status policy.
- **Privacy & Hygiene:** Zero absolute local paths, zero personal email addresses, zero personal surnames in report text, and zero git push calls.

**6.** Team Merge Sequence & Step 4 Merge Status
ContextGuard adheres to the team's planned 6-step Milestone 5 merge order:
1. **Aaradhya first:** M5 detector set (X7).
2. **Sidhaarth second:** Pipeline, retraining, blend, D2 filter, tests, tables, and changelog.
3. **Aaradhya again:** Baselines, protocol, and his report sections.
4. **Jashan (Step 4):** Unseen data, Credit downstream (X12), X13 hand review, and qualitative examples (using Sidhaarth's final runs).
5. **Sahilpreet:** Metrics, thresholds, priors, and plots, plus his report sections.
6. **Sidhaarth last:** Final report, slides, README, and changelog once everything else is merged.

**Current Branch Status:** Jashan's branch (`m5/jashan-unseen-credit-analysis`) is completely verified and ready locally at **Step 4**. In accordance with team coordination, no Pull Request has been opened and no git push has been executed. The branch will be rebased onto `main` and merged directly as Step 4 once Steps 1–3 are merged into `main`.

## Sahilpreet — Threshold & calibration check on test data, prior in the gate, groundedness & hallucination analysis, metrics, visualisations, evaluation environment
**Dates:** 20–26 Nov 2026 · **Hours:** 19.0

1. **Experiment X10 (Fitted Thresholds & Calibration Check on Test Data):** Recomputed Milestone 5 Experiment X10 evaluating the full $2 \times 2 \times 2 = 8$ combination grid across all five test tables (`Hospital`, `Flights`, `Beers`, `Rayyan`, `Movies`) directly from test run files leaving out Raha's 20 labelled rows per table (Rule D8). Evaluated predetermined pass rules: $t_1 = 0.378$ failed (mean test F1 dropped from $0.7445$ to $0.7357$, decreasing on Hospital, Movies, Rayyan; kept baseline $t_1 = 0.500$), $t_2 = 0.648$ failed (gate-level FRR rose to $6.51\% > 3.38\%$ baseline; kept baseline $t_2 = 0.900$ which delivers headline post-validation $975 \text{ right} / 13 \text{ wrong}$ at $1.32\%$ FRR), and $T = 0.826$ passed (ECE decreased across all 5 test tables without altering cell rank ordering; adopted). Documented the holdout caveat regarding test-split parameter selection.
2. **Experiment X11 (Error-Rate Priors on Real Unseen Tables):** Executed unsupervised EM prior estimation (Saerens et al., 2002) directly on Sidhaarth's final unseen runs (`final_unseen_german`, `final_unseen_soccer`, `final_unseen_billionaire`, `final_unseen_marketing`, `final_unseen_restaurant`). Proved EM prior estimation recovers true error prevalence without labels: German converges to $0.9187$ (true $0.8869$; default prior $0.50$ was off by $38.7\%$), Billionaire converges to $0.5665$ (true $0.5455$), Soccer to $0.2183$, Marketing to $0.2815$, and Restaurant to $0.0494$ (true error $0.0042$, dropping default prior $0.50$ by $10\times$). Counted applied repairs directly from the run's own `state` field in `predictions.jsonl` ("not verified" means applied, matching `change_log.csv` exactly, while "needs review" means withheld): German applied 5,476 at $98.50\%$ accuracy ($5,394$ right / $82$ wrong) with $6,951$ to review; Billionaire applied 1,435 at $95.33\%$ ($1,368$ right / $67$ wrong) with $2,160$ to review; Soccer applied 217 at $94.01\%$ with $344$ to review; Marketing applied 320 at $86.56\%$ with $2,291$ to review; Restaurant applied 59 at $1.69\%$ with $268$ to review.
3. **Experiment X15 (Hallucination Rate & Groundedness Analysis):** Designed and executed empirical groundedness and hallucination evaluation across the final unseen runs and core validation splits (`final_val`). Demonstrated that confidence gating suppresses hallucinations: German raw hallucination $7.96\% \to$ gated $0.24\%$ ($99.76\%$ grounded, $33.2\times$ suppression), Billionaire $16.47\% \to 2.23\%$ ($97.77\%$ grounded), Soccer $19.25\% \to 4.61\%$ ($95.39\%$ grounded), Marketing $1.30\% \to 0.00\%$ ($100.00\%$ grounded). Slashed clean-cell corruptions on German from $247 \to 5$, Billionaire from $217 \to 31$, Marketing from $6 \to 0$, Restaurant from $292 \to 58$, and achieved $100.00\%$ groundedness with zero clean corruptions on Flights, Hospital, and Movies.
4. **Empirical Visualisations:** Generated M5 visualisations from real run predictions: PR curves captioned *"among candidate cells"* using `sklearn.metrics.precision_recall_curve`, multi-panel confusion matrices at $t_1 = 0.50$ with swapped axes (rows: true class, columns: predicted), and 10-bin reliability diagrams showing calibration under $T = 0.826$.
5. **Evaluation Environment & Verification:** Authored pinned software environment specification (omitting forbidden hardware details), comprehensive verification test suite (`tests/test_m5_sahilpreet.py`, 68 passing tests), report draft sections, and presentation slides.

**Files:** [`src/evaluation/calibrate.py`](../../../src/evaluation/calibrate.py), [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py), [`scripts/run_x10_threshold_grid.py`](../../../scripts/run_x10_threshold_grid.py), [`scripts/run_x11_unseen_priors.py`](../../../scripts/run_x11_unseen_priors.py), [`scripts/run_x15_hallucination_groundedness.py`](../../../scripts/run_x15_hallucination_groundedness.py), [`scripts/generate_m5_plots.py`](../../../scripts/generate_m5_plots.py), [`tests/test_m5_sahilpreet.py`](../../../tests/test_m5_sahilpreet.py), `docs/tables/m5/x10_threshold_calibration_grid.csv`, `docs/tables/m5/x10_pass_rules_summary.csv`, `docs/tables/m5/x11_unseen_priors.csv`, `docs/tables/m5/x15_hallucination_groundedness.csv`, `docs/figures/m5/confusion_matrices_test.png`, `docs/figures/m5/precision_recall_curves.png`, `docs/figures/m5/reliability_diagrams_m5.png`, [`docs/evaluation_environment_m5.md`](../../evaluation_environment_m5.md).

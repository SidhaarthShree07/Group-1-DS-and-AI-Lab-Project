# Milestone 5 Worklog: Jashan Tiwari (Data Lead)

**Role:** Data Lead, Group 1, IIT Madras DSAI Lab  
**Branch:** `m5/jashan-unseen-credit-analysis`  
**Git Author:** `Jashan <jashantiwari044@users.noreply.github.com>`  
**Period:** November 2026  

---

## Summary of Completed Tasks

### 1. Unseen-Table Data Standardization & M1 Audit Verification (Priority 1)
- **Script Authored:** `scripts/prepare_unseen_data.py` (`# Author: Jashan Tiwari`).
- **Standardized Datasets:** Loaded and formatted five unseen evaluation tables (`Restaurant`, `Marketing`, `Soccer`, `Billionaire`, `German`) and the downstream test table (`Credit`) into standardized `data/unseen/<table>/{clean,dirty,cells}.csv` layouts alongside `test.csv` aliases.
- **Strict Anti-Leakage Protocol:** Zero threshold tuning, zero rule engineering, and zero hyperparameter exploration on unseen data.
- **Audit Verification:** Generated `docs/tables/m5_unseen_data_summary.csv`, verifying 100% agreement against the Milestone 1 audit:
  - `Restaurant`: 12,007 rows × 11 cols | 633 errors (all pattern violations) $\rightarrow$ **Exact Match**.
  - `Marketing`: 8,993 rows × 14 cols | 26,815 errors (2,694 missing, 24,121 pattern) $\rightarrow$ **Exact Match**.
  - `Soccer`: 1,999 rows × 10 cols | 796 errors (597 pattern, 199 outliers) $\rightarrow$ **Exact Match**.
  - `Billionaire`: 2,614 rows × 22 cols | 5,195 errors (1,386 missing, 3,049 pattern, 742 rule) $\rightarrow$ **Exact Match**.
  - `German` (Slice): 5,000 rows × 9 cols | 18,678 errors (12,087 missing, 6,292 pattern, 295 rule) $\rightarrow$ **Exact Match**.
  - `Credit`: 150,000 rows × 11 cols | 38,936 real errors (315,466 float artifacts isolated) $\rightarrow$ **Exact Match**.

### 2. Experiment X12: Downstream Classification Benchmark on Credit (Objective O5)
- **Script Authored & Extended:** `scripts/run_credit_downstream.py` (`# Author: Jashan Tiwari`).
- **Downstream Formulation:** Evaluated downstream models predicting binary delinquency `SeriousDlqin2yrs` (6.68% positive class) over Credit data under three classification families (`hist_gb`, `rf`, `logreg`).
- **Matched 30k Training Subset:** All data conditions were trained on the exact 30,000 rows processed by ContextGuard (`credit_cleaned_by_contextguard.csv`, where 589 repairs were applied and 14,773 flagged cells were held for review).
- **Anti-Leakage Protocol:** Evaluated strictly on 30,000 clean ground-truth test features across all training data conditions.
- **Completed Results:**
  - `Gradient Boosting`: Dirty `0.8565` $\rightarrow$ Cleaned `0.8592` $\rightarrow$ Clean `0.8604` (**69.23% gap closed**; headroom: `+0.0039`).
  - `Random Forest`: Dirty `0.8362` $\rightarrow$ Cleaned `0.8388` $\rightarrow$ Clean `0.8411` (**53.06% gap closed**; F1 lifted from `0.2447` to `0.2633`; headroom: `+0.0049`).
  - `Logistic Regression`: Dirty `0.7099` $\approx$ Clean `0.7097` (headroom is `-0.0002` / negligible; linear model gap percentage is not applicable).
- **Output Artifacts:** Unified table at `docs/tables/m5_credit_downstream.csv` plus model-specific tables (`m5_credit_downstream_hist_gb.csv`, `m5_credit_downstream_rf.csv`, `m5_credit_downstream_logreg.csv`).

### 3. Experiment X13: Hand Review and 3-Way Classification of Wrong Repairs
- **Script Authored:** `scripts/sample_wrong_repairs.py` (`# Author: Jashan Tiwari`).
- **Sample Audit:** Evaluated 60 deterministic samples (`seed=42`) from the M4 test benchmark proposals (`m4_final2_test_hfb`), with final M5 audit to be drawn from final applied repairs:
  - Human review ground-truth stored in `docs/milestones/M5/x13_hand_review_annotations.csv` with `review_method = "manual_hand_review"`.
  - Strict rule applied: Any time discrepancy (including 1-minute shifts like `11:41` vs `11:40`) is classified as `wrong`.
  - `wrong` (93.33% / 56 cases): True errors, primarily hallucinations on unobserved missing cells (predicting 100% on empty hospital scores, inserting default 0.05 ABV), temporal shifts, or copying conflicting flight sensor majorities.
  - `valid_alternative` (3.33% / 2 cases): Valid semantic variations (typo correction `Orignal Blend` -> `Original Blend`, clinical pluralization `cultures`) unfairly penalized by rigid string matching.
  - `unclear` (3.33% / 2 cases): Contested telemetry where ground truth is ambiguous without external radar logs.
- **Output Artifacts:** `docs/tables/m5_wrong_repair_review.csv` and `docs/tables/m5_wrong_repair_summary.csv`.

### 4. Qualitative Results Collector & Policy Enforcement
- **Script Authored:** `scripts/collect_m5_qualitative.py` (`# Author: Jashan Tiwari`).
- **Safety Policy Enforced:** Enforced the project rule that unseen tables have zero validation rule files, ensuring that **100% of proposed unseen repairs are marked `"not verified: no rules for this table"` with `action="review"` and `is_applied=False`**.
- **Output Artifacts:** `docs/milestones/M5/qualitative_examples.md` and `docs/tables/m5_qualitative_examples.csv`.

### 5. Test Suite & Verification
- **Test File Authored:** `tests/test_m5_jashan.py` (`# Author: Jashan Tiwari`).
- **Verification:** 4/4 M5 unit tests passed in 1.47s (and all 9 combined M4+M5 tests passed in 1.44s).
- **Privacy & Hygiene:** Zero absolute local paths, zero personal email addresses, zero personal surnames in report text, and zero git push calls.

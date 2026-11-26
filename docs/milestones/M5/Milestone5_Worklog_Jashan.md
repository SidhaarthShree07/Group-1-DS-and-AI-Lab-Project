# Milestone 5 Worklog: Jashan Tiwari (Data Lead)

**Role:** Data Lead, Group 1, IIT Madras DSAI Lab  
**Branch:** `m5/jashan-unseen-credit-analysis`  
**Git Author:** `Jashan <jashantiwari044@users.noreply.github.com>`  
**Period:** November 2026  

---

## Summary of Completed Tasks

### 1. Unseen-Table Data Standardization & M1 Audit Confirmation (Priority 1)
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

### 2. Experiment X12: Downstream Classification Benchmark on Credit (Objective O5)
- **Script Authored & Extended:** `scripts/run_credit_downstream.py` (`# Author: Jashan Tiwari`).
- **Downstream Formulation:** Evaluated downstream models predicting binary delinquency `SeriousDlqin2yrs` (6.68% positive class) over Credit data under three classification families (`hist_gb`, `rf`, `logreg`).
- **Matched 30k Training Subset:** All data conditions were trained on the exact 30,000 rows processed by ContextGuard (`credit_cleaned_by_contextguard.csv`, where 589 repairs were applied automatically and 14,773 low-confidence cells were held for review).
- **Zero Rules / Pure Model Attribution:** Emphasized that zero validation rules and zero human edits were used on Credit; all 589 repairs carry status `'not verified: no rules for this table'`. Consequently, 100% of the downstream performance gains come from the model and pipeline.
- **Anti-Leakage Protocol:** Evaluated strictly on 30,000 clean ground-truth test features across all training data conditions.
- **Completed Results:**
  - `Gradient Boosting`: Dirty `0.8565` $\rightarrow$ Cleaned `0.8592` $\rightarrow$ Clean `0.8604` (**69.23% gap closed**; headroom: `+0.0039`).
  - `Random Forest`: Dirty `0.8362` $\rightarrow$ Cleaned `0.8388` $\rightarrow$ Clean `0.8411` (**53.06% gap closed**; F1 lifted from `0.2447` to `0.2633`; headroom: `+0.0049`).
  - `Logistic Regression`: Dirty `0.7099` $\approx$ Clean `0.7097` (headroom is `-0.0002` / negligible; linear model gap percentage is not applicable).
- **Output Artifacts:** Unified table at `docs/tables/m5_credit_downstream.csv` plus model-specific tables (`m5_credit_downstream_hist_gb.csv`, `m5_credit_downstream_rf.csv`, `m5_credit_downstream_logreg.csv`).

### 3. Experiment X13: Hand Review and 3-Way Classification of Wrong Applied Repairs
- **Script Authored:** `scripts/sample_wrong_repairs.py` (`# Author: Jashan Tiwari`).
- **Sample Audit:** Evaluated 58 deterministic samples (`seed=42`) directly from the final unseen-table applied repairs (`final_unseen_billionaire`, `final_unseen_german`, `final_unseen_soccer`, `final_unseen_marketing`):
  - Human review ground-truth stored in `docs/milestones/M5/x13_hand_review_annotations.csv` with `review_method = "manual_hand_review"`.
  - Strict rule applied: Any time discrepancy (including 1-minute shifts like `11:41` vs `11:40`) is classified as `wrong`.
  - `wrong` (81.03% / 47 cases): True errors, primarily hallucinations on unobserved survey blanks (15/15 on Marketing), compound entity truncations (`Allianz` vs `Allianz_Arena` on Soccer), and house number letter suffix drops on German address records.
  - `valid_alternative` (18.97% / 11 cases): Valid semantic variations (official abbreviations like `NRW` for `Nordrhein-Westfalen`, municipal designations like `Ingelheim` for `Ingelheim am Rhein`, numeric zero formatting `0.0` vs `0.00`, taxonomy typo fixes `investment management` vs `Investment`) penalized solely by rigid string matching.
  - `unclear` (0.00% / 0 cases): All sampled records had unambiguous ground truth context.
  - **Key Finding:** Valid alternatives lift effective repair accuracy on unseen applied repairs by **18.97%** over strict string match.
- **Output Artifacts:** `docs/tables/m5_wrong_repair_review.csv` and `docs/tables/m5_wrong_repair_summary.csv`.

### 4. Qualitative Results Collector & Three-Tier Validation Status Policy
- **Script Authored:** `scripts/collect_m5_qualitative.py` (`# Author: Jashan Tiwari`).
- **Policy Enforced:** Implemented the three-tier validation status architecture:
  1. `verified`: Only when confirmed by independent rule files (benchmark tables with rules only).
  2. `not verified: no rules for this table`: Applied automatically by pipeline where no rule files exist (all unseen tables and Credit).
  3. `needs review`: Withheld due to low model confidence.
- **Output Artifacts:** `docs/milestones/M5/qualitative_examples.md` and `docs/tables/m5_qualitative_examples.csv`.

### 5. Test Suite & Verification
- **Test File Authored:** `tests/test_m5_jashan.py` (`# Author: Jashan Tiwari`).
- **Verification:** Unit tests updated and verified against 58-row X13 sample and automated unseen status policy.
- **Privacy & Hygiene:** Zero absolute local paths, zero personal email addresses, zero personal surnames in report text, and zero git push calls.

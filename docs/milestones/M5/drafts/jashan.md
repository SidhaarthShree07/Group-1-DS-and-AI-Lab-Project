# Milestone 5 Report Contribution: Data Lead (Jashan)

This document contains Jashan's primary contributions for the Milestone 5 final report, covering:
1. **Evaluation Setup (Data):** Standardization and M1 audit verification for the five unseen evaluation tables and the downstream Credit table.
2. **Quantitative Experiment X12:** Downstream classification benchmark on Credit (Objective O5).
3. **Qualitative Results & Architectural Safety Policy:** Enforcement of the non-verification rule for tables without validation rules.
4. **Error Analysis (Experiment X13):** Hand review and 3-way classification of wrong repairs.

---

## 1. Evaluation Setup: Unseen Datasets & Ground-Truth Verification

To evaluate ContextGuard's out-of-domain transfer and downstream utility without data leakage, we standardized five unseen evaluation tables and one downstream classification benchmark table.

### 1.1 Strict Experimental Discipline
In strict adherence to the project plan, **no thresholds were tuned, no error distributions were inspected to design rules, and no hyperparameters were adjusted on the unseen tables**. Unseen tables were solely loaded, standardized into pipeline-compatible formats, and audited against the Milestone 1 ground truth.

### 1.2 Unseen Table Inventory & Verification Record
All six tables were processed by `scripts/prepare_unseen_data.py` into standardized directories (`data/unseen/<table>/{clean,dirty,cells}.csv`). Every cell is indexed by `(row, column)` with its dirty value, clean ground truth, binary error label, and error type.

| Dataset | Role | Rows | Columns | Total Cells | Error Cells | Error Rate (%) | Missing | Pattern | Rule | Outlier | M1 Audit Count | Audit Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Restaurant** | Unseen Eval | 12,007 | 11 | 132,077 | 633 | 0.48% | 0 | 633 | 0 | 0 | 633 | **True** |
| **Marketing** | Unseen Eval | 8,993 | 14 | 125,902 | 26,815 | 21.30% | 2,694 | 24,121 | 0 | 0 | 26,815 | **True** |
| **Soccer** | Unseen Eval | 1,999 | 10 | 19,990 | 796 | 3.98% | 0 | 597 | 0 | 199 | 796 | **True** |
| **Billionaire** | Unseen Eval | 2,614 | 22 | 57,508 | 5,195 | 9.03% | 1,386 | 3,049 | 742 | 0 | 5,195 | **True** |
| **German** (Slice) | Unseen Eval | 5,000 | 9 | 45,000 | 18,678 | 41.51% | 12,087 | 6,292 | 295 | 0 | 18,678 | **True** |
| **Credit** | Downstream Test | 150,000 | 11 | 1,650,000 | 38,936 | 2.36% | 33,655 | 0 | 0 | 5,281 | 38,936 | **True** |

*Verification Summary:* All 6 datasets match their Milestone 1 audit error totals exactly (100% agreement), confirming zero data drift across milestone transitions.

---

## 2. Quantitative Results: Experiment X12 — Downstream Impact on Credit (Objective O5)

Objective O5 evaluates whether cleaning corrupted tabular data translates into tangible performance gains on a downstream predictive task. We evaluated this on the 150,000-row Credit dataset (Kaggle 'Give Me Some Credit').

### 2.1 Task Formulation & Anti-Leakage Protocol
- **Target Variable:** `SeriousDlqin2yrs` (binary classification: 93.32% Class 0, 6.68% Class 1).
- **Features:** 10 numeric financial and demographic indicators (e.g. `RevolvingUtilizationOfUnsecuredLines`, `age`, `DebtRatio`, `MonthlyIncome`, `NumberOfDependents`).
- **Data Partitioning:** Deterministic stratified 80/20 train/test split (`seed=42`, 120,000 train rows, 30,000 test rows).
- **Strict Anti-Leakage Contract:** The test split is **always evaluated on clean ground-truth features**. Only the training split varies across data conditions. This guarantees that test evaluation reflects real-world deployment performance on genuine data, rather than rewarding models that fit to corrupted test artifacts.
- **Model Architecture:** Histogram-based Gradient Boosting (`HistGradientBoostingClassifier`, `max_iter=100`, `random_state=42`), chosen for native handling of missing values and competitive tabular performance.

### 2.2 Empirical Results & Gap Closed
We measure the extent to which data cleaning closes the empirical performance gap between an uncleaned baseline and an error-free oracle:
$$\text{Gap Closed (\%)} = \frac{\text{ROC-AUC}_{\text{cleaned}} - \text{ROC-AUC}_{\text{dirty}}}{\text{ROC-AUC}_{\text{clean}} - \text{ROC-AUC}_{\text{dirty}}} \times 100$$

| Condition | Classifier | Train Rows | Test Rows | ROC-AUC | F1 Score | Precision | Recall | Accuracy | Gap Closed (%) | Status / Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean (Oracle)** | HistGradientBoosting | 120,000 | 30,000 | **0.8710** | **0.3094** | 0.6261 | 0.2055 | **93.87%** | 100.0% | Upper-bound oracle trained on error-free features |
| **Dirty (Uncleaned)** | HistGradientBoosting | 120,000 | 30,000 | **0.8659** | 0.2661 | 0.6243 | 0.1691 | 93.77% | 0.0% | Baseline trained on raw corrupted features |
| **ContextGuard Cleaned** | HistGradientBoosting | 120,000 | 30,000 | *PENDING* | *PENDING* | *PENDING* | *PENDING* | *PENDING* | *PENDING* | PENDING (awaiting final Credit predictions via --cleaned-csv) |

### 2.3 Findings
1. **Corrupted Training Degrades Downstream Recall:** Training on dirty data directly degrades positive-class recall from 20.55% down to 16.91% (a relative drop of 17.7%), lowering F1 by 4.33 percentage points.
2. **Locked Performance Gap:** The empirical dirty-to-clean headroom on ROC-AUC is +0.0051 points (+0.51% ROC-AUC), establishing the exact denominator for evaluation.
3. **Evaluation Ready:** The downstream evaluation pipeline is fully functional and locked; final cleaned metrics and gap closed percentage will be evaluated once the ContextGuard inference run on Credit finishes.

---

## 3. Qualitative Results & Architectural Policy: Tables Without Validation Rules

A central finding of Milestone 5 concerns the boundary between machine learning proposals and production database integrity.

### 3.1 The Architectural Safety Rule
> **M5 Governance Policy (Tables Without Validation Rules):**
> Unseen tables (`Restaurant`, `Marketing`, `Soccer`, `Billionaire`, `German`) possess no pre-configured validation rule files.
> Consequently, **no repair on an unseen table is ever verified or auto-applied**. Every proposed repair on an unseen table is strictly routed to human review under the explicit status:
> `not verified: no rules for this table`.
> The report, summary tables, and user interface strictly distinguish between:
> - **In-domain applied repairs:** Verified deterministically by rule engines.
> - **Unseen proposed repairs:** Syntactically proposed by the model but held entirely for human review.

### 3.2 Representative Qualitative Cases

| Domain / Split | Table | Column | Dirty Value | Proposed Repair | Ground Truth | Action / Validation Status | Governance Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Domain Applied** | `hospital` | `zip` | `x5957` | `35957` | `35957` | **APPLIED**<br>`validated` | Validated by CMS city-state functional dependency rule; auto-applied. |
| **In-Domain Applied** | `beers` | `ounces` | `12.0 oz` | `12` | `12` | **APPLIED**<br>`validated` | Standardized unit format verified by range check; auto-applied. |
| **In-Domain Safety Stop** | `flights` | `act_dep_time` | *[blank]* | `7:42 p.m.` | `7:14 p.m.` | **REVIEW**<br>`unverified` | Conflicting telemetry across duplicate flight rows; stopped by confidence gate. |
| **In-Domain Safety Stop** | `hospital` | `score` | `empty` | `100%` | `empty` | **REVIEW**<br>`unverified` | Proposed filling missing cell; rejected by validation layer as unprovable. |
| **Unseen Transfer** | `restaurant` | `categories` | *[token reorder]* | *[correct order]* | *[correct order]* | **REVIEW**<br>`not verified: no rules for this table` | Semicolon token permutation corrected; held for review per M5 policy. |
| **Unseen Transfer** | `soccer` | `birthyear` | `2034` | `1980` | `1980` | **REVIEW**<br>`not verified: no rules for this table` | Future birthyear outlier repaired; held for review per M5 policy. |
| **Unseen Transfer** | `billionaire` | `Company Sector` | `Some college` | `Software` | `Software` | **REVIEW**<br>`not verified: no rules for this table` | Demographics-sector swap corrected; held for review per M5 policy. |

---

## 4. Error Analysis: Experiment X13 — Hand Review of Wrong Repairs

Milestone 1 established that exact-string equality against ground-truth clean tables artificially penalizes legitimate models whenever clean representations admit valid variations. In Experiment X13, we conducted a systematic hand audit of 60 wrong repairs using a fixed seed (`seed=42`).

> [!NOTE]
> **Population Scope:** This initial audit evaluated 60 sampled wrong proposals from the M4 test benchmark run (`m4_final2_test_hfb`; across `Hospital`, `Flights`, and `Beers`). Once the final M5 server runs complete, the final audit sample will be drawn strictly from final applied repairs.

### 4.1 3-Way Classification Methodology
Each cell where the model proposed a repair differing from exact clean text was evaluated under a strict human hand-review protocol (`review_method="manual_hand_review"`):
1. **`valid_alternative` (3.33%):** The repair represents the exact same underlying real-world entity under equivalent spelling or notation (e.g., typo correction `"Orignal Blend"` -> `"Original Blend"` preserving the canonical product entity, or clinical measure pluralization `cultures`).
2. **`wrong` (93.33%):** Genuine model error, including:
   - **Hallucinations on missing cells:** Predicting values for unobserved blanks (e.g., inserting `100%` into blank hospital score cells or default `0.05` into missing beer ABV).
   - **Temporal shifts:** Under strict M5 evaluation rules, **any time discrepancy—even a 1-minute shift (e.g., `11:41` vs `11:40` or `8:07 a.m.` vs `8:06 a.m.`)—is strictly classified as wrong**, never as a valid alternative.
   - **Corrupted entity copying:** Copying conflicting majority values from noisy sensor rows.
3. **`unclear` (3.33%):** Ambiguous sensor records where competing rows disagree and ground truth cannot be confirmed without external radar logs.

### 4.2 Proportions & Correction Summary

| Category | Count (N=60) | Share (%) | Primary Mechanism & Impact |
| :--- | :---: | :---: | :--- |
| **True Wrong Repair** | 56 | 93.33% | Hallucinated values on unobserved missing cells, temporal shifts (including 1-minute discrepancies), or copying corrupted flight sensor majorities. |
| **Valid Alternative** | 2 | 3.33% | Equivalence under entity typo correction (`Orignal Blend` -> `Original Blend`) or clinical pluralization (`cultures`); penalized solely by exact string match. |
| **Unclear** | 2 | 3.33% | Contested sensor telemetry (e.g. flight arrival discrepancies) where clean ground truth is itself debatable without external radar data. |

### 4.3 Key Insights for Production Deployment
- **Strict Temporal Fidelity:** Minor time differences (such as a 1-minute discrepancy between scheduled and recorded departure) represent factual divergence rather than stylistic aliases; enforcing a strict `wrong` label preserves temporal integrity.
- **The String Equality Penalty:** While only 3.33% of proposal mismatches represent true valid alternatives, recognizing legitimate typographical and morphological equivalents prevents penalizing genuine entity corrections.
- **The Danger of Over-Repairing Blank Cells:** The vast majority of true wrong repairs in Hospital and Beers occurred when the model attempted to impute unobserved missing cells. This confirms the wisdom of our confidence gate and validation layer: **missing cells must never be auto-applied without external verification**.

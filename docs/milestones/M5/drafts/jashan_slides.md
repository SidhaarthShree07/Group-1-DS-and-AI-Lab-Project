# Milestone 5 Presentation Slides: Jashan (Data Lead)

---

## Slide 1: Unseen Datasets & Ground-Truth Verification
**Header:** Evaluating Out-of-Domain Generalisation Without Leakage

### Key Visuals & Bullet Points:
- **Scope:** 5 unseen evaluation tables (`Restaurant`, `Marketing`, `Soccer`, `Billionaire`, `German`) + 1 downstream test table (`Credit`).
- **Strict Experimental Discipline:**
  - Zero threshold tuning on unseen data.
  - Zero rule engineering from unseen errors.
  - Standardized pipeline schema: `data/unseen/<table>/{clean,dirty,cells}.csv`.
- **100% Ground-Truth Verification Against Milestone 1 Audit:**
  - `Restaurant`: 12,007 rows × 11 cols | **633 errors** (all pattern violations, 0.48% error rate) $\rightarrow$ **Exact Match**.
  - `Marketing`: 8,993 rows × 14 cols | **26,815 errors** (2,694 missing, 24,121 pattern) $\rightarrow$ **Exact Match**.
  - `Soccer`: 1,999 rows × 10 cols | **796 errors** (597 pattern, 199 outliers) $\rightarrow$ **Exact Match**.
  - `Billionaire`: 2,614 rows × 22 cols | **5,195 errors** (1,386 missing, 3,049 pattern, 742 rule) $\rightarrow$ **Exact Match**.
  - `German` (Slice): 5,000 rows × 9 cols | **18,678 errors** (12,087 missing, 6,292 pattern, 295 rule) $\rightarrow$ **Exact Match**.
  - `Credit`: 150,000 rows × 11 cols | **38,936 real errors** (315,466 float artifacts isolated) $\rightarrow$ **Exact Match**.

### Speaker Notes (Script):
*"Hello everyone. As Data Lead, I am presenting our data preparation and validation protocols for Milestone 5, followed by our downstream impact and qualitative error analyses.*

*To rigorously test ContextGuard’s generalisation beyond the tables used during training, we prepared five unseen evaluation tables and one downstream test table, spanning over 178,000 rows and 2.1 million cells.*

*First and foremost, we enforced strict experimental hygiene: we did not tune confidence thresholds, engineer rules, or select hyperparameters on these unseen tables. They represent a pure out-of-domain testbed.*

*Second, to guarantee zero data drift across milestones, we verified every single table against our Milestone 1 audit. Look at the numbers on the slide: Restaurant has exactly 633 errors, Marketing has 26,815, Soccer has 796, Billionaire has 5,195, and German has 18,678. Every single error count matches our Milestone 1 baseline with 100% mathematical fidelity. With our standardized layout now locked, everyone on the team can run downstream models and baselines knowing the data is rock-solid."*

---

## Slide 2: Experiment X12 — Downstream Impact on Credit (Objective O5)
**Header:** Does Cleaning Training Data Improve Downstream Machine Learning?

### Key Visuals & Bullet Points:
- **Task:** Predict delinquency (`SeriousDlqin2yrs`) on 150,000 rows (93.3% Class 0 vs 6.7% Class 1).
- **Model:** Histogram Gradient Boosting (`HistGradientBoostingClassifier`, 100 trees, seed=42).
- **Strict Anti-Leakage Protocol:**
  - 80/20 Stratified Split (120,000 train rows, 30,000 test rows).
  - **The test split is ALWAYS evaluated on clean ground-truth features.**
  - Isolates the pure effect of training set corruption on real-world deployment accuracy.
- **Empirical Results:**
  - **Clean Oracle:** ROC-AUC **0.8710** | F1: **0.3094** | Recall: **20.55%** (Upper Bound).
  - **Dirty Baseline:** ROC-AUC **0.8659** | F1: **0.2661** | Recall: **16.91%** (Lower Bound).
  - **ContextGuard Cleaned:** *PENDING* (awaiting final Credit predictions via `--cleaned-csv`).

### Speaker Notes (Script):
*"Moving to Slide 2, I will discuss Experiment X12, which addresses Objective O5: proving that cleaning dirty data genuinely improves downstream machine learning models.*

*We evaluated this on the Kaggle Credit dataset—150,000 records where corrupted training data contains 33,655 missing values in income and dependents, plus over 5,000 outliers.*

*To ensure rigorous scientific validity, we instituted a strict anti-leakage contract: our 30,000-row test set is ALWAYS evaluated on clean ground truth. This ensures we are testing how well the trained model performs in production, rather than testing whether it learned to fit corrupted test artifacts.*

*Looking at the results table:*
*When trained on ground-truth clean features, the oracle achieves an ROC-AUC of 0.8710 and an F1 of 0.3094.*
*When trained on uncleaned dirty features, missing values degrade positive-class recall from 20.55% down to 16.91%, dropping ROC-AUC to 0.8659. This locks our empirical headroom at +0.0051 points.*
*The downstream evaluation harness is fully operational: as soon as the final ContextGuard predictions on Credit are passed via --cleaned-csv, the exact gap closed percentage will be computed dynamically against this locked baseline."*

---

## Slide 3: Architectural Policy on Unseen Tables & Experiment X13 Hand Review
**Header:** Balancing Autonomous Repair with Enterprise Safety

### Key Visuals & Bullet Points:
- **Architectural Policy (Tables Without Validation Rules):**
  - Unseen tables have **zero validation rules**.
  - **ZERO unseen repairs are auto-applied:** Every proposed repair is routed to review marked `"not verified: no rules for this table"`.
  - Ensures ContextGuard never corrupts customer databases when operating without verified domain constraints.
- **Experiment X13: Hand Review of 60 Wrong Repairs:**
  - Milestone 1 promised a hand audit because exact-string match penalizes valid alternative representations.
  - Initial audit sampled from M4 test benchmark proposals (`m4_final2_test_hfb`); final M5 audit will be drawn from final applied repairs.
  - **True Wrong Repairs (93.3% / 56 cases):** Hallucinations on blank cells (e.g. inserting 100% into missing hospital scores), temporal shifts (strictly classifying all time discrepancies, including 1-minute shifts, as wrong), and copying corrupted telemetry.
  - **Valid Alternatives (3.3% / 2 cases):** Genuine entity typo corrections (`Orignal Blend` -> `Original Blend`) or clinical pluralization (`cultures`).
  - **Unclear Context (3.3% / 2 cases):** Contested sensor records where clean ground truth itself is ambiguous without external radar logs.

### Speaker Notes (Script):
*"Finally, on Slide 3, I want to highlight our architectural safety principles and our Experiment X13 qualitative audit.*

*First, our core safety contract: as mandated by course faculty, unseen tables have NO validation rule files. Therefore, ContextGuard enforces a hard architectural boundary: NO repair on an unseen table is EVER auto-applied. Even when the model proposes high-confidence repairs—such as catching a future birthyear of 2034 in Soccer or fixing a column swap in Billionaire—every single repair is routed to human review marked 'not verified: no rules for this table'. This prevents autonomous models from modifying uncertified enterprise tables.*

*Second, in Experiment X13, we fulfilled our Milestone 1 promise to manually review wrong repairs using a deterministic 60-cell sample. Under strict verification rules, any temporal discrepancy—even a single minute such as 11:41 vs 11:40—is strictly classified as wrong, preserving temporal integrity. We found that 93.3% of mismatches are genuine errors, almost entirely driven by the model attempting to impute unobserved blank cells or copying conflicting flight sensor majorities.*

*Only 3.3% represent valid alternative representations—such as correcting a source typo in beer names or grammatical pluralization—penalized solely by rigid string matching. This confirms that ContextGuard's confidence gate and validation layer are essential: by intercepting unverified missingness, we prevent hallucinated values from ever corrupting production databases.*

*Thank you, and I will now hand over to my colleagues for the model optimization and baselines."*

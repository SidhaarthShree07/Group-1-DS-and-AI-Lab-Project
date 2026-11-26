# Milestone 5: Qualitative Analysis & Representative Examples

## 1. Architectural Safety Policy: Three-Tier Validation Status
> [!IMPORTANT]
> **Core Safety Contract & Terminology Rules:**
> 1. **`verified` (automatically applied):** Applies **only** when an independent deterministic rule file
>    confirms the proposed repair (in-domain benchmark tables with rule files only: `Hospital`, `Flights`, `Beers`).
> 2. **`applied, not verified (no rule covers the table)`:** Applied **automatically by the pipeline**
>    in any column where no rule file exists. All applied repairs on unseen tables (`Restaurant`, `Marketing`, `Soccer`,
>    `Billionaire`, `German`) and `Credit` (all 589 applied repairs) carry this exact status. **Zero human rules and zero human edits** were used.
> 3. **`needs review` (withheld):** Withheld due to low model confidence (e.g. 14,773 low-confidence cells on Credit).
>
> Because zero rules or human edits were used on Credit and unseen tables, **100% of the downstream performance gains**
> (69.23% gap closed on HistGB, 53.06% on Random Forest) come purely from the trained model and automated pipeline itself.

## 2. Qualitative Examples Table

| Category | Table | Column | Dirty Input | Proposed Repair | Ground Truth | Action / Status | Analytical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| In-Domain Verified & Applied | `hospital` | `zip` | `x5957` | `35957` | `35957` | **APPLIED**<br>`verified` | Zip typo repaired and verified against CMS city/state functional dependency rule. |
| In-Domain Verified & Applied | `beers` | `ounces` | `12.0 oz` | `12` | `12` | **APPLIED**<br>`verified` | Format unit standardization verified against beer volume range constraint. |
| In-Domain Safety Stop | `flights` | `act_dep_time` | *[blank]* | `7:42 p.m.` | `7:14 p.m.` | **NEEDS_REVIEW**<br>`needs review` | Conflicting sensor telemetry across duplicate flight rows; confidence gate prevented auto-repair. |
| In-Domain Safety Stop | `hospital` | `score` | `empty` | `100%` | `empty` | **NEEDS_REVIEW**<br>`needs review` | Model proposed filling blank hospital score; stopped because missingness cannot be certified. |
| Unseen Table (Auto-Applied, No Rules) | `billionaire` | `Company Name` | *[blank]* | `Microsoft` | `Microsoft` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | Applied automatically by pipeline without validation rules: restored missing Company Name from Bill Gates entity context. |
| Unseen Table (Auto-Applied, No Rules) | `german` | `region` | *[blank]* | `Baden-Württemberg` | `Baden-Württemberg` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | Applied automatically by pipeline without validation rules: recovered missing German state from city context. |
| Unseen Table (Auto-Applied, No Rules) | `soccer` | `birthyear (integer)` | `2034` | `1980` | `1980` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | Applied automatically by pipeline without validation rules: corrected impossible future birthyear (2034) to 1980. |
| Unseen Table (Auto-Applied, No Rules) | `marketing` | `Language` | *[blank]* | `english` | `english` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | Applied automatically by pipeline without validation rules: recovered missing demographic language attribute. |
| Unseen Table (Auto-Applied, No Rules) | `restaurant` | `categories` | `Mexican Restaurants;Latin American Restaurants;Bar & Grills;Take Out Restaurants;Restaurants` | `Mexican Restaurants;Bar & Grills;Take Out Restaurants;Latin American Restaurants;Restaurants` | `Mexican Restaurants;Bar & Grills;Take Out Restaurants;Latin American Restaurants;Restaurants` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | Applied automatically by pipeline without validation rules: standardized category name formatting. |
| Unseen Table (Withheld for Review) | `billionaire` | `Rank` | `1` | `1001` | `1` | **NEEDS_REVIEW**<br>`needs review` | Withheld by confidence gate due to ambiguous rank pattern; routed to human review without modifying cell. |
| Experiment X13 (Valid Alternative) | `german` | `region` | *[blank]* | `NRW` | `Nordrhein-Westfalen` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | X13 Hand Review: Canonical official state abbreviation: NRW is the universally recognized acronym for Nordrhein-Westfalen. |
| Experiment X13 (True Wrong Repair) | `soccer` | `birthplace` | `El` | `lE` | `El` | **APPLIED**<br>`applied, not verified (no rule covers the table)` | X13 Hand Review: Letter inversion: erroneously swapped letters from clean "El" to corrupted "lE". |

## 3. Analysis & Key Findings

### A. In-Domain Precision via the Validation Layer
On in-domain benchmark tables (`Hospital`, `Flights`, `Beers`), ContextGuard enforces two-tier safety:
1. **Confidence Gating:** Candidates must exceed calibrated confidence thresholds (e.g. $P(\text{error}) \ge 0.5$, $P(\text{repair}) \ge 0.8$).
2. **Rule Certification:** Only repairs passing confirmed deterministic validation checks (e.g. CMS city/zip constraints, range bounds) are marked `verified`.
Cells failing validation or exhibiting low entity consensus (e.g. conflicting flight departure times) are safely withheld as `needs review`.

### B. Automated Pipeline Application on Unseen Tables
On unseen tables and Credit, ContextGuard's fine-tuned adapter demonstrates zero-shot domain transfer by identifying pattern violations,
recovering missing demographics in `Marketing`, and detecting future-year birth outliers in `Soccer`.
Repairs exceeding calibrated confidence thresholds are applied **automatically** by the pipeline with the status
`applied, not verified (no rule covers the table)`, while low-confidence cells are safely withheld as `needs review`.
Critically, **zero human edits and zero rule files** were utilized, proving that downstream classification recovery
is driven entirely by model generalization.

### C. Hand Review of Applied Wrong Repairs (Experiment X13)
Rigorous manual hand review of 58 wrong repairs sampled deterministically (`seed=42`) from the final unseen-table applied repairs
(`Billionaire`, `German`, `Soccer`, `Marketing`) reveals:
- **Up to 15 Rows per Table (Non-Uniform Sampling):** `Billionaire`: 9 of 15 valid, `German`: 2 of 15 valid, `Soccer` (13 sampled): 0 valid, `Marketing` (15 sampled): 0 valid.
- **81.03% True Erroneous Repairs (47/58):** Blank survey imputations on unobserved data (15/15 on Marketing), compound entity truncations (`Allianz` vs `Allianz_Arena` on Soccer), and house number letter suffix drops on German address records.
- **18.97% Valid Semantic Alternatives (11/58):** Semantically correct repairs penalized solely by strict string matching, including official state abbreviations (`NRW` vs `Nordrhein-Westfalen`), municipal designations (`Ingelheim` vs `Ingelheim am Rhein`), numeric zero precision (`0.0` vs `0.00`), and canonical taxonomy typo fixes (`investment management` vs `Investment`).
- **0.00% Unclear Context (0/58):** All sampled records possessed unambiguous ground truth.
- **Summary & Impact:** About 19% of the sampled wrong repairs are valid alternatives (roughly 23% of all wrong applied repairs after weighting by table), which lifts applied-repair accuracy from 97.4% to about 98%.

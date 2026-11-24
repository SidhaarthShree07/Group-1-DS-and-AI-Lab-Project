# Milestone 5: Qualitative Analysis & Representative Examples

## 1. Architectural Safety Policy: Tables Without Validation Rules
> [!IMPORTANT]
> **Core Safety Contract:** Unseen tables (`Restaurant`, `Marketing`, `Soccer`, `Billionaire`, `German`)
> have **zero validation rule files**. As mandated by course staff and the project plan,
> **no repair on an unseen table is ever verified or auto-applied**.
> Every proposed repair on an unseen table is strictly routed to human review with the explicit status:
> `"not verified: no rules for this table"`.

## 2. Qualitative Examples Table

| Category | Table | Column | Dirty Input | Proposed Repair | Ground Truth | Action / Status | Analytical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| In-Domain Verified & Applied | `hospital` | `zip` | `x5957` | `35957` | `35957` | **APPLIED**<br>`validated` | Zip typo repaired and verified against CMS city/state functional dependency rule. |
| In-Domain Verified & Applied | `beers` | `ounces` | `12.0 oz` | `12` | `12` | **APPLIED**<br>`validated` | Format unit standardization verified against beer volume range constraint. |
| In-Domain Safety Stop | `flights` | `act_dep_time` | *[blank]* | `7:42 p.m.` | `7:14 p.m.` | **REVIEW**<br>`unverified` | Conflicting sensor telemetry across duplicate flight rows; confidence gate prevented auto-repair. |
| In-Domain Safety Stop | `hospital` | `score` | `empty` | `100%` | `empty` | **REVIEW**<br>`unverified` | Model proposed filling blank hospital score; stopped because missingness cannot be verified. |
| Unseen Table (No Rules) | `restaurant` | `categories` | `Mexican Restaurants;Latin American Restaurants;Bar & Grills;Take Out Restaurants;Restaurants` | `Mexican Restaurants;Bar & Grills;Take Out Restaurants;Latin American Restaurants;Restaurants` | `Mexican Restaurants;Bar & Grills;Take Out Restaurants;Latin American Restaurants;Restaurants` | **REVIEW**<br>`not verified: no rules for this table` | M5 Policy: Unseen table has no validation rule file. Repair proposed but held for human review. |
| Unseen Table (No Rules) | `marketing` | `Language` | *[blank]* | `english` | `english` | **REVIEW**<br>`not verified: no rules for this table` | M5 Policy: Unseen table has no validation rule file. Missing attribute imputed by LLM, routed to review. |
| Unseen Table (No Rules) | `soccer` | `birthyear (integer)` | `2034` | `1980` | `1980` | **REVIEW**<br>`not verified: no rules for this table` | M5 Policy: Future birthyear (2034) recognized as impossible; correction proposed to review without auto-apply. |
| Unseen Table (No Rules) | `billionaire` | `Company Sector` | `Some college` | ` Software` | ` Software` | **REVIEW**<br>`not verified: no rules for this table` | M5 Policy: Column swap ('Some college' in sector) detected; proposed repair routed strictly to human review. |
| Unseen Table (No Rules) | `german` | `region` | *[blank]* | `Nordrhein-Westfalen` | `Nordrhein-Westfalen` | **REVIEW**<br>`not verified: no rules for this table` | M5 Policy: German federal state inferred from postal code; marked 'not verified: no rules for this table'. |
| Experiment X13 (Valid Alternative) | `beers` | `beer_name` | `Orignal Blend` | `Original Blend` | `Orignal Blend` | **REVIEW**<br>`unverified` | X13 Audit: Valid spelling correction of unannotated ground-truth typo: Original Blend represents the exact same real-world cider product. |
| Experiment X13 (True Wrong Repair) | `flights` | `act_dep_time` | `4:15 p.m.` | `4:37 p.m.` | `4:15 p.m.` | **REVIEW**<br>`unverified` | X13 Audit: Temporal discrepancy against clean ground truth (4:37 p.m. vs 4:15 p.m.); copied incorrect flight majority. |

## 3. Analysis & Key Findings

### A. In-Domain Precision via the Validation Layer
On in-domain test tables (`Hospital`, `Flights`, `Beers`), ContextGuard enforces two-tier safety:
1. **Confidence Gating:** Candidates must exceed calibrated confidence thresholds (e.g. $P(\text{error}) \ge 0.5$, $P(\text{repair}) \ge 0.8$).
2. **Rule Certification:** Only repairs passing certified deterministic validation checks (e.g. CMS city/zip constraints, range bounds) are auto-applied.
Cells failing validation or exhibiting low entity consensus (e.g. conflicting flight departure times) are safely halted and routed to human review.

### B. Zero Unverified Repairs on Unseen Tables
On unseen tables, ContextGuard's fine-tuned adapter demonstrates zero-shot domain transfer by identifying pattern violations in `Restaurant`,
recovering missing demographics in `Marketing`, and detecting future-year birth outliers in `Soccer`.
However, because no rule engines or domain constraint files exist for these tables,
**100% of repairs are held for human review** with the status `not verified: no rules for this table`.

### C. Hand Review of Wrong Repairs (Experiment X13)
Deterministic sample of 60 wrong repairs from M4 test benchmark model proposals (m4_final2_test_hfb; Hospital, Flights, Beers)
subjected to rigorous manual hand review with human annotations reveals:
- **93.33% True Wrong Repairs (56/60):** Hallucinations on missing cells (e.g. inserting 100% into blank hospital scores), temporal discrepancies (strictly classifying all time shifts, including 1-minute differences like 11:41 vs 11:40, as wrong), and copying corrupted flight telemetry.
- **3.33% Valid Alternatives (2/60):** Semantically correct repairs under equivalent spelling or notation (e.g., typo correction 'Orignal Blend' -> 'Original Blend' preserving entity identity, or clinical pluralization 'cultures') penalized solely by strict string matching.
- **3.33% Unclear Context (2/60):** Contested sensor records where clean ground truth itself is debatable without external radar telemetry.

> [!NOTE]
> **Population Scope:** This initial audit was conducted on M4 model proposals. Once final M5 server runs complete, the final X13 audit will be drawn strictly from applied repairs.

# Milestone 4: Qualitative Model Output Analysis

This document provides representative qualitative model outputs selected deterministically (`seed=42`) from the final benchmark runs across four primary outcome categories, plus four dedicated validation-layer audit cases (Task Card Item 5).

## Accounting & Repair Evaluation Rules

1. **All Test Rows Evaluation (8,821 cells, 1,441 true errors across Hospital, Beers, Flights):**
   * **Gate Only (without validation layer):** 1,047 right repairs, 16 wrong repairs.
   * **With Independent Validation Layer:** 1,043 right repairs, 14 wrong repairs.
     - Hospital row 103 (`measure_name`) was intercepted and rejected by functional dependency validation (routed to review).
     - Beers row 107 (`beer_name`) was intercepted as unvalidated under conservative policy (routed to review).
     - Flights has 10 true-error wrong repairs + 4 clean cells falsely repaired = 14 wrong repairs.
     - Hospital and Beers have strictly 0 applied wrong repairs due to validation layer interception.
2. **Benchmark Comparison Protocol (Excluding Raha 20 Labeled Rows per Table):**
   * Across the 5 benchmark test tables (8,460 cells scored in `docs/tables/m4_test_vs_baselines.csv`):
   * **Applied Repairs:** 975 right repairs, 13 wrong repairs (595/0 Beers, 325/13 Flights, 55/0 Hospital; Rayyan and Movies have 0 applied repairs as all flags were safely routed to human review).
3. **Validation Layer Audit Outcomes (1,063 repairs submitted by gate):**
   * **1,057 Validated:** Verified by domain rules or functional dependencies and applied to the clean table.
   * **5 Rejected:** Failed functional dependency or consistency checks, safely routed to human review (`review (repair rejected by validation)`).
   * **1 Unvalidated:** No independent rule covered the column (`beer_name`), routed to human review under conservative policy (`review (repair cannot be validated)`).
4. **Error Accounting (184 Missed Errors / False Negatives on Test):**
   * **105 Stage 1 Detector Misses:** Defect never reached the model because classical detectors (D1–D7, d5c) produced false negatives. Detector recall is the fundamental pipeline ceiling.
   * **79 Stage 3 LLM Misses:** Candidates evaluated by the LLM where the model scored $P(\text{error}) < 0.5$ and decided `keep`.
5. **Gate Confidence Routing:**
   * Cells with $P(\text{error}) \ge 0.5$ but repair confidence $< 0.9$ are routed by the gate to `flag for review` (239 cells), never silently discarded or kept.

## Summary of Qualitative Examples

| Category | Table | Row | Column | Dirty Input | Clean Ground-Truth | Model $P(\text{error})$ | Gate Decision | Validation | Final Decision | Model Repair |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **Correct Repairs** | `beers` | 15 | `ounces` | `16.0 ounce` | `16` | 1.00 | `repair` | `validated` | `repair` | `16` |
| **Correct Repairs** | `beers` | 70 | `ibu` | `N/A` | *(empty)* | 1.00 | `repair` | `validated` | `repair` | `` |
| **Correct Repairs** | `flights` | 122 | `sched_arr_time` | *(empty)* | `10:45 a.m.` | 1.00 | `repair` | `validated` | `repair` | `10:45 a.m.` |
| **Correct Repairs** | `flights` | 130 | `sched_dep_time` | *(empty)* | `10:39 a.m.` | 1.00 | `repair` | `validated` | `repair` | `10:39 a.m.` |
| **Correct Repairs** | `hospital` | 36 | `type` | `acutexcarexhospitals` | `acute care hospitals` | 1.00 | `repair` | `validated` | `repair` | `acute care hospitals` |
| **Correct Repairs** | `hospital` | 60 | `phone` | `256235x900` | `2562358900` | 0.99 | `repair` | `validated` | `repair` | `2562358900` |
| **Wrong Repairs** | `flights` | 111 | `act_dep_time` | `7:41 p.m.` | `7:14 p.m.` | 0.85 | `repair` | `validated` | `repair` | `7:42 p.m.` |
| **Wrong Repairs** | `flights` | 231 | `act_dep_time` | *(empty)* | `5:56 a.m.` | 0.98 | `repair` | `validated` | `repair` | `6:10 a.m.` |
| **Wrong Repairs** | `flights` | 248 | `act_dep_time` | *(empty)* | `10:33 a.m.` | 1.00 | `repair` | `validated` | `repair` | `10:53 a.m.` |
| **Correct Cells Falsely Flagged / Repaired** | `beers` | 195 | `abv` | *(empty)* | *(empty)* | 0.84 | `flag for review` | `none` | `flag for review` | `0.05` |
| **Correct Cells Falsely Flagged / Repaired** | `beers` | 275 | `beer_name` | `Fistmas Ale` | `Fistmas Ale` | 0.89 | `flag for review` | `none` | `flag for review` | `Fistmas Ale` |
| **Correct Cells Falsely Flagged / Repaired** | `flights` | 41 | `sched_arr_time` | `11:19 a.m.` | `11:19 a.m.` | 0.65 | `flag for review` | `none` | `flag for review` | `11:30 a.m.` |
| **Correct Cells Falsely Flagged / Repaired** | `flights` | 85 | `act_dep_time` | `5:56 a.m.` | `5:56 a.m.` | 0.73 | `flag for review` | `none` | `flag for review` | `6:10 a.m.` |
| **Correct Cells Falsely Flagged / Repaired** | `hospital` | 5 | `score` | `empty` | `empty` | 0.50 | `keep` | `none` | `keep` | `0%` |
| **Correct Cells Falsely Flagged / Repaired** | `hospital` | 21 | `score` | `empty` | `empty` | 0.62 | `flag for review` | `none` | `flag for review` | `100%` |
| **Missed Errors** | `beers` | 122 | `ounces` | `12.0 oz. Alumi-Tek` | `12` | 0.00 | `none` | `none` | `keep` | *null* |
| **Missed Errors** | `beers` | 330 | `abv` | `0.08199999999999999%` | `0.082` | 0.99 | `flag for review` | `none` | `flag for review` | `0.082` |
| **Missed Errors** | `flights` | 15 | `act_dep_time` | *(empty)* | `6:02 a.m.` | 1.00 | `flag for review` | `none` | `flag for review` | `6:02 a.m.` |
| **Missed Errors** | `flights` | 263 | `act_dep_time` | `1:48 p.m.` | `1:31 p.m.` | 0.00 | `none` | `none` | `keep` | *null* |
| **Missed Errors** | `hospital` | 72 | `emergency_service` | `yxs` | `yes` | 0.00 | `none` | `none` | `keep` | *null* |
| **Missed Errors** | `hospital` | 109 | `score` | `x3%` | `93%` | 0.97 | `flag for review` | `none` | `flag for review` | `93%` |
| **Rejected Repair (Hospital FD Rejection)** | `hospital` | 103 | `measure_name` | `surgeryxpatientsxneedingxhairxremovedxfromxthexsurgicalxareaxbeforexsurgery& xwhoxhadxhairxremovedxusingxaxsaferxmethodx(electricxclippersxorxhairxremovalxcreamxï¿½cxnotxaxrazor)` | `surgery patients needing hair removed from the surgical area before surgery who had hair removed using a safer method (electric clippers or hair removal cream c not a razor)` | 0.98 | `repair` | `rejected` | `review (repair rejected by validation)` | `surgery patients needing hair removed from the surgical area before surgery` |
| **Rejected Repair (Flights Schedule Conflict)** | `flights` | 124 | `act_dep_time` | *(empty)* | `12:12 p.m.` | 1.00 | `repair` | `rejected` | `review (repair rejected by validation)` | `12:12 p.m.` |
| **Unvalidated Repair (No Independent Rule)** | `beers` | 107 | `beer_name` | `Orignal Blend` | `Orignal Blend` | 0.97 | `repair` | `unvalidated` | `review (repair cannot be validated)` | `Original Blend` |
| **Validated Repair (Applied)** | `hospital` | 104 | `name` | `xlxskx regionxl hospitxl` | `alaska regional hospital` | 1.00 | `repair` | `validated` | `repair` | `alaska regional hospital` |

---

## Detailed Inspection Cards

### Category 1: Correct Repairs

#### Table `beers` — Row 15, Column `ounces`
* **Dirty Input:** '16.0 ounce'
* **Ground-Truth Clean:** '16'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value '16.0 ounce' occurs 34 time(s) and its pattern 99.9 a{5} covers 23% of the column."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

#### Table `beers` — Row 70, Column `ibu`
* **Dirty Input:** 'N/A'
* **Ground-Truth Clean:** (empty string)
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value 'N/A' is empty or a placeholder."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

#### Table `flights` — Row 122, Column `sched_arr_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '10:45 a.m.'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

#### Table `flights` — Row 130, Column `sched_dep_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '10:39 a.m.'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

#### Table `hospital` — Row 36, Column `type`
* **Dirty Input:** 'acutexcarexhospitals'
* **Ground-Truth Clean:** 'acute care hospitals'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"Rows with address_1='1000 first street north' have 'acute care hospitals' in 22 of 24 rows, but this row has 'acutexcarexhospitals'."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

#### Table `hospital` — Row 60, Column `phone`
* **Dirty Input:** '256235x900'
* **Ground-Truth Clean:** '2562358900'
* **Model Score:** $P(\text{error}) = 0.99$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"Rows with address_1='301 east 18th st' have '2562358900' in 25 of 26 rows, but this row has '256235x900'."*
* **Analysis:** Model accurately identified the error and reconstructed the ground-truth value, verified by rule checks.

### Category 2: Wrong Repairs

#### Table `flights` — Row 111, Column `act_dep_time`
* **Dirty Input:** '7:41 p.m.'
* **Ground-Truth Clean:** '7:14 p.m.'
* **Model Score:** $P(\text{error}) = 0.85$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"Rows with flight='CO-1586-IAH-MCO' have '7:42 p.m.' in 9 of 23 rows, but this row has '7:41 p.m.'."*
* **Analysis:** Model flagged the error but proposed an incorrect repair ('7:42 p.m.' vs clean '7:14 p.m.') that passed the validation layer.

#### Table `flights` — Row 231, Column `act_dep_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '5:56 a.m.'
* **Model Score:** $P(\text{error}) = 0.98$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Model flagged the error but proposed an incorrect repair ('6:10 a.m.' vs clean '5:56 a.m.') that passed the validation layer.

#### Table `flights` — Row 248, Column `act_dep_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '10:33 a.m.'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Model flagged the error but proposed an incorrect repair ('10:53 a.m.' vs clean '10:33 a.m.') that passed the validation layer.

### Category 3: Correct Cells Falsely Flagged / Repaired

#### Table `beers` — Row 195, Column `abv`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** (empty string)
* **Model Score:** $P(\text{error}) = 0.84$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

#### Table `beers` — Row 275, Column `beer_name`
* **Dirty Input:** 'Fistmas Ale'
* **Ground-Truth Clean:** 'Fistmas Ale'
* **Model Score:** $P(\text{error}) = 0.89$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value 'Fistmas Ale' occurs 1 time(s) and its pattern Aa{6} Aaa covers 0% of the column."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

#### Table `flights` — Row 41, Column `sched_arr_time`
* **Dirty Input:** '11:19 a.m.'
* **Ground-Truth Clean:** '11:19 a.m.'
* **Model Score:** $P(\text{error}) = 0.65$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value '11:19 a.m.' occurs 2 time(s) and its pattern 99:99 a.a. covers 31% of the column."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

#### Table `flights` — Row 85, Column `act_dep_time`
* **Dirty Input:** '5:56 a.m.'
* **Ground-Truth Clean:** '5:56 a.m.'
* **Model Score:** $P(\text{error}) = 0.73$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"Rows with flight='UA-414-LAX-SFO' have '6:10 a.m.' in 11 of 23 rows, but this row has '5:56 a.m.'."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

#### Table `hospital` — Row 5, Column `score`
* **Dirty Input:** 'empty'
* **Ground-Truth Clean:** 'empty'
* **Model Score:** $P(\text{error}) = 0.50$
* **Gate Decision:** `keep` | **Validation Outcome:** `none` | **Final Decision:** `keep`
* **Model Stated Evidence:** *"The value 'empty' is empty or a placeholder."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

#### Table `hospital` — Row 21, Column `score`
* **Dirty Input:** 'empty'
* **Ground-Truth Clean:** 'empty'
* **Model Score:** $P(\text{error}) = 0.62$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value 'empty' is empty or a placeholder."*
* **Analysis:** Clean ground-truth cell falsely flagged as defect due to pattern anomaly or minority value.

### Category 4: Missed Errors

#### Table `beers` — Row 122, Column `ounces`
* **Dirty Input:** '12.0 oz. Alumi-Tek'
* **Ground-Truth Clean:** '12'
* **Model Score:** $P(\text{error}) = 0.00$
* **Gate Decision:** `none` | **Validation Outcome:** `none` | **Final Decision:** `keep`
* **Analysis:** Defect never reached the model: no classical detector (D1-D7, d5c) fired on this cell (Stage 1 FN).

#### Table `beers` — Row 330, Column `abv`
* **Dirty Input:** '0.08199999999999999%'
* **Ground-Truth Clean:** '0.082'
* **Model Score:** $P(\text{error}) = 0.99$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value '0.08199999999999999%' occurs 1 time(s) and its pattern 9.9{17}% covers 6% of the column."*
* **Analysis:** Defect reached the model and was flagged, but gate or validation layer routed it to review (flag for review).

#### Table `flights` — Row 15, Column `act_dep_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '6:02 a.m.'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** Defect reached the model and was flagged, but gate or validation layer routed it to review (flag for review).

#### Table `flights` — Row 263, Column `act_dep_time`
* **Dirty Input:** '1:48 p.m.'
* **Ground-Truth Clean:** '1:31 p.m.'
* **Model Score:** $P(\text{error}) = 0.00$
* **Gate Decision:** `none` | **Validation Outcome:** `none` | **Final Decision:** `keep`
* **Analysis:** Defect never reached the model: no classical detector (D1-D7, d5c) fired on this cell (Stage 1 FN).

#### Table `hospital` — Row 72, Column `emergency_service`
* **Dirty Input:** 'yxs'
* **Ground-Truth Clean:** 'yes'
* **Model Score:** $P(\text{error}) = 0.00$
* **Gate Decision:** `none` | **Validation Outcome:** `none` | **Final Decision:** `keep`
* **Analysis:** Defect never reached the model: no classical detector (D1-D7, d5c) fired on this cell (Stage 1 FN).

#### Table `hospital` — Row 109, Column `score`
* **Dirty Input:** 'x3%'
* **Ground-Truth Clean:** '93%'
* **Model Score:** $P(\text{error}) = 0.97$
* **Gate Decision:** `flag for review` | **Validation Outcome:** `none` | **Final Decision:** `flag for review`
* **Model Stated Evidence:** *"The value 'x3%' occurs 1 time(s) and its pattern a9% covers 0% of the column."*
* **Analysis:** Defect reached the model and was flagged, but gate or validation layer routed it to review (flag for review).

### Validation Layer: Rejected Repair (Hospital FD Rejection)

#### Table `hospital` — Row 103, Column `measure_name`
* **Dirty Input:** 'surgeryxpatientsxneedingxhairxremovedxfromxthexsurgicalxareaxbeforexsurgery& xwhoxhadxhairxremovedxusingxaxsaferxmethodx(electricxclippersxorxhairxremovalxcreamxï¿½cxnotxaxrazor)'
* **Ground-Truth Clean:** 'surgery patients needing hair removed from the surgical area before surgery who had hair removed using a safer method (electric clippers or hair removal cream c not a razor)'
* **Model Score:** $P(\text{error}) = 0.98$
* **Gate Decision:** `repair` | **Validation Outcome:** `rejected` | **Final Decision:** `review (repair rejected by validation)`
* **Model Stated Evidence:** *"Rows with measure_code='scip-inf-6' have 'surgery patients needin…' in 4 of 5 rows, but this row has 'surgeryxpatientsxneeding…'."*
* **Analysis:** The model proposed a truncated repair that dropped the second clause of the CMS description. The validation layer checked the functional dependency measure_code -> measure_name+condition, detected that this repair disagreed with consistent rows sharing the same measure_code, and rejected the repair. This prevented an incorrect cell modification and safely routed the cell to human review.

### Validation Layer: Rejected Repair (Flights Schedule Conflict)

#### Table `flights` — Row 124, Column `act_dep_time`
* **Dirty Input:** (empty string)
* **Ground-Truth Clean:** '12:12 p.m.'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `rejected` | **Final Decision:** `review (repair rejected by validation)`
* **Model Stated Evidence:** *"The value '' is empty or a placeholder."*
* **Analysis:** The model proposed a departure time repair, but the flight dependency check detected inconsistent timing with related rows for that flight number and rejected the repair, preventing an unverified modification.

### Validation Layer: Unvalidated Repair (No Independent Rule)

#### Table `beers` — Row 107, Column `beer_name`
* **Dirty Input:** 'Orignal Blend'
* **Ground-Truth Clean:** 'Orignal Blend'
* **Model Score:** $P(\text{error}) = 0.97$
* **Gate Decision:** `repair` | **Validation Outcome:** `unvalidated` | **Final Decision:** `review (repair cannot be validated)`
* **Model Stated Evidence:** *"The value 'Orignal Blend' occurs 1 time(s) and its pattern Aa{6} Aaaaa covers 0% of the column."*
* **Analysis:** Under our conservative validation policy, repairs on columns with no independent functional dependencies or domain rules cannot be verified, and are safely routed to human review rather than silently applied.

### Validation Layer: Validated Repair (Applied)

#### Table `hospital` — Row 104, Column `name`
* **Dirty Input:** 'xlxskx regionxl hospitxl'
* **Ground-Truth Clean:** 'alaska regional hospital'
* **Model Score:** $P(\text{error}) = 1.00$
* **Gate Decision:** `repair` | **Validation Outcome:** `validated` | **Final Decision:** `repair`
* **Model Stated Evidence:** *"Rows with address_1='2801 debarr road' have 'alaska regional hospital' in 17 of 18 rows, but this row has 'xlxskx regionxl hospitxl'."*
* **Analysis:** The corrupted hospital name ('xlxskx regionxl hospitxl') was corrected to 'alaska regional hospital' and verified by the provider_number functional dependency consistency check.

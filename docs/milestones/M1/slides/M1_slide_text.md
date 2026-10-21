# Milestone 1 deck: slide text

Text for a 10-slide deck (title, eight member slides, next steps). Each member presents their own two slides; the order follows the questions in the TA's approval note (problem, landscape, data, contribution). Copy the text into the slide template; tables can stay tables. This file is the single record of the slide text.

Order: 1 title · 2 Sidhaarth · 3–4 Aaradhya · 5–6 Jashan · 7–8 Sahilpreet · 9 Sidhaarth · 10 next steps.

---

## Slide 1: ContextGuard: finding and repairing cell-level errors in tables  
*Presenter: Sidhaarth*

**Milestone 1: problem definition, literature review and data audit**
DSAI Lab Project, IIT Madras BS · Group 1 · Sidhaarth Shree, Jashan, Sahilpreet, Aaradhya
Repository: github.com/SidhaarthShree07/Group-1-DS-and-AI-Lab-Project

*Speaker notes:* We are building a system that finds cell-level errors in tables, names the error type, proposes a repair and says how confident it is. Today we cover the problem, what exists, the data we will use and how we will measure success.

---

## Slide 2: The problem, and what is new  
*Presenter: Sidhaarth*

**A cell can look fine and still be wrong.** Hospital: `City = "birminghxm"` with `ZipCode = 35233` and `State = "al"`. 48 rows share that zip; 45 say `birmingham`.

| Approach today | Limit |
|---|---|
| Rules (NADEEF, HoloClean) | An expert writes constraints for every dataset; HoloClean repair F1 is ~0 on Beers and Rayyan (Baran paper) |
| Statistics | A rare value is often correct; a wrong value that looks ordinary is missed |
| Raha, Baran | Need labelled rows for every new table |
| LLM cleaners (ZeroED, Cocoon, ForestED) | Large hosted models, detection only, no calibrated confidence |

**ContextGuard:** cheap detectors choose the cells, a fine-tuned open 8B model decides (error? type? repair?), a calibrated gate applies only confident repairs.
**Why it matters:** cleaning can help or hurt a downstream model (CleanML, 14 datasets), so we repair conservatively and measure the effect.

*Speaker notes:* Machine-learning models are only as good as their training tables. A cell can look valid and still be wrong given its row, as the Hospital example shows. Rules need an expert per dataset, statistics miss ordinary-looking wrong values, Raha and Baran need labels for every table, and recent LLM cleaners need large hosted models and give no confidence. ContextGuard combines cheap detectors, a fine-tuned open model and a confidence gate, and repairs conservatively because cleaning can hurt a downstream model.

---

## Slide 3: What exists before LLMs  
*Presenter: Aaradhya*

**Rules and constraints (HoloClean, NADEEF, HyFD).** They rely on functional dependencies and denial constraints. Rules are written by hand (9 constraints for Hospital); exact dependency discovery breaks on noisy tables; HoloClean's repair F1 is about 0.01 on Beers and 0 on Rayyan.

**Statistical and learning-based cleaners (Raha, Baran, BClean).** Ensembles of detectors plus up to 20 labelled tuples. They need fresh labels for every new table, and repair quality falls when detection is imperfect (Rayyan F1 0.52 with perfect detection, 0.35 end to end).

**Industry tools (Great Expectations, Deequ, Pandera, AWS Glue, Dataplex, Monte Carlo and others).** Their documentation shows cross-column SQL checks, but every check is written by a person and detects only what is specified; we found no documented automatic repair of cell values.

*Speaker notes:* Before LLMs there were three families: constraint engines such as HoloClean, statistical and learning-based cleaners such as Raha and Baran, and industry validation tools. The first two need hand-written rules or fresh labels for every table, and repair falls when detection is imperfect. The industry tools check what a person has specified.

---

## Slide 4: What we must beat, and how we measure it  
*Presenter: Aaradhya*

**Numbers to beat (from the original papers, 20 labels):**
- Raha detection F1 (Table 5): Hospital 0.72, Flights 0.81, Beers 0.99, Rayyan 0.79, Movies 0.86.
- Raha + Baran end-to-end repair F1 (Table 7): Hospital 0.66, Flights 0.66, Beers 0.90, Rayyan 0.35, Tax 0.74.
- dBoost alone, detection F1: Hospital 0.49, Flights 0.66, Beers 0.55.

**How we measure:** a cell is an error if it differs from the clean value after Raha's normalisation; a repair counts only on an exact match. Raha and Baran get 20 labelled tuples per table, ContextGuard gets none on unseen tables. Results are mean ± std over 10 seeds (42 to 51), reported on our own splits.

**Baselines:** B1 HoloClean, B2 dBoost, B3 Raha + Baran, B4 Gemma 4 E4B-it zero-shot, B5 Gemma 4 E4B-it with 3 examples.

*Speaker notes:* These are the numbers we have to beat, taken from the original papers with 20 labels. We never present a paper's number as our own result: we rerun every baseline on our splits, with Raha's normalisation, exact-match repair, 20 labels for Raha and Baran on every table and none for ContextGuard on unseen tables, over ten seeds.

---

## Slide 5: Our benchmark tables and what we found in them  
*Presenter: Jashan*

**18,414 verified error cells in five tables** (Raha rule: dirty ≠ clean after unescaping, whitespace collapse, strip).

| Table | Size | Error cells | Rate |
|---|---|---|---|
| Hospital | 1,000 × 19 | 509 | 2.68% |
| Flights | 2,376 × 6 | 4,920 | 34.51% |
| Beers | 2,410 × 10 | 4,362 | 18.10% |
| Rayyan | 1,000 × 11 | 948 | 8.62% |
| Movies | 7,390 × 17 | 7,675 | 6.11% |

- The public TableEG copies had **lost errors**: Beers 1,005 (`N/A` in the Raha dirty file, blank in TableEG's) and Movies a net 668 (`RatingValue` `7.0` became `7`). We use the Raha originals.
- Clean and dirty headers differ (Hospital 19, Movies 17, Beers 2), so columns are aligned by position, not by name.

*Speaker notes:* I audited the five core benchmark tables with `scripts/data_inventory.py`, using Raha's own comparison rule. Two of the TableEG copies had silently lost errors: 1,005 in Beers because `N/A` was turned into a blank, and a net 668 in Movies because `RatingValue` 7.0 was saved as 7. We therefore use the Raha originals as ground truth, which also means our baseline numbers can be compared with the published ones. Because headers differ between the clean and dirty files, every loader aligns columns by position.

---

## Slide 6: Error types, weaknesses of the "clean" files, and data sufficiency  
*Presenter: Jashan*

**81,596 true error cells across the nine audited tables** (Credit's 315,466 float re-saving differences are kept apart): missing 39,975 (49%), pattern 37,810 (46%), rule 2,329 (3%), outlier 1,482 (2%).

- Rule violations are scarce (Flights 1,149, Rayyan 619, Hospital 408, Beers 153), so training adds synthetic dependency violations on clean UCI tables (at most 20% of training data).
- The "clean" gold files still contain blanks (Beers 1,072, Movies 2,680, Rayyan 1,609 cells), which limits achievable repair recall.
- Soccer: 796 errors typed by us (597 pattern, 199 outlier; BART's log has no types). Restaurant: 633 errors, all pattern. Credit: imputed income `88.44070676` repeats 17,435 times; target 6.68% positive.
- The five core tables give an estimated 25,000–35,000 training examples, inside the 15,000–40,000 budget for LoRA fine-tuning of Gemma 4 E4B-it.

*Speaker notes:* Across the audited tables the errors split into about 49% missing values, 46% pattern violations, 3% rule violations and 2% outliers. Rule violations are rare, so we add synthetic ones. The "clean" files are not perfectly clean: Beers has 1,072 blank cells, which caps repair recall. For the unseen tables I typed Soccer's errors myself, because BART's log has no types, and computed Restaurant's 633 errors from the files. The training pool is large enough for fine-tuning.

---

## Slide 7: The extra datasets that test generalisation  
*Presenter: Sahilpreet*

| Table | Role | Size | Error cells | Rate |
|---|---|---|---|---|
| Tax (Raha) | Training | 200,000 × 15 | 121,219 | 4.04% |
| Inpatient (BClean, 10%) | Training only | 4,017 × 11 | 4,465 | 10.10% |
| Facilities (BClean, 10%) | Training only | 7,992 × 10 | 8,318 | 10.41% |
| Billionaire (ZeroED) | Unseen test | 2,614 × 22 | 5,195 | 9.03% |
| Clean Me If You Can, Germany | Unseen test | 5,000 of 374,533 × 9 | 18,678 | 41.51% |

- Every pair was aligned and checked; BClean's dirty files have an extra row-index column that we drop.
- The sources give no error types, so we type each changed cell: missing, format artifact, rule (a valid value in the wrong place) or pattern.
- Unseen tables are never used for training, augmentation or profiling; Inpatient and Facilities share Hospital's domain, so they are training only.

*Speaker notes:* These five tables are the extra data beyond the benchmark tables. Tax, Inpatient and Facilities are for training; Billionaire and a German address slice are tests on data the model has never seen. I downloaded each from its source, recorded the commit or Zenodo version and checksums, checked that clean and dirty line up, and counted errors with Raha's rule. Because these sources have no labels for error types, I typed every changed cell with a simple rule.

---

## Slide 8: What we found, and the limits  
*Presenter: Sahilpreet*

- **Tax is 97.6% re-formatting:** `3` against `3.0` and a leading zero on zip codes. Only 2,966 of 121,219 error cells are real injected errors, so these differences are kept out of the training positives.
- **Billionaire's clean file** has 200 `nan (wait to be cleaned)` placeholders.
- **Germany slice:** already published on Zenodo; 5,000 rows drawn with seed 42; 41.5% of cells are errors, mostly missing values.
- **No row overlap** with training tables: no shared ids between Hospital and Inpatient or Facilities (one hospital name and address appears in both) and none between Billionaire and the other tables.
- **Licences:** BClean and ZeroED state none, so they are cited and not redistributed; the address data is OpenStreetMap-derived (ODbL, share-alike, attribution); UCI tables for augmentation are CC BY 4.0.

*Speaker notes:* The main finding is that almost all of Tax's errors are number formatting, which would teach the wrong lesson, so we count them separately. The German slice is the hardest test: four in ten cells differ. We found no leakage into the test tables. Two sources have no licence, so we only cite them and keep raw data out of the repository.

---

## Slide 9: Objectives, evidence and the model choice  
*Presenter: Sidhaarth*

| Objective | Target |
|---|---|
| O1 Detection | Best macro-F1 against all baselines |
| O2 Safe repair | Only a small share of applied repairs wrong |
| O3 Unseen tables | Above zero-shot LLM and Raha with 20 labels |
| O4 Cost | Keeps most errors, sends only a minority to the LLM |
| O5 Downstream | Share of the dirty-versus-clean gap closed |
| O6 End-to-end tool | Upload a CSV, get the cleaned file and a change log |

**Base model (zero-shot, 3 seeds of 1,500 cells, mean ± std):** Gemma 4 E4B-it 0.645 ± 0.007, Qwen3.5-9B 0.603 ± 0.005, Qwen3-4B 0.548 ± 0.009. Gemma won every seed and makes far fewer wrong repairs (0.39 against 0.68–0.70).
**Data:** 18,414 verified error cells in the five core tables; 5 unseen test tables; Credit for the downstream test.

*Speaker notes:* These are our six objectives. The first three are about accuracy, safety and unseen tables, and the last is the end-to-end tool: a user uploads a CSV and receives the cleaned data plus a change log. We chose the base model by running three open models zero-shot on the same 1,500-cell samples, with three sampling seeds. Gemma 4 E4B-it won every seed and makes far fewer wrong repairs, so it is the model we fine-tune. This is a zero-shot comparison only; fine-tuning may change the order.

---

## Slide 10: Plan and risks  
*Presenter: Sidhaarth*

| Milestone | Due | Work |
|---|---|---|
| M2 Dataset preparation | 29 Oct | Verified sources, splits with leakage checks, data card, first instruction–response examples |
| M3 Model architecture | 5 Nov | Pipeline end to end on a subset, input formats, justification, loss and metrics |
| M4 Model training | 19 Nov | Fine-tuning runs, ablations, several seeds |
| M5 Evaluation and analysis | 26 Nov | Baseline comparison, precision–coverage curve, unseen tables, downstream experiment |
| M6 Deployment and documentation | 10 Dec | Gradio demo and API, model and data cards, final report |

**Main risks:** the cheap stage misses many errors (recall ceiling); the model over-flags rare but correct values; unclear licences (raw data stays out of the repository); Credit's clean income values were imputed, so the downstream test is repeated on Beers; a new model family.

*Speaker notes:* The timeline follows the course milestones. The two risks that matter most are recall of the cheap stage, which caps the whole system, and over-flagging, which hurts precision. We address both in training and report the recall ceiling honestly.

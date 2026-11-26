# Milestone 5: detector and baseline sections (Aaradhya)

Sections for the M5 report: X7, X8, X9, the missed-error analysis, and corrections to M4. Sidhaarth merges them.
**State, 26 Nov 2026.** Done: X8, X9, baselines on the unseen tables, missed errors on validation, missed errors on test, X7 (precision rule: six of eight validation tables decided; on test it holds on all five core tables). Open: see Section 6.
Numbers come from `docs/tables/m5_*.csv` and, for ContextGuard M4, `m4_test_vs_baselines.csv`. Counted directly from run files: errors in the 20 labelled rows and the cells HoloClean fills (Sections 2 and 2b). Rules written before the runs, and their amendments, are in Section 7.

## 1. X7: detector generalisation

**Change.** All settings were chosen on validation splits; ideas were confirmed on train; detectors read the dirty table only.
- **D2 free-text gate.** In a column where at least 90% of the values are distinct, shape rarity is ignored. Cells are still flagged for encoding damage (replacement, control or loose combining characters) or a character found in at most 2% of the column's cells.
- **D7 extension.** Thousands separators, trailing `.0`, dates with a two-digit year whose last two parts can be swapped, and unit text with a hyphen.

**Result** (validation, `m5_candidate_recall.csv`; `*` = generality check, not used for the first choice):

| Table | Errors | M4 recall / cells sent | M5 recall / cells sent | Gained / lost | New cells |
|---|---|---|---|---|---|
| Hospital | 79 | 91.14% / 16.4% | 91.14% / 16.4% | 0 / 0 | 0 |
| Flights | 723 | 78.98% / 29.4% | 78.98% / 29.4% | 0 / 0 | 0 |
| Beers | 679 | 98.09% / 36.5% | 100.00% / 27.8% | 13 / 0 | 13 |
| Rayyan | 141 | 46.81% / 50.8% | 97.16% / 26.7% | 72 / 1 | 76 |
| Movies | 1054 | 42.79% / 48.9% | 97.34% / 24.6% | 597 / 22 | 600 |
| Tax * | 51 | 90.20% / 7.3% | 90.20% / 10.2% | 0 / 0 | 1301 |
| Inpatient * | 695 | 78.85% / 15.3% | 78.85% / 15.3% | 0 / 0 | 0 |
| Facilities * | 1250 | 94.32% / 31.5% | 94.32% / 31.5% | 0 / 0 | 0 |

| Rule | Result |
|---|---|
| Recall +2 points on at least one table | **pass** (Rayyan +50.4, Movies +54.6) |
| No table loses more than 1 point | **pass** on all eight tables |
| At most 35% of cells sent, every table | **pass** (highest: Facilities 31.5%) |
| Total calls at most M4's | **pass** on the core five: 7,198 against 12,498 (−42%) |
| Pipeline precision falls by at most 0.01 | **pass** on six of eight validation tables without the model; Rayyan and Movies have no validation run, but on **test** the rule holds on all five core tables even in the worst case (below) |

**Precision rule without the model** (`m5_precision_rule.csv`). A prompt depends on the dirty table and the cell, not on the detector that sent it, so the model's answer for a cell is the same in both configurations (up to numerical effects of batching). Where M5 only removes non-errors, or only adds errors or format artifacts (which are not scored), precision cannot fall:

| Table | Errors removed | Non-errors removed | Errors added | Non-errors added | Format artifacts added | Precision rule |
|---|---|---|---|---|---|---|
| Hospital | 0 | 0 | 0 | 0 | 0 | pass: identical candidates |
| Flights | 0 | 0 | 0 | 0 | 0 | pass: identical candidates |
| Beers | 0 | 328 | 13 | 0 | 0 | pass: only non-errors or artifacts change, errors only added |
| Rayyan | 1 | 472 | 72 | 4 | 0 | not decided: needs the model |
| Movies | 22 | 5,166 | 597 | 3 | 0 | not decided: needs the model |
| Tax | 0 | 0 | 0 | 0 | 1,301 | pass: only non-errors or artifacts change, errors only added |
| Inpatient | 0 | 0 | 0 | 0 | 0 | pass: identical candidates |
| Facilities | 0 | 0 | 0 | 0 | 0 | pass: identical candidates |

Rayyan and Movies also lose true errors or add non-errors, and M4 has no validation run on them.

**The M4 final runs, replayed with arm A** (`scripts/replay_m4_runs.py`, `m5_precision_replay.csv`). The runs first reproduce the published M4 test scores exactly. For a cell that both detector sets send, the M4 answer is used; for a cell only M5 sends, the worst case is assumed: every new error is missed and every new non-error is flagged. All test rows:

| Table (test) | M4 precision | False alarms removed | True positives lost | New errors / non-errors | M5 precision, worst case | Model flagged, same columns in M4 |
|---|---|---|---|---|---|---|
| Hospital | 0.837 | 0 | 0 | 0 / 0 | 0.837 | – |
| Flights | 0.955 | 0 | 0 | 0 / 0 | 0.955 | – |
| Beers | 0.966 | 2 | 0 | 14 / 0 | 0.969 | ounces 347/347 |
| Rayyan | 0.663 | 11 | 0 | 75 / 3 | 0.726 | article_jcreated_at 34/34 |
| Movies | 0.755 | 69 | 64 | 688 / 3 | 0.832 | rating_count 245/245; rating_value 30/30 |

- **Precision cannot fall on any of the five tables.** On Rayyan and Movies it rises in the worst case, because the D2 gate removes 11 and 69 false alarms.
- This is test data and is used for reporting only: arm A was fixed before, and nothing is changed because of it.
- **The cost is on Movies:** 64 errors that M4 flagged are no longer sent (`release_date` 28, `description` 15, `creator` 19, `name` 2).
- The last column is the model's answer on errors M4 did send in the columns of the new errors: 347 of 347 on Beers, 34 of 34 on Rayyan, 275 of 275 on Movies. These cells reached the model through other detectors, so they may be easier than the new ones.

A replay on the M3 subset (`m5_e3_replay.csv`), where `m4_e3_all_ft` has an answer for every cell (early M4 prompt settings, Hospital, Flights and Beers only), gives the same precision and recall for the M4 and the M5 detectors on all three tables, and 84 fewer calls on Beers (384 to 300).

On train the same settings also meet the recall and cost rules on all eight tables; Movies is close to the limit (34.4%).

**Ablation** (recall, %, validation, `m5_detector_ablation.csv`). The plan's "D1–D6 plus D5 alone" is read as D1–D6 plus the D5 entity-consensus extension, because D1–D6 already holds D5:

| Detectors | Hospital | Flights | Beers | Rayyan | Movies | Tax | Inpatient | Facilities | Cells, core five |
|---|---|---|---|---|---|---|---|---|---|
| D1-D6 | 82.3 | 59.3 | 37.1 | 46.8 | 42.8 | 90.2 | 78.8 | 94.3 | 11,876 |
| D1-D6 + D5 consensus | 91.1 | 79.0 | 37.1 | 46.8 | 42.8 | 90.2 | 78.8 | 94.3 | 12,084 |
| D1-D6 + D7 | 82.3 | 59.3 | 98.1 | 46.8 | 42.8 | 90.2 | 78.8 | 94.3 | 12,290 |
| M4 (D1-D6, D7, D5 consensus) | 91.1 | 79.0 | 98.1 | 46.8 | 42.8 | 90.2 | 78.8 | 94.3 | 12,498 |
| M4 + D7 extension, no D2 gate | 91.1 | 79.0 | 100.0 | 95.7 | 99.4 | 90.2 | 78.8 | 94.3 | 13,179 |
| M4 + D2 gate, no D7 extension | 91.1 | 79.0 | 98.1 | 48.2 | 40.7 | 90.2 | 78.8 | 94.3 | 6,517 |
| M5 | 91.1 | 79.0 | 100.0 | 97.2 | 97.3 | 90.2 | 78.8 | 94.3 | 7,198 |
| M5, D3/D4 without column names | 91.1 | 79.0 | 100.0 | 97.2 | 97.3 | 92.2 | 80.9 | 94.6 | 7,311 |

**Reading.**
- **The D7 extension gives the recall, the D2 gate pays for it.** Without the gate three tables exceed 35%; without the extension recall barely changes (Rayyan +1.4, Movies −2.1).
- **The gain is on three core tables.** On Tax, Inpatient and Facilities recall does not change. On Tax the extension adds 1,301 cells, none of them a scored error (they are format artifacts).
- **The extension encodes a convention.** In Movies every `rating_count` of 1,000 or more has a comma and `rating_value` has no plain integer, so nothing is inconsistent inside the dirty column; in Rayyan no date shows its order. The flags mean "non-canonical or ambiguous", not "inconsistent". An unseen table that uses commas correctly will send those cells to the model too.
- **The D2 gate can hide typos in near-unique name columns.** Arms that gated by D2's own fire rate lost 7–12 points on Facilities with the D7 extension on. The chosen arm does not gate those columns, but an unseen table may have columns above the 90% limit.
- **D3 and D4 still skip columns by name** (zip, phone, code, ...). Without that rule recall rises on Tax (+2.0), Inpatient (+2.0) and Facilities (+0.2), does not change on the core five, and the core five need 113 more calls. It was listed as "reported, not selected" and is not applied; recommended as the next change.

**How the arm was chosen.** The first choice, made on the core five only, broke the recall floor on Facilities (−7.8) and Inpatient (−1.3). All arms were then judged on eight tables and a family with a 35% table switch was added. The written selection rule picks a member of that later family (`fire>0.2`, switch on; 5,133 core calls). The arm used here was in the original grid, has higher recall on Movies train (95.6 against 90.8) and has no table-level switch. All 82 arms are in `m5_x7_sweep.csv`.

## 1b. Options suggested after a look at the unseen tables

**Provenance.** On 25 Nov, after the arm above was fixed, Sidhaarth suggested four detector changes from the labelled errors of Soccer, German and Billionaire. The plan says these tables are touched once, with the frozen configuration. So:
- **Arm A (Section 1) is the frozen set.** It was chosen on validation before these suggestions and uses nothing from the unseen tables. It gives the headline unseen result.
- **A configuration built from the suggestions is post-hoc on Soccer, German and Billionaire.** If it is run there, it is reported as a second row and not as a held-out result.
- The options were judged on validation and train only, with fixed settings, one variant of O1 and no sweep. I ran no detector on the unseen tables.

Sidhaarth's rule, read per table: recall goes up on at least one table, candidates grow by less than 10% on every table, no table above 35%. All eight validation tables, then train (`m5_x7_options.csv`, `m5_x7_options_train.csv`):

| Option, on top of arm A | Val: cells / errors added | Val: largest growth | Train: cells / errors added | Train: largest growth | Rule, val / train |
|---|---|---|---|---|---|
| O1 composite key (D5-consensus thresholds) | 4,450 / 91 | 85.2% (Tax) | 21,592 / 866 | 73.6% (Tax) | fail / fail |
| O1b composite key (plain D5 thresholds) | 89 / 13 | 1.2% (Movies) | 559 / 469 | 4.3% (Inpatient) | pass / pass |
| O2 dominant value (80%) | 0 / 0 | 0% | 4 / 2 | 0.2% (Hospital) | fail / pass |
| O3 abbreviation mix | 0 / 0 | 0% | 152 / 0 | 7.0% (Hospital) | fail / fail |
| O4 rare category | 35 / 8 | 1.5% (Hospital) | 34 / 0 | 0.7% (Hospital) | pass / fail |

- **O1b is the only one that passes on both.** It stays off by default (`m5_detectors(table, composite_key=True)`). The gain is on Inpatient (train recall 72.9 to 79.9%) and Facilities (81.4 to 85.8%); on validation it is small (13 errors in 89 cells, 54 of them non-errors on Movies). It does nothing on the other core tables.
- **O1** finds more errors but sends 3,911 Tax cells that hold none.
- **O2 and O3 cannot be chosen on our data:** no validation table has the pattern. O3 sends 152 Hospital cells on train with no error.
- **O4** passes on validation by 8 errors and finds none on train, so it is not adopted.
- **Limiting D2 is already in arm A:** on the core five D2 sends 3,248 cells instead of 9,242, with 236 errors instead of 259 (`m5_detector_contribution.csv`, which also lists what each detector adds). The effect on Restaurant is not known, because the detectors were not run there.
- **Arm A under the 10% rule:** candidates fall by 42% on the core five and 19% on the eight tables together. Tax alone grows by 40% (1,301 cells, all format artifacts), so it fails if every table must stay below 10%.

**Limitation.** Swaps inside a valid domain ("Consumer" for "Technology-Computer") leave no trace in the cell or its column. They need a check across columns and are out of reach for these detectors.

## 2. X8: full baselines on the five test tables

Same cells and scorer (`score_flags`) for every method; the 20 M4 rows are left out for all. Raha and Baran receive a clean file that equals the dirty file outside the labelled rows. No run failed or timed out.

**Detection F1** (`m5_baselines_full.csv`, `m5_raha_label_budgets.csv`). ContextGuard is the **M4** configuration; the possible range for M5 is given below.

| Table | ContextGuard, M4 config | HoloClean | dBoost, 34 settings | dBoost, 36 settings † | Raha, 20 fixed rows | Raha, own 20 rows ‡ |
|---|---|---|---|---|---|---|
| Hospital | 0.886 | 0.653 | 0.518 | 0.518 | 0.625 | 0.888 ± 0.061 |
| Flights | 0.852 | 0.634 | 0.622 | 0.622 | 0.826 | 0.850 ± 0.016 |
| Beers | 0.972 | 0.397 | 0.839 | 0.839 | 0.954 | 0.970 ± 0.032 |
| Rayyan | 0.493 | 0.039 | 0.314 | 0.299 | 0.667 | 0.794 ± 0.044 |
| Movies | 0.503 | 0.017 | 0.390 | 0.422 | 0.881 | 0.896 ± 0.023 |
| **Mean** | 0.741 | 0.348 | 0.537 | 0.540 | 0.791 | 0.880 |

† Raha's 34 settings plus M4's two others, chosen on train; added after the headline results because M4's dBoost was better on Movies. ‡ Raha chooses its 20 rows; mean ± sd over five seeds. Its own labelled rows are left out as well, so the cells differ slightly.

**Repairs on the same cells:**

| Table | Errors | ContextGuard M4, applied (right / wrong) | Raha + Baran (right / executed) | HoloClean (right / executed) |
|---|---|---|---|---|
| Hospital | 66 | 55 / 0 | 23 / 29 | 32 / 32 |
| Flights | 681 | 325 / 13 | 485 / 561 | 223 / 316 |
| Beers | 613 | 595 / 0 | 518 / 551 | 21 / 163 |
| Rayyan | 131 | 0 / 0 | 4 / 13 | 0 / 227 |
| Movies | 1145 | 0 / 0 | 676 / 711 | 0 / 370 |

ContextGuard applies no repair on Rayyan and Movies (no rule file), by design.

**What M5 can reach on these cells** (detection F1, the 20 rows left out; `m5_contextguard_x9_cells.csv`). The lower end assumes that the model flags none of the new errors and all new non-errors, the upper end the reverse. **This is a range, not a result**; the final run decides.

| Table | ContextGuard M4 | ContextGuard M5, possible range | Raha, 20 fixed rows | Raha, own 20 rows |
|---|---|---|---|---|
| Hospital | 0.886 | 0.886 to 0.886 | 0.625 | 0.888 |
| Flights | 0.852 | 0.852 to 0.852 | 0.826 | 0.850 |
| Beers | 0.972 | 0.973 to 0.985 | 0.954 | 0.970 |
| Rayyan | 0.493 | 0.507 to 0.893 | 0.667 | 0.794 |
| Movies | 0.503 | 0.463 to 0.924 | 0.881 | 0.896 |
| **Mean** | 0.741 | 0.736 to 0.908 | 0.791 | 0.880 |

On Movies the lower end is below M4 (lost errors); the model's answers on the same columns (above) point to the upper end.

**Reading.**
- **The M4 statement "ContextGuard beats every baseline on four of five tables" does not hold against the real Raha.** With the same 20 fixed rows, ContextGuard (M4) leads on Hospital, Flights and Beers, and Raha leads on Rayyan (+0.17) and Movies (+0.38). Mean F1: 0.791 against 0.741.
- **Raha with its own rows is stronger:** mean 0.880. Hospital, Flights and Beers are then ties (differences of 0.002, far inside the spread). Fixed random rows understate Raha, because choosing rows is part of its method; both must be shown.
- **M4's reduced Raha was too weak a baseline:** 0.014 on Rayyan and 0.000 on Movies, against 0.667 and 0.881.
- **ContextGuard needs no labels from the test table, but its adapter was trained on the train splits of these tables.** The unseen tables are the fair comparison.
- **Repairs.** Baran repairs more cells than ContextGuard on Flights and Movies (485 and 676 right) with more wrong ones (76 and 35). ContextGuard's applied repairs stay far more precise (975 right, 13 wrong).
- **HoloClean is precise where it has constraints** (Hospital: 32 of 32 right), but has none on Rayyan and one on Movies, where all 220 and 357 of its wrong changes fill cells that are blank in the clean table too. On Hospital it is below M4's majority vote (0.653 against 0.842). Scored on every cell it marks noisy, F1 is lower: Hospital 0.073, Flights 0.425, Beers 0.239, Rayyan 0.076, Movies 0.018. Run time: Hospital 8 s, Flights 3 s, Beers 6 s, Rayyan 4 s, Movies 618 s.

**Deviations.** HoloClean: commit `d4f5929`, its example parameters and default seed (45), one run; `torch` 1.0.1 (its `1.0.1.post2` is no longer published), newer `python-Levenshtein` and `smart-open`, Unix-socket connection. It lowercases its input, so repairs are mapped back to the most frequent original spelling in the column. Constraints are mined on the dirty test table, as in M4. Raha 1.26: every strategy runs; KBVD returns output only on Hospital and Beers. Baran has no Wikipedia-pretrained value models.

## 2b. Baselines on the unseen tables (objective O3)

The five tables come from Jashan's files. Rows, columns and error counts were recomputed from the dirty and clean files and equal the M1 audit (633, 26,815, 796, 5,195 and 18,678). The settings and the code hashes were written down before the run (Section 7). ContextGuard and its detectors have not been run on these tables. The baselines ran before the team's config freeze; they use none of its settings.

| Table | Errors scored (rate) | Errors in the 20 rows | HoloClean F1 | dBoost F1 | Raha F1, 20 fixed rows | Raha F1, own 20 rows | Baran right / executed | HoloClean right / executed |
|---|---|---|---|---|---|---|---|---|
| Restaurant | 633 (0.5%) | 0 | 0.000 | 0.002 | 0.000 | 0.044 ± 0.076 | 0 / 0 | 0 / 0 |
| Marketing | 26,772 (21.3%) | 43 | 0.026 | 0.413 | 0.366 | 0.339 ± 0.026 | 5,566 / 23,741 | 348 / 359 |
| Soccer | 789 (4.0%) | 7 | 0.023 | 0.108 | 0.458 | 0.625 ± 0.036 | 99 / 431 | 9 / 9 |
| Billionaire | 5,131 (9.0%) | 46 | 0.423 | 0.353 | 0.572 | 0.666 ± 0.023 | 1,404 / 2,633 | 791 / 1,378 |
| German | 18,599 (41.5%) | 76 | 0.675 | 0.433 | 0.873 | 0.873 ± 0.020 | 8,207 / 10,450 | 4,236 / 9,470 |
| **Mean** |  |  | 0.229 | 0.262 | 0.454 | 0.509 |  |  |

Detection F1, 20 labelled rows left out for every method (`m5_baselines_unseen.csv`, `m5_raha_unseen_labelling.csv`). Raha's 20 fixed rows are one random draw per table; "own rows" is the mean ± sd over five seeds.

- **The bar for O3** is Raha with 20 labels: mean F1 0.454 with fixed rows and 0.509 when it chooses its rows. Both should be shown, as on the test tables.
- **Restaurant:** 0.48% of the cells are errors and the 20 fixed rows hold none, so Raha flags nothing. With its own rows it reaches 0.044. No baseline is useful here.
- **Marketing:** random category swaps. Raha flags 24,318 cells and 14,972 of them are not errors. HoloClean mines no constraint and only fills blanks.
- **German:** Raha is strong (0.873). HoloClean's precision is 1.00 because all 9,470 of its changes fill blank cells, and the clean table has no blank cell; 4,236 of the fills are right.
- **HoloClean on Billionaire** ran out of memory in the first attempt (our 120 GB limit, reported as failed) and completed in the second (280 GB): F1 0.423; 1,242 of its 1,378 changes fill blank cells. Its run time on the other four tables was 96–227 s.
- **dBoost** uses one setting fixed on the core tables; it has no train split here.
- The error *types* in Jashan's Billionaire and German files follow the M1 typing rule, not the extended M2 rule (Billionaire 742 rule violations against 1,456). F1 is not affected; per-type numbers would be.

## 3. X9: Raha label budgets

Nested label sets (5 ⊂ 10 ⊂ 20 ⊂ 50; the 20 are the M4 rows). The same 50 rows are left out at every budget. F1, mean ± sd over five worker seeds; the label sets are fixed, so the spread is algorithm noise only and not the effect of which rows are labelled.

| Table | Rows (errors) scored | 5 labels | 10 labels | 20 labels | 50 labels | Baran repair accuracy, 5 / 10 / 20 / 50 |
|---|---|---|---|---|---|---|
| Hospital | 109 (55) | 0.264 ± 0.038 | 0.380 ± 0.047 | 0.612 ± 0.028 | 0.898 ± 0.018 | 0.15 / 0.20 / 0.37 / 0.72 |
| Flights | 315 (624) | 0.789 ± 0.011 | 0.826 ± 0.001 | 0.832 ± 0.000 | 0.953 ± 0.001 | 0.53 / 0.71 / 0.72 / 0.94 |
| Beers | 311 (560) | 0.906 ± 0.045 | 0.955 ± 0.000 | 0.955 ± 0.000 | 0.986 ± 0.003 | 0.53 / 0.73 / 0.82 / 0.93 |
| Rayyan | 100 (98) | 0.595 ± 0.004 | 0.611 ± 0.012 | 0.674 ± 0.018 | 0.818 ± 0.003 | 0.01 / 0.01 / 0.12 / 0.49 |
| Movies | 1059 (1115) | 0.837 ± 0.000 | 0.827 ± 0.000 | 0.883 ± 0.000 | 0.948 ± 0.003 | 0.07 / 0.41 / 0.59 / 0.79 |
| **Mean** |  | 0.678 | 0.720 | 0.791 | 0.920 | 0.26 / 0.41 / 0.52 / 0.77 |

- Raha keeps improving with labels, most on Hospital (0.26 to 0.90) where errors are rare and 20 random rows hold few.
- Five labels already give 0.79–0.91 on Flights, Beers and Movies. Probably because their errors repeat a few patterns.
- Baran needs more labels than Raha: mean repair accuracy 0.26 at 5 and 0.77 at 50.
- Hospital and Rayyan are small here (109 and 100 rows scored), so single values move easily.
- ContextGuard (M4 configuration) on the same 50-row cells, without labels from the test table: Hospital 0.879, Flights 0.848, Beers 0.972, Rayyan 0.484, Movies 0.501 (mean 0.737). Raha passes that mean with 20 labels (0.791).

**Reproducibility.** Two jobs each on Rayyan and Beers were run again: labelled rows, flags and repairs identical. M4's dBoost scores on Rayyan (0.2989) and Movies (0.4220) come out again exactly.

## 4. Missed errors

Errors no M5 detector sends to the model, validation splits (`m5_missed_errors.csv`). Kinds are assigned by comparing the dirty and clean values, not by column name.

| Table | Kind | Missed | Columns |
|---|---|---|---|
| Hospital | typo (1-3 edits) | 6 | sample, measure_code |
| Hospital | other | 1 | measure_name |
| Flights | same format, different number or time | 140 | act_dep_time, act_arr_time |
| Flights | other | 7 | act_dep_time, sched_dep_time |
| Flights | extra text | 5 | sched_arr_time |
| Rayyan | date order | 2 | article_jcreated_at |
| Rayyan | other | 2 | article_pagination |
| Movies | other | 15 | description, creator |
| Movies | truncated | 6 | creator |
| Movies | valid value, wrong for this row | 6 | genre |
| Movies | extra text | 1 | creator |
| Tax | same format, different number or time | 5 | zip |
| Inpatient | other | 74 | lst_nm, frst_nm, adrs_id, cty |
| Inpatient | same format, different number or time | 40 | zip, org_pac_id, prvdr_id, NPI |
| Inpatient | typo (1-3 edits) | 20 | lst_nm, adrs_id, cty, frst_nm |
| Inpatient | valid value, wrong for this row | 9 | cty, org_pac_id, st, frst_nm, adr_ln_1 |
| Inpatient | truncated | 4 | st |
| Facilities | valid value, wrong for this row | 44 | Condition, City, CCN, State, ZipCode, PhoneNumber |
| Facilities | other | 14 | City |
| Facilities | same format, different number or time | 12 | ZipCode, PhoneNumber, CCN |
| Facilities | truncated | 1 | State |

- **Flights is the largest core group: 152 missed, 140 of them another valid time** in the two actual-time columns. The sources disagree there (X3), so no format rule can find them.
- **Hospital:** typos in `sample` and `measure_code`, below D6's limits.
- **Movies:** rewritten descriptions, author lists and shorter genre lists. 22 of the 28 were found by M4 and are the cost of the D2 gate.
- **Inpatient and Facilities:** valid values in the wrong row or column, and changed digits in identifiers. These need dependencies, not formats.

**Where errors are lost on test** (M4 final runs, all test rows; `m5_missed_errors_by_stage.csv`):

| Table (test) | Errors | M4: never sent | M4: model said "no error" | M5: never sent | Main kinds M5 never sends |
|---|---|---|---|---|---|
| Hospital | 76 | 3 | 1 | 3 | sample 2; emergency_service 1 |
| Flights | 717 | 88 | 77 | 88 | act_dep_time 67; act_arr_time 15; sched_arr_time 3; sched_dep_time 3 |
| Beers | 648 | 14 | 1 | 0 | – |
| Rayyan | 146 | 81 | 4 | 6 | article_pagination 6 |
| Movies | 1161 | 726 | 1 | 102 | release_date 28; year 28; creator 21; description 15 |

- **Once a cell is sent, the model rarely rejects a real error, except on Flights** (77 on test, 40 on validation), all in the two actual-time columns: these are the contested values of X3.
- **M5 leaves 199 of the 2,748 test errors unsent, M4 912.** Flights and Hospital are unchanged.
- By type (TableEG types, `m5_candidate_recall_by_type.csv`), the gain on test is Rayyan rule errors (24.5% to 100%) and Movies pattern errors (36.4% to 91.1%). Flights rule errors stay at 61.0%.
- **A threshold at the edge.** On Movies test, `year` ("1971 1972 1973") and `release_date` hold 28 errors each, 2.5% of the rows. D2 and the rare-character check cut at 2%, so they miss them. On validation the same kinds were 1.8% and were caught. The M4 detectors miss the `year` ones too. The cut was not changed after seeing this.

The M4 script `analyze_missed_errors.py` never matched the type names in the data (`pattern_violation` against `pattern`), so its type-based rows fell into "none"; it also read empty cells as NaN. Both are fixed. On `m4_final2_val_all` the script gives the same 15 groups and 172 errors; the 24 errors M4 listed under "none" now go to D2 (23) and D5 (1). The submitted M4 table is not changed.

## 4b. M5 final runs (Sidhaarth's runs, 26 Nov)

Final pipeline: the blended model, the M5 detectors (arm A, no composite key), blank-context lines and Sidhaarth's D2 fire-rate filter. Scored with the baseline scorer (`scripts/score_m5_final_runs.py`, `m5_final_rows.csv`): same cells as the baselines, the 20 labelled rows left out, format artifacts skipped. The runs' cells equal our data cell for cell. Restaurant is not in yet.

**Two flag rules.** The official scorer and M4's table count the model's `is_error` answer. Sidhaarth's table counts the final decision (repair, flag or review). His four rows come out exactly with his rule; both are listed. One rule must be chosen for M4 and M5 alike.

| Unseen table | ContextGuard P / R / F1 (is_error) | F1 (decision ≠ keep) | Applied repairs right / wrong | Raha, 20 fixed rows | Raha, own rows | Best of HoloClean, dBoost |
|---|---|---|---|---|---|---|
| Soccer | 0.977 / 0.696 / **0.813** | 0.807 | 202 / 13 | 0.458 | 0.625 | 0.108 |
| Billionaire | 0.935 / 0.646 / **0.764** | 0.766 | 1,361 / 57 | 0.572 | 0.666 | 0.423 |
| German | 0.977 / 0.651 / **0.781** | 0.780 | 5,369 / 82 | 0.873 | 0.873 | 0.675 |
| Marketing | 0.998 / 0.097 / **0.177** | 0.177 | 277 / 43 | 0.366 | 0.339 | 0.413 |
| **Mean** | **0.634** | 0.633 |  | 0.567 | 0.626 |  |

- **ContextGuard leads on Soccer and Billionaire, Raha on German; dBoost and Raha both beat it on Marketing** (random category swaps: 23,157 of its 26,772 errors are never sent; see below).
- **Mean over the four tables: ContextGuard 0.634 (is_error), Raha 0.567 with fixed rows and 0.626 with its own rows.** The comparison is for detection. **These runs apply repairs on tables without rule files** (`apply_unverified`): 7,209 right and 195 wrong over the four tables. The M5 plan says no repair is applied there and each goes to review marked "not verified"; the report has to follow one of the two and say which.
- The D2 fire-rate filter was described as chosen on the core validation splits; Sidhaarth's message reports its effect on German and Billionaire, so the report should say when it was fixed.

**Validation, final pipeline** (all rows):

| Validation table | P / R / F1 (is_error) | Applied repairs right / wrong |
|---|---|---|
| Hospital | 0.923 / 0.911 / 0.917 | 66 / 1 |
| Flights | 0.947 / 0.747 / 0.835 | 282 / 0 |
| Beers | 0.988 / 1.000 / 0.994 | 676 / 2 |
| Rayyan | 0.792 / 0.943 / 0.861 | 59 / 9 |
| Movies | 0.967 / 0.958 / 0.963 | 914 / 2 |

**X14, detector half** (`m5_x14_detector_ablation.csv`). From the runs before the D2 filter, the candidates that only one detector found are removed and the run is scored again (the other answers are unchanged). Rows with no such candidate are left out:

| Table | Without | Candidates removed (errors) | F1 | Change in F1 |
|---|---|---|---|---|
| German | D7 | 96 (92) | 0.792 | -0.003 |
| Hospital | D5 consensus | 7 (7) | 0.867 | -0.051 |
| Flights | D5 consensus | 201 (142) | 0.744 | -0.091 |
| Beers | D7 | 427 (427) | 0.537 | -0.457 |
| Rayyan | D7 | 71 (69) | 0.544 | -0.317 |
| Movies | D7 | 597 (597) | 0.560 | -0.404 |

- **D7 carries Beers, Rayyan and Movies; D5 consensus carries Flights and Hospital.** On the unseen tables neither matters much: D7 alone adds 92 errors on German; D5 consensus adds no candidate of its own there.

**X7, D2 fire-rate filter as an extra arm** (`scripts/evaluate_d2_fire_filter.py`, `m5_d2_fire_filter*.csv`). On top of arm A, no model:

| Table | Validation recall, arm A → + filter | Train recall, arm A → + filter | Validation candidates |
|---|---|---|---|
| Beers | 100.0 → 100.0 (+0.0) | 100.0 → 100.0 (+0.0) | 1,008 → 1,008 |
| Movies | 97.3 → 95.8 (-1.5) | 95.6 → 91.9 (-3.7) | 4,636 → 2,585 |
| Inpatient | 78.8 → 78.8 (+0.0) | 72.9 → 70.3 (-2.6) | 1,021 → 1,021 |
| Facilities | 94.3 → 86.2 (-8.1) | 81.4 → 70.8 (-10.7) | 3,796 → 1,622 |

- **The filter costs recall on Facilities as the earlier fire-rate arms did** (−8.1 on validation, −10.7 on train), and on Inpatient on train (−2.6); both break the 1-point floor. It saves calls where D2 floods text columns (Movies, Facilities, Beers on train).
- Hospital, Flights, Rayyan and Tax are unchanged.

**Missed errors by stage, final pipeline** (`m5_missed_errors_final.csv`; unseen: 20 labelled rows left out):

| Table | Errors | Flagged | Model said no error | Never a candidate | Withheld by the D2 filter |
|---|---|---|---|---|---|
| Hospital | 79 | 72 | 0 | 7 | 0 |
| Flights | 723 | 540 | 31 | 152 | 0 |
| Beers | 679 | 679 | 0 | 0 | 0 |
| Rayyan | 141 | 133 | 4 | 4 | 0 |
| Movies | 1054 | 1010 | 0 | 28 | 16 |
| Soccer | 789 | 549 | 105 | 135 | 0 |
| Billionaire | 5131 | 3315 | 302 | 1210 | 304 |
| German | 18599 | 12111 | 821 | 2837 | 2830 |
| Marketing | 26772 | 2606 | 1009 | 23157 | 0 |

- **Marketing:** all 23,157 errors never sent are valid categories in the wrong row, which no detector can see (Section 1b, limitation); of the 1,009 the model rejects, 684 are missing values.
- **German:** 2,830 errors are withheld by the D2 filter, 1,640 of them typos; without the filter the model flagged 763 more errors there (F1 0.795 before, 0.782 after).
- **Soccer:** the model says "no error" to 96 typos it is shown.
- **Billionaire:** 419 valid values in the wrong row are never sent; 143 more are sent and rejected.

## 5. Corrections to the M4 report

| Point | Where | Correction |
|---|---|---|
| B1 constraints "mined from the training split" | §10.5 | Mined on the dirty test split (`export_baselines.py`), as M2 §12.4 and the code say |
| dBoost settings | §10.5 | 4 settings were tried, not the 34 of `config.json` |
| "Best detection F1 on four of the five tables" | §10.5, conclusions | True only against the reduced baselines; see Section 2 |
| Raha leads on Flights by 0.046 | §10.5 | Real Raha: 0.826 with the fixed rows, 0.850 ± 0.016 with its own rows; ContextGuard 0.852 |

**For X10 (Sahilpreet).** The official detection score counts the model's `is_error` answer; the flag threshold on P(error) is not applied (`score_like_baselines.py`, `metrics.py`). Requiring P(error) ≥ 0.5 as well would change M4's Hospital test precision from 0.838 to 0.873 and Rayyan's from 0.650 to 0.675. A comparison of thresholds needs the score to use them.

## 6. Open

| Item | Needs |
|---|---|
| X7 precision on Rayyan and Movies **validation** | a model run (10,062 M4 calls, 676 new M5 cells). On test the rule already holds in the worst case |
| Restaurant | Sidhaarth's run (due about 23:30); then `score_m5_final_runs.py` again |
| One flag rule for M4 and M5 | team decision (Section 4b) |
| Error types on Rayyan and Movies validation | TableEG annotation files (the M4 runs carry the types for test and for Hospital, Flights and Beers validation) |

## 7. Protocol and commands

Written before the runs: pass rules and arms of X7; scorer, masks, seeds, HoloClean scoring, dBoost rule and the unseen dBoost rule. Changed after seeing results, all listed above: the eight-table floor and the switch family (X7), a bug fix in the hyphen rule (flagged 128 Hospital addresses, full sweep rerun), dBoost with 36 settings (X8). HoloClean used its own seed, not 0. Unseen tables: HoloClean on Billionaire ran out of memory at the frozen 120 GB limit and was run a second time at 280 GB (no result had been seen); both attempts are in the file. After the runs one line of the runner changed (the note of a failed run). The arm was chosen over the rule's literal pick.

```bash
python scripts/evaluate_m5_candidates.py                          # X7 tables, missed errors (no model)
python scripts/evaluate_x7_options.py                             # what each detector adds, options of Section 1b
python scripts/check_precision_rule.py --e3 <m4_e3_all_ft>        # precision rule without the model
python scripts/replay_m4_runs.py --runs <m4_final runs>           # precision bounds, errors lost by stage
python scripts/run_baselines_m5.py --raha-python <raha env> --hc-python <HoloClean env> \
    --hc-src <HoloClean clone> --hc-socket <PostgreSQL socket dir>   # X8, X9; HoloClean needs PostgreSQL
python scripts/score_m5_final_runs.py --runs <final runs> --unseen <data> # final rows, X14, missed errors by stage
python scripts/evaluate_d2_fire_filter.py                          # D2 fire-rate filter arm
python scripts/rescore_m5_baselines.py --raw <saved outputs>       # scores the saved baseline outputs again
python scripts/run_baselines_m5.py --raha-python <raha env> --methods dboost --dboost-grid union36
```
Unseen tables: add `--splits <folder> --tables restaurant marketing soccer billionaire german --dboost-setting histogram 0.7 0.5 --budgets 20 --fixed-seeds 0 --baran-seconds 5400 --hc-timeout 5400`. The dBoost setting is `histogram 0.7 0.5` (mean train F1 0.391 over the five core tables).
Versions: Raha 1.26 (Python 3.10, pandas 1.5.3, numpy 1.23.5, scikit-learn 1.3.2); HoloClean environment Python 3.6, pandas 0.24.1, numpy 1.16.1, scikit-learn 0.20.0, torch 1.0.1, PostgreSQL 16.
Code: `src/detectors/{candidates,d2_pattern,d7_format_mix}.py` (M4 behaviour unchanged by default; `m5_detectors()` is the new set), `src/detectors/x7_options.py`, `scripts/evaluate_m5_candidates.py`, `evaluate_x7_options.py`, `check_precision_rule.py`, `replay_m4_runs.py`, `run_baselines_m5.py`, `analyze_missed_errors.py`, `scripts/baseline_workers/`, `tests/test_m5_*.py`.

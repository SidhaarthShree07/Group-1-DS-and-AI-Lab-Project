# Milestone 2 slide text

Text of the 10-slide review deck (`ContextGuard_M2_Slides.pptx`, PDF copy `ContextGuard_M2_Slides.pdf`): the words on each slide and the speaker notes. All numbers come from the scripts and tables in this repository (`docs/tables/m2_*.csv`).

---

## Slide 1: From benchmark tables to training-ready examples

**On the slide:** MILESTONE 2 · DATASET PREPARATION · HOSPITAL · THREE ROWS · City · State · ZipCode · birmingham · al · 35233 · ContextGuard · birmingham · al · 35233 · birmingxam · al · 35233 · From benchmark tables to training-ready examples · Proposed repair: birmingham · 73,979 · 13 · question-and-answer examples · source tables: 5 core, 3 external, 5 synthetic · Sidhaarth Shree · Jashan · Sahilpreet · Aaradhya / DSAI Lab Project, IIT Madras BS · Group 1 / github.com/SidhaarthShree07/Group-1-DS-and-AI-Lab-Project · 0 · 1 · row or entity overlap between splits · command rebuilds everything from public sources

*Speaker notes:* Presenter: Sidhaarth. In Milestone 1 we defined the problem and checked the data. In this milestone we turned the data into something a model can be trained on. Five core benchmark tables, three external training tables and five synthetic tables become 73,979 question-and-answer examples. They are split without leakage, a check of 29 conditions passes, and one command rebuilds everything from the public sources, so another team could start training immediately. Each of us presents our part.

---

## Slide 2: Core tables: ingestion, quality audit and ground-truth policy

**On the slide:** CORE BENCHMARK TABLES · Core tables: ingestion, quality audit and ground-truth policy · 5 · 193,986 · 18,414 · 100% · core tables · total cells · true errors (9.49%) · deterministic · Hospital · Flights · Beers · Rayyan · Movies · 14,176 rows across 63 data columns · Exact match to the M1 audit · Pinned Raha commit 7be1334b8c, SHA-256 verified · TABLE · SIZE · ERRORS · ERROR RATE · WHAT THE ERRORS ARE · CLEAN-FILE GAPS · Hospital · 1,000 × 19 · 509 · 2.68% · 408 rule, 101 pattern; clinical measure columns · Missing clean cells: Beers 1,072 · Movies 2,680 · Rayyan 1,609 · Hospital 0 · Flights 0. Reported openly, never imputed. · Flights · 2,376 × 6 · 4,920 · 34.51% · Time columns only (scheduled and actual departure and arrival) · RECOVERED ERRORS · Beers · 2,410 × 10 · 4,362 · 18.10% · Ounces format (100%), IBU missing (41.7%), ABV (28.8%) · 1,005 Beers “N/A” and 668 Movies RatingValue errors lost in TableEG’s copies are restored from the Raha originals. · Rayyan · 1,000 × 11 · 948 · 8.62% · 619 rule, 254 pattern, 75 missing; creation date, ISSN · HEADERS · Movies · 7,390 × 17 · 7,675 · 6.11% · 5,505 RatingCount commas, 856 RatingValue decimals · Columns aligned by position, not by name (Hospital 19, Movies 17, Beers 2).

*Speaker notes:* Presenter: Jashan. Good morning everyone. In Milestone 2, my primary task was transforming the five core Raha benchmark tables into fully reproducible, leakage-free data splits for our LLM pipeline.

First, the numbers on this slide: across Hospital, Flights, Beers, Rayyan, and Movies, we audit 14,176 rows and 193,986 cells. Using our load_core.py script, we verified the exact SHA-256 checksums of every clean and dirty file against the pinned Raha repository commit 7be1334b8c. Applying the strict Raha comparison rule, we conserve exactly 18,414 error cells, matching our M1 audit down to the single cell.

Three critical data quality findings shaped our ingestion design:
1. In TableEG's packaged release, preprocessing bugs caused severe data loss: 1,005 real missing-value errors in Beers were deleted when 'N/A' was blanked out, and 668 errors in Movies were lost due to float truncation on RatingValue. We bypassed TableEG's corrupted copies and restored the canonical Raha originals, recovering 100% of these errors.
2. On the clean side, our audit confirmed significant missing ground truth: Beers has 1,072 missing clean cells, Movies has 2,680, and Rayyan has 1,609. We report these gaps transparently and do not impute fake ground truth. Cells are labelled by the Raha rule, and the gaps limit what repair accuracy means on those three tables.
3. Header casing mismatches, such as Hospital's 19 PascalCase versus snake_case headers, are resolved by positional column alignment, discarding metadata columns like tuple_id and index.

All cell-level ground-truth records are generated dynamically by src/data/load_core.py with zero hard-coding.

---

## Slide 3: Group-aware splits with zero leakage

**On the slide:** DECISIONS D1 & D2 · GROUP-AWARE 70/15/15 · SEED 42 · Group-aware splits with zero leakage · WHY SPLIT BY ENTITY (D2) · ROWS PER SPLIT (TRUE ERRORS) · Train · Val · Test · ZERO-LEAKAGE CHECKS · A random row split leaks when one real entity spans many rows. · 509 · Hospital · 68.8% · 15.3% · 15.9% · 0 · Train 688 (354) · Val 153 (79) · Test 159 (76) · errors · 99.9% · Hospital · row overlap, all 5 tables · 4,920 · Flights · 69.5% · 15.2% · 15.4% · provider_number · 45 entities · max 28 rows each · 0 · Train 1,651 (3,480) · Val 360 (723) · Test 365 (717) · errors · entity overlap, clean key · Flights · 100% · Beers · 70.0% · 15.1% · 15.0% · 4,362 · 18,414 · flight · 100 entities · max 29 rows each · Train 1,686 (3,035) · Val 363 (679) · Test 361 (648) · errors · errors conserved in splits · Rayyan · 70% · 15% · 15% · 948 · 94.7% · Beers · 29 / 29 · errors · Train 700 (661) · Val 150 (141) · Test 150 (146) · brewery_id · 558 entities · max 62 rows each · Movies · 70% · 15% · 15% · 7,675 · leakage checks pass · errors · Share of rows with an entity key used by other rows. Rayyan and Movies never repeat an id (0%), so each row is its own entity. · Train 5,172 (5,460) · Val 1,109 (1,054) · Test 1,109 (1,161) · Error rates stay close across splits, e.g. Flights 35.1 / 33.5 / 32.7%. · All five tables: 18,414 errors, all conserved across splits

*Speaker notes:* Presenter: Jashan. Now: how did we split the data, and how do we prove zero data leakage?

Under Decision D1, we target a 70% Train, 15% Validation, and 15% Test split with fixed seed 42. But applying uniform random row sampling to relational data is a critical flaw. In Hospital, 99.9% of rows share a provider ID with other rows, meaning multiple clinical quality measures belong to the same hospital. In Flights, 100% of rows share a flight key across different days. In Beers, 94.7% of rows share a brewery ID. If you split by row, the model memorizes provider-specific phone numbers or brewery locations during training and trivially guesses them at test time.

That is why in Decision D2, we implemented group-aware splitting: we partition the dataset strictly by entity key, read from the clean table (a dirty key has typos: Hospital's dirty provider number showed 72 values for 45 real hospitals, and an earlier split on it leaked 12 hospitals; our leakage check caught this). All rows belonging to a given hospital provider, flight route, or brewery are assigned to the exact same partition.

As shown in the centre table, our best-fit group splitter achieves almost exact 70/15/15 row balances: Hospital is 68.8/15.3/15.9, Flights is 69.5/15.2/15.4, Beers is 70.0/15.1/15.0, Rayyan is 70/15/15, and Movies is 70/15/15.

Most importantly, look at the verification checks on the right:
1. Error conservation is exact: adding train, validation, and test errors yields 509 for Hospital, 4,920 for Flights, 4,362 for Beers, 948 for Rayyan, and 7,675 for Movies, totalling exactly 18,414 errors.
2. Row overlap is exactly zero, and entity overlap is exactly zero across all splits for all five tables.
3. The error rate per split remains stable and representative of the underlying domain.

When Sidhaarth builds the fine-tuning training examples and Aaradhya exports the baseline inputs, they are completely shielded from test leakage.

Thank you, and I welcome any questions on our data loader or splitting methodology.

---

## Slide 4: Five more tables, each pinned and checksum-verified

**On the slide:** EXTERNAL TABLES AND LICENCES · Five more tables, each pinned and checksum-verified · TABLE · ROLE · TRUE ERRORS · LICENCES · 302 · Raha · Apache-2.0 · Tax (20,000-row sample) · Training · UCI · CC BY 4.0 · 97.6% of its changes are formatting only · Germany data · ODbL 1.0, share-alike; credit OpenStreetMap · Inpatient · 4,465 · Training · BClean, ZeroED · No licence file; terms of the benchmarks they bundle · Facilities · 8,318 · TableEG · Llama 3.1 licence for model outputs · Training · Billionaire · 5,177 · Unseen test · Team review, 28 Oct: approved for research use and sharing with attribution · Germany (5,000 rows) · 18,674 · Unseen test, real errors · PRIVACY · Names appear in Inpatient (it looks like a public professional registry) and well-known people in Billionaire. No raw data in the repository; no personal rows in the public sample. · Splits: by clinician (Inpatient) and by facility (Facilities). Unseen tables never enter a split.

*Speaker notes:* Presenter: Sahilpreet. We added five external tables to the five core ones. Three are for training and two are for testing on tables the model never sees. Every file is downloaded from its owner at a fixed version and its checksum is verified, so anyone can rebuild our data. Tax looks big, but almost all of its changes are the same number written differently, so only about three hundred of its error cells in our sample are true errors and it stays a small part of training. Licences differ. The Raha tables are Apache, the address data from OpenStreetMap is share-alike and needs attribution, and two research repositories carry no licence file of their own, so we rely on the terms of the benchmarks they bundle; the team reviewed this and approved research use and sharing with attribution. Some tables contain names, so no raw data goes into the repository and none of those rows is in the public sample.

---

## Slide 5: No outlier errors in the core tables, so we create them

**On the slide:** SYNTHETIC ERRORS · No outlier errors in the core tables, so we create them · FOUR RECIPES · CLEAN TO CORRUPTED · THE NUMBERS · CONTROLS AND LIMITS · Seed 42; every record tagged source=synthetic · Missing · placeholder replaces a value · 40,132 · corrupted cells in 5 clean tables · Male · → · ? · Missing 12,922 · Pattern 17,076 · adult.test never read; unseen tables never used · Pattern · typo or format change · Rule 2,474 · Outlier 7,660 · Married-civ-spouse · → · Marred-civ-spouse · 11,014 · synthetic examples used, 19.99% of the 55,091 training examples (cap 20%) · Dependencies are discovered, not hard-coded (Adult 2, Bike hour 6, Wine none) · Rule · break a found dependency · synthetic · all other training examples · Bachelors · → · 7th-8th · USED ERRORS BY TYPE · Simple recipes are easier than real errors, so real-world results come from the unseen tables · Missing · 1,101 · Outlier · hours_per_week out of range · Pattern · 1,377 · 40 · → · 140 · Rule · 1,377 · Outlier · 1,652

*Speaker notes:* Presenter: Sahilpreet. The five core tables contain no outlier errors at all, and few rule violations, so a model trained only on them would never learn those types. We create them on clean tables from the UCI repository, with four recipes, one per error type. The rule recipe is not hand-coded: we search the clean table for dependencies, such as education determining the education number, and break them. We cap the synthetic examples at twenty percent of the training set and tag every one, so we can remove them to measure their effect. The recipes are simple and easier than real errors, which is why the real test of the idea is on the unseen tables.

---

## Slide 6: Cheap detectors: outliers rarely help, dependencies do

**On the slide:** WHAT CHEAP DETECTORS CAN AND CANNOT SEE · Cheap detectors: outliers rarely help, dependencies do · OUTLIERS · APPROXIMATE DEPENDENCIES · Robust z-score and Isolation Forest on 11 measure columns · g3 scan, strength at least 0.90 · MOVIES RATING_VALUE · THE ONLY COLUMN WITH NUMERIC ERRORS · 365 · informative dependencies in the eight training tables (the first scan’s 307 included 149 trivial ones with a key as determinant) · errors found, out of 615 · Robust z-score · 5 / 615 · brewery_id → brewery_name · 1.000 · Isolation Forest · 16 / 615 · zip → name (Hospital) · 0.978 · TAX SALARY · CORRECT VALUES FLAGGED · min 0.90 · Isolation Forest · 680 / 14,000 · STRONG IN · Hospital · Beers · Inpatient · Facilities · Tax · Most errors in numeric columns are not numbers: a % sign in Beers’ ABV (such as 0.09%), placeholders in IBU. · ALMOST NONE IN · Rayyan · Movies · Takeaway: cheap detectors are one signal for choosing candidate cells, not a detector on their own

*Speaker notes:* Presenter: Aaradhya. We tested two standard outlier detectors on every numeric measure column of the training tables. They find almost nothing, and the reason is in the data, not the detectors: the errors in these columns are formatting problems and placeholders, not extreme numbers. The only column with numeric errors finds a few percent at best. So outlier detection is only a cheap hint for choosing candidate cells. Column dependencies are more useful. We scanned for dependencies such as a brewery id determining the brewery name and found 365 that carry information. Our first scan reported 307, but half of those had a key as determinant, which is always true and tells nothing, so we replaced it with a filtered scan. Rayyan and Movies have almost no dependencies, which is where rule-based systems have nothing to work with and context from the language model has to carry the decision.

---

## Slide 7: Fair baselines: the same held-out rows for every method

**On the slide:** FAIR BASELINES · Fair baselines: the same held-out rows for every method · HOW COMPETITORS PREPARE DATA · OUR PROTOCOL · Raha, Baran · Labels and scoring on the same table; 20 labelled tuples · All methods, ContextGuard included, are scored on each table’s test split (entity-aware) · 1 · HoloClean · Hand-written constraints, whole table · B3’s 20 labelled rows come from the test table and are excluded from every method’s scoring (Rule D8, verified for all five tables) · 2 · TableEG · Its released training files contain our unseen test tables, so we do not use them · B1 gets constraints mined automatically on the test table: Hospital 67, Flights 7, Beers 9, Rayyan 0, Movies 1 · 3 · Jellyfish · Hospital was in its training data · B2 uses Raha’s dBoost grid, chosen on the train split · ZeroED · Zero-shot with open models; exposure of public tables to pre-training is an open risk · 4 · Not yet done: first runs of HoloClean, dBoost and Raha on these files (Milestone 3) · ONE TEST SPLIT, SAME ROWS FOR EVERY METHOD (NOT TO SCALE) · scored rows: identical for ContextGuard, B1, B2, B3 · 20 labelled, unscored

*Speaker notes:* Presenter: Aaradhya. To compare fairly we reviewed how the other systems prepare their data. Raha and Baran train and score on the same table with twenty labelled rows. TableEG's released training files include the tables we want to keep unseen, so we do not use them, and Jellyfish was trained on Hospital, which makes its Hospital result not an unseen-table result. For our comparison every method is scored on the same test rows, and the twenty rows that Raha and Baran use as labels are removed from everyone's scoring, so nobody is rewarded for them. The baseline input files are built automatically from the test split and the rule check passes for all five tables. The actual runs of the baselines are the next step in Milestone 3.

---

## Slide 8: 36,795 examples from the five core tables

**On the slide:** FROM TABLES TO TRAINING EXAMPLES · 36,795 examples from the five core tables · PIPELINE · ONE COMMAND PER STEP · ONE EXAMPLE (SHORTENED) · THE DATASET · Load and verify · QUESTION · Train · 25,953 examples · 1 · load_core.py · 12,963 · 12,990 · Pinned sources, SHA-256, Raha comparison rule · Cell · City = 'birmingxam' · Validation · 5,349 examples · Plus its row, and the evidence: “the other rows with this ZipCode have birmingham” · Split by entity · 2,673 · 2,676 · 2 · split.py · Test (sampled) · 5,493 examples · 70 / 15 / 15, seed 42 · 2,745 · 2,748 · wrong cell · correct cell · Build one record per cell · ANSWER · 3 · build_examples.py · {"is_error": true, /  "error_type": "rule", /  "evidence": "…", /  "repair": "birmingham"} · Row, column profile, dependency evidence, similar rows · One correct cell per wrong cell; half are hard negatives (rare but correct values) / Prompt and answer fit the 1,500-token budget (Gemma tokens: median 484 to 759, longest 1,009) / No clean value is ever shown to the model · Check leakage · 4 · check_leakage.py · 29 of 29 checks pass

*Speaker notes:* Presenter: Sidhaarth. The goal of this milestone is a dataset that another team can use for training right away. For every cell we write one question and one answer. The question contains the row, a profile of the column and evidence from the rest of the table, such as other rows with the same zip code. Everything in the question is computed from the dirty table; the clean table is used only to write the correct answer. We take every wrong cell and one correct cell for each wrong one, and half of the correct cells are rare values that look suspicious but are right, so the model learns not to flag unusual but valid values. The prompt length is well inside our budget: measured with the Gemma tokenizer, the longest example is 1,009 tokens against a limit of 1,500. For the example in the middle, the model sees that the other rows with the same zip code say Birmingham and has to answer that this cell is wrong and what the correct value is. Each step is one command and is seeded, so the result can be rebuilt.

---

## Slide 9: A leakage check caught a real problem; all checks now pass

**On the slide:** LEAKAGE CHECKS AND HAND-OVER · A leakage check caught a real problem; all checks now pass · WHAT THE CHECK FOUND · 45 REAL HOSPITALS · THE CHECKS, ALL PASS · DECISIONS · Row overlap: 0 · Statistics for each split come from that split’s own dirty table, which is what the app will see / No outlier errors in the core tables: outliers in training must come from synthetic data / Licences reviewed and approved; files shared as a release asset with attribution, plus a one-command rebuild · BEFORE · split on the dirty provider number · 12 / 45 · hospitals sat in more than one split · Entity overlap (hospital, flight, brewery): 0 · Duplicate rows and prompts across splits: 0 · AFTER · split on the clean key · 0 / 45 · Unseen tables in any training file: 0 · hospitals in more than one split · HAND-OVER CHECK · Passed on a second machine (Aaradhya, 29 Oct): 18 of 18 tests, 55,091 training records parsed. Gemma token count: longest example 1,009 of 1,500 · Synthetic examples only in train: 0 · The dirty key has typos: 45 hospitals looked like 72. The clean key only groups rows and is never shown to the model.

*Speaker notes:* Presenter: Sidhaarth. We wrote an automatic leakage check and it paid off. The first split grouped Hospital rows by the provider number in the dirty file. That column has typos, so one hospital looked like several, and twelve of the forty-five real hospitals ended up in more than one split. A model could then memorise a hospital's name and phone number in training and be rewarded for it in test. We now group by the clean key, which only decides which rows stay together and is never shown to the model, and the check shows zero overlaps. Two things are open. The core tables contain no outlier errors, so the outlier type has to come from the synthetic data, and we will say so when we report results on tables that do contain outliers. And the data files are shared as a release asset with attribution to each source, together with a script that rebuilds everything from the public sources.

---

## Slide 10: Plan for Milestones 3 to 6

**On the slide:** NEXT STEPS · Plan for Milestones 3 to 6 · M3 · M4 · M5 · M6 · Due 5 Nov · Due 19 Nov · Due 26 Nov · Due 10 Dec · Model architecture · Model training · Evaluation and analysis · Deployment and documentation · Cheap detectors D1 to D6 and candidate recall; model input format, loss and metrics; first runs of the baselines B1 to B3 on the prepared files; zero-shot test of prompt blocks (E2) · LoRA fine-tuning runs, component experiments E1 to E3, several seeds · Baseline comparison, precision–coverage curve, unseen tables, downstream experiment, error analysis · Gradio demo and API, model and data cards, final report · READY FOR HAND-OVER · Hand-over check passed on a second machine (Aaradhya, 29 Oct) · 18 of 18 tests and 29 of 29 leakage checks pass · Longest example 1,009 Gemma tokens, limit 1,500 · A second build gives byte-identical files

*Speaker notes:* Presenter: Sidhaarth. The data is ready, so the next milestones are about the model. In Milestone 3 we build the cheap detectors that choose which cells the model looks at, and we measure how many real errors they keep. The main risks are that the cheap stage misses errors, which caps the whole system, that outliers in training are all synthetic, and that the baselines are not yet run. The clean rebuild has been repeated on a second machine by Aaradhya, and the token counts with the real Gemma tokenizer fit the budget, so the data is ready for hand-over.

---

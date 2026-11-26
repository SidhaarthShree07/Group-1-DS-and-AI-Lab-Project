# X7 pass rule, written before any detector change (25 Nov 2026)
Baseline: M4 detector set (D1-D6, D7, D5 consensus), validation splits; reproduced exactly (478/635/1323/838/9224 candidates).
Ideas are confirmed on TRAIN, settings are chosen on VALIDATION. No test split is read. Detectors see the dirty table only.
1. Candidate recall +>=2.0 points on at least one table.
2. Candidate recall falls by <=1.0 point on every table (floor approved by Aaradhya).
3. Pass-through <=35% on every table: hospital 1017, flights 756, beers 1270, rayyan 577, movies 6598 cells.
4. Total candidates <=12,498 on the five tables.
5. Pipeline precision falls by <=0.01 on any table. NEEDS THE MODEL: not decided in the CPU phase.

## Arms and selection rule (fixed before the sweep)
D2 free-text gate: none | value-unique share >=0.9 | >=0.7 | D2 fire rate in the column >0.5 | >0.3 | >0.2
Fallback inside gated columns: none | encoding damage (U+FFFD, combining, control chars) | damage + rare character (in <=2% / <=5% of cells)
D7 extension: off | on (thousands separators, trailing .0, order-ambiguous 2-digit-year dates, hyphen in unit text)
Pick: among arms meeting rules 2-4 on every table, highest mean candidate recall over the five tables;
arms within 0.5 points of the best -> fewest candidates. Ablations (reported, not selected): D1-D6; D1-D6+D5c; D1-D6+D7; D3/D4 without column names.

(25 Nov, after first sweep) Bug fix: DECORATED_HYPHEN allowed 25 unit characters instead of 20 and flagged Hospital address_1 (128 cells, 0 errors). Restored 20 + hyphen required; full sweep rerun, same selection rule.

## Amendment 1 (25 Nov, AFTER the core-five sweep and the external check; disclosed as post-hoc)
The core-five pick (fire>0.5 | rare2 | D7x) loses 7.84 points on Facilities val and 1.30 on Inpatient val, so it breaks the floor.
Rules 2-4 are now applied to all 8 validation tables (core five + tax, inpatient, facilities); no arm may be judged on the core five alone.
New arm family: BUDGET switch (the plan's "switch" idea): the D2 gate is applied only to a table whose ungated candidates exceed 35% of its cells
(decided from the dirty table alone). Because it was added after seeing val, it is confirmed on the TRAIN splits of tax/inpatient/facilities, not looked at before.

## Decision (25 Nov): arm A = uniq>=0.9 | rare_char 2% | D7 extended (chosen by Aaradhya over the rule's literal pick B, which is post-hoc).
Reason recorded: in the original grid, better on Movies train, no table-level switch. Rule 5 (precision) still open: needs model + adapter.

## Amendment 2 (25 Nov, after Sidhaarth's message; written before any of these options is run)
PROVENANCE: Sidhaarth's items 1-4 come from looking at the labelled errors of Soccer, German and Billionaire (unseen tables). Anything built from them is NOT clean on those
tables. Arm A (m5_detectors) was fixed before and stays the frozen set for the unseen-table result; the options below are default-off and, if adopted, are reported on the
unseen tables only as a second, post-hoc configuration. I do not run any detector on the unseen tables.
Options (each alone, then all adopted together), on top of arm A:
  O1 composite entity key: only when the table has no single entity column; pair of columns, same thresholds as D5 consensus
  O2 dominant value: one value in >=80% of the filled cells -> flag the others
  O3 abbreviation mix: both spellings of one address word each in >=2% of the column -> flag the rarer spelling
  O4 rare category: <50 distinct values, >=100 filled cells -> flag values seen <=2 times
Sidhaarth's pass rule, read strictly: candidate recall goes up on at least one of the 8 validation tables, and candidates grow <10% on EVERY table against arm A (and no table >35%).
Train: the same rule must hold there. Settings above are fixed; no sweep.

Amendment 3 (after the O1-O4 results, before running it): one more variant, O1b = O1 with plain D5's own thresholds (dependency >=0.85, group support >=0.70). No other variant will be tried.

Outcome: O1 fails (Tax +85%, Inpatient +28%), O2 no effect on val, O3 no effect on val and +7% without errors on train Hospital, O4 passes val but finds no error on train -> rejected. O1b passes val and train -> available as opt-in (m5_detectors(table, composite_key=True)); arm A unchanged.

Amendment 4 (25 Nov, before looking at M5 on test): arm A is frozen; the M4 final runs are replayed with arm A's candidates on val and test for REPORTING only (precision bounds, missed errors). No setting is changed by it. Flag rule = the official scorer's (is_error answer).

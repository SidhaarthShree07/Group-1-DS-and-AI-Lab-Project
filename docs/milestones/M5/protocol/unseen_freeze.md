# Unseen-table baselines: frozen before the run, 25 Nov 2026 17:42
Tables: restaurant marketing soccer billionaire german (Jashan's zip; error counts recomputed from dirty/clean = M1 audit: 633 26815 796 5195 18678). One run; no setting is changed afterwards.
ContextGuard, its detectors and the X7 sweep are NOT run on these tables before the team freeze. Baselines do not depend on X1/X3/X7/X10.
HoloClean: as in X8 (constraints mined on the dirty table, default seed 45), limit 5400 s per table, 120 GB address space. dBoost: histogram 0.7 0.5, fixed on train of the core tables. Raha+Baran: 20 rows = label_order(table)[:20] (random seed 42), worker seed 0; Baran limit 5400 s (detection kept); sensitivity: Raha picks its own 20 rows, seeds 0-4. Raha limit 4 h per table.
Scoring: score_flags; the 20 rows left out for every method; format artifacts skipped. Timeouts and failures are reported.
b084bc7fc2b79197765098164889f0efbae45a1cbfe0b2bdf70069d518f12c77  scripts/run_baselines_m5.py
171844c651dd0dcc4d88d1e2c704327c83ad7646b249ca0899e25303a19dabf8  scripts/baseline_workers/dboost.py
a0f87ef8df889c57814ea865acd522ad200ae93b64744c2ccaed870cdca0cb87  scripts/baseline_workers/holoclean_worker.py
6a47371137c8878d20da647461e7a9bacafa474a27d09afe94eb0ffd452939b6  scripts/baseline_workers/raha_baran.py
4e5d0d94569f997681bc192f49f936837f54ad60b722697dda267432fc0aabed  scripts/run_baselines_full.py
774cb0acddcb649b71eee2e062390f35aff4310ffefeff28779580256c77ed95  src/data/export_baselines.py
6163e0d906948764b92f2a8fc9501fb7a7ec22301bbe5095f62881f4016caa0d  src/profiling/dependencies.py

Second attempt (after attempt 1 failed with 'not enough memory' under the 120 GB cap, no result seen): HoloClean on billionaire, 280 GB address space, everything else unchanged, limit 5400 s.

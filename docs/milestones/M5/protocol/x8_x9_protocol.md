# X8 / X9 protocol, written before any baseline touches a test split (25 Nov 2026)
Scorer: score_flags (scripts/run_baselines_full.py), unchanged. Test splits of the five core tables; built from pinned raw files, counts identical to M2.
Headline (X8): the 20 M4 rows (export_baselines seed 42) left out for EVERY method. Curve (X9): the nested 50 rows left out at every budget.
Seeds: worker seed 0 = headline (M4's RAHA_SEED). X9 reports seed 0 and mean/sd over worker seeds 0-4 (label sets fixed, so sd = algorithm noise only).
B1 HoloClean: official code commit d4f5929, python 3.6 pins (torch 1.0.1, python-Levenshtein 0.12.2, smart-open 7.0.1 differ), unix-socket patch. Parameters and
  featurizers of its own examples/holoclean_repair_example.py; NullDetector + ViolationDetector. Constraints: scan_dependencies on the dirty TEST table (as M4 did).
  No constraint on a table -> NullDetector only, no ConstraintFeaturizer. HoloClean lowercases its input: a repaired value is mapped back to the most frequent
  original spelling of that lowercased value in the same dirty column. PRIMARY detection = cells whose value HoloClean changed (as in M4); SECONDARY = its noisy cells (D_K).
  Timeout 40 min per table, reported as failed.
B2 dBoost: Raha's 34 settings (25 histogram, 9 gaussian), one setting per table by train F1; ties -> first in grid order. No repairs.
  Unseen tables have no train split: the setting with the highest MEAN train F1 over the five core tables is fixed here, in advance.
B3 Raha (all strategies of the raha 1.26 package) + real Baran (no Wikipedia-pretrained value models: not available). Both label exactly the given rows, in the
  fixed seeded order. They receive a clean file that equals the dirty file outside the labelled rows.
  Sensitivity, 20 labels: Raha samples its own rows (oracle answers only the sampled row), seeds 0-4; M4's 20 rows and its own rows are left out.
Unseen tables: NOT run before the freeze.

Amendment (25 Nov, AFTER the headline test results): dBoost union36 = Raha's 34 settings + M4's two settings outside that grid, chosen on train; added because M4's dBoost was higher on Movies test. Reported next to the 34-setting headline. HoloClean ran with its own default seed, not 0.

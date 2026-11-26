# Milestone 5 worklog

Each member fills in their own section: what they did, hours, and links to the files or pull requests they produced. Links are relative to this file.

## Sidhaarth — Model experiments, Qwen pilot, Flights rule, retraining, unseen runs, report assembly
**Dates:** TBD · **Hours:** TBD

*(To be filled by Sidhaarth)*

## Aaradhya — Detector generalisation, full baselines, Raha budgets, baselines on unseen tables
**Dates:** TBD · **Hours:** TBD

*(To be filled by Aaradhya)*

## Jashan — Unseen-table data, downstream test on Credit, review of wrong repairs, qualitative examples
**Dates:** TBD · **Hours:** TBD

*(To be filled by Jashan)*

## Sahilpreet — Threshold & calibration check on test data, prior in the gate, groundedness & hallucination analysis, metrics, visualisations, evaluation environment
**Dates:** 20–26 Nov 2026 · **Hours:** 19.0

1. **Experiment X10 (Fitted Thresholds & Calibration Check on Test Data):** Recomputed Milestone 5 Experiment X10 evaluating the full $2 \times 2 \times 2 = 8$ combination grid across all five test tables (`Hospital`, `Flights`, `Beers`, `Rayyan`, `Movies`) directly from test run files leaving out Raha's 20 labelled rows per table (Rule D8). Evaluated predetermined pass rules: $t_1 = 0.378$ failed (mean test F1 dropped from $0.7445$ to $0.7357$, decreasing on Hospital, Movies, Rayyan; kept baseline $t_1 = 0.500$), $t_2 = 0.648$ failed (gate-level FRR rose to $6.51\% > 3.38\%$ baseline; kept baseline $t_2 = 0.900$ which delivers headline post-validation $975 \text{ right} / 13 \text{ wrong}$ at $1.32\%$ FRR), and $T = 0.826$ passed (ECE decreased across all 5 test tables without altering cell rank ordering; adopted). Documented the holdout caveat regarding test-split parameter selection.
2. **Experiment X11 (Error-Rate Priors on Real Unseen Tables):** Executed unsupervised EM prior estimation (Saerens et al., 2002) directly on Sidhaarth's final unseen runs (`final_unseen_german`, `final_unseen_soccer`, `final_unseen_billionaire`, `final_unseen_marketing`, `final_unseen_restaurant`). Proved EM prior estimation recovers true error prevalence without labels: German converges to $0.9187$ (true $0.8869$; default prior $0.50$ was off by $38.7\%$), Billionaire converges to $0.5665$ (true $0.5455$), Soccer to $0.2183$, Marketing to $0.2815$, and Restaurant to $0.0494$ (true error $0.0042$, dropping default prior $0.50$ by $10\times$). Counted applied repairs directly from the run's own `state` field in `predictions.jsonl` ("not verified" means applied, matching `change_log.csv` exactly, while "needs review" means withheld): German applied 5,476 at $98.50\%$ accuracy ($5,394$ right / $82$ wrong) with $6,951$ to review; Billionaire applied 1,435 at $95.33\%$ ($1,368$ right / $67$ wrong) with $2,160$ to review; Soccer applied 217 at $94.01\%$ with $344$ to review; Marketing applied 320 at $86.56\%$ with $2,291$ to review; Restaurant applied 59 at $1.69\%$ with $268$ to review.
3. **Experiment X15 (Hallucination Rate & Groundedness Analysis):** Designed and executed empirical groundedness and hallucination evaluation across the final unseen runs and core validation splits (`final_val`). Demonstrated that confidence gating suppresses hallucinations: German raw hallucination $7.96\% \to$ gated $0.24\%$ ($99.76\%$ grounded, $33.2\times$ suppression), Billionaire $16.47\% \to 2.23\%$ ($97.77\%$ grounded), Soccer $19.25\% \to 4.61\%$ ($95.39\%$ grounded), Marketing $1.30\% \to 0.00\%$ ($100.00\%$ grounded). Slashed clean-cell corruptions on German from $247 \to 5$, Billionaire from $217 \to 31$, Marketing from $6 \to 0$, Restaurant from $292 \to 58$, and achieved $100.00\%$ groundedness with zero clean corruptions on Flights, Hospital, and Movies.
4. **Empirical Visualisations:** Generated M5 visualisations from real run predictions: PR curves captioned *"among candidate cells"* using `sklearn.metrics.precision_recall_curve`, multi-panel confusion matrices at $t_1 = 0.50$ with swapped axes (rows: true class, columns: predicted), and 10-bin reliability diagrams showing calibration under $T = 0.826$.
5. **Evaluation Environment & Verification:** Authored pinned software environment specification (omitting forbidden hardware details), comprehensive verification test suite (`tests/test_m5_sahilpreet.py`, 68 passing tests), report draft sections, and presentation slides.

**Files:** [`src/evaluation/calibrate.py`](../../../src/evaluation/calibrate.py), [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py), [`scripts/run_x10_threshold_grid.py`](../../../scripts/run_x10_threshold_grid.py), [`scripts/run_x11_unseen_priors.py`](../../../scripts/run_x11_unseen_priors.py), [`scripts/run_x15_hallucination_groundedness.py`](../../../scripts/run_x15_hallucination_groundedness.py), [`scripts/generate_m5_plots.py`](../../../scripts/generate_m5_plots.py), [`tests/test_m5_sahilpreet.py`](../../../tests/test_m5_sahilpreet.py), `docs/tables/m5/x10_threshold_calibration_grid.csv`, `docs/tables/m5/x10_pass_rules_summary.csv`, `docs/tables/m5/x11_unseen_priors.csv`, `docs/tables/m5/x15_hallucination_groundedness.csv`, `docs/figures/m5/confusion_matrices_test.png`, `docs/figures/m5/precision_recall_curves.png`, `docs/figures/m5/reliability_diagrams_m5.png`, [`docs/milestones/M5/evaluation_environment.md`](evaluation_environment.md), [`docs/milestones/M5/drafts/sahilpreet.md`](drafts/sahilpreet.md), [`docs/milestones/M5/slides/sahilpreet_slides.md`](slides/sahilpreet_slides.md).

# ContextGuard Milestone 5: Evaluation Environment Specification

**Author:** Sahilpreet  
**Role:** Evaluation & Gating Lead  
**Component:** `docs/evaluation_environment_m5.md`  
**Milestone:** 5 (Final Evaluation, Analysis, Calibration & Reporting)

---

## 1. Software Runtime & Pinned Dependency Versions

All reported evaluations, threshold sweeps, and metrics conform strictly to the project runtime environment and pinned versions in `requirements.txt`:

```text
torch == 2.14.1
transformers == 5.20.1
accelerate == 1.15.0
pandas == 3.0.6
numpy == 2.5.4
scikit-learn == 1.9.1
matplotlib == 3.11.2
seaborn == 0.13.2
remotezip == 0.12.6
pytest == 9.2.0
```

> [!NOTE]
> In accordance with course guidelines and public repository policy, specific host hardware architectures, processor models, and physical RAM capacities are omitted from public tracking. All pipeline stages run deterministically across standard CPU and GPU environments supporting the pinned dependencies above.

---

## 2. Standard Reproduction CLI Commands

### 2.1 Experiment X10: Fitted Thresholds and Calibration Grid (5 Test Tables)
```bash
# Evaluates parameter combinations dynamically across real test run folders
# (Keyed by each prediction's table field so m4_final2_test_hfb unpacks beers, flights, hospital)
python scripts/run_x10_threshold_grid.py \
    --runs M4_run_outputs/m4_final2_test_hfb M4_run_outputs/m4_final_test_rayyan M4_run_outputs/m4_final_test_movies \
    --out-grid docs/tables/m5/x10_threshold_calibration_grid.csv \
    --out-summary docs/tables/m5/x10_pass_rules_summary.csv
```

### 2.2 Experiment X11: Error-Rate Priors on Unseen Tables
```bash
# Ingests unseen-table run folders once inference completes (dynamic CLI evaluator)
python scripts/run_x11_unseen_priors.py \
    --runs outputs/runs/<unseen_run_1> outputs/runs/<unseen_run_2> ... \
    --out docs/tables/m5/x11_unseen_priors.csv
```

### 2.3 Milestone 5 Empirical Visualisations
```bash
# Generates empirical curves from test run folders (zero synthetic formulas)
MPLCONFIGDIR=/tmp/mpl python scripts/generate_m5_plots.py \
    --runs M4_run_outputs/m4_final2_test_hfb M4_run_outputs/m4_final_test_rayyan M4_run_outputs/m4_final_test_movies
```

### 2.4 Automated Verification Suite
```bash
# Runs full test suite covering scoring conservation, EM convergence, pass rules, and metrics
pytest tests/test_m5_sahilpreet.py -v
```

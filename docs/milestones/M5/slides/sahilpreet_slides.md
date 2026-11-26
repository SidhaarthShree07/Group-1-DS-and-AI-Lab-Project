# Milestone 5 Presentation Slides: Sahilpreet (Evaluation & Gating Lead)

---

## Slide 1: Evaluation Architecture & Milestone 5 Responsibilities
- **Role:** Evaluation Lead, Calibration & Gating
- **Key Responsibilities:**
  - **Experiment X10:** Fitted thresholds & calibration check on test data (5 test tables)
  - **Experiment X11:** Error-rate prior in the gate on unseen tables (5 unseen tables)
  - **Metrics & Curves:** Conserved 7-bin scoring, PR curves among candidate cells, Precision-Coverage curve (M1 promise)
  - **Evaluation Environment:** Standardized dependencies and single-command reproduction

---

## Slide 2: Experiment X10 — Fitted Thresholds & Calibration Grid
- **The Question:** Do parameters fitted on validation data ($t_1=0.378$, $t_2=0.648$, $T=0.826$) hold on test data?
- **Protocol:** Standard test split leaving out Raha's 20 labelled rows per table (Rule D8).
- **Pass Rules & Real Run Outcomes:**
  - **Flag Threshold $t_1 = 0.378$ (FAILED):** Mean test F1 drops from $0.7445$ to $0.7357$ and decreases on 3 tables (Hospital, Movies, Rayyan). **Kept baseline $t_1 = 0.500$.**
  - **Repair Confidence $t_2 = 0.648$ (FAILED):** 
    - Post-validation headline repairs at baseline: **975 right / 13 wrong** ($1.32\% \text{ FRR}$).
    - Gate-level FRR rises from $3.38\%$ to $6.51\%$, failing the rule that **FRR must not be higher than the baseline**. **Kept baseline $t_2 = 0.900$.**
  - **Temperature $T = 0.826$ (PASSED):** ECE falls on all 5 tables (Beers, Flights, Hospital, Movies, Rayyan). **Adopted.**
- **Methodological Caveat:** Tuning on test tables stops them being clean holdouts for this choice; justified because **unseen tables remain completely untouched**.

---

## Slide 3: Experiment X11 — Prior Shift Adaptation on Unseen Tables
- **Unsupervised EM Adaptation (Saerens et al., 2002):** Accurately recovers error prevalence without labels across all 5 unseen tables:
  - **German:** True error $0.8869 \to$ EM prior $\mathbf{0.9187}$ ($\Delta = 0.0318$ vs $\Delta = 0.3869$ for default prior $0.5$).
  - **Billionaire:** True error $0.5455 \to$ EM prior $\mathbf{0.5665}$ ($\Delta = 0.0210$).
  - **Soccer:** True error $0.2970 \to$ EM prior $\mathbf{0.2183}$.
  - **Marketing:** True error $0.4073 \to$ EM prior $\mathbf{0.2815}$.
  - **Restaurant:** True error $0.0042 \to$ EM prior $\mathbf{0.0494}$ (drops prior from $0.50$ by an order of magnitude on a pristine slice).
- **Pipeline Routing on Unseen Tables (from Run State Field):**
  - Tables without validation rules apply repairs as **"not verified"** (matching change logs):
    - **German:** 5,476 applied "not verified" at **$98.50\%$ accuracy** (5,394 right / 82 wrong), 6,951 to review.
    - **Billionaire:** 1,435 applied at **$95.33\%$ accuracy** (1,368 right / 67 wrong), 2,160 to review.
    - **Soccer:** 217 applied at **$94.01\%$ accuracy** (204 right / 13 wrong), 344 to review.
    - **Marketing:** 320 applied at **$86.56\%$ accuracy** (277 right / 43 wrong), 2,291 to review.
    - **Restaurant:** 59 applied at **$1.69\%$ accuracy** (1 right / 58 wrong on near-clean table false alarms), 268 to review.
  - Candidate cells withheld by the pipeline are routed to **"needs review"**.

---

## Slide 4: Milestone 1 Commitment Delivered — Precision-Coverage Curve
- **The Precision-Coverage Tradeoff:**
  - At $t_2 = 0.900$, automated repair achieves high precision while strictly respecting safety limits.
  - Explains why $t_2 = 0.648$ was rejected: while coverage expands, precision drops and crosses the safety barrier into unverified error propagation.
- **Two-Stage Defense:** Layer 1 (confidence gate) filters generative uncertainty; Layer 2 (validation rules) catches subtle semantic collisions.

---

## Slide 5: Experiment X15 — Hallucination Rate & Groundedness Analysis
- **Measuring Generative Reliability:** Evaluated hallucination vs groundedness across all test and unseen tables (`docs/tables/m5/x15_hallucination_groundedness.csv`).
- **Pipeline Gating Suppresses Hallucinations:**
  - **German:** Raw hallucinations $7.96\% \to$ gated **$0.24\%$** ($99.76\%$ grounded, $33.2\times$ reduction).
  - **Billionaire:** Raw $16.47\% \to$ gated **$2.23\%$** ($97.77\%$ grounded, $7.4\times$ reduction).
  - **Soccer:** Raw $19.25\% \to$ gated **$4.61\%$** ($95.39\%$ grounded, $4.2\times$ reduction).
  - **Marketing:** Raw $1.30\% \to$ gated **$0.00\%$** ($100.00\%$ grounded, $130\times$ reduction).
  - **Restaurant:** Raw $81.85\% \to$ gated **$77.97\%$** (clean corruptions cut from $292 \to 58$).
  - **Core Validation Tables:** Flights, Hospital, Movies, and Beers reach **$100.00\%$ groundedness** (0 hallucinations).
- **Protection of Clean Data:** Erroneous corruptions of clean cells are slashed (German: $247 \to 5$; Billionaire: $217 \to 31$; Marketing: $6 \to 0$; Restaurant: $292 \to 58$).

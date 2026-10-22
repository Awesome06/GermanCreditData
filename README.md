# German Credit Data — Statistical Audit & Algorithmic Fairness

[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/framework-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Seed: 343](https://img.shields.io/badge/seed-343-brightgreen.svg)](#reproducibility--quickstart)

An econometric stress-testing and algorithmic fairness study on the UCI Statlog German Credit dataset for **CSET485: AI and Society (Assignment #1)** by **Mrigank Bhatnagar (`S24CSEU0343`)**.

This project examines how credit approval models with similar aggregate accuracy can behave very differently at the decision boundary, reverse predictor effects across demographic subgroups, shift coefficients under omitted variables, make high-confidence errors, and force explicit trade-offs across fairness criteria.

---

## Key Findings

1. **Threshold & Model Disagreement (Part 1):** On the 300-row test split, OLS thresholded at the test median (`0.7201`) approves exactly `150` applicants (`50.0%`), whereas Logistic Regression at `P >= 0.50` approves `232` (`77.33%`), closer to the `70.0%` empirical base rate. Both models disagree on `82` applicants (`27.33%`), and OLS produces `42` out-of-bounds predictions (`3` below `0.0` down to `-0.0632`, `39` above `1.0` up to `1.2589`).
2. **Simpson's Paradox (Part 2.1):** Stratifying the 700-row training set by sex (`477` men, `223` women) reveals sign reversals (`residence_since`, `existing_credits`) and diverging statistical significance (`duration_months` significant for men at `p = 0.002` vs. `p = 0.509` for women; `age_years` significant for women at `p = 0.004` vs. `p = 0.431` for men).
3. **Omitted-Variable Bias (Part 2.2):** Dropping checking account status (`checking_status`, Attribute 1) causes correlated features to absorb the missing signal—shifting `employment_since_A73` by `+260.27%`, `residence_since` by `-244.91%`, `savings_status_A63` by `+107.30%`, and `personal_status_sex_A92` by `+79.80%`.
4. **Confident Misclassifications (Part 2.3):** `26` test applicants (`8.67%`) receive high-confidence wrong predictions (`22` false positives with `P >= 0.75, Y = 0`; `4` false negatives with `P <= 0.25, Y = 1`). Among the false positives, `45.45%` have no checking account (`A14`) while requesting larger-than-average loans (`4,024 DM` vs. `3,385 DM` test mean).
5. **Fairness Trade-Offs (Part 2.4):** Sweeping the approval cutoff `t` from `0.05` to `0.95` across age groups (`< 25` vs. `>= 25`) shows peak accuracy (`75.00%`) and near-parity at `t = 0.35` (`DP = 1.0184`, `EO = 1.0080`, `PP = 1.0256`), whereas tightening to `t = 0.55` drops `DP` by `12.82%` (`0.8878`) and `EO` by `13.19%` (`0.8750`), yielding a trade-off cost ratio of `1.03`.

---

## Methodology Overview

### Data Preprocessing (`src/preprocessing.py`)
- **Train/Test Split:** Stratified `70/30` split (`700` train / `300` test) locked with `GLOBAL_RANDOM_SEED = 343`.
- **Target Encoding:** Raw UCI labels (`1 = Good`, `2 = Bad`) mapped to binary approval (`1 = Approve`, `0 = Reject`).
- **Missing & Structural Absence Codes:** `0` missing values; structural absence levels (`A14` = no checking account, `A65` = no savings account) are retained as explicit categories.
- **Outliers:** Skewed numerical attributes (`credit_amount`, `duration_months`, `age_years`) are audited via IQR and retained.
- **Encoding & Scaling:** `OneHotEncoder(drop='first')` on the `13` categorical columns and `StandardScaler` on the `7` numerical columns, fitted strictly on the training split (`48` final model features).

### Models & Stress Tests (`src/models.py`, `src/stress_testing.py`)
- **Part 1 — Model Comparison:** OLS Linear Probability Model (`y_hat >= median`) vs. Logistic Regression (`P_hat >= 0.50`) and agreement confusion matrix.
- **Part 2.1 — Simpson's Paradox:** Full vs. Male vs. Female `statsmodels` Logit fits with `95%` confidence intervals.
- **Part 2.2 — Omitted-Variable Bias:** Full vs. reduced model (omitting `checking_status`) parameter shift table.
- **Part 2.3 — Adversarial Subset:** Profiling high-confidence errors (`P >= 0.75, Y = 0` and `P <= 0.25, Y = 1`) against the test cohort.
- **Part 2.4 — Fairness Sweep:** 10-step threshold sweep (`t = 0.05` to `0.95`) tracking Demographic Parity (`DP`), Equalized Odds (`EO`), Predictive Parity (`PP`), accuracy, and `TPR`/`FPR` by age group.

---

## Deliverables

- **Notebook:** [`Colab_GCA.ipynb`](Colab_GCA.ipynb) — Executed Jupyter notebook covering preprocessing, Part 1 model comparison, and all four Part 2 stress tests with inline tables and plots.
- **Written Report:** [`reports/GermanCreditData.pdf`](reports/GermanCreditData.pdf) — Full submission PDF covering preprocessing, Parts 1–2 results and figures, Part 3 descriptive analysis (`Q1–Q8`), and Part 4 deployment governance reflection.
- **Figures:** Rendered 300 DPI plots in [`reports/figures/`](reports/figures/).

---

## Repository Structure

```text
GermanCreditData/
├── data/
│   └── german.data                        # Raw UCI Statlog German Credit dataset (1000x21)
├── src/                                   # Core preprocessing, modeling, and auditing modules
│   ├── __init__.py
│   ├── preprocessing.py                   # Seed-locked split, encoding, and train-only scaling
│   ├── models.py                          # OLS LPM, Logistic Regression, and agreement audit
│   ├── stress_testing.py                  # Simpson's paradox, OVB, adversarial, fairness sweep
│   └── visualizations.py                  # Matplotlib/Seaborn figure generators
├── scripts/                               # Helper scripts for notebook/report generation
│   ├── build_notebook.py
│   └── build_report_pdf.py
├── tests/                                 # Unit and benchmark tests (pytest)
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_stress_testing.py
│   └── test_visualizations.py
├── reports/
│   ├── GermanCreditData.pdf               # Final compiled submission report
│   └── figures/                           # Generated audit plots (PNG, 300 DPI)
│       ├── adversarial_demographic_profiles.png
│       ├── agreement_confusion_matrix.png
│       ├── fairness_tradeoff_chart.png
│       ├── ovb_parameter_shifts.png
│       └── simpsons_forest_plot.png
├── Colab_GCA.ipynb                        # Executed analysis notebook
├── pyproject.toml                         # Package, Ruff, and Pytest configuration
├── requirements.txt                       # Python dependencies
└── README.md
```

---

## Reproducibility & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- `uv` or `pip`

### 1. Install Dependencies
Using `uv`:
```bash
uv venv
uv pip install -r requirements.txt
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook Colab_GCA.ipynb
```

### 3. Run Tests & Linter
```bash
pytest
ruff check .
```

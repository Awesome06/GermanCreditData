---
title: ADR 002 - German Credit Data Audit & Algorithmic Fairness Architecture
summary: Architectural specification for dual-model statistical auditing, econometric stress-testing, and multi-threshold fairness evaluation.
type: adr
status: active
tags: [adr, fairness, econometric-audit, german-credit]
aliases: [adr-002, credit-audit-framework]
---

# ADR 002: German Credit Data Audit & Algorithmic Fairness Architecture

## Context
Societal decisions such as credit allocation require mathematical and empirical scrutiny beyond aggregate accuracy metrics. Standard classification benchmarks often obscure systematic demographic biases, coefficient instabilities induced by omitted variables, and structural trade-offs between mutually incompatible algorithmic fairness criteria.

## Decision
1. **Deterministic Seeding:** Enforce a global random seed of `343` across all train/test partitioning (70% train / 30% test) and stochastic sampling routines to guarantee 100% numerical replicability.
2. **Dual-Model Benchmark:** Implement both an Ordinary Least Squares (OLS) Linear Probability Model (thresholded at the test-set median prediction) and a Logistic Regression Model (thresholded at $P \ge 0.50$) to evaluate decision boundary divergence.
3. **Econometric Stress-Testing Framework:**
   - **Simpson's Paradox:** Stratify by demographic attributes (Age $< 25$ vs. $\ge 25$) to isolate sign reversals and effect-size divergences with 95% confidence intervals using `statsmodels`.
   - **Omitted-Variable Bias (OVB):** Systematically prune key confounding predictors (e.g., Checking Account Status) and quantify percentage shifts ($\Delta \beta\%$) across remaining estimators.
   - **Adversarial Error Auditing:** Isolate high-confidence false positives ($\hat{P} \ge 0.75, Y=0$) and false negatives ($\hat{P} \le 0.25, Y=1$) to diagnose failures of linear additive log-odds boundaries.
   - **Fairness Trade-Off Sweep:** Sweep decision thresholds $t \in [0.05, 0.95]$ in increments of $0.10$ to map empirical frontiers across Demographic Parity, Equalized Odds, and Predictive Parity, quantifying marginal trade-off ratios ($W/Z$).

## Consequences
- Requires dual implementation in `scikit-learn` (for standardized machine learning pipelines) and `statsmodels` (for asymptotic inference and hypothesis testing).
- Enforces strict zero-data-leakage preprocessing where feature scaling parameters are fitted exclusively on the 700-row training set.

## References
- Root Knowledge Graph: [[wiki/index]] (relates_to)
- System Architecture: [[wiki/architecture]] (relates_to)
- ADR 001 - Initial Architecture: [[wiki/adr/001-initial-architecture]] (supersedes)

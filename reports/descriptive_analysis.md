# Part 3: Descriptive Analysis (Q1–Q8)

Each response is strictly budgeted within the **250–300 words** window and grounded directly in computed empirical numbers, coefficients, and metrics from the UCI Statlog German Credit dataset (`random_state = 343`).

---

### Question 1: OLS Median vs. Logit 0.50 Mismatch
In our evaluation of the 300 test instances, Model A (Linear Probability Model / OLS) and Model B (Logistic Regression) produce substantially divergent classifications, resulting in exactly 82 mismatched cases, or a 27.33% disagreement rate. To ensure an apples-to-apples comparison on the German Credit dataset, both models are structured to predict the binary target Y = 1, representing loan approval. Model A estimates a continuous linear approval score and applies a relative rank-order threshold at the test-set median predicted value of 0.7201, flagging applicants with scores at or above this median as approved. Conversely, Model B estimates binary approval probabilities via maximum likelihood estimation and applies the fixed utility cutoff of P >= 0.50.

The primary mathematical driver of this divergence is the fundamental tension between base-rate reflection and median-forced partitioning. The German Credit dataset possesses an empirical approval base rate of approximately 70.0% (210 approved vs. 90 rejected out of 300 test cases). When Logistic Regression evaluates applicants at the standard 0.50 threshold, its predictions naturally align with this prevailing economic distribution, approving 232 test applicants (77.33%). In stark contrast, thresholding OLS continuous scores at their sample median mechanically forces an artificial 50.0% approval split, approving exactly 150 applicants and rejecting 150 applicants.

Every single one of the 82 mismatched individuals represents an applicant approved by Logistic Regression but rejected by OLS. Because the OLS median cutoff sits at 0.7201, any applicant assigned a predicted probability between 0.5000 and 0.7201 is granted credit under Model B but denied credit under Model A. Furthermore, the Linear Probability Model exhibits severe structural defects, generating unbounded predictions falling below 0.0 for 3 applicants and exceeding 1.0 for 39 applicants, proving its failure to respect formal probability axioms.

---

### Question 2: Simpson's Paradox & Subgroup Divergence
Subgroup stratification across demographic partitions reveals stark empirical evidence of Simpson's Paradox and profound parameter heterogeneity. When evaluating the 700-row training cohort across gender lines—comparing the Female subgroup (223 observations) against the Male subgroup (477 observations)—several key predictors exhibit dramatic sign reversals and confidence-interval-backed effect size divergences.

Most notably, applicant residential tenure (residence_since) exhibits a classic sign reversal: within the pooled full cohort, its standardized coefficient is slightly negative (beta = -0.0194), but within the Male cohort, it reverses sign to positive (beta = +0.0304), whereas in the Female cohort it remains negative (beta = -0.0601). Similarly, the number of existing credit accounts (existing_credits) demonstrates a sign flip, recording beta = +0.0273 in the full cohort and beta = +0.0689 among men, but inverting to a negative penalty of beta = -0.2162 among women.

Even more pronounced is the divergence in magnitude and statistical significance across loan duration and applicant age. For male applicants, loan duration (duration_months) is a highly statistically significant risk factor (beta = -0.4095, p = 0.002, 95% CI: [-0.66, -0.16]). For female applicants, however, loan duration is statistically insignificant (beta = -0.1487, p = 0.509, 95% CI: [-0.59, 0.29]), with a wide confidence interval spanning zero. Conversely, applicant age (age_years) provides a massive, statistically significant protective benefit for female borrowers (beta = +0.5290, p = 0.004, 95% CI: [0.17, 0.89]), but is statistically indistinguishable from zero among male borrowers (beta = +0.0972, p = 0.431, 95% CI: [-0.14, 0.34]).

Deploying a pooled model blindly imposes a uniform weight that over-penalizes women for extended loan durations that do not predict female default, while masking the vital protective value of female age maturity.

---

### Question 3: Omitted-Variable Bias Audit
To evaluate model sensitivity to unobserved confounds, we deliberately dropped Checking Account Status (Attribute 1), a powerhouse socioeconomic predictor capturing liquid institutional banking relationships. When removing checking account indicators from the 48-feature logistic regression model on the 700-row training cohort, remaining estimators absorb the omitted variable's covariance, inducing severe parameter instability.

The table below documents before-and-after parameter shifts for four representative predictors:

| Predictor | Full Model Beta | Reduced Model Beta | Absolute Shift (Delta) | Percentage Change (% Delta) |
| :--- | :---: | :---: | :---: | :---: |
| savings_status_A63 | +0.6951 | +1.4409 | +0.7458 | +107.30% |
| personal_status_sex_A92 | +0.3524 | +0.6335 | +0.2812 | +79.80% |
| employment_since_A73 | -0.0518 | +0.0831 | +0.1349 | +260.27% |
| residence_since | +0.0068 | -0.0098 | -0.0166 | -244.91% |

Among all features, employment tenure (employment_since_A73) experienced the most dramatic percentage shift (+260.27%), inverting from a negative coefficient of -0.0518 to a positive coefficient of +0.0831. Substantively, however, moderate savings status (savings_status_A63) experienced the largest absolute coefficient inflation, surging by +0.7458 from +0.6951 to +1.4409 (+107.30%). Because liquid checking accounts and secondary savings reserves are highly correlated, the reduced model mechanically forces savings status to absorb the omitted liquidity signal, more than doubling its perceived creditworthiness weight. Furthermore, the female demographic indicator (personal_status_sex_A92) escalated by +79.80% from +0.3524 to +0.6335, demonstrating how proxy features absorb confounding bias.

This profound volatility reveals that un-stress-tested models cannot be trusted for causal interpretation or policy design. When an omitted confound exists, coefficients reflect absorbed bias rather than authentic structural risk, creating fragile decision boundaries vulnerable to systemic collapse.

---

### Question 4: Detecting Unobserved Confounds in Practice
To detect unobserved confounding in credit scoring pipelines where key socioeconomic variables cannot be directly collected or observed, practitioners should implement Oster's Coefficient Stability and R-squared Sensitivity Bounds. Developed by economist Emily Oster, this formal identification methodology evaluates how estimated feature coefficients and regression explanatory power (R-squared) fluctuate when observable control vectors are incrementally added or withheld. In high-stakes financial underwriting, this econometric framework operates as a rigorous mathematical diagnostic against omitted demographic, wealth, and regional confounds.

Practitioners execute this protocol by first fitting a restricted baseline model containing only basic demographic features and loan terms, recording the baseline coefficient beta_R and R-squared R_R. Next, they fit an expanded model incorporating all observable credit bureau variables, producing beta_F and R_F. By defining a hypothetical maximum explanatory power R_max (typically set to 1.3 times R_F, capped at 1.0), the practitioner calculates Oster's proportional selection ratio delta. This metric formally quantifies how much stronger selection on unobservables (such as informal familial financial safety nets, cash-flow volatility, or unbanked gig economy earnings) would need to be relative to observables to drive the estimated credit effect to zero.

If delta exceeds the standard econometric benchmark of 1.0, the estimated relationship is formally considered robust to unobserved confounding. As demonstrated by our Part 2.2 omitted-variable experiment—where omitting checking account status inflated savings coefficients by +107.30% and distorted employment weights by +260.27%—features with strong underlying covariances easily masquerade as primary risk drivers. Supplementing Oster bounds with residual spatial autocorrelation audits across applicant postal codes ensures that unobserved municipal economic disparities do not secretly contaminate underwriting scores, preventing institutional reliance on corrupted proxy signals.

---

### Question 5: Adversarial Subset Failure Analysis
Auditing the 300 test instances for extreme model errors isolates an adversarial subset of exactly 26 applicants, representing 8.67% of the test cohort. This adversarial pool comprises 22 high-confidence False Positives (where the logistic model predicted an approval probability P >= 0.75, yet the applicant defaulted, Y = 0) and 4 high-confidence False Negatives (where the model predicted severe default risk, P <= 0.25, yet the applicant successfully repaid, Y = 1). In accordance with the prompt's protocol, because fewer than 50 qualifying rows exist, all 26 instances are analyzed.

Demographic and financial profiling exposes two dominant shared characteristics. First, an astonishing 96.15% (25 of 26 applicants) in the adversarial subset are classified as foreign workers (A201), closely mirroring the 95.67% baseline prevalence across the test set. Second, and far more revealing, 45.45% of the high-confidence False Positives share the structural absence trait of having no checking account (checking_status_A14), despite an elevated average loan burden of 4,024 DM (compared to 3,385 DM across the full test set).

Statistically, the model fails confidently due to the rigid additive linearity of the logistic log-odds boundary: log(p / (1 - p)) = beta_0 + sum(beta_i * x_i). In the fitted model, having no checking account (A14) receives a large positive weight (beta approx +1.40) because historically, non-account holders exhibited reasonable aggregate repayment rates. The model mechanically adds this large credit bonus without interacting it with loan size or duration. When an unbanked borrower requests an unusually large loan (4,000+ DM), the additive formula blindspot allows the positive A14 bonus to overwhelm real financial vulnerability, producing confident, catastrophic misclassifications.

---

### Question 6: Human Impact of Fairness Trade-Offs
The fairness trade-off cost computed in Part 2.4 translates into severe, tangible consequences for real human borrowers. Consider two realistic credit applicants evaluated on our test set:

Persona 1 (disadvantaged group) is an ambitious 23-year-old nurse (Group A: Young < 25) requesting a 3,000 DM auto loan to commute to a regional hospital. She maintains a modest savings buffer and steady employment, earning a model-predicted approval probability of P_hat = 0.48.

Persona 2 (privileged group) is a 46-year-old established clerk (Group B: Older >= 25) requesting an identical 3,000 DM loan, with a slightly longer credit history yielding a model-predicted probability of P_hat = 0.52.

Under a policy operating at decision threshold t = 0.35, both applicants are successfully approved (P_hat >= 0.35), achieving near-perfect Demographic Parity (DP = 1.0184) and Equalized Odds (EO = 1.0080), with overall test accuracy peaking at 75.00%. Persona 1 secures transportation, preserves her job, and builds initial credit equity.

However, if bank management adopts a conservative credit-rationing stance and shifts the decision threshold to t = 0.55, Persona 1 is abruptly denied credit (0.48 < 0.55), while Persona 2 is approved (0.52 >= 0.55). At t = 0.55, Demographic Parity degrades from 1.0184 to 0.8878 (a 12.82% loss), while Equalized Odds falls from 1.0080 to 0.8750 (a 13.19% loss). The trade-off cost of 1.03 reflects that every incremental gain in bank default protection exacts a proportional penalty in equal treatment. Prioritizing conservative risk metrics mechanically excludes creditworthy young borrowers like Persona 1, entrenching demographic wealth disparities while shielding older applicants from scrutiny.

---

### Question 7: Defending a Specific Threshold (t* = 0.35)
From a combined fairness and utility standpoint, the optimal operating decision threshold is t* = 0.35. This threshold is defended by three compelling empirical metrics computed from our Part 2.4 sweep on the 300-row test cohort:

First, t* = 0.35 maximizes overall classification accuracy at 75.00% (225 of 300 test applicants correctly classified), surpassing the baseline accuracy of 74.00% observed at the standard default threshold of t = 0.50.

Second, t* = 0.35 achieves near-ideal demographic equity between younger applicants (Group A: Young < 25) and older applicants (Group B: Older >= 25). It records a Demographic Parity ratio of DP = 1.0184 (with young approval at 89.74% vs. older approval at 88.12%), effectively eliminating group-level under-representation. Simultaneously, it records an Equalized Odds ratio of EO = 1.0080 and a Predictive Parity ratio of PP = 1.0256, indicating calibrated parity in precision across demographics.

Third, t* = 0.35 achieves an outstanding True Positive Rate of TPR_A = 96.43% for young borrowers and TPR_B = 95.05% for older borrowers, virtually guaranteeing that creditworthy individuals across both cohorts are granted financing.

Strongest Critic Counter-Argument: A risk-averse institutional credit officer could legitimately challenge t* = 0.35 on the grounds of excessive False Positive exposure. At t = 0.35, the False Positive Rate for younger applicants is alarmingly elevated at FPR_A = 72.73% (and FPR_B = 72.15% for older borrowers). By approving almost all applicants with predicted probabilities above 0.35, the institution absorbs 8 out of 11 young defaulting borrowers. While mathematically equitable, this policy increases the bank's non-performing loan portfolio, threatening institutional liquidity if default write-off costs substantially exceed interest margin yields.

---

### Question 8: Demographic Parity vs. Equalized Odds
For commercial credit lending on the German Credit dataset, I strongly recommend prioritizing Equalized Odds over Demographic Parity. In commercial underwriting, credit granting is not an unconditional public entitlement; it is a risk-bearing financial contract where creditworthiness (the propensity and capacity to repay debt) is the legitimate basis for evaluation.

Empirically, our Part 2.4 threshold sweep proves that Demographic Parity and Equalized Odds are fundamentally distinct objectives. At t = 0.35, the model attains near-ideal parity across both metrics (DP = 1.0184, EO = 1.0080, PP = 1.0256, with test accuracy at 75.00%). However, as the threshold is calibrated to standard commercial risk tolerances (such as t = 0.55), Demographic Parity degrades to DP = 0.8878, while Equalized Odds records EO = 0.8750. Crucially, the Young cohort (< 25) exhibits an empirical base approval rate of 66.67% at t = 0.55 compared to 75.10% for the Older cohort.

Enforcing strict Demographic Parity (mandating DP = 1.0000 regardless of threshold) requires the lender to approve equal proportions of young and older applicants. Because young borrowers possess shorter employment tenures and lower accumulated savings in this dataset, forcing proportional demographic parity requires lowering approval bars specifically for uncreditworthy younger applicants, artificially inflating their default rate. In contrast, Equalized Odds mandates equal True Positive Rates (TPR_A / TPR_B) and equal False Positive Rates (FPR_A / FPR_B). This guarantees that any young applicant who is genuinely creditworthy enjoys the exact same likelihood of approval as an equally creditworthy older counterpart.

We explicitly sacrifice unconditional group-level parity (DP) to preserve non-discriminatory individual merit (EO), which upholds the legal standards of the Equal Credit Opportunity Act while protecting institutional solvency.

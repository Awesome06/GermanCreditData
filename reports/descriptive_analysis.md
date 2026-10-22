# Part 3: Descriptive Analysis

### Question 1: Linear Regression Median vs. Logistic Regression Cutoff Mismatch
When we compare the predictions on the 300 test applicants, Model A (an Ordinary Least Squares [OLS] linear regression model) and Model B (a standard logistic regression model) make different decisions on 82 people, which is a 27.33% mismatch rate. Both models were trained on the 700 training rows to predict loan approval, where 1 means approved and 0 means rejected. Model A predicts a continuous number, and following the assignment instructions, we set its approval cutoff at the test set median of 0.7201. Because a median always splits a group in half, Model A approves exactly 150 applicants (50.0%) and rejects 150 applicants (50.0%).

Model B, on the other hand, predicts a probability between 0 and 1, and approves anyone with a probability of 0.50 or higher. In our test data, 70.0% of applicants actually have good credit (210 approved vs. 90 rejected out of 300). Logistic regression picks up on this overall rate and approves 232 applicants (77.33%).

This difference in how the cutoffs work explains all 82 disagreements. Every single mismatched applicant was approved by logistic regression but rejected by linear regression. These 82 individuals had predicted probabilities between 0.5000 and 0.7201. That was high enough to pass Model B's 0.50 cutoff, but not high enough to make it into the top 50% required by Model A's median cutoff.

In addition, linear regression produces impossible probabilities: 3 test applicants received negative scores below 0.0 (down to -0.0632), and 39 applicants received scores above 1.0 (up to 1.2589). Logistic regression avoids this problem because its S-shaped probability curve always keeps predictions between 0 and 1.

---

### Question 2: Simpson's Paradox & Subgroup Divergence
To test for Simpson's Paradox—where an overall trend in data flips when looking at smaller groups—we split our 700 training records by applicant sex: 223 female applicants and 477 male applicants. We then fit separate logistic regression models on the full dataset, on women, and on men.

Comparing the estimated regression coefficients (beta) shows clear sign reversals and differences in feature importance. The clearest sign reversal happens with how long someone has lived at their current address (residence_since). In the full model, living longer at an address has a slight negative effect (beta = -0.0194). For men, this coefficient flips to positive (beta = +0.0304), improving approval odds. For women, it stays negative (beta = -0.0601). Similarly, existing bank credit accounts (existing_credits) is positive overall (beta = +0.0273) and for men (beta = +0.0689), but turns into a penalty for women (beta = -0.2162).

Even larger differences show up for loan duration and age. For men, loan duration in months (duration_months) is a strong risk factor that hurts approval odds (beta = -0.4095, p-value = 0.002, 95% confidence interval [CI]: [-0.66, -0.16]). For women, loan duration has almost no statistical relationship with approval (beta = -0.1487, p = 0.509, 95% CI: [-0.59, 0.29]). Meanwhile, applicant age in years (age_years) strongly protects female applicants (beta = +0.5290, p = 0.004, 95% CI: [0.17, 0.89]), but has no real effect for men (beta = +0.0972, p = 0.431, 95% CI: [-0.14, 0.34]).

A single combined model misses these differences. If a bank uses only the combined model, it will unfairly penalize women for longer loans that do not predict female default, while failing to reward older women for their proven financial stability.

---

### Question 3: Omitted-Variable Bias Audit
To test how omitted variables affect model stability, we deliberately removed Checking Account Status (Attribute 1), a key financial indicator that captures whether an applicant holds funds in a bank account. We then refit our logistic regression model on the 700 training observations and compared the before and after regression coefficients (beta) across the remaining predictors.

When an important variable is left out, the remaining correlated variables absorb its effect, causing their estimated weights to shift noticeably:

| Predictor | Full Model Beta | Reduced Model Beta | Absolute Shift | Percentage Change |
| :--- | :---: | :---: | :---: | :---: |
| savings_status_A63 | +0.6951 | +1.4409 | +0.7458 | +107.30% |
| personal_status_sex_A92 | +0.3524 | +0.6335 | +0.2812 | +79.80% |
| employment_since_A73 | -0.0518 | +0.0831 | +0.1349 | +260.27% |
| residence_since | +0.0068 | -0.0098 | -0.0166 | -244.91% |

In percentage terms, employment tenure between 1 and 4 years (employment_since_A73) shifted the most (+260.27%), flipping from a small negative coefficient of -0.0518 to a positive +0.0831. In absolute terms, moderate savings status (savings_status_A63) experienced the largest jump, climbing by +0.7458 from +0.6951 to +1.4409 (+107.30%). Because checking account balance and savings reserves often move together, removing checking accounts forces the savings feature to absorb the missing liquidity signal, artificially doubling its apparent weight. Additionally, the female applicant indicator (personal_status_sex_A92) increased by +79.80% from +0.3524 to +0.6335, acting as a demographic stand-in.

This instability proves that without stress-testing, an unverified model cannot be trusted for credit decisions. When key drivers are missing, the remaining coefficients do not measure true independent risk; they simply capture borrowed correlations, creating a fragile decision system that could collapse in real deployment.

---

### Question 4: Detecting Unobserved Confounds in Practice
If I were evaluating this model in a real-world setting, the most practical way to catch a hidden confounding variable is to test the model on data from a different branch or region, combined with an error residual check. When a model relies on an unseen factor, its predictions usually fall apart when it encounters a new group of applicants where that hidden factor is distributed differently.

To run this test, I would take our trained model and evaluate it on loan records from another bank branch without retraining it. First, I would calculate the prediction errors—the difference between actual loan outcomes and the model's predicted probabilities. Then, I would group these errors by variables that the model was not given, such as applicant postal code, branch location, or local unemployment rates.

If the model is genuinely measuring creditworthiness, its errors should look like random noise across all areas. However, if there is a missing variable—like local factory layoffs or strong community financial networks—the model will make systematic mistakes in specific locations. For example, if borrowers in one specific city default more often than expected despite having the same savings and employment numbers as borrowers elsewhere, that unexplained gap is a clear sign that a local economic factor was left out.

As our test in Part 2.2 showed, leaving out checking accounts caused savings coefficients to artificially jump by +107.30% and employment weights to flip by +260.27%. Testing the model across different regions helps ensure that the bank is not making decisions based on borrowed correlations.

---

### Question 5: Adversarial Subset Failure Analysis
When we search the 300 test applicants for confident mistakes, we find an adversarial group of exactly 26 individuals, which is 8.67% of the test set. This group consists of 22 False Positives (FP)—cases where the logistic regression model was confident the applicant would repay (approval probability P >= 0.75), but the borrower actually defaulted (outcome 0)—and 4 False Negatives—cases where the model predicted severe risk (P <= 0.25), but the borrower successfully repaid their loan (outcome 1). Since there were fewer than 50 cases in our test set, we examined all 26.

Looking closely at these 26 applicants shows two common characteristics. First, 96.15% (25 of the 26 applicants) are foreign workers (code A201), which is very close to the 95.67% rate in the entire test set. Second, and much more importantly, 45.45% of the 22 FP cases had no checking account (code A14), yet applied for an average loan of 4,024 Deutsche Mark, well above the test average of 3,385 Deutsche Mark.

The model fails confidently here because of how logistic regression works. Logistic regression simply adds up separate weights for each feature. In our trained model, not having a checking account gets a large positive weight (beta approx +1.40) because historically, applicants without accounts who borrowed small amounts had reasonable repayment rates. However, the model adds this bonus blindly without checking the loan size. When someone with no bank account asks for a large loan of 4,000+ Deutsche Mark, the fixed positive bonus overpowers the risk of debt, tricking the model into assigning an undeserved high approval score.

---

### Question 6: Human Impact of Fairness Trade-Offs
The trade-off calculations from Part 2.4 have direct, life-altering effects on real borrowers. To see how decision thresholds change individual outcomes, consider two applicants from our test evaluation:

Applicant 1 (younger group) is a 23-year-old nurse (Group A: age under 25) requesting a 3,000 Deutsche Mark (DM) car loan to travel to work. She has steady entry-level employment and modest personal savings, earning a model-predicted approval probability of P = 0.48.

Applicant 2 (older group) is a 46-year-old administrative clerk (Group B: age 25 or older) applying for an identical 3,000 DM loan. He has a longer credit history, yielding a model-predicted probability of P = 0.52.

If the bank chooses an inclusive decision threshold of t = 0.35, both applicants receive credit because both probabilities exceed 0.35. At this threshold, the bank achieves strong overall accuracy (75.00%) and near-perfect fairness between groups: Demographic Parity (DP), which measures the ratio of approval rates between groups, reaches 1.0184, and Equalized Odds (EO), which ensures equal true and false positive rates, reaches 1.0080. Applicant 1 buys her car, keeps her job, and builds initial credit.

However, if bank management decides to be conservative and raises the approval threshold to t = 0.55, Applicant 1 is suddenly rejected (0.48 < 0.55), while Applicant 2 is approved (0.52 >= 0.55). At t = 0.55, DP falls by 12.82% (from 1.0184 down to 0.8878), and EO drops by 13.19% (from 1.0080 down to 0.8750). The trade-off cost of 1.03 means that tightening credit to avoid default risk harms creditworthy young applicants like Applicant 1 disproportionately, cutting them off from financing while older applicants face lower barriers.

---

### Question 7: Defending a Specific Threshold (t* = 0.35)
From a combined fairness and business perspective, the most defensible decision threshold in our sweep is t = 0.35. This choice is supported by three concrete results from our 300-row test evaluation:

First, t = 0.35 yields the highest overall test accuracy across our entire threshold sweep at 75.00% (225 out of 300 applicants correctly classified), outperforming the standard default cutoff of t = 0.50, which achieved 74.00% accuracy.

Second, t = 0.35 produces near-perfect balance between younger applicants (Group A: age under 25) and older applicants (Group B: age 25 or older). It achieves a Demographic Parity ratio of 1.0184 (with younger approval at 89.74% compared to 88.12% for older applicants), effectively eliminating approval disparities. At the same time, it attains an Equalized Odds ratio of 1.0080 and a Predictive Parity ratio of 1.0256, proving that approval accuracy is virtually identical across both age groups.

Third, t = 0.35 ensures that creditworthy applicants are not left behind, achieving a True Positive Rate—the share of good borrowers approved—of 96.43% for young applicants and 95.05% for older applicants.

Strongest Counter-Argument: A skeptical credit risk officer could reasonably argue that setting the approval bar at 0.35 exposes the bank to substantial default losses. At t = 0.35, the False Positive Rate—the share of defaulting borrowers mistakenly approved—reaches 72.73% for young applicants (8 out of 11 young defaulters approved) and 72.15% for older applicants. This criticism is valid: approving nearly all applicants who clear a modest 0.35 score expands financial inclusion, but directly increases bad debt write-offs if the bank cannot absorb the unpaid principal.

---

### Question 8: Demographic Parity vs. Equalized Odds
For commercial bank lending on the German Credit dataset, I recommend prioritizing Equalized Odds (EO) over Demographic Parity (DP). A bank loan is a binding financial contract that requires repayment; therefore, individual creditworthiness—the genuine ability and willingness to repay debt—must serve as the guiding standard for approval decisions.

Our threshold analysis in Part 2.4 demonstrates that Demographic Parity and Equalized Odds measure completely different concepts. At threshold t = 0.35, the model temporarily satisfies both metrics (DP = 1.0184, EO = 1.0080, Predictive Parity = 1.0256, accuracy = 75.00%). However, as the bank tightens approval standards toward standard industry levels like t = 0.55, DP falls to 0.8878, while EO holds at 0.8750. In our data, younger applicants naturally have shorter job tenures and smaller savings balances, leading to an empirical approval rate of 66.67% at t = 0.55 compared to 75.10% for older applicants.

Demographic Parity requires the bank to approve equal percentages of applicants across groups, regardless of whether they can afford repayment. Enforcing strict DP would force the lender to lower approval standards for younger applicants who are likely to default, setting them up for financial distress and damaging their future credit standing. In contrast, EO requires equal accuracy: it ensures that creditworthy young applicants have the same True Positive Rate as creditworthy older applicants, while keeping False Positive Rates balanced.

By prioritizing EO, we intentionally sacrifice DP under tighter lending conditions. This trade-off is ethically defensible because it treats every qualified applicant fairly based on actual merit, avoiding both discriminatory exclusion and irresponsible over-lending that threatens bank solvency.

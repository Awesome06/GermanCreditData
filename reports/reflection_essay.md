# Part 4: Deployment Governance Reflection

> "If a policymaker asked you: 'Just give me one number: should we deploy this model or not?', what would you say, and what is the limitation of that 'one number' answer that you discovered yourself through your analysis?"

---

### 1. Direct Recommendation
I recommend conditional deployment with mandatory human review. While the model achieves a seemingly respectable 74.67% test accuracy, that single summary metric conceals an 8.67% confident error rate and severe performance disparities across demographic subgroups. Automated scoring must never operate autonomously on high-stakes lending decisions without structural human safeguards.

### 2. Why a Single Number Fails
First, an aggregate accuracy score masks deep divergences across demographic groups. In the Simpson's Paradox audit, overall accuracy remained stable, yet underlying feature weights completely flipped across genders. Address tenure exhibited a negative weight overall (beta = -0.0194) but flipped positive for men (beta = +0.0304). Similarly, loan duration predicted defaults strongly for men (p = 0.002) while showing no statistical relationship for women (p = 0.509), and applicant age strongly protected female borrowers (p = 0.004) while having no meaningful effect for men (p = 0.431). A single summary number conceals these conflicting patterns, imposing a uniform formula upon groups with fundamentally distinct financial dynamics.

Second, selecting a decision cutoff is an institutional policy decision rather than a neutral mathematical fact. An aggregate metric implicitly assumes that all classification errors carry equal societal and financial costs. A cutoff of t = 0.35 maximizes accuracy at 75.00% and balances approval rates (Demographic Parity [DP] = 1.0184), yet tolerates a 72.73% false positive rate among young applicants, exposing the bank to substantial unpaid debt. Conversely, a strict cutoff of t = 0.55 shields the portfolio from defaults but rejects 33.33% of creditworthy young borrowers. Furthermore, mathematical impossibility theorems prove that no scoring system can satisfy Demographic Parity, Equalized Odds, and Predictive Parity simultaneously when baseline repayment rates differ across groups. Finally, aggregate scores conceal fragility from omitted variables; dropping checking accounts caused savings weights to surge by +107.30% and employment weights to flip by +260.27%, demonstrating how clean headline metrics can mask borrowed correlations.

### 3. Supporting Evidence from the Analysis
The empirical stress-tests ground these governance boundaries in concrete metrics. First, linear and logistic architectures disagreed on 82 test cases (a 27.33% mismatch rate). Second, the model generated 26 high-confidence classification errors (8.67% error rate); 45.45% of false positives lacked checking accounts yet secured loans averaging 4,024 Deutsche Mark. Third, omitting checking status produced a +260.27% coefficient swing in employment tenure. Finally, threshold sweeps demonstrated a 1.03 trade-off cost ratio between demographic parity and equalized odds.

### 4. Key Takeaways for Deployment
Responsible credit underwriting cannot rely on a single performance score. Sustainable deployment demands transparent policy choices regarding solvency versus generational equity, ongoing subgroup residual audits across branch networks, and accessible human appeals for adversely affected borrowers.

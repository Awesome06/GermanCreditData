# Part 4: Deployment Governance Reflection

> *"If a policymaker asked you: 'Just give me one number: should we deploy this model or not?', what would you say, and what is the limitation of that 'one number' answer that you discovered yourself through your analysis?"*

---

### 1. Direct Recommendation
I recommend conditional deployment with mandatory human review for borderline applicants and separate subgroup evaluations, because while the model achieves 74.67% test accuracy, it conceals an 8.67% high-confidence error rate and severe differences across demographic groups.

### 2. Why a Single Number Fails
First, a single overall accuracy number hides deep performance differences between demographic groups. In our Simpson's Paradox audit, the model achieved stable overall accuracy, yet underlying feature weights completely flipped across genders. For example, how long an applicant lived at their residence reversed from a slightly negative coefficient in the full sample (beta = -0.0194) to a positive benefit for men (beta = +0.0304). Similarly, loan duration strongly predicted defaults for male borrowers (p = 0.002) but had no statistical relationship for female borrowers (p = 0.509). A single accuracy score gives a false sense of security while applying a one-size-fits-all formula that misjudges distinct groups.

Second, choosing a decision cutoff is a policy value judgment rather than a neutral mathematical fact. An overall score assumes all errors carry equal costs, but changing the threshold shifts who bears the financial burden. Lowering the cutoff to t = 0.35 maximizes test accuracy (75.00%) and achieves balanced approval rates between young and older borrowers (Demographic Parity [DP] = 1.0184), but exposes the bank to an elevated young false positive rate of 72.73%. Conversely, setting a strict cutoff of t = 0.55 shields the bank from unpaid debt but rejects 33.33% of creditworthy younger borrowers. A single metric cannot tell a policymaker which group should absorb risk.

Third, different fairness standards directly clash with one another. When demographic groups have different baseline repayment rates, mathematical theorems prove that no model can satisfy Demographic Parity, Equalized Odds (EO), and Predictive Parity at the same time. At threshold t = 0.35, the model balances approval rates across age groups, but as the bar is raised to t = 0.55, DP falls to 0.8878. A single approval number hides the reality that helping one group under one definition inevitably harms another under a different definition.

Finally, an aggregate score conceals how models fail when key variables are omitted. When we removed checking account records, savings coefficients artificially doubled (+107.30% shift), proving that clean top-line numbers can hide borrowed correlations and blind-spot failures.

### 3. Grounding in Empirical Evidence
Our stress-tests anchor these limitations in exact numbers: the 27.33% decision disagreement between linear and logistic models (82 mismatched cases); the 26 high-confidence test mistakes (8.67% error rate), where 45.45% of false positives lacked checking accounts; parameter swings reaching +260.27% under omitted variable testing; and a 1.03 trade-off cost ratio between DP and EO during threshold sweeps.

### 4. Closing Meta-Lesson
Responsible deployment of artificial intelligence cannot be answered with a single pass or fail score. Real-world deployment requires clear transparency about unavoidable trade-offs between bank solvency and social equity, continuous subgroup auditing, and accessible human appeal processes for individuals flagged by automated systems.

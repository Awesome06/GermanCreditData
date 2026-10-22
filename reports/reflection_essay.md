# Part 4: Deployment Governance Reflection

*"If a policy-maker asked you: 'just give me one number: should we deploy this model or not?', what would you say, and what is the limitation of that 'one number' answer that you discovered yourself through your analysis?"*

---

### 1. Direct Recommendation
I recommend Conditional Deployment strictly under mandatory human-in-the-loop oversight and subgroup-specific calibration, because while the model achieves 74.67% aggregate test accuracy, it exhibits an 8.67% high-confidence adversarial failure rate and systematic subgroup parameter instability.

### 2. Why a Single Number Fails
First, an aggregate metric like overall accuracy dangerously obscures severe subgroup performance gaps. In our Simpson's Paradox audit, the pooled model reported stable average accuracy, yet underlying feature coefficients diverged profoundly across demographic cohorts. For example, residential tenure reversed sign from beta = -0.0194 in the pooled population to beta = +0.0304 among male applicants, while loan duration was strongly predictive for men (p = 0.002) but completely insignificant for women (p = 0.509). Relying on a single pooled metric blinds policymakers to the fact that the algorithm evaluates distinct demographic groups under flawed, one-size-fits-all assumptions.

Second, choosing an operating threshold is a normative ethical value judgment rather than an objective mathematical fact. A single accuracy metric assumes equal utility between error types. However, shifting the cutoff threshold redistributes financial harm: lowering the threshold to t = 0.35 achieves peak test accuracy (75.00%) and near-perfect demographic equity (DP = 1.0184, EO = 1.0080), but inflates the young false positive rate to 72.73%. Conversely, a conservative threshold of t = 0.55 protects bank liquidity but mechanically excludes 33.33% of creditworthy younger borrowers. A single metric cannot declare which stakeholder group should bear the cost of uncertainty.

Third, statistical fairness criteria are mathematically incompatible. Under Kleinberg and Chouldechova's impossibility theorems, when base repayment rates differ between demographic groups, no predictive algorithm can concurrently satisfy Demographic Parity, Equalized Odds, and Predictive Parity. Demanding a single metric ignores this structural reality: optimizing for Demographic Parity at t = 0.35 requires sacrificing institutional default discipline, whereas calibrating for Predictive Parity at t = 0.65 (PP = 1.0841) severely degrades Equalized Odds (EO = 0.4758 for female applicants).

Finally, aggregate metrics conceal vulnerability to omitted confounders and confident adversarial failures. When checking account status was omitted, savings coefficients artificially doubled (+107.30% shift) and employment weights inverted by +260.27%, proving that a seemingly accurate model can be founded on corrupted proxy variables.

### 3. Grounding in Empirical Evidence
Our empirical stress-tests firmly anchor these four limitations in exact computations: the 27.33% classification mismatch between OLS and Logit models; the 26 high-confidence adversarial test errors (8.67% failure rate), where 45.45% of false positives lacked checking accounts; the parameter shifts of up to +260.27% under omitted confound testing; and the 1.03 trade-off cost ratio between Demographic Parity and Equalized Odds across decision sweeps.

### 4. Closing Meta-Lesson
Responsible AI governance cannot be reduced to a binary go/no-go score. True algorithmic accountability requires transparently exposing trade-offs between solvency and equity, maintaining continuous multi-metric stress-testing, and establishing formal human recourse for edge-case applicants.

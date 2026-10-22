# Workspace Agent Directives: German Credit Data Audit & Algorithmic Fairness

## Bootstrap Protocol
- On session initialization:
  1. Verify Git hook configuration by checking if `.git/hooks/pre-commit` exists. If missing, execute `.agent/setup.ps1` (or verify that `.agent/.registry` is present).
  2. Check if the local `.agents/skills/` directory contains active domain skills (`python-ml`, `readme-craft`).
  3. Automatically detect the application stack from repository manifest files (`pyproject.toml`, `requirements.txt`) and synchronize skills silently without pausing to ask the user for confirmation to analyze.
  4. Once skills are linked, strictly enforce their constraints during all subsequent code generation and review tasks.
  5. Note: Global skills (`code-review`, `doc-sync`, `error-discipline`, `git-release`, `incident-triaging`, `security-auditor`, `test-discipline`, `workspace-initializer`, `repo-wiki`) are pre-loaded at the machine level (`~/.gemini/config/skills/`). Do not attempt to symlink them into `.agents/skills/`.
  6. Repository Knowledge Graph: Check for `wiki/`. When architectural decisions, design contracts, or runbooks change, maintain atomic records using the `repo-wiki` protocol (`[[wikilinks]]`, YAML frontmatter). Use `.agent/wiki.ps1 query <topic>` (or `.agent/wiki.sh`) to slice context on demand without bloating the prompt window.

## General Constraints
- The `.agent/` and `.agents/` directories (including `.agent/.registry`) are internal registry, hook, and customization storage. NEVER treat them as application source code, and NEVER allow internal registry files to trigger skills.
- **Strict Zero-Legacy-Skills Invariant:** Antigravity IDE discovers workspace skills exclusively from `.agents/skills/` (plural). The legacy directory `.agent/skills/` (singular) is permanently DEPRECATED and PROHIBITED. Agents and scripts MUST NEVER create, link, or reference `.agent/skills/`. All active domain skills and local customizations MUST reside strictly in `.agents/skills/`.
- Do not add or remove third-party dependencies without explicit user confirmation.
- Keep the context window lean: do not load files into context unless actively modifying them.

## Data Science & Econometric Auditing Invariants

### 1. Deterministic Seeding & Train-Test Partitioning
- **Global Seed Invariant:** Lock `random_state = 343` across all data splitting, model training, and stochastic sampling operations.
- **Strict Partition Dimensions:** Use a 70% train / 30% test split resulting in exactly 700 training observations and 300 test observations.
- **Data Leakage Prohibition:** Fit all scaling transformations (`StandardScaler`) exclusively on the 700-row training set; transform the 300-row test set without refitting.

### 2. Preprocessing & Feature Hygiene
- **Dataset Source:** Use raw `german.data` (20 features: 7 numerical, 13 categorical). Do not use `german.data-numeric`.
- **Target Encoding:** Re-encode the target variable so that **Approve = 1** (formerly 1 in UCI) and **Reject = 0** (formerly 2 in UCI).
- **Structural Absence Codes:** Maintain structural absence categories (`A14` = no checking account, `A65` = no savings account) as distinct categorical levels. Never impute or drop them.
- **Multicollinearity Discipline:** When applying One-Hot Encoding to categorical features, strictly enforce `drop='first'` to avoid the dummy variable trap and preserve non-singular matrices for OLS/Logit.

### 3. Statistical Modeling & Econometric Rigor
- **Dual Model Implementation:**
  - **Model A (Linear Probability Model / OLS):** Predict continuous approval probabilities using OLS. Decision rule: flag as positive if $\hat{y} \ge \text{median}(\hat{y}_{\text{test}})$, else negative.
  - **Model B (Logistic Regression / Logit):** Predict binary approval probabilities via Maximum Likelihood Estimation. Decision rule: flag as positive if $\hat{P} \ge 0.50$, else negative.
- **Inference Verification:** Employ `statsmodels` (Formula/Logit/OLS APIs) in tandem with `scikit-learn` to extract robust standard errors, $z$/$t$-statistics, $p$-values, and 95% confidence intervals for all stress tests.

### 4. Stress-Testing Protocols ("The Four Traps")
- **Trap 2.1 (Simpson's Paradox):** Partition by sensitive demographic attributes (e.g., Age Group: Young $<25$ vs. Older $\ge 25$, or Sex). Fit models across Full, Subgroup A, and Subgroup B. Document both sign reversals and confidence-interval-backed effect size divergences.
- **Trap 2.2 (Omitted-Variable Bias):** Deliberately remove a dominant confounder (e.g., Checking Account Status `Attribute 1` or Credit Amount `Attribute 5`). Construct comparative tables reporting $\beta_{\text{Full}}$, $\beta_{\text{Reduced}}$, $\Delta \beta$, and $\% \Delta$.
- **Trap 2.3 (Adversarial Error Auditing):** Filter the test set for high-confidence false positives ($\hat{P} \ge 0.75, Y=0$) and high-confidence false negatives ($\hat{P} \le 0.25, Y=1$). Profile feature distributions and diagnose linear log-odds boundary failures.
- **Trap 2.4 (Multi-Threshold Fairness Auditing):** Sweep decision thresholds $t \in [0.05, 0.95]$ with step $0.10$. Compute Demographic Parity (DP), Equalized Odds (EO), Predictive Parity (PP), and the Trade-Off Cost ($W/Z$).

### 5. Descriptive Analysis & Written Defense Standards
- **Word Count Windows:** Maintain strict adherence to prompt targets:
  - Part 3 (Q1–Q8): Each response must strictly sit within **250–300 words**.
  - Part 4 (Reflection Essay): Total response length must sit tightly around **500 words** structured into the 4 mandated subsections.
- **Zero-Ungrounded Claims Invariant:** Every analytical statement, trade-off defense, or policy takeaway MUST cite exact empirical numbers, coefficients, sample counts, or confidence intervals computed directly from the dataset. Generic, unanchored prose is strictly prohibited.

## Git Staging & Commit Protocol
- **Pre-Staging Verification:** Run active linters (`ruff check .`) and tests via `test-discipline` before staging any changes.
- **Autonomous Staging & Commit Formulation:** Agents are authorized to stage modified files (`git add`) and formulate structured commit messages adhering to Conventional Commits (via `git-release`) with a comprehensive technical description:
  - Header: `<type>(<scope>): <subject>` (maximum 72 characters, imperative mood, lowercase, no trailing period).
  - Multi-line body: Explicit motivation/intent, bulleted key changes, and verification evidence.
- **Review Gate (Not Committed Until Review):** Commits MUST NOT be executed autonomously. Always leave changes staged and present the drafted commit message and diff summary to the user for review. Execute `git commit` ONLY after explicit user review and approval.
- **Strict Zero-Unprompted-Push Guardrail:** NEVER run `git push` without explicit user instruction.

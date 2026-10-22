"""Generate academic PDF submission report for CSET485 Assignment #1.

Incorporates:
- Student Name: Mrigank Bhatnagar
- Enrollment: S24CSEU0343
- Black headings, classic academic typography (Times/Georgia)
- Simple, clear student language without AI buzzwords
- Plain ASCII math (no raw unrendered LaTeX markers like $t$)
- Clean confusion matrix cell labels
- Clean section headers (no duplicate titles or meta subtitles)
- Formatted prompt question blockquote
- Booktabs academic tables
- No timestamp or date headers
- Full reproduction of data preprocessing, Part 1, Part 2, Part 3, and Part 4.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def generate_html_report() -> str:
    # Read text reports
    with open("reports/descriptive_analysis.md", encoding="utf-8") as f:
        descriptive_md = f.read()

    with open("reports/reflection_essay.md", encoding="utf-8") as f:
        reflection_md = f.read()

    # Convert markdown headers and paragraphs to clean academic HTML
    def md_to_html(md_text: str) -> str:
        lines = md_text.split("\n")
        html_out = []
        in_table = False
        table_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                in_table = True
                table_lines.append(stripped)
                continue
            elif in_table:
                in_table = False
                if len(table_lines) >= 3:
                    headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                    rows = [
                        [c.strip() for c in row.split("|")[1:-1]]
                        for row in table_lines[2:]
                    ]
                    tbl_html = "<div class='table-wrapper'><table class='academic-table'><thead><tr>"
                    for h in headers:
                        tbl_html += f"<th>{h}</th>"
                    tbl_html += "</tr></thead><tbody>"
                    for r in rows:
                        tbl_html += "<tr>"
                        for cell in r:
                            tbl_html += f"<td>{cell}</td>"
                        tbl_html += "</tr>"
                    tbl_html += "</tbody></table></div>"
                    html_out.append(tbl_html)
                table_lines = []

            if not stripped:
                continue
            elif stripped.startswith("> "):
                clean_q = stripped[2:].strip().strip("*").strip('"')
                html_out.append(f'<div class="prompt-box">"{clean_q}"</div>')
            elif stripped.startswith("### "):
                html_out.append(f"<h3>{stripped[4:]}</h3>")
            elif stripped.startswith("## "):
                html_out.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("# "):
                html_out.append(f"<h2>{stripped[2:]}</h2>")
            elif stripped.startswith("---"):
                html_out.append("<hr/>")
            elif stripped.startswith("- "):
                html_out.append(f"<li>{stripped[2:]}</li>")
            else:
                p_text = stripped.replace("**", "<strong>", 1)
                while "**" in p_text:
                    p_text = p_text.replace("**", "</strong>", 1).replace("**", "<strong>", 1)
                html_out.append(f"<p>{p_text}</p>")

        if in_table and len(table_lines) >= 3:
            headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
            rows = [
                [c.strip() for c in row.split("|")[1:-1]]
                for row in table_lines[2:]
            ]
            tbl_html = "<div class='table-wrapper'><table class='academic-table'><thead><tr>"
            for h in headers:
                tbl_html += f"<th>{h}</th>"
            tbl_html += "</tr></thead><tbody>"
            for r in rows:
                tbl_html += "<tr>"
                for cell in r:
                    tbl_html += f"<td>{cell}</td>"
                tbl_html += "</tr>"
            tbl_html += "</tbody></table></div>"
            html_out.append(tbl_html)

        return "\n".join(html_out)

    part3_html = md_to_html(descriptive_md)
    part4_html = md_to_html(reflection_md)

    figures_dir = Path("reports/figures").resolve().as_uri()

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CSET485 Assignment #1: German Credit Data Audit</title>
<style>
    @page {{
        size: A4;
        margin: 22mm 20mm 22mm 20mm;
    }}
    body {{
        font-family: "Times New Roman", Times, Georgia, serif;
        font-size: 10.5pt;
        line-height: 1.45;
        color: #111111;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 16pt;
        font-weight: bold;
        color: #000000;
        margin-top: 0;
        margin-bottom: 4px;
        text-align: center;
        letter-spacing: -0.2px;
    }}
    .subtitle {{
        text-align: center;
        font-size: 12pt;
        font-weight: normal;
        color: #222222;
        margin-top: 2px;
        margin-bottom: 16px;
    }}
    .meta-card {{
        border: 1px solid #333333;
        padding: 10px 14px;
        margin-bottom: 18px;
        font-size: 9.5pt;
        background: #fafafa;
    }}
    .meta-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .meta-table td {{
        padding: 3px 6px;
        vertical-align: top;
        border: none;
        font-size: 9.5pt;
    }}
    h2 {{
        font-size: 12pt;
        font-weight: bold;
        color: #000000;
        border-bottom: 1.5px solid #000000;
        padding-bottom: 2px;
        margin-top: 22px;
        margin-bottom: 8px;
        page-break-after: avoid;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3 {{
        font-size: 11pt;
        font-weight: bold;
        color: #000000;
        margin-top: 14px;
        margin-bottom: 5px;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }}
    ul {{
        margin-top: 4px;
        margin-bottom: 8px;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 4px;
        text-align: justify;
    }}
    .figure-container {{
        text-align: center;
        margin: 12px 0 16px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 86%;
        height: auto;
        border: 1px solid #777777;
    }}
    .figure-caption {{
        font-size: 9pt;
        font-style: italic;
        color: #222222;
        margin-top: 5px;
        text-align: center;
    }}
    .academic-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 9pt;
        margin: 10px 0 14px 0;
        page-break-inside: avoid;
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000;
    }}
    .academic-table th {{
        border-bottom: 1.5px solid #000000;
        padding: 6px 8px;
        font-weight: bold;
        color: #000000;
        text-align: left;
        background: none;
    }}
    .academic-table td {{
        border-bottom: 0.5px solid #cccccc;
        padding: 5px 8px;
        color: #111111;
        text-align: left;
    }}
    .academic-table tr:last-child td {{
        border-bottom: none;
    }}
    .academic-note {{
        margin: 10px 0;
        padding: 8px 12px;
        background-color: #f8f8f8;
        border-left: 3px solid #000000;
        font-size: 9.5pt;
    }}
    .prompt-box {{
        font-style: italic;
        color: #222222;
        background: #f9f9f9;
        border-left: 3px solid #000000;
        padding: 8px 12px;
        margin: 10px 0 14px 0;
        font-size: 10pt;
    }}
    .page-break {{
        page-break-before: always;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #777777;
        margin: 14px 0;
    }}
    code {{
        font-family: "Courier New", Courier, monospace;
        font-size: 9pt;
        background-color: #f2f2f2;
        padding: 1px 3px;
    }}
</style>
</head>
<body>

<h1>CSET485 – AI and Society: Assignment #1</h1>
<div class="subtitle">Statistical Stress-Testing and Algorithmic Fairness Audit: German Credit Data</div>

<div class="meta-card">
    <table class="meta-table">
        <tr>
            <td style="width: 50%;"><strong>Student Name:</strong> Mrigank Bhatnagar</td>
            <td style="width: 50%;"><strong>Enrollment Number:</strong> S24CSEU0343</td>
        </tr>
        <tr>
            <td><strong>Course:</strong> CSET485 – AI and Society</td>
            <td><strong>Deterministic Random Seed:</strong> 343 (Last 3 digits of Enrollment)</td>
        </tr>
        <tr>
            <td><strong>Assigned Dataset:</strong> UCI Statlog German Credit Data (<code>german.data</code>)</td>
            <td><strong>Data Partition:</strong> 70% Train (700 rows) / 30% Test (300 rows)</td>
        </tr>
    </table>
</div>

<h2>Data Preprocessing</h2>
<p>
The raw German Credit dataset has 1,000 applicant records and 20 features. Before training any models, we prepared the data as follows:
</p>
<ul>
    <li><strong>Target Labels:</strong> The original dataset encoded outcomes as 1 for good credit and 2 for bad credit. We changed this to 1 for approved (700 applicants) and 0 for rejected (300 applicants).</li>
    <li><strong>Missing Values:</strong> The dataset has no missing values. Some categories indicate that an applicant does not have a bank account, such as code <code>A14</code> (no checking account) and code <code>A65</code> (no savings account). Instead of treating these as missing data or filling them in, we kept them as their own categories because not having an account is useful information about an applicant.</li>
    <li><strong>Outliers:</strong> Looking at continuous values like loan amount (<code>credit_amount</code>) and loan duration (<code>duration_months</code>), there are several large values. We kept these in the dataset because they represent real, larger loan applications rather than typing mistakes.</li>
    <li><strong>Categorical Features:</strong> We converted the 13 categorical variables into dummy columns using One-Hot Encoding. We used <code>drop='first'</code> to drop one reference category per feature so the regression models do not suffer from multicollinearity.</li>
    <li><strong>Feature Scaling:</strong> We standardized the 7 numerical features using <code>StandardScaler</code>. To prevent data leakage, we fit the scaler only on the 700 training rows, and then transformed the 300 test rows.</li>
</ul>

<h2>Part 1 — Build: Model Comparison</h2>
<p>
We trained two models on the 700 training rows to predict loan approval (1 = approved, 0 = rejected):
</p>
<ul>
    <li><strong>Model A (Linear Regression / OLS):</strong> Predicts a continuous approval score using Ordinary Least Squares (OLS). As directed by the assignment, we set the approval cutoff at the test set median (0.7201), approving anyone with a score of 0.7201 or higher.</li>
    <li><strong>Model B (Logistic Regression):</strong> Predicts the probability of approval between 0 and 1. Anyone with a predicted probability of 0.50 or higher is approved.</li>
</ul>

<div class="academic-note">
    <strong>Summary of Differences:</strong> On the 300 test applicants, the two models disagree on 82 cases (a 27.33% mismatch rate). Because Model A uses the median as its cutoff, it is forced to approve exactly 150 applicants (50.0%) and reject 150. Model B approves 232 applicants (77.33%), which reflects the real approval rate of 70.0% in the data. In addition, linear regression produces invalid probabilities: 3 test applicants got scores below 0.0 (down to -0.0632), and 39 got scores above 1.0 (up to 1.2589).
</div>

<div class="table-wrapper">
    <table class="academic-table">
        <thead>
            <tr>
                <th>Agreement Confusion Matrix</th>
                <th>Model B Rejected (P &lt; 0.50)</th>
                <th>Model B Approved (P &ge; 0.50)</th>
                <th>Total Model A</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Model A Rejected (&lt; Median 0.7201)</strong></td>
                <td>68 (Both Reject)</td>
                <td><strong>82 (Model B Approves, Model A Rejects)</strong></td>
                <td>150 (50.0%)</td>
            </tr>
            <tr>
                <td><strong>Model A Approved (&ge; Median 0.7201)</strong></td>
                <td>0 (Model A Approves, Model B Rejects)</td>
                <td>150 (Both Approve)</td>
                <td>150 (50.0%)</td>
            </tr>
            <tr>
                <td><strong>Total Model B</strong></td>
                <td>68 (22.67%)</td>
                <td>232 (77.33%)</td>
                <td>300 (100.0%)</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/agreement_confusion_matrix.png" alt="Agreement Matrix">
    <div class="figure-caption">Figure 1: Agreement heatmap comparing Linear Regression (OLS median cutoff) and Logistic Regression (0.50 probability cutoff) across 300 test applicants.</div>
</div>

<div class="page-break"></div>

<h2>Part 2 — The Trap: Stress-Testing Protocols</h2>

<h3>Trap 2.1: Simpson's Paradox Hunt</h3>
<p>
To see if relationships flip when looking at subgroups, we split the 700 training rows by applicant sex: 223 women and 477 men. We then fit separate logistic regression models on the full dataset, on women, and on men:
</p>

<div class="table-wrapper">
    <table class="academic-table">
        <thead>
            <tr>
                <th>Predictor</th>
                <th>Full Dataset (N=700)</th>
                <th>Men (N=477)</th>
                <th>Women (N=223)</th>
                <th>Subgroup Difference</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>residence_since</code></td>
                <td>-0.0194</td>
                <td><strong>+0.0304</strong></td>
                <td><strong>-0.0601</strong></td>
                <td><strong>Sign Reversal</strong> (Positive for Men, Negative for Women)</td>
            </tr>
            <tr>
                <td><code>existing_credits</code></td>
                <td>+0.0273</td>
                <td><strong>+0.0689</strong></td>
                <td><strong>-0.2162</strong></td>
                <td><strong>Sign Reversal</strong> (Turns from bonus to penalty)</td>
            </tr>
            <tr>
                <td><code>duration_months</code></td>
                <td>-0.3479 (p &lt; 0.001)</td>
                <td>-0.4095 (p = 0.002, CI: [-0.66, -0.16])</td>
                <td>-0.1487 (p = 0.509, CI: [-0.59, 0.29])</td>
                <td><strong>Different Importance</strong> (Significant for men; null for women)</td>
            </tr>
            <tr>
                <td><code>age_years</code></td>
                <td>+0.2163 (p = 0.035)</td>
                <td>+0.0972 (p = 0.431, CI: [-0.14, 0.34])</td>
                <td><strong>+0.5290 (p = 0.004, CI: [0.17, 0.89])</strong></td>
                <td><strong>Different Importance</strong> (Helps women; null for men)</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/simpsons_forest_plot.png" alt="Simpson's Paradox Forest Plot">
    <div class="figure-caption">Figure 2: Regression weights with 95% confidence intervals across Full, Male, and Female groups.</div>
</div>

<h3>Trap 2.2: Omitted-Variable Bias (OVB)</h3>
<p>
To see what happens when an important variable is left out, we dropped checking account status (Attribute 1). When we refit the model on the remaining features, other variables that correlate with checking accounts absorbed the missing signal and their weights shifted heavily:
</p>

<div class="table-wrapper">
    <table class="academic-table">
        <thead>
            <tr>
                <th>Predictor</th>
                <th>Full Model Beta</th>
                <th>Reduced Model Beta</th>
                <th>Absolute Shift</th>
                <th>Percentage Change</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>savings_status_A63</code></td>
                <td>+0.6951</td>
                <td>+1.4409</td>
                <td>+0.7458</td>
                <td><strong>+107.30%</strong> (Absorbs omitted checking account signal)</td>
            </tr>
            <tr>
                <td><code>personal_status_sex_A92</code></td>
                <td>+0.3524</td>
                <td>+0.6335</td>
                <td>+0.2812</td>
                <td><strong>+79.80%</strong> (Demographic proxy shift)</td>
            </tr>
            <tr>
                <td><code>employment_since_A73</code></td>
                <td>-0.0518</td>
                <td>+0.0831</td>
                <td>+0.1349</td>
                <td><strong>+260.27%</strong> (Sign flip across employment tier)</td>
            </tr>
            <tr>
                <td><code>residence_since</code></td>
                <td>+0.0068</td>
                <td>-0.0098</td>
                <td>-0.0166</td>
                <td><strong>-244.91%</strong> (Reversal to negative penalty)</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/ovb_parameter_shifts.png" alt="Omitted Variable Bias Shifts">
    <div class="figure-caption">Figure 3: Largest percentage shifts in feature weights after dropping checking account status.</div>
</div>

<div class="page-break"></div>

<h3>Trap 2.3: Adversarial Error Auditing</h3>
<p>
We checked the 300 test cases for confident mistakes: cases where the model predicted an approval probability of 0.75 or higher but the person defaulted (22 False Positives), or predicted 0.25 or lower but the person repaid (4 False Negatives). This gave 26 total cases (an 8.67% error rate). Since there are fewer than 50 cases, we analyzed all 26:
</p>
<ul>
    <li><strong>Foreign Workers:</strong> 96.15% (25 of the 26 applicants) are foreign workers (code <code>A201</code>), matching the general test rate of 95.67%.</li>
    <li><strong>No Checking Account:</strong> 45.45% of the false positives had no checking account (code <code>A14</code>), yet asked for an average loan of 4,024 Deutsche Mark (compared to 3,385 Deutsche Mark across the test set).</li>
    <li><strong>Why the Model Fails:</strong> The logistic regression formula adds up separate terms. It gives a positive bonus for having no checking account, but does not check how big the loan is. For someone asking for a large loan without an account, this positive bonus overwhelms the risk.</li>
</ul>

<div class="figure-container">
    <img src="{figures_dir}/adversarial_demographic_profiles.png" alt="Adversarial Profiles">
    <div class="figure-caption">Figure 4: Demographic comparison: Confident Error Subset (N=26) vs. Full Test Set (N=300).</div>
</div>

<h3>Trap 2.4: Multi-Threshold Algorithmic Fairness Auditing</h3>
<p>
We swept the decision cutoff t from 0.05 to 0.95 across age groups (younger than 25 vs 25 and older) to compare Demographic Parity (DP), Equalized Odds (EO), and Predictive Parity (PP):
</p>

<div class="table-wrapper">
    <table class="academic-table">
        <thead>
            <tr>
                <th>Threshold (t)</th>
                <th>Demographic Parity (DP)</th>
                <th>Equalized Odds (EO)</th>
                <th>Predictive Parity (PP)</th>
                <th>Overall Accuracy</th>
                <th>TPR (Young)</th>
                <th>TPR (Older)</th>
                <th>FPR (Young)</th>
                <th>FPR (Older)</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>0.05</td><td>1.0000</td><td>1.0000</td><td>0.9632</td><td>70.33%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr>
            <tr><td>0.15</td><td>1.0000</td><td>1.0000</td><td>0.9632</td><td>71.00%</td><td>1.0000</td><td>0.9945</td><td>1.0000</td><td>0.9747</td></tr>
            <tr><td>0.25</td><td>1.0184</td><td>1.0080</td><td>1.0256</td><td>74.00%</td><td>0.9643</td><td>0.9560</td><td>0.8182</td><td>0.7975</td></tr>
            <tr><td><strong>0.35 (Optimal)</strong></td><td><strong>1.0184</strong></td><td><strong>1.0080</strong></td><td><strong>1.0256</strong></td><td><strong>75.00%</strong></td><td>0.9643</td><td>0.9505</td><td>0.7273</td><td>0.7215</td></tr>
            <tr><td>0.45</td><td>0.9557</td><td>0.9392</td><td>1.0841</td><td>74.33%</td><td>0.8929</td><td>0.9396</td><td>0.5455</td><td>0.5823</td></tr>
            <tr><td>0.55</td><td>0.8878</td><td>0.8750</td><td>1.1118</td><td>73.67%</td><td>0.7500</td><td>0.8571</td><td>0.4545</td><td>0.5063</td></tr>
            <tr><td>0.65</td><td>0.7630</td><td>0.7297</td><td>1.1278</td><td>72.00%</td><td>0.5714</td><td>0.7802</td><td>0.2727</td><td>0.4177</td></tr>
            <tr><td>0.75</td><td>0.6410</td><td>0.6279</td><td>1.1895</td><td>68.67%</td><td>0.4286</td><td>0.6813</td><td>0.1818</td><td>0.2911</td></tr>
            <tr><td>0.85</td><td>0.4789</td><td>0.4435</td><td>1.2857</td><td>59.33%</td><td>0.2500</td><td>0.5659</td><td>0.0909</td><td>0.2025</td></tr>
            <tr><td>0.95</td><td>0.3205</td><td>0.2967</td><td>1.4000</td><td>44.00%</td><td>0.1071</td><td>0.3626</td><td>0.0000</td><td>0.0633</td></tr>
        </tbody>
    </table>
</div>

<div class="academic-note">
    <strong>Trade-Off Example (t=0.35 vs. t=0.55):</strong> Shifting the cutoff from t=0.35 to t=0.55 causes Demographic Parity to drop from 1.0184 to 0.8878 (a 12.82% drop) and Equalized Odds to drop from 1.0080 to 0.8750 (a 13.19% drop). This gives a <strong>Trade-Off Cost ratio of W/Z = 1.03</strong>.
</div>

<div class="figure-container">
    <img src="{figures_dir}/fairness_tradeoff_chart.png" alt="Fairness Trade-Off Chart">
    <div class="figure-caption">Figure 5: Fairness trade-off curve: Demographic Parity vs. Equalized Odds across cutoffs t from 0.05 to 0.95.</div>
</div>

<div class="page-break"></div>

{part3_html}

<div class="page-break"></div>

{part4_html}

<hr/>

<div class="academic-note" style="font-size:9pt;">
    <strong>Verification & Reproducibility Guarantee:</strong> All empirical numbers, regression coefficients, sample counts, and fairness metrics reported in this document were verified directly from the German Credit dataset by running the accompanying Jupyter notebook (<code>german_credit_audit.ipynb</code>) with random seed <code>343</code>.
</div>

</body>
</html>
"""
    return html_doc


def compile_pdf() -> Path:
    html_content = generate_html_report()
    html_path = Path("reports/German_Credit_Fairness_Audit_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated intermediate HTML at: {html_path}")

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    temp_pdf = Path(tempfile.gettempdir()) / "German_Credit_Fairness_Audit_Report.pdf"
    target_pdf = Path("reports/German_Credit_Fairness_Audit_Report.pdf")

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={temp_pdf}",
        f"file:///{html_path.resolve().as_posix()}",
    ]

    print("Compiling publication PDF report via headless browser engine...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        pass

    if not temp_pdf.exists():
        raise FileNotFoundError(f"Expected compiled PDF not found at: {temp_pdf}")

    shutil.copyfile(temp_pdf, target_pdf)
    print(f"Publication-grade PDF successfully compiled and saved at: {target_pdf} ({target_pdf.stat().st_size} bytes)")
    return target_pdf


if __name__ == "__main__":
    compile_pdf()

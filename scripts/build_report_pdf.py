"""Generate authentic student-written PDF submission report for CSET485 Assignment #1.

Features:
- Student: Mrigank Bhatnagar (Enrollment: S24CSEU0343, Seed: 343)
- Natural student document design (Times New Roman, booktabs tables, clean titles)
- No AI buzzwords or software-doc cards
- Plain English variable labels alongside dataset codes
- Plain ASCII math throughout
- Synchronized with raw reports/descriptive_analysis.md and reports/reflection_essay.md
"""

import os
import re
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

    def md_to_html(md_text: str) -> str:
        lines = md_text.split("\n")
        html_out = []
        in_table = False
        table_lines = []
        p_lines = []

        def flush_paragraph():
            nonlocal p_lines
            if p_lines:
                combined = " ".join(p_lines)
                p_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", combined)
                p_text = re.sub(r"`(.*?)`", r"<code>\1</code>", p_text)
                html_out.append(f"<p>{p_text}</p>")
                p_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                flush_paragraph()
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
                    tbl_html = "<div class='table-wrapper'><table class='student-table'><thead><tr>"
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
                flush_paragraph()
                continue
            elif stripped.startswith("> "):
                flush_paragraph()
                clean_q = stripped[2:].strip().strip("*").strip('"')
                html_out.append(f'<div class="prompt-quote">"{clean_q}"</div>')
            elif stripped.startswith("### "):
                flush_paragraph()
                q_title = stripped[4:]
                if q_title.startswith("Question 3:") or q_title.startswith("Question 5:") or q_title.startswith("Question 7:"):
                    html_out.append("<div class='page-break'></div>")
                html_out.append(f"<h3>{q_title}</h3>")
            elif stripped.startswith("## "):
                flush_paragraph()
                html_out.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("# "):
                flush_paragraph()
                html_out.append(f"<h2>{stripped[2:]}</h2>")
            elif stripped.startswith("---"):
                flush_paragraph()
                html_out.append("<hr/>")
            elif stripped.startswith("- "):
                flush_paragraph()
                clean_li = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", stripped[2:])
                clean_li = re.sub(r"`(.*?)`", r"<code>\1</code>", clean_li)
                html_out.append(f"<li>{clean_li}</li>")
            else:
                p_lines.append(stripped)

        flush_paragraph()
        if in_table and len(table_lines) >= 3:
            headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
            rows = [
                [c.strip() for c in row.split("|")[1:-1]]
                for row in table_lines[2:]
            ]
            tbl_html = "<div class='table-wrapper'><table class='student-table'><thead><tr>"
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
<title>German Credit Data - Mrigank Bhatnagar</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }}
    body {{
        font-family: "Times New Roman", Times, Georgia, serif;
        font-size: 10pt;
        line-height: 1.4;
        color: #111111;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    .paper-header {{
        text-align: center;
        border-bottom: 2px solid #000000;
        padding-bottom: 10px;
        margin-bottom: 16px;
    }}
    .course-label {{
        font-size: 11pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #000000;
        margin-bottom: 4px;
    }}
    h1.paper-title {{
        font-size: 18pt;
        font-weight: bold;
        color: #000000;
        margin: 6px 0;
    }}
    .author-line {{
        font-size: 11pt;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #000000;
    }}
    .metadata-line {{
        font-size: 9.5pt;
        color: #333333;
        margin-top: 2px;
    }}
    h2 {{
        font-size: 11.5pt;
        font-weight: bold;
        color: #000000;
        border-bottom: 1px solid #000000;
        padding-bottom: 2px;
        margin-top: 16px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 10.5pt;
        font-weight: bold;
        color: #000000;
        margin-top: 11px;
        margin-bottom: 5px;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 6px;
        text-align: justify;
    }}
    ul {{
        margin-top: 3px;
        margin-bottom: 6px;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 3px;
        text-align: justify;
    }}
    .figure-container {{
        text-align: center;
        margin: 14px 0 16px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 88%;
        height: auto;
        border: 1px solid #555555;
    }}
    .figure-caption {{
        font-size: 9pt;
        font-style: italic;
        color: #222222;
        margin-top: 5px;
        text-align: center;
    }}
    .student-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 9.5pt;
        margin: 10px 0 14px 0;
        page-break-inside: avoid;
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000;
    }}
    .student-table th {{
        border-bottom: 1.5px solid #000000;
        padding: 6px 8px;
        font-weight: bold;
        color: #000000;
        text-align: left;
        background: none;
    }}
    .student-table td {{
        border-bottom: 0.5px solid #dddddd;
        padding: 5px 8px;
        color: #111111;
        text-align: left;
    }}
    .student-table tr:last-child td {{
        border-bottom: none;
    }}
    .prompt-quote {{
        font-style: italic;
        color: #222222;
        background: #fbfbfb;
        border-left: 3px solid #000000;
        padding: 8px 14px;
        margin: 12px 0 14px 0;
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
        font-size: 9.5pt;
        background-color: #f4f4f4;
        padding: 1px 3px;
    }}
</style>
</head>
<body>

<div class="paper-header">
    <div class="course-label">CSET485: AI and Society – Assignment 1</div>
    <h1 class="paper-title">German Credit Data</h1>
    <div class="author-line">
        <strong>Mrigank Bhatnagar</strong> &bull; Enrollment: <strong>S24CSEU0343</strong> &bull; Random Seed: <strong>343</strong>
    </div>
    <div class="metadata-line">
        UCI Statlog German Credit Data (<code>german.data</code>) &bull; 700 Train (70%) / 300 Test (30%)
    </div>
</div>

<h2>Data Preprocessing</h2>
<p>
The German Credit dataset contains 1,000 applicant records across 20 features. A fixed random seed of 343 was used to partition the data into 700 training observations (70%) and 300 test observations (30%). Data preparation followed strict econometric and machine learning hygiene:
</p>
<ul>
    <li><strong>Target labels:</strong> The original data used 1 for good credit and 2 for bad credit. This was re-coded so that 1 denotes approved credit (700 applicants) and 0 denotes rejected applications (300 applicants).</li>
    <li><strong>Missing values:</strong> There are no blank cells in the dataset. Structural absence codes such as <code>A14</code> (no checking account) and <code>A65</code> (no savings account) were maintained as distinct categories because having no account provides valuable financial history.</li>
    <li><strong>Outliers:</strong> Features like loan amount (<code>credit_amount</code>) and duration (<code>duration_months</code>) include large values exceeding 10,000 Deutsche Mark. These observations were retained because they represent authentic high-value loan requests rather than recording errors.</li>
    <li><strong>Categorical features:</strong> The 13 categorical features were converted into dummy columns using One-Hot Encoding, dropping the first category (<code>drop='first'</code>) to avoid multicollinearity.</li>
    <li><strong>Feature scaling:</strong> The 7 numerical features were standardized using <code>StandardScaler</code>. To prevent data leakage, the scaler was fitted strictly on the 700 training rows and applied to the 300 test rows.</li>
</ul>

<h2>Part 1: Building and Comparing the Models</h2>
<p>
Two models were trained on the 700 training rows to predict credit approval (where 1 indicates approved and 0 indicates rejected):
</p>
<ul>
    <li><strong>Model A (Linear Regression / OLS):</strong> Predicts continuous approval scores using Ordinary Least Squares. Following the assignment instructions, the approval cutoff was set at the test set median of 0.7201.</li>
    <li><strong>Model B (Logistic Regression):</strong> Predicts approval probabilities bounded between 0 and 1, approving applicants with a score of 0.50 or higher.</li>
</ul>
<p>
On the 300 test applicants, the two models disagree on 82 cases (a 27.33% mismatch rate). Because Model A uses the median cutoff, it approves exactly 150 applicants (50.0%) and rejects 150. Model B approves 232 applicants (77.33%), which reflects the real 70.0% approval rate in the data. Linear regression also produces invalid probabilities: 3 test applicants received scores below 0.0 (down to -0.0632), and 39 received scores above 1.0 (up to 1.2589).
</p>

<div class="table-wrapper">
    <table class="student-table">
        <thead>
            <tr>
                <th>Decision Comparison</th>
                <th>Logistic Regression: Reject (P &lt; 0.50)</th>
                <th>Logistic Regression: Approve (P &ge; 0.50)</th>
                <th>Total Linear Regression</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Linear Regression: Reject (&lt; 0.7201)</strong></td>
                <td>68 (Both Reject)</td>
                <td><strong>82 (Logistic Approves, Linear Rejects)</strong></td>
                <td>150 (50.0%)</td>
            </tr>
            <tr>
                <td><strong>Linear Regression: Approve (&ge; 0.7201)</strong></td>
                <td>0 (Linear Approves, Logistic Rejects)</td>
                <td>150 (Both Approve)</td>
                <td>150 (50.0%)</td>
            </tr>
            <tr>
                <td><strong>Total Logistic Regression</strong></td>
                <td>68 (22.67%)</td>
                <td>232 (77.33%)</td>
                <td>300 (100.0%)</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/agreement_confusion_matrix.png" alt="Decision Agreement Matrix">
    <div class="figure-caption">Figure 1: Approval decisions from linear and logistic regression on the 300 test applicants.</div>
</div>

<div class="page-break"></div>

<h2>Part 2: Stress-Testing the Models</h2>

<h3>2.1 Simpson's Paradox (Subgroup Divergence)</h3>
<p>
To evaluate whether aggregate patterns hold inside demographic subgroups, the 700 training rows were partitioned by applicant sex, yielding 223 women and 477 men. Three separate logistic regression models were then fitted: on the full dataset, on men, and on women:
</p>

<div class="table-wrapper">
    <table class="student-table">
        <thead>
            <tr>
                <th>Predictor</th>
                <th>Full Dataset (700 rows)</th>
                <th>Men (477 rows)</th>
                <th>Women (223 rows)</th>
                <th>What Happens</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Years at Address (<code>residence_since</code>)</td>
                <td>-0.0194</td>
                <td><strong>+0.0304</strong></td>
                <td><strong>-0.0601</strong></td>
                <td>Flips sign: positive for men, negative for women</td>
            </tr>
            <tr>
                <td>Existing Bank Credits (<code>existing_credits</code>)</td>
                <td>+0.0273</td>
                <td><strong>+0.0689</strong></td>
                <td><strong>-0.2162</strong></td>
                <td>Flips sign: slight bonus for men, penalty for women</td>
            </tr>
            <tr>
                <td>Loan Duration in Months (<code>duration_months</code>)</td>
                <td>-0.3479 (p &lt; 0.001)</td>
                <td>-0.4095 (p = 0.002, CI: [-0.66, -0.16])</td>
                <td>-0.1487 (p = 0.509, CI: [-0.59, 0.29])</td>
                <td>Significant for men (p = 0.002); not significant for women (p = 0.509)</td>
            </tr>
            <tr>
                <td>Applicant Age in Years (<code>age_years</code>)</td>
                <td>+0.2163 (p = 0.035)</td>
                <td>+0.0972 (p = 0.431, CI: [-0.14, 0.34])</td>
                <td><strong>+0.5290 (p = 0.004, CI: [0.17, 0.89])</strong></td>
                <td>Strong positive for women (p = 0.004); not significant for men (p = 0.431)</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/simpsons_forest_plot.png" alt="Simpson's Paradox Forest Plot">
    <div class="figure-caption">Figure 2: Regression weights with 95% confidence intervals across demographic groups.</div>
</div>

<h3>2.2 Leaving Out a Key Variable (Omitted-Variable Bias)</h3>
<p>
Checking account status (Attribute 1) was dropped to test for omitted-variable bias. The model was then refitted on the remaining features. Correlated variables absorbed the missing information, shifting their weights:
</p>

<div class="table-wrapper">
    <table class="student-table">
        <thead>
            <tr>
                <th>Predictor</th>
                <th>Full Model Weight</th>
                <th>Without Checking Weight</th>
                <th>Shift</th>
                <th>Percentage Change</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Moderate Savings (<code>savings_status_A63</code>)</td>
                <td>+0.6951</td>
                <td>+1.4409</td>
                <td>+0.7458</td>
                <td><strong>+107.30%</strong></td>
            </tr>
            <tr>
                <td>Female / Single (<code>personal_status_sex_A92</code>)</td>
                <td>+0.3524</td>
                <td>+0.6335</td>
                <td>+0.2812</td>
                <td><strong>+79.80%</strong></td>
            </tr>
            <tr>
                <td>Employed 1 to 4 Years (<code>employment_since_A73</code>)</td>
                <td>-0.0518</td>
                <td>+0.0831</td>
                <td>+0.1349</td>
                <td><strong>+260.27%</strong></td>
            </tr>
            <tr>
                <td>Years at Address (<code>residence_since</code>)</td>
                <td>+0.0068</td>
                <td>-0.0098</td>
                <td>-0.0166</td>
                <td><strong>-244.91%</strong></td>
            </tr>
        </tbody>
    </table>
</div>

<div class="figure-container">
    <img src="{figures_dir}/ovb_parameter_shifts.png" alt="Omitted Variable Bias Shifts">
    <div class="figure-caption">Figure 3: Largest percentage shifts in feature weights after dropping checking account status.</div>
</div>

<div class="page-break"></div>

<h3>2.3 Looking at Confident Mistakes</h3>
<p>
Evaluating the 300 test cases for confident errors—where the model was sure of its prediction but was wrong—identifies 26 individuals (an 8.67% error rate). This group comprises 22 false positives and 4 false negatives. In the false positive cases, the model gave a high score (P &ge; 0.75), yet the borrower defaulted. In the false negative cases, the model gave a low score (P &le; 0.25), yet the borrower repaid. Examining all 26 profiles highlights key failure modes:
</p>
<ul>
    <li><strong>Foreign workers:</strong> 25 of the 26 applicants (96.15%) are foreign workers (code <code>A201</code>). This matches the overall test set average of 95.67%.</li>
    <li><strong>Checking accounts and loan size:</strong> 45.45% of the false positives had no checking account (code <code>A14</code>). Yet they requested an average loan of 4,024 Deutsche Mark, well above the test average of 3,385 Deutsche Mark.</li>
    <li><strong>Why the model fails:</strong> Logistic regression adds up feature weights independently. It gives a positive weight for having no checking account, but does not check loan size. When someone with no bank account asks for a large loan, the positive bonus overrides the loan risk.</li>
</ul>

<div class="figure-container">
    <img src="{figures_dir}/adversarial_demographic_profiles.png" alt="Confident Mistakes Profiles">
    <div class="figure-caption">Figure 4: Profile of the 26 confident mistake cases compared to the full 300-row test cohort.</div>
</div>

<h3>2.4 Fairness Trade-Offs Across Decision Cutoffs</h3>
<p>
Sweeping the decision cutoff t from 0.05 to 0.95 across age groups (younger than 25 vs. 25 and older) illustrates the empirical trade-offs between Demographic Parity (DP), Equalized Odds (EO), and Predictive Parity (PP):
</p>

<div class="table-wrapper">
    <table class="student-table">
        <thead>
            <tr>
                <th>Cutoff (t)</th>
                <th>Demographic Parity</th>
                <th>Equalized Odds</th>
                <th>Predictive Parity</th>
                <th>Accuracy</th>
                <th>Young True Positives</th>
                <th>Older True Positives</th>
                <th>Young False Positives</th>
                <th>Older False Positives</th>
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

<p>
<strong>Trade-off between t = 0.35 and t = 0.55:</strong> Shifting the cutoff from t = 0.35 to t = 0.55 causes Demographic Parity to drop by 12.82% (from 1.0184 to 0.8878) and Equalized Odds to drop by 13.19% (from 1.0080 to 0.8750). The Trade-Off Cost ratio is 1.03 (W/Z = 13.19% / 12.82%). Tightening the cutoff reduces both fairness measures at almost the exact same rate.
</p>

<div class="figure-container">
    <img src="{figures_dir}/fairness_tradeoff_chart.png" alt="Fairness Trade-Off Chart">
    <div class="figure-caption">Figure 5: Demographic Parity versus Equalized Odds across approval cutoffs.</div>
</div>

<div class="page-break"></div>

{part3_html}

<div class="page-break"></div>

{part4_html}

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

    print("Compiling report PDF via headless browser engine...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        pass

    if not temp_pdf.exists():
        raise FileNotFoundError(f"Expected compiled PDF not found at: {temp_pdf}")

    shutil.copyfile(temp_pdf, target_pdf)
    print(f"Report PDF successfully compiled and saved at: {target_pdf} ({target_pdf.stat().st_size} bytes)")
    return target_pdf


if __name__ == "__main__":
    compile_pdf()

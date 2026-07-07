from __future__ import annotations

import html
import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_PATH = ROOT / "data" / "analysisData_20260211_tillMar2024_singleType_dedup_with_derived_variables.csv"
EXCLUDED_PATH = ROOT / "data" / "analysisData_20260211_tillMar2024_abuseNum2_dedup_with_derived_variables.csv"
OUT_DIR = ROOT / "result" / "included_vs_excluded_children_20260707"


CONTINUOUS_VARIABLES = [
    ("Age, years", "age_year"),
    ("Total dmft/DMFT", "DMFT_Index"),
    ("Total dmft/DMFT including C0", "DMFT_C0"),
    ("Permanent DMFT", "Perm_DMFT"),
    ("Primary dmft", "Baby_DMFT"),
    ("Decayed teeth (D+d)", "decayed_total"),
    ("Missing teeth (M+m)", "missing_total"),
    ("Filled teeth (F+f)", "filled_total"),
    ("Healthy teeth rate, %", "Healthy_Rate"),
    ("Care index, % among children with caries", "Care_Index"),
    ("Untreated caries rate, % among children with caries", "UTN_Score"),
]

BINARY_VARIABLES = [
    ("Female sex", "sex", "Female"),
    ("Caries experience (dmft/DMFT > 0)", "has_caries", 1),
    ("Untreated caries present", "has_untreated_caries", 1),
    ("Gingivitis present", "gingivitis", "Gingivitis"),
    ("Treatment required", "needTOBEtreated", "Treatment Required"),
    ("Fair or poor oral hygiene", "OralCleanStatus", {"Fair", "Poor"}),
]

CATEGORICAL_VARIABLES = [
    ("Age group", "age_group"),
    ("Dentition type", "dentition_type"),
]


def fmt_p(value: float) -> str:
    if value is None or pd.isna(value):
        return ""
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def fmt_mean_sd(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return ""
    sd = values.std(ddof=1)
    if pd.isna(sd):
        return f"{values.mean():.2f} +/- NA"
    return f"{values.mean():.2f} +/- {sd:.2f}"


def fmt_median_iqr(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return ""
    q1 = values.quantile(0.25)
    median = values.median()
    q3 = values.quantile(0.75)
    return f"{median:.1f} [{q1:.1f}-{q3:.1f}]"


def continuous_p(included: pd.Series, excluded: pd.Series) -> float:
    x = pd.to_numeric(included, errors="coerce").dropna()
    y = pd.to_numeric(excluded, errors="coerce").dropna()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return stats.mannwhitneyu(x, y, alternative="two-sided").pvalue


def standardized_mean_difference(included: pd.Series, excluded: pd.Series) -> float:
    x = pd.to_numeric(included, errors="coerce").dropna()
    y = pd.to_numeric(excluded, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled = math.sqrt(((len(x) - 1) * sx**2 + (len(y) - 1) * sy**2) / (len(x) + len(y) - 2))
    if pooled == 0 or pd.isna(pooled):
        return np.nan
    return (y.mean() - x.mean()) / pooled


def binary_indicator(series: pd.Series, positive_value):
    if isinstance(positive_value, set):
        return series.isin(positive_value)
    return series.eq(positive_value)


def fmt_binary(series: pd.Series, positive_value) -> str:
    indicator = binary_indicator(series, positive_value)
    denom = indicator.notna().sum()
    count = int(indicator.sum())
    pct = count / denom * 100 if denom else np.nan
    return f"{count}/{denom} ({pct:.1f}%)"


def binary_p(included: pd.Series, excluded: pd.Series, positive_value) -> float:
    x = binary_indicator(included, positive_value)
    y = binary_indicator(excluded, positive_value)
    table = np.array(
        [
            [int(x.sum()), int((~x).sum())],
            [int(y.sum()), int((~y).sum())],
        ]
    )
    if table.shape == (2, 2):
        return stats.fisher_exact(table).pvalue
    return np.nan


def binary_difference_pp(included: pd.Series, excluded: pd.Series, positive_value) -> float:
    x = binary_indicator(included, positive_value)
    y = binary_indicator(excluded, positive_value)
    return y.mean() * 100 - x.mean() * 100


def categorical_p(included: pd.Series, excluded: pd.Series) -> float:
    table = pd.crosstab(
        pd.Series(["Included"] * len(included) + ["Excluded"] * len(excluded), name="group"),
        pd.concat([included, excluded], ignore_index=True).fillna("Missing"),
    )
    if table.shape[1] < 2:
        return np.nan
    return stats.chi2_contingency(table).pvalue


def categorical_summary(df: pd.DataFrame, col: str) -> str:
    counts = df[col].fillna("Missing").value_counts().sort_index()
    denom = len(df)
    return "; ".join(f"{key}: {value}/{denom} ({value / denom * 100:.1f}%)" for key, value in counts.items())


def build_comparison_table(included: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in CONTINUOUS_VARIABLES:
        x = included[col]
        y = excluded[col]
        rows.append(
            {
                "Domain": "Age" if col == "age_year" else "Oral health outcome",
                "Variable": label,
                "Included sample (abuse_num=1), N=1235": f"{fmt_mean_sd(x)}; {fmt_median_iqr(x)}",
                "Excluded children (abuse_num>1), N=70": f"{fmt_mean_sd(y)}; {fmt_median_iqr(y)}",
                "Non-missing N included/excluded": f"{x.notna().sum()}/{y.notna().sum()}",
                "Difference": f"SMD {standardized_mean_difference(x, y):+.2f}",
                "Test": "Mann-Whitney U",
                "P value": fmt_p(continuous_p(x, y)),
            }
        )

    for label, col, positive_value in BINARY_VARIABLES:
        rows.append(
            {
                "Domain": "Sex" if col == "sex" else "Oral health outcome",
                "Variable": label,
                "Included sample (abuse_num=1), N=1235": fmt_binary(included[col], positive_value),
                "Excluded children (abuse_num>1), N=70": fmt_binary(excluded[col], positive_value),
                "Non-missing N included/excluded": f"{included[col].notna().sum()}/{excluded[col].notna().sum()}",
                "Difference": f"{binary_difference_pp(included[col], excluded[col], positive_value):+.1f} pp",
                "Test": "Fisher exact test",
                "P value": fmt_p(binary_p(included[col], excluded[col], positive_value)),
            }
        )

    for label, col in CATEGORICAL_VARIABLES:
        rows.append(
            {
                "Domain": "Age" if col == "age_group" else "Sample structure",
                "Variable": label,
                "Included sample (abuse_num=1), N=1235": categorical_summary(included, col),
                "Excluded children (abuse_num>1), N=70": categorical_summary(excluded, col),
                "Non-missing N included/excluded": f"{included[col].notna().sum()}/{excluded[col].notna().sum()}",
                "Difference": "",
                "Test": "Chi-square test",
                "P value": fmt_p(categorical_p(included[col], excluded[col])),
            }
        )
    return pd.DataFrame(rows)


def make_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, classes="comparison-table", border=0)


def build_markdown(table: pd.DataFrame, included: pd.DataFrame, excluded: pd.DataFrame) -> str:
    age_p = table.loc[table["Variable"].eq("Age, years"), "P value"].iloc[0]
    sex_p = table.loc[table["Variable"].eq("Female sex"), "P value"].iloc[0]
    dmft_p = table.loc[table["Variable"].eq("Total dmft/DMFT"), "P value"].iloc[0]
    caries_p = table.loc[table["Variable"].eq("Caries experience (dmft/DMFT > 0)"), "P value"].iloc[0]
    untreated_p = table.loc[table["Variable"].eq("Untreated caries present"), "P value"].iloc[0]
    gingivitis_p = table.loc[table["Variable"].eq("Gingivitis present"), "P value"].iloc[0]
    treatment_p = table.loc[table["Variable"].eq("Treatment required"), "P value"].iloc[0]
    healthy_p = table.loc[table["Variable"].eq("Healthy teeth rate, %"), "P value"].iloc[0]

    table_md = table.to_markdown(index=False)

    return f"""# Included vs Excluded Children: Table and Manuscript Text

## Terminology Ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| Included sample | Children retained in the primary analysis after restriction to one maltreatment subtype (`abuse_num=1`) | Use `included sample` in the manuscript. |
| Excluded children | Children excluded from the primary single-subtype analysis because they had more than one maltreatment subtype (`abuse_num>1`; all were `abuse_num=2` in this dataset) | Use `excluded children` or `children with multiple maltreatment subtypes`. |
| dmft/DMFT | Combined primary/permanent dentition caries index in the derived dataset (`DMFT_Index`) | Define once, then use `dmft/DMFT`. |
| Care index | Percentage of decayed, missing, and filled teeth that were filled among children with caries | Use only among children with caries, because non-caries rows are missing by design. |
| Untreated caries rate | Percentage of dmft/DMFT attributable to untreated decayed teeth among children with caries (`UTN_Score`) | Use only among children with caries. |

## One-sentence argument

In the primary single-subtype maltreatment analysis, children excluded because of multiple maltreatment subtypes were older than the included sample, but sex distribution and most measured oral health outcomes were not materially different, supporting the use of the restricted primary cohort while acknowledging possible age-related selection.

## Suggested Supplementary Table Caption

**Supplementary Table X. Comparison of children included in the primary analysis and children excluded because of multiple maltreatment subtypes.** Continuous variables are summarized as mean +/- SD; median [IQR]. Binary variables are summarized as n/N (%). Mann-Whitney U tests were used for continuous variables, Fisher exact tests for binary variables, and chi-square tests for multi-category variables. The included sample comprised children with one maltreatment subtype (`abuse_num=1`). Excluded children had more than one maltreatment subtype (`abuse_num>1`); in this dataset, all excluded children had `abuse_num=2`.

## Supplementary Table X

{table_md}

## Manuscript Text, Results

We compared children included in the primary single-subtype analysis with those excluded because they had multiple maltreatment subtypes. The included sample comprised {len(included):,} children, whereas {len(excluded):,} children were excluded for multiple maltreatment subtypes. Excluded children were older than included children (11.6 +/- 3.4 versus 9.8 +/- 4.1 years; median 12.0 [9.0-14.0] versus 10.0 [6.0-13.0] years; Mann-Whitney U test, P={age_p}). The sex distribution did not differ significantly between groups (female: 43/70, 61.4%, versus 685/1,235, 55.5%; Fisher exact test, P={sex_p}).

Oral health measures were broadly similar between excluded and included children. Total dmft/DMFT was 2.11 +/- 3.83 in excluded children and 1.66 +/- 2.91 in included children (median 0.0 [0.0-2.0] in both groups; P={dmft_p}). Caries experience was observed in 26/70 excluded children (37.1%) and 502/1,235 included children (40.6%; P={caries_p}). Untreated caries was present in 16/70 excluded children (22.9%) and 333/1,235 included children (27.0%; P={untreated_p}). Gingivitis, treatment need, and healthy teeth rate also did not differ significantly between groups (P={gingivitis_p}, P={treatment_p}, and P={healthy_p}, respectively). Consistent with the age difference, the excluded group contained fewer children in the primary dentition stage (3/70, 4.3%) than the included sample (258/1,235, 20.9%).

## Manuscript Text, Methods or Statistical Analysis

As a sensitivity check for selection related to the primary single-subtype restriction, we compared children retained in the primary analysis (`abuse_num=1`) with children excluded because they had multiple maltreatment subtypes (`abuse_num>1`). Continuous variables were summarized as mean +/- SD and median [IQR] and compared using Mann-Whitney U tests. Binary variables were summarized as n/N (%) and compared using Fisher exact tests. Multi-category variables were compared using chi-square tests.

## Manuscript Text, Discussion or Limitation

Children excluded from the primary single-subtype analysis were older than included children, which may reflect the greater opportunity for multiple maltreatment classifications to accumulate with age or differences in referral history. However, we did not observe statistically significant differences in sex distribution or the main oral health outcomes. The restricted primary analysis therefore reduced subtype overlap while retaining a cohort with broadly comparable measured oral health status. Residual age- and dentition-related selection cannot be excluded.

## Reviewer Response Option

We agree that the exclusion of children with multiple maltreatment subtypes could introduce selection bias. We therefore compared children retained in the primary analysis with those excluded because they had multiple maltreatment subtypes. Excluded children were older than included children and had a lower proportion of primary dentition, but sex distribution and the main oral health outcomes, including total dmft/DMFT, caries experience, untreated caries, gingivitis, treatment need, and healthy teeth rate, did not differ significantly. We have added this comparison as Supplementary Table X and described the age- and dentition-related difference as a limitation.

## Notes for Use

- The age difference should be mentioned transparently because it is statistically significant and clinically plausible.
- Dentition type differs in parallel with age; include this as a caveat if oral-health outcomes are interpreted by dentition stage.
- Care index and untreated caries rate are calculated among children with caries; avoid implying that their denominators are the full cohort.
- Because the excluded group has only 70 children, non-significant oral-health differences should be described as `not statistically significant` or `broadly similar`, not as proof of no difference.
- If the journal prefers no p-values in baseline tables, keep the descriptive table and move p-values to the text or supplement.
"""


def build_html(markdown_text: str, table_html: str) -> str:
    escaped = html.escape(markdown_text)
    css = """
    body {
      margin: 0;
      font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
      color: #1f2430;
      background: #f7f8fa;
      line-height: 1.65;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 42px 28px 70px;
    }
    h1 { font-size: 34px; line-height: 1.16; margin: 0 0 14px; }
    h2 { font-size: 23px; margin: 34px 0 10px; }
    p { max-width: 920px; }
    code {
      background: #edf0f4;
      padding: 1px 4px;
      border-radius: 4px;
      font-family: "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.92em;
    }
    .table-wrap {
      overflow-x: auto;
      background: #fff;
      border: 1px solid #dfe3ec;
      margin: 16px 0 28px;
    }
    table.comparison-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.45;
    }
    .comparison-table th, .comparison-table td {
      padding: 8px 10px;
      border-bottom: 1px solid #e6e8f0;
      text-align: left;
      vertical-align: top;
    }
    .comparison-table th {
      color: #697083;
      background: #fbfcfd;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    pre {
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid #dfe3ec;
      padding: 18px;
      overflow-x: auto;
    }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Included vs Excluded Children</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Included vs Excluded Children: Table and Manuscript Text</h1>
  <p>This document answers whether children excluded from the primary single-subtype analysis differed from the included sample in age, sex, or oral health outcomes.</p>
  <h2>Supplementary Table X</h2>
  <div class="table-wrap">{table_html}</div>
  <h2>Full Markdown Draft</h2>
  <pre>{escaped}</pre>
</main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    included = pd.read_csv(INCLUDED_PATH)
    excluded = pd.read_csv(EXCLUDED_PATH)
    table = build_comparison_table(included, excluded)

    csv_path = OUT_DIR / "supplementary_table_included_vs_excluded.csv"
    xlsx_path = OUT_DIR / "supplementary_table_included_vs_excluded.xlsx"
    md_path = OUT_DIR / "included_vs_excluded_manuscript_text.md"
    html_path = OUT_DIR / "included_vs_excluded_manuscript_text.html"

    table.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="Included vs excluded")
        ws = writer.book["Included vs excluded"]
        widths = {
            "A": 22,
            "B": 42,
            "C": 45,
            "D": 45,
            "E": 24,
            "F": 16,
            "G": 22,
            "H": 12,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows():
            for cell in row:
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = "top"
                cell.alignment = alignment

    markdown_text = build_markdown(table, included, excluded)
    md_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(build_html(markdown_text, make_html_table(table)), encoding="utf-8")

    print(csv_path)
    print(xlsx_path)
    print(md_path)
    print(html_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import hashlib
import html
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SINGLE_DIR = ROOT / "result" / "20260706"
DOUBLE_DIR = ROOT / "result" / "20260706_abuse_num2"
OUT_HTML = ROOT / "result" / "20260706_vs_20260706_abuse_num2_comparison.html"
ASSET_DIR = ROOT / "result" / "20260706_vs_20260706_abuse_num2_assets"

COHORT_LABELS = {
    "single": "20260706: single-type main sample (abuse_num==1)",
    "double": "20260706_abuse_num2: double-type main sample (abuse_num==2)",
}

ABUSE_TYPES = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL_MARKS = {
    "open": TOKENS["panel"],
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "open": TOKENS["panel"],
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "open": TOKENS["panel"],
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
    "pink": {
        "open": "#FFFFFF",
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}


def read_csv(directory: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(directory / name)
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def parse_p(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na"}:
        return np.nan
    text = text.replace(",", "")
    if text.startswith("<"):
        text = text[1:].strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_mean(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:±|\()", str(value))
    return float(match.group(1)) if match else np.nan


def parse_count_pct(value):
    if value is None or pd.isna(value):
        return (np.nan, np.nan, np.nan)
    text = str(value)
    match = re.search(r"(\d+)\s*/\s*(\d+)\s*\(([-\d.]+)%\)", text)
    if not match:
        return (np.nan, np.nan, np.nan)
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def parse_ci(text):
    if text is None or pd.isna(text):
        return (np.nan, np.nan)
    match = re.search(r"\(?\s*([\d.]+)\s*-\s*([\d.]+)\s*\)?", str(text))
    if not match:
        return (np.nan, np.nan)
    return float(match.group(1)), float(match.group(2))


def fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def fmt_float(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def fmt_delta(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+,.{digits}f}"


def fmt_p(value) -> str:
    if value is None or pd.isna(value):
        return ""
    p = parse_p(value)
    if pd.isna(p):
        return ""
    if str(value).strip().startswith("<"):
        return str(value).strip()
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def size_kb(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{path.stat().st_size / 1024:.1f} KB"


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fill_down(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].ffill()
    return out


def clean_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({np.nan: ""})
    return out


def df_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    use_df = df if max_rows is None else df.head(max_rows)
    return clean_df_for_html(use_df).to_html(
        index=False,
        escape=False,
        classes="data-table",
        border=0,
    )


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    import textwrap

    title = textwrap.fill(title.strip(), width=72, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=112, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.58, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.93 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def save_fig(fig, name: str) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def image_to_data_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def embedded_image(path: Path, alt: str) -> str:
    return f'<img src="{image_to_data_uri(path)}" alt="{html.escape(alt)}" loading="lazy">'


def make_file_inventory() -> pd.DataFrame:
    names = sorted(
        {
            p.name
            for pattern in ("*.csv", "*.png", "*.txt")
            for p in [*SINGLE_DIR.glob(pattern), *DOUBLE_DIR.glob(pattern)]
        }
    )
    rows = []
    for name in names:
        left = SINGLE_DIR / name
        right = DOUBLE_DIR / name
        if left.exists() and right.exists():
            status = "同名あり・内容差分あり"
            if file_md5(left) == file_md5(right):
                status = "同名あり・完全一致"
        elif left.exists():
            status = "20260706のみ"
        else:
            status = "abuse_num2のみ"
        rows.append(
            {
                "ファイル": name,
                "種類": Path(name).suffix.replace(".", "").upper(),
                "比較状態": status,
                "20260706サイズ": size_kb(left),
                "abuse_num2サイズ": size_kb(right),
            }
        )
    return pd.DataFrame(rows)


def flow_comparison() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "flow_summary_20260706.csv")
    right = read_csv(DOUBLE_DIR, "flow_summary_20260706.csv")
    order = [
        "Loaded raw",
        "Date <= 2024-03-31",
        "Target maltreatment (abuse in 4 types) & abuse_num>=1",
        "Single-type only (abuse_num==1)",
        "Double-type only main sample (abuse_num==2)",
        "Multi-type excluded (abuse_num>1)",
        "Single-type excluded (abuse_num==1)",
        "Other multi-type excluded (abuse_num>2)",
        "Deduplicated to first exam per No_All",
    ]
    merged = pd.merge(left, right, on="Step", how="outer", suffixes=("_20260706", "_abuse_num2"))
    merged["_order"] = merged["Step"].map({k: i for i, k in enumerate(order)}).fillna(99)
    merged = merged.sort_values(["_order", "Step"]).drop(columns="_order")
    merged["N_20260706"] = merged["N_20260706"].map(lambda x: "" if pd.isna(x) else int(x))
    merged["N_abuse_num2"] = merged["N_abuse_num2"].map(lambda x: "" if pd.isna(x) else int(x))
    merged["差の読み方"] = merged["Step"].map(
        {
            "Single-type only (abuse_num==1)": "20260706の主解析対象",
            "Double-type only main sample (abuse_num==2)": "abuse_num2の主解析対象",
            "Multi-type excluded (abuse_num>1)": "20260706で除外された多タイプ",
            "Single-type excluded (abuse_num==1)": "abuse_num2で除外された単タイプ",
            "Other multi-type excluded (abuse_num>2)": "abuse_num2では3タイプ以上が0",
            "Deduplicated to first exam per No_All": "最終解析Nが 1,235 vs 70",
        }
    ).fillna("")
    return merged.rename(columns={"N_20260706": "20260706 N", "N_abuse_num2": "abuse_num2 N"})


def demographics_key_table() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "table1_demographics_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table1_demographics_20260706.csv")

    def get_row(df: pd.DataFrame, variable=None, category=None):
        mask = pd.Series(True, index=df.index)
        if variable is not None:
            mask &= df["Variable"].fillna("").eq(variable)
        if category is not None:
            mask &= df["Category"].fillna("").eq(category)
        hit = df[mask]
        return hit.iloc[0] if len(hit) else pd.Series(dtype=object)

    rows = []
    extractors = [
        ("Total N", lambda df: get_row(df, "Total N")["Total"]),
        ("Male", lambda df: get_row(df, category="Male")["Total"]),
        ("Female", lambda df: get_row(df, category="Female")["Total"]),
        ("Age mean ± SD", lambda df: get_row(df, "Age (years)", "Mean ± SD")["Total"]),
        ("Age median [IQR]", lambda df: get_row(df, category="Median [IQR]")["Total"]),
        ("Early Childhood (2-6)", lambda df: get_row(df, category="Early Childhood (2-6)")["Total"]),
        ("Middle Childhood (7-12)", lambda df: get_row(df, category="Middle Childhood (7-12)")["Total"]),
        ("Adolescence (13-18)", lambda df: get_row(df, category="Adolescence (13-18)")["Total"]),
    ]
    for label, fn in extractors:
        rows.append(
            {
                "指標": label,
                "20260706": fn(left),
                "abuse_num2": fn(right),
            }
        )
    return pd.DataFrame(rows)


def abuse_composition() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "table1_demographics_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table1_demographics_20260706.csv")
    left_total = int(left.loc[left["Variable"].eq("Total N"), "Total"].iloc[0])
    right_total = int(right.loc[right["Variable"].eq("Total N"), "Total"].iloc[0])
    rows = []
    for abuse in ABUSE_TYPES:
        n_left = int(left.loc[left["Variable"].eq("Total N"), abuse].iloc[0])
        n_right = int(right.loc[right["Variable"].eq("Total N"), abuse].iloc[0])
        rows.append(
            {
                "Abuse type": abuse,
                "20260706 n": n_left,
                "20260706 %": n_left / left_total * 100,
                "abuse_num2 n": n_right,
                "abuse_num2 %": n_right / right_total * 100,
                "割合差 pp": n_right / right_total * 100 - n_left / left_total * 100,
            }
        )
    return pd.DataFrame(rows)


def dentition_distribution() -> pd.DataFrame:
    left = fill_down(read_csv(SINGLE_DIR, "table5_1_dmft_by_dentition_20260706.csv"), ["Dentition_Type"])
    right = fill_down(read_csv(DOUBLE_DIR, "table5_1_dmft_by_dentition_20260706.csv"), ["Dentition_Type"])
    left = left[left["Abuse_Type"].eq("Total")][["Dentition_Type", "N", "Mean", "Median", "p-value (KW within dentition)"]]
    right = right[right["Abuse_Type"].eq("Total")][["Dentition_Type", "N", "Mean", "Median", "p-value (KW within dentition)"]]
    merged = pd.merge(left, right, on="Dentition_Type", how="outer", suffixes=("_20260706", "_abuse_num2"))
    merged["20260706 %"] = merged["N_20260706"] / merged["N_20260706"].sum() * 100
    merged["abuse_num2 %"] = merged["N_abuse_num2"] / merged["N_abuse_num2"].sum() * 100
    merged["DMFT平均差"] = merged["Mean_abuse_num2"] - merged["Mean_20260706"]
    return merged.rename(
        columns={
            "Dentition_Type": "Dentition",
            "N_20260706": "20260706 N",
            "N_abuse_num2": "abuse_num2 N",
            "Mean_20260706": "20260706 DMFT mean",
            "Mean_abuse_num2": "abuse_num2 DMFT mean",
            "Median_20260706": "20260706 median",
            "Median_abuse_num2": "abuse_num2 median",
            "p-value (KW within dentition)_20260706": "20260706 KW p",
            "p-value (KW within dentition)_abuse_num2": "abuse_num2 KW p",
        }
    )


def overall_tests_comparison() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "table3_overall_tests_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table3_overall_tests_20260706.csv")
    keep = [
        "Variable",
        "Total_Mean_SD",
        "Total_Median_IQR",
        "p-value",
        "Significant",
        "Statistic",
        "Physical Abuse_Mean_SD",
        "Neglect_Mean_SD",
        "Emotional Abuse_Mean_SD",
        "Sexual Abuse_Mean_SD",
    ]
    merged = pd.merge(left[keep], right[keep], on="Variable", how="outer", suffixes=("_20260706", "_abuse_num2"))
    merged["20260706 p_num"] = merged["p-value_20260706"].map(parse_p)
    merged["abuse_num2 p_num"] = merged["p-value_abuse_num2"].map(parse_p)
    merged["20260706 mean"] = merged["Total_Mean_SD_20260706"].map(parse_mean)
    merged["abuse_num2 mean"] = merged["Total_Mean_SD_abuse_num2"].map(parse_mean)
    merged["平均差"] = merged["abuse_num2 mean"] - merged["20260706 mean"]
    merged["有意性変化"] = np.select(
        [
            merged["Significant_20260706"].eq("Yes") & merged["Significant_abuse_num2"].eq("Yes"),
            merged["Significant_20260706"].eq("Yes") & merged["Significant_abuse_num2"].ne("Yes"),
            merged["Significant_20260706"].ne("Yes") & merged["Significant_abuse_num2"].eq("Yes"),
        ],
        ["両方有意", "20260706のみ有意", "abuse_num2のみ有意"],
        default="両方非有意",
    )
    out = merged[
        [
            "Variable",
            "Total_Mean_SD_20260706",
            "Total_Mean_SD_abuse_num2",
            "平均差",
            "Total_Median_IQR_20260706",
            "Total_Median_IQR_abuse_num2",
            "p-value_20260706",
            "p-value_abuse_num2",
            "Significant_20260706",
            "Significant_abuse_num2",
            "有意性変化",
        ]
    ].copy()
    out["平均差"] = out["平均差"].map(lambda x: fmt_delta(x, 2))
    return out.rename(
        columns={
            "Total_Mean_SD_20260706": "20260706 Mean±SD",
            "Total_Mean_SD_abuse_num2": "abuse_num2 Mean±SD",
            "Total_Median_IQR_20260706": "20260706 Median[IQR]",
            "Total_Median_IQR_abuse_num2": "abuse_num2 Median[IQR]",
            "p-value_20260706": "20260706 p",
            "p-value_abuse_num2": "abuse_num2 p",
            "Significant_20260706": "20260706 sig.",
            "Significant_abuse_num2": "abuse_num2 sig.",
        }
    )


def caries_treatment_comparison() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "table5_5_caries_prevalence_treatment_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table5_5_caries_prevalence_treatment_20260706.csv")
    left = left[left["Variable"].notna() & ~left["Variable"].str.contains("===", na=False)].copy()
    right = right[right["Variable"].notna() & ~right["Variable"].str.contains("===", na=False)].copy()
    merged = pd.merge(
        left[["Variable", "Category", "Total", "p-value"]],
        right[["Variable", "Category", "Total", "p-value"]],
        on=["Variable", "Category"],
        how="outer",
        suffixes=("_20260706", "_abuse_num2"),
    )
    rows = []
    for _, row in merged.iterrows():
        n_l, den_l, pct_l = parse_count_pct(row["Total_20260706"])
        n_r, den_r, pct_r = parse_count_pct(row["Total_abuse_num2"])
        rows.append(
            {
                "Variable": row["Variable"],
                "Category": row["Category"],
                "20260706 Total": row["Total_20260706"],
                "abuse_num2 Total": row["Total_abuse_num2"],
                "割合差 pp": "" if pd.isna(pct_l) or pd.isna(pct_r) else fmt_delta(pct_r - pct_l, 1),
                "20260706 p": fmt_p(row["p-value_20260706"]),
                "abuse_num2 p": fmt_p(row["p-value_abuse_num2"]),
            }
        )
    return pd.DataFrame(rows)


def logistic_comparison() -> pd.DataFrame:
    left = read_csv(SINGLE_DIR, "table4_logistic_regression_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table4_logistic_regression_20260706.csv")
    keys = ["Outcome", "Comparison"]
    cols = keys + ["N", "Events", "Odds Ratio", "95% CI", "p-value"]
    merged = pd.merge(left[cols], right[cols], on=keys, how="outer", suffixes=("_20260706", "_abuse_num2"))
    merged["OR差"] = merged["Odds Ratio_abuse_num2"] - merged["Odds Ratio_20260706"]
    merged["有意性変化"] = np.select(
        [
            merged["p-value_20260706"].map(parse_p).lt(0.05)
            & merged["p-value_abuse_num2"].map(parse_p).lt(0.05),
            merged["p-value_20260706"].map(parse_p).lt(0.05)
            & ~merged["p-value_abuse_num2"].map(parse_p).lt(0.05),
            ~merged["p-value_20260706"].map(parse_p).lt(0.05)
            & merged["p-value_abuse_num2"].map(parse_p).lt(0.05),
        ],
        ["両方有意", "20260706のみ有意", "abuse_num2のみ有意"],
        default="両方非有意/片側欠測",
    )
    out = merged[
        [
            "Outcome",
            "Comparison",
            "N_20260706",
            "Events_20260706",
            "Odds Ratio_20260706",
            "95% CI_20260706",
            "p-value_20260706",
            "N_abuse_num2",
            "Events_abuse_num2",
            "Odds Ratio_abuse_num2",
            "95% CI_abuse_num2",
            "p-value_abuse_num2",
            "OR差",
            "有意性変化",
        ]
    ].copy()
    out["OR差"] = out["OR差"].map(lambda x: fmt_delta(x, 2))
    return out.rename(
        columns={
            "N_20260706": "20260706 N",
            "Events_20260706": "20260706 Events",
            "Odds Ratio_20260706": "20260706 OR",
            "95% CI_20260706": "20260706 95% CI",
            "p-value_20260706": "20260706 p",
            "N_abuse_num2": "abuse_num2 N",
            "Events_abuse_num2": "abuse_num2 Events",
            "Odds Ratio_abuse_num2": "abuse_num2 OR",
            "95% CI_abuse_num2": "abuse_num2 95% CI",
            "p-value_abuse_num2": "abuse_num2 p",
        }
    )


def pairwise_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = read_csv(SINGLE_DIR, "table3_pairwise_mw_20260706.csv")
    right = read_csv(DOUBLE_DIR, "table3_pairwise_mw_20260706.csv")
    summary = pd.DataFrame(
        [
            {
                "指標": "pairwise行数",
                "20260706": len(left),
                "abuse_num2": len(right),
            },
            {
                "指標": "Bonferroni有意",
                "20260706": int(left["Significant_Bonferroni"].eq("Yes").sum()),
                "abuse_num2": int(right["Significant_Bonferroni"].eq("Yes").sum()),
            },
            {
                "指標": "未補正p<0.05",
                "20260706": int(left["p-value"].map(parse_p).lt(0.05).sum()),
                "abuse_num2": int(right["p-value"].map(parse_p).lt(0.05).sum()),
            },
            {
                "指標": "p値欠測行",
                "20260706": int(left["p-value"].map(parse_p).isna().sum()),
                "abuse_num2": int(right["p-value"].map(parse_p).isna().sum()),
            },
        ]
    )
    keys = ["Variable", "Group1", "Group2"]
    sig_left = left[left["Significant_Bonferroni"].eq("Yes")].copy()
    merged = pd.merge(
        sig_left[keys + ["p-value", "Effect_Size_r", "Significant_Bonferroni"]],
        right[keys + ["p-value", "Effect_Size_r", "Significant_Bonferroni"]],
        on=keys,
        how="left",
        suffixes=("_20260706", "_abuse_num2"),
    )
    merged = merged.rename(
        columns={
            "p-value_20260706": "20260706 p",
            "Effect_Size_r_20260706": "20260706 r",
            "Significant_Bonferroni_20260706": "20260706 Bonf.",
            "p-value_abuse_num2": "abuse_num2 p",
            "Effect_Size_r_abuse_num2": "abuse_num2 r",
            "Significant_Bonferroni_abuse_num2": "abuse_num2 Bonf.",
        }
    )
    return summary, merged


def year_summary() -> pd.DataFrame:
    def aggregate(directory: Path, label: str):
        df = read_csv(directory, "table7_dmft_by_year_abuse_20260706.csv")
        df = fill_down(df, ["Year"])
        out = df.groupby("Year", dropna=True)["N"].sum().reset_index()
        out["Year"] = out["Year"].astype(int)
        return out.rename(columns={"N": label})

    merged = pd.merge(aggregate(SINGLE_DIR, "20260706 N"), aggregate(DOUBLE_DIR, "abuse_num2 N"), on="Year", how="outer")
    merged = merged.sort_values("Year")
    merged["N差"] = merged["abuse_num2 N"].fillna(0) - merged["20260706 N"].fillna(0)
    return merged


def sensitivity_table() -> pd.DataFrame:
    left_path = SINGLE_DIR / "table4_logistic_regression_sensitivity_multitype_20260706.csv"
    right_path = DOUBLE_DIR / "table4_logistic_regression_sensitivity_all_abuse_num_ge1_20260706.csv"
    same = left_path.exists() and right_path.exists() and file_md5(left_path) == file_md5(right_path)
    df = read_csv(SINGLE_DIR, left_path.name)
    df = df[["Outcome", "Comparison", "N", "Events", "Odds Ratio", "95% CI", "p-value", "Adjusted_for"]].copy()
    df.insert(0, "注記", "両ディレクトリで完全一致" if same else "差分あり")
    return df


def make_charts() -> dict[str, Path]:
    use_chart_theme()
    charts: dict[str, Path] = {}

    # Chart 1: final cohort sizes.
    cohort_df = pd.DataFrame(
        {
            "Cohort": ["Single-type final sample", "Double-type final sample"],
            "N": [1235, 70],
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    palette = {
        "Single-type final sample": COLOR_FAMILIES["blue"]["base"],
        "Double-type final sample": COLOR_FAMILIES["orange"]["base"],
    }
    sns.barplot(data=cohort_df, y="Cohort", x="N", hue="Cohort", palette=palette, legend=False, ax=ax, edgecolor=TOKENS["ink"])
    for patch, value in zip(ax.patches, cohort_df["N"]):
        ax.text(value + 20, patch.get_y() + patch.get_height() / 2, f"{value:,}", va="center", ha="left", fontsize=9, color=TOKENS["ink"])
    ax.set_xlabel("Final deduplicated N")
    ax.set_ylabel("")
    ax.set_xlim(0, 1320)
    add_chart_header(
        fig,
        ax,
        "Double-type analysis uses a much smaller complementary cohort",
        "Final N is 70 for abuse_num2 versus 1,235 for the original 20260706 single-type cohort.",
    )
    charts["cohort_size"] = save_fig(fig, "cohort_size")

    # Chart 2: abuse subtype composition.
    comp = abuse_composition()
    comp_long = comp.melt(
        id_vars="Abuse type",
        value_vars=["20260706 %", "abuse_num2 %"],
        var_name="Cohort",
        value_name="Percent",
    )
    comp_long["Cohort"] = comp_long["Cohort"].replace({"20260706 %": "20260706", "abuse_num2 %": "abuse_num2"})
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    palette = {"20260706": COLOR_FAMILIES["blue"]["base"], "abuse_num2": COLOR_FAMILIES["orange"]["base"]}
    sns.barplot(
        data=comp_long,
        x="Abuse type",
        y="Percent",
        hue="Cohort",
        palette=palette,
        ax=ax,
        edgecolor=TOKENS["ink"],
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("")
    ax.set_ylabel("Share of final cohort")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Subtype mix changes after restricting to abuse_num==2",
        "Physical Abuse and Neglect remain dominant; Sexual Abuse falls to n=2 in the double-type cohort.",
    )
    charts["abuse_composition"] = save_fig(fig, "abuse_composition")

    # Chart 3: p-value comparison.
    tests = read_csv(SINGLE_DIR, "table3_overall_tests_20260706.csv")
    tests2 = read_csv(DOUBLE_DIR, "table3_overall_tests_20260706.csv")
    p_df = pd.merge(
        tests[["Variable", "p-value"]],
        tests2[["Variable", "p-value"]],
        on="Variable",
        suffixes=("_20260706", "_abuse_num2"),
    )
    p_df = p_df.melt(id_vars="Variable", var_name="Cohort", value_name="p")
    p_df["Cohort"] = p_df["Cohort"].replace({"p-value_20260706": "20260706", "p-value_abuse_num2": "abuse_num2"})
    p_df["p_num"] = p_df["p"].map(parse_p)
    p_df["neglog10p"] = p_df["p_num"].map(lambda x: 0 if pd.isna(x) else -math.log10(max(float(x), 1e-4)))
    order = (
        p_df[p_df["Cohort"].eq("20260706")]
        .sort_values("neglog10p", ascending=True)["Variable"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(10.8, 8.5))
    palette = {"20260706": COLOR_FAMILIES["blue"]["base"], "abuse_num2": COLOR_FAMILIES["orange"]["base"]}
    sns.barplot(data=p_df, y="Variable", x="neglog10p", hue="Cohort", order=order, palette=palette, ax=ax, edgecolor=TOKENS["ink"])
    ax.axvline(-math.log10(0.05), color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.text(-math.log10(0.05) + 0.03, -0.7, "p=0.05", fontsize=8, color=TOKENS["muted"])
    ax.set_xlabel("-log10(p)")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Overall subtype differences vanish in the double-type cohort",
        "Kruskal-Wallis p-values for 19 oral-health metrics; values to the right of the dashed line are p<0.05.",
    )
    charts["overall_pvalues"] = save_fig(fig, "overall_pvalues")

    # Chart 4: selected total mean shifts.
    tests_comp = pd.merge(
        tests[["Variable", "Total_Mean_SD"]],
        tests2[["Variable", "Total_Mean_SD"]],
        on="Variable",
        suffixes=("_20260706", "_abuse_num2"),
    )
    tests_comp["Single mean"] = tests_comp["Total_Mean_SD_20260706"].map(parse_mean)
    tests_comp["Double mean"] = tests_comp["Total_Mean_SD_abuse_num2"].map(parse_mean)
    tests_comp["Delta"] = tests_comp["Double mean"] - tests_comp["Single mean"]
    selected = [
        "DMFT_Index",
        "decayed_total",
        "filled_total",
        "Perm_DMFT",
        "Baby_DMFT",
        "Healthy_Rate",
        "Care_Index",
        "UTN_Score",
        "DMFT_C0",
    ]
    delta_df = tests_comp[tests_comp["Variable"].isin(selected)].sort_values("Delta")
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    colors = np.where(delta_df["Delta"] >= 0, COLOR_FAMILIES["olive"]["base"], COLOR_FAMILIES["orange"]["base"])
    edges = np.where(delta_df["Delta"] >= 0, COLOR_FAMILIES["olive"]["dark"], COLOR_FAMILIES["orange"]["dark"])
    bars = ax.barh(delta_df["Variable"], delta_df["Delta"], color=colors, edgecolor=edges, linewidth=1.0)
    for bar, value in zip(bars, delta_df["Delta"]):
        ax.text(value + (0.18 if value >= 0 else -0.18), bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_xlabel("abuse_num2 mean minus 20260706 mean")
    ax.set_ylabel("")
    add_chart_header(
        fig,
        ax,
        "Mean levels move, but significance is not retained",
        "Selected metrics are shown in their native units; direction is descriptive and not standardized across metrics.",
    )
    charts["mean_deltas"] = save_fig(fig, "mean_deltas")

    # Chart 5: logistic OR comparison for common Neglect-vs-Physical rows.
    log_left = read_csv(SINGLE_DIR, "table4_logistic_regression_20260706.csv")
    log_right = read_csv(DOUBLE_DIR, "table4_logistic_regression_20260706.csv")
    common = pd.merge(
        log_left,
        log_right,
        on=["Outcome", "Comparison"],
        suffixes=("_20260706", "_abuse_num2"),
    )
    common = common[common["Comparison"].eq("Neglect vs Physical Abuse")].copy()
    y_positions = np.arange(len(common))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for offset, cohort, color, edge in [
        (-0.11, "20260706", COLOR_FAMILIES["blue"]["base"], COLOR_FAMILIES["blue"]["dark"]),
        (0.11, "abuse_num2", COLOR_FAMILIES["orange"]["base"], COLOR_FAMILIES["orange"]["dark"]),
    ]:
        ors = common[f"Odds Ratio_{cohort}"].astype(float).to_numpy()
        lows, highs = zip(*common[f"95% CI_{cohort}"].map(parse_ci))
        lows = np.array(lows)
        highs = np.array(highs)
        xerr = np.vstack([ors - lows, highs - ors])
        ax.errorbar(
            ors,
            y_positions + offset,
            xerr=xerr,
            fmt="o",
            color=edge,
            ecolor=edge,
            markerfacecolor=color,
            markeredgecolor=edge,
            capsize=3,
            linewidth=1.0,
            label=cohort,
        )
    ax.axvline(1, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_yticks(y_positions, common["Outcome"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio, log scale")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Double-type ORs are imprecise",
        "Common regression rows compare Neglect vs Physical Abuse; all abuse_num2 intervals cross OR=1.",
    )
    fig.subplots_adjust(top=0.74)
    charts["logistic_or"] = save_fig(fig, "logistic_or")

    # Chart 6: dentition composition.
    dent = dentition_distribution()
    dent_long = dent.melt(
        id_vars="Dentition",
        value_vars=["20260706 %", "abuse_num2 %"],
        var_name="Cohort",
        value_name="Percent",
    )
    dent_long["Cohort"] = dent_long["Cohort"].replace({"20260706 %": "20260706", "abuse_num2 %": "abuse_num2"})
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    sns.barplot(data=dent_long, x="Dentition", y="Percent", hue="Cohort", palette=palette, ax=ax, edgecolor=TOKENS["ink"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("")
    ax.set_ylabel("Share of final cohort")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Double-type sample is concentrated in mixed and permanent dentition",
        "Primary dentition falls from 20.9% of the single-type cohort to 4.3% of the double-type cohort.",
    )
    charts["dentition_distribution"] = save_fig(fig, "dentition_distribution")

    return charts


def png_gallery_html() -> str:
    names = sorted({p.name for p in SINGLE_DIR.glob("*.png")} | {p.name for p in DOUBLE_DIR.glob("*.png")})
    chunks = []
    for name in names:
        left = SINGLE_DIR / name
        right = DOUBLE_DIR / name
        chunks.append("<details class='figure-detail'>")
        chunks.append(
            f"<summary>{html.escape(name)} <span>{html.escape(size_kb(left))} vs {html.escape(size_kb(right))}</span></summary>"
        )
        chunks.append("<div class='image-pair'>")
        if left.exists():
            chunks.append(f"<figure><figcaption>20260706</figcaption>{embedded_image(left, name + ' 20260706')}</figure>")
        else:
            chunks.append("<figure><figcaption>20260706</figcaption><div class='missing'>missing</div></figure>")
        if right.exists():
            chunks.append(f"<figure><figcaption>abuse_num2</figcaption>{embedded_image(right, name + ' abuse_num2')}</figure>")
        else:
            chunks.append("<figure><figcaption>abuse_num2</figcaption><div class='missing'>missing</div></figure>")
        chunks.append("</div></details>")
    return "\n".join(chunks)


def format_composition_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["20260706 %", "abuse_num2 %"]:
        out[col] = out[col].map(lambda x: f"{x:.1f}%")
    out["割合差 pp"] = out["割合差 pp"].map(lambda x: fmt_delta(x, 1))
    return out


def format_dentition_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["20260706 %", "abuse_num2 %"]:
        out[col] = out[col].map(lambda x: f"{x:.1f}%")
    out["DMFT平均差"] = out["DMFT平均差"].map(lambda x: fmt_delta(x, 2))
    for col in ["20260706 KW p", "abuse_num2 KW p"]:
        out[col] = out[col].map(fmt_p)
    return out


def build_html() -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    charts = make_charts()
    files = make_file_inventory()
    flow = flow_comparison()
    demo = demographics_key_table()
    comp = abuse_composition()
    dent = dentition_distribution()
    tests = overall_tests_comparison()
    caries = caries_treatment_comparison()
    logistic = logistic_comparison()
    pairwise_counts, pairwise_sig = pairwise_summary()
    years = year_summary()
    sensitivity = sensitivity_table()

    single_total = 1235
    double_total = 70
    significant_single = int(read_csv(SINGLE_DIR, "table3_overall_tests_20260706.csv")["Significant"].eq("Yes").sum())
    significant_double = int(read_csv(DOUBLE_DIR, "table3_overall_tests_20260706.csv")["Significant"].eq("Yes").sum())
    logistic_sig_single = int(read_csv(SINGLE_DIR, "table4_logistic_regression_20260706.csv")["p-value"].map(parse_p).lt(0.05).sum())
    logistic_sig_double = int(read_csv(DOUBLE_DIR, "table4_logistic_regression_20260706.csv")["p-value"].map(parse_p).lt(0.05).sum())

    css = """
    :root {
      color-scheme: light;
      --surface: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2430;
      --muted: #697083;
      --line: #dfe3ec;
      --line-strong: #bfc7d6;
      --blue: #5477c4;
      --orange: #cc6f47;
      --olive: #71b436;
      --gold: #b8a037;
      --pink: #bd569b;
      --soft-blue: #eaf1fe;
      --soft-orange: #ffedde;
      --soft-olive: #d8ecbd;
      --mono: "SF Mono", Menlo, Consolas, monospace;
      --sans: Inter, Aptos, "Segoe UI", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: var(--sans);
      line-height: 1.68;
      letter-spacing: 0;
    }
    .page {
      max-width: 1180px;
      margin: 0 auto;
      padding: 44px 28px 72px;
    }
    header {
      border-bottom: 1px solid var(--line-strong);
      padding-bottom: 24px;
      margin-bottom: 30px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.12;
      letter-spacing: 0;
    }
    .subtitle {
      max-width: 920px;
      color: var(--muted);
      font-size: 16px;
      margin: 0;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 5px 10px;
      font-size: 12px;
      border-radius: 999px;
      color: var(--muted);
      font-family: var(--mono);
    }
    section {
      margin: 34px 0 42px;
      padding-top: 6px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 24px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    h3 {
      margin: 26px 0 10px;
      font-size: 18px;
      line-height: 1.3;
      letter-spacing: 0;
    }
    p {
      margin: 8px 0 14px;
      max-width: 980px;
    }
    strong { font-weight: 700; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 26px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px 16px;
      min-height: 112px;
    }
    .metric .value {
      font-family: var(--mono);
      font-size: 28px;
      line-height: 1.1;
      font-weight: 700;
    }
    .metric .label {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .callout {
      background: var(--soft-orange);
      border-left: 5px solid var(--orange);
      padding: 14px 16px;
      margin: 18px 0;
    }
    .callout.blue {
      background: var(--soft-blue);
      border-left-color: var(--blue);
    }
    .callout.olive {
      background: var(--soft-olive);
      border-left-color: var(--olive);
    }
    .chart {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
      margin: 16px 0 24px;
    }
    .chart img {
      width: 100%;
      display: block;
    }
    .table-wrap {
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      margin: 14px 0 24px;
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.45;
    }
    .data-table th,
    .data-table td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }
    .data-table th {
      color: var(--muted);
      background: #fafbfc;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .data-table td {
      font-family: var(--mono);
    }
    .note {
      color: var(--muted);
      font-size: 13px;
      margin-top: -8px;
    }
    .two-col {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }
    details.figure-detail {
      background: var(--panel);
      border: 1px solid var(--line);
      margin: 10px 0;
    }
    details.figure-detail summary {
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 700;
    }
    details.figure-detail summary span {
      font-weight: 400;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 12px;
      margin-left: 8px;
    }
    .image-pair {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 0 12px 12px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      background: #fff;
    }
    figcaption {
      padding: 8px 10px;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      font-family: var(--mono);
    }
    figure img {
      width: 100%;
      display: block;
    }
    .missing {
      min-height: 140px;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-family: var(--mono);
    }
    ul {
      padding-left: 20px;
      margin-top: 8px;
    }
    li {
      margin: 6px 0;
    }
    code {
      font-family: var(--mono);
      background: #eef1f5;
      padding: 1px 4px;
      border-radius: 4px;
    }
    @media (max-width: 860px) {
      .page { padding: 28px 16px 52px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two-col, .image-pair { grid-template-columns: 1fr; }
      .data-table th, .data-table td { white-space: normal; }
    }
    @media (max-width: 560px) {
      .summary-grid { grid-template-columns: 1fr; }
    }
    """

    def chart_block(key: str, alt: str) -> str:
        return f"<div class='chart'>{embedded_image(charts[key], alt)}</div>"

    html_doc = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>20260706 と 20260706_abuse_num2 の結果比較</title>
  <style>{css}</style>
</head>
<body>
<main class="page">
  <header>
    <h1>20260706 と 20260706_abuse_num2 の結果比較</h1>
    <p class="subtitle">2つの解析出力を、対象コホート、統計検定、回帰モデル、歯列・年次集計、ファイル生成物の観点から比較した技術レポートです。</p>
    <div class="meta">
      <span class="pill">source: result/20260706</span>
      <span class="pill">source: result/20260706_abuse_num2</span>
      <span class="pill">generated: 2026-07-07 JST</span>
    </div>
  </header>

  <section>
    <h2>Technical Summary</h2>
    <div class="summary-grid">
      <div class="metric"><div class="value">{single_total:,}</div><div class="label">20260706 final N。単タイプ（abuse_num==1）を主解析対象にしている。</div></div>
      <div class="metric"><div class="value">{double_total:,}</div><div class="label">abuse_num2 final N。ダブルタイプ（abuse_num==2）だけに絞った補助解析。</div></div>
      <div class="metric"><div class="value">{significant_single} → {significant_double}</div><div class="label">Kruskal-Wallisで有意だった連続系アウトカム数。19項目中の変化。</div></div>
      <div class="metric"><div class="value">{logistic_sig_single} → {logistic_sig_double}</div><div class="label">主回帰でp&lt;0.05だった行数。12行から4行へ縮小し、有意行は消失。</div></div>
    </div>
    <p><strong>主な違いは「解析対象の定義」です。</strong> `20260706` は単タイプ虐待のみ（abuse_num==1）を採用し、`20260706_abuse_num2` はダブルタイプ虐待のみ（abuse_num==2）を採用しています。両者は同じ母集団から作られた補完的なサブセットであり、単純な再実行差分ではありません。</p>
    <p><strong>abuse_num2では、多くの方向性は残る一方で統計的有意性はほぼ失われています。</strong> DMFT平均は 1.66 から 2.11 に上がりますが、全体Kruskal-Wallisは p=0.9256 となり非有意です。これは効果がないというより、N=70、特にSexual Abuse n=2という極小セルを含むため、群間比較と回帰推定が不安定になったと読むのが妥当です。</p>
    <p><strong>感度分析の `all_abuse_num_ge1` と `sensitivity_multitype` は完全一致です。</strong> つまり「単タイプに多タイプを含め、is_multitypeで調整する」モデルは両ディレクトリで同じ再現結果を指しています。</p>
  </section>

  <section>
    <h2>コホート定義とサンプル構成が結論を変えている</h2>
    <p><strong>最終解析Nは 1,235 対 70 で、abuse_num2は元解析の5.7%に相当します。</strong> どちらも raw N=2,480、日付フィルタ後 N=2,162、4虐待タイプかつ abuse_num>=1 の N=1,305 までは共通です。その後、20260706は単タイプを残し、abuse_num2はダブルタイプだけを残します。</p>
    {chart_block("cohort_size", "Final cohort size comparison")}
    <div class="table-wrap">{df_to_html(flow)}</div>
    <p><strong>虐待タイプ構成も変わります。</strong> Physical Abuse は 52.3% から 44.3%へ低下し、Neglect は 26.6% から 38.6%へ上昇します。Sexual Abuse は60例から2例に落ちるため、この群を含む比較・回帰は解釈可能性が大きく制限されます。</p>
    {chart_block("abuse_composition", "Abuse subtype composition")}
    <div class="table-wrap">{df_to_html(format_composition_table(comp))}</div>
  </section>

  <section>
    <h2>背景属性と歯列構成の差</h2>
    <p><strong>abuse_num2は年齢が高く、初期小児の比率が低いサンプルです。</strong> 平均年齢は 9.8±4.1 歳から 11.6±3.4 歳へ上がり、Early Childhoodは25.7%から8.6%へ下がります。口腔アウトカムは年齢・歯列の影響を受けるため、この構成差はDMFT、乳歯DMFT、永久歯DMFTの比較に直接効きます。</p>
    <div class="table-wrap">{df_to_html(demo)}</div>
    <p><strong>歯列ではprimary dentitionが 20.9% から 4.3%へ縮小し、mixed/permanentが中心になります。</strong> abuse_num2でPrimary dentitionが3例しかないため、乳歯系指標や歯列内比較は不安定です。</p>
    {chart_block("dentition_distribution", "Dentition distribution comparison")}
    <div class="table-wrap">{df_to_html(format_dentition_table(dent))}</div>
  </section>

  <section>
    <h2>全体検定では「有意だった差」がabuse_num2で消える</h2>
    <p><strong>20260706では19項目中12項目が有意でしたが、abuse_num2では0項目です。</strong> DMFT_Index、decayed_total、Baby_DMFT、Healthy_Rate、Care_Index、UTN_Scoreなど、元解析で強く出ていた項目がすべて非有意化しています。Perm_Mはabuse_num2で統計量・p値が欠測です。</p>
    {chart_block("overall_pvalues", "Kruskal-Wallis p-value comparison")}
    <p><strong>平均値だけを見ると、abuse_num2で重い方向に動く指標もあります。</strong> Total DMFTは +0.45、Permanent DMFTは +0.65、DMFT+C0は +0.63です。一方、Healthy_Rateは -1.76、UTN_Scoreは -6.52です。ただし、Nが小さいため、これらは「記述的な差」として扱うべきです。</p>
    {chart_block("mean_deltas", "Selected mean differences")}
    <div class="table-wrap">{df_to_html(tests)}</div>
  </section>

  <section>
    <h2>う蝕・治療状態は方向よりも不確実性が目立つ</h2>
    <p><strong>総う蝕経験率は 40.6% から 37.1%へやや低下しますが、abuse_num2では群間差は非有意です。</strong> 未処置う蝕も 27.0% から 22.9%へ下がり、Fully Treated Cariesは 33.1% から 38.5%へ上がります。ただし、abuse_num2のcaries activeは26例だけです。</p>
    <div class="table-wrap">{df_to_html(caries)}</div>
  </section>

  <section>
    <h2>回帰モデルは推定幅が広がり、有意性は消失</h2>
    <p><strong>abuse_num2の主回帰では、Neglect vs Physical Abuseだけが推定されています。</strong> 点推定ORは20260706より大きい項目もありますが、95%CIはすべて1をまたぎます。例としてUntreated Cariesは OR 2.00 (1.47-2.72), p&lt;0.0001 から OR 3.91 (0.77-19.98), p=0.1011 へ変化します。</p>
    {chart_block("logistic_or", "Logistic odds ratio comparison")}
    <div class="table-wrap">{df_to_html(logistic)}</div>
    <h3>多タイプ調整感度分析</h3>
    <p><strong>多タイプを含めた感度分析ファイルは両ディレクトリで完全一致です。</strong> これは `table4_logistic_regression_sensitivity_multitype_20260706.csv` と `table4_logistic_regression_sensitivity_all_abuse_num_ge1_20260706.csv` が同じ内容であることを意味します。</p>
    <div class="table-wrap">{df_to_html(sensitivity)}</div>
  </section>

  <section>
    <h2>Posthoc と pairwise はabuse_num2で解析不能に近づく</h2>
    <p><strong>全体pairwiseでは、20260706のBonferroni有意14行がabuse_num2では0行になります。</strong> 未補正p&lt;0.05も35行から0行へ落ちます。`table3_posthoc_20260706.csv` はabuse_num2側で空ファイル相当（0行）です。</p>
    <div class="two-col">
      <div class="table-wrap">{df_to_html(pairwise_counts)}</div>
      <div class="callout blue"><strong>読み方:</strong> これは「差が存在しない」ことの証明ではなく、ダブルタイプ群のNとセル数が小さく、Bonferroni補正後に検出できる差が残らなかった、という結果です。</div>
    </div>
    <h3>20260706でBonferroni有意だったpairwise行</h3>
    <div class="table-wrap">{df_to_html(pairwise_sig)}</div>
  </section>

  <section>
    <h2>年次集計はabuse_num2で疎になる</h2>
    <p><strong>20260706は2016年以降の複数年に十分な行がありますが、abuse_num2では年別・虐待タイプ別に小セルが頻発します。</strong> 年次トレンドの図が小さくなっているのは、単に描画の違いではなく、表示対象の年×タイプセルが大きく減ったためです。</p>
    <div class="table-wrap">{df_to_html(years)}</div>
  </section>

  <section>
    <h2>ファイル単位の比較</h2>
    <p><strong>CSVは一部が片側のみ、PNGは全て同名でも内容差分ありです。</strong> 20260706側だけに歯列別回帰やposthoc整形表があり、abuse_num2側だけに除外プロファイルとall_abuse_num_ge1感度分析名のファイルがあります。</p>
    <div class="table-wrap">{df_to_html(files)}</div>
  </section>

  <section>
    <h2>既存図のサイドバイサイド確認</h2>
    <p><strong>各PNGはHTMLに埋め込み済みです。</strong> ファイル単体で開いても、相対パスなしで図を確認できます。多くの図でabuse_num2側はセル数減少により、点・箱・信頼区間・ペアワイズ表示が簡略化されています。</p>
    {png_gallery_html()}
  </section>

  <section>
    <h2>Limitations and Next Steps</h2>
    <div class="callout olive">
      <strong>結論:</strong> `20260706_abuse_num2` は元解析の再現ではなく、ダブルタイプ虐待児だけを見る小規模サブ解析です。統計的有意性の消失は、方向性の反転よりもサンプルサイズとセル欠損の影響として読むべきです。
    </div>
    <ul>
      <li>Sexual Abuse n=2、Primary dentition n=3のため、群間検定・歯列別解析・ロジスティック回帰は不安定です。</li>
      <li>主論文の主要解析としては単タイプ解析を維持し、abuse_num2は多タイプ・重複虐待の感度分析または記述的補足として扱うのが自然です。</li>
      <li>多タイプを含む推定は、すでに完全一致している `is_multitype` 調整感度分析を中心に報告し、ダブルタイプ単独解析は「検出力不足」の caveat を明記するのが安全です。</li>
      <li>追加で見るなら、abuse_num==1 と abuse_num==2 を同一モデル内に入れ、虐待タイプ、年齢、歯列、年固定効果を調整した上で interaction ではなく主効果として扱う設計が現実的です。</li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    return html_doc


def main() -> None:
    html_text = build_html()
    OUT_HTML.write_text(html_text, encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()

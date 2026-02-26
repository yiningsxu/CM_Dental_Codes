"""
plotting_patch_CAN.py
Publication-ready plotting helpers for Child Abuse & Neglect submission.

Key changes vs current Functions_refactored.py:
- Uses matplotlib only (no seaborn dependency for core figures).
- Exports both PNG (300 dpi) and TIFF (600 dpi) suitable for journal submission.
- Supports Dunn post-hoc with Holm (recommended) or Bonferroni (sensitivity).
- Forest plot supports log-scale x-axis and dynamic limits.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scikit_posthocs as sp
except Exception as e:
    sp = None


# -----------------------------
# Global style (safe defaults)
# -----------------------------
def set_pub_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_dual(fig, out_base: str):
    # PNG for drafts; TIFF for submission (many journals accept both; TIFF is safest).
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".tif", dpi=600)


# -----------------------------
# Dunn + boxplot (Figure 1/2)
# -----------------------------
def dunn_posthoc(data: pd.DataFrame, group_col: str, val_col: str, p_adjust: str = "holm") -> pd.DataFrame:
    """
    Returns symmetric matrix of adjusted p-values.
    """
    if sp is None:
        raise ImportError("scikit-posthocs is required for Dunn's test. Install: pip install scikit-posthocs")
    return sp.posthoc_dunn(data, val_col=val_col, group_col=group_col, p_adjust=p_adjust)


def boxplot_with_points_and_mean(
    df: pd.DataFrame,
    val_col: str,
    group_col: str = "abuse",
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    p_adjust: str = "holm",
    annotate_pairs: Optional[List[Tuple[str, str]]] = None,
    alpha_points: float = 0.35,
    jitter: float = 0.15,
    out_dir: str = "./",
    out_name: Optional[str] = None,
):
    """
    Publication-ready boxplot:
    - median/IQR; points; group mean (dashed).
    - optional pairwise p-value annotations (Dunn + p_adjust).
    """
    set_pub_style()
    _ensure_dir(out_dir)

    data = df[[group_col, val_col]].dropna().copy()
    if data.empty:
        return

    if order is None:
        order = list(pd.unique(data[group_col]))
    order = [g for g in order if g in set(data[group_col])]

    # compute dunn
    pmat = None
    if annotate_pairs:
        pmat = dunn_posthoc(data, group_col, val_col, p_adjust=p_adjust)

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    positions = np.arange(len(order))
    grouped = [data.loc[data[group_col] == g, val_col].values for g in order]

    ax.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        showfliers=False,
        medianprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # points + mean
    rng = np.random.default_rng(42)
    for i, g in enumerate(order):
        vals = data.loc[data[group_col] == g, val_col].values
        x = np.full_like(vals, positions[i], dtype=float)
        x = x + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x, vals, s=10, alpha=alpha_points, linewidths=0)

        m = np.nanmean(vals) if len(vals) else np.nan
        if np.isfinite(m):
            ax.hlines(m, positions[i]-0.28, positions[i]+0.28, linestyles="--", linewidth=1.0)

        ax.text(positions[i], ax.get_ylim()[1], f"n={len(vals)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel(y_label or val_col)
    if title:
        ax.set_title(title)

    # annotations
    if annotate_pairs and pmat is not None:
        # simple bracket annotations above plot; stack them
        y_max = np.nanmax(data[val_col].values)
        y = y_max + 0.08 * (y_max if y_max > 0 else 1.0)
        step = 0.08 * (y_max if y_max > 0 else 1.0)

        def fmt_p(p):
            if p < 0.001:
                return "<0.001"
            if p < 0.01:
                return "<0.01"
            return f"{p:.2f}"

        for (g1, g2) in annotate_pairs:
            if g1 not in order or g2 not in order:
                continue
            p = float(pmat.loc[g1, g2])
            x1, x2 = order.index(g1), order.index(g2)
            ax.plot([x1, x1, x2, x2], [y, y+step/3, y+step/3, y], linewidth=1.0)
            ax.text((x1+x2)/2, y+step/3, f"Dunn ({p_adjust}): {fmt_p(p)}", ha="center", va="bottom", fontsize=9)
            y += step

    fig.tight_layout()
    if out_name is None:
        out_name = f"figure_{val_col}_{p_adjust}"
    out_base = os.path.join(out_dir, out_name)
    _save_dual(fig, out_base)
    plt.close(fig)


# -----------------------------
# Forest plot (Figure 3)
# -----------------------------
def forest_plot_or(
    df_or: pd.DataFrame,
    out_dir: str,
    out_name: str = "figure_forest_or",
    x_log: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
):
    """
    Expects a tidy df with columns:
      Outcome, Comparison, OR, CI_lower, CI_upper, significant (bool)
    """
    set_pub_style()
    _ensure_dir(out_dir)

    df = df_or.copy()
    required = {"Outcome", "Comparison", "OR", "CI_lower", "CI_upper"}
    if not required.issubset(df.columns):
        raise ValueError(f"forest_plot_or requires columns: {sorted(required)}")

    # ordering: group by outcome
    df["Outcome"] = df["Outcome"].astype(str)
    df["Comparison"] = df["Comparison"].astype(str)

    outcomes = list(pd.unique(df["Outcome"]))
    rows = []
    for o in outcomes:
        sub = df[df["Outcome"] == o].copy()
        rows.append(sub)
    dfp = pd.concat(rows, axis=0)

    y = np.arange(len(dfp))
    fig, ax = plt.subplots(figsize=(7.0, max(3.8, 0.35*len(dfp) + 1.5)))

    ax.axvline(1.0, linestyle="--", linewidth=1.0)

    orv = dfp["OR"].astype(float).values
    lo = dfp["CI_lower"].astype(float).values
    hi = dfp["CI_upper"].astype(float).values

    ax.errorbar(orv, y, xerr=[orv-lo, hi-orv], fmt="none", capsize=3, linewidth=1.2)
    ax.scatter(orv, y, s=35)

    ax.set_yticks(y)
    ax.set_yticklabels(dfp["Comparison"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted Odds Ratio (95% CI)")

    if x_log:
        ax.set_xscale("log")

    if xlim is None:
        finite = np.isfinite(np.r_[lo, hi])
        lo_m = np.min(np.r_[lo, hi][finite])
        hi_m = np.max(np.r_[lo, hi][finite])
        # pad
        if x_log:
            xlim = (max(0.05, lo_m/1.8), hi_m*1.8)
        else:
            xlim = (max(0.0, lo_m-0.5), hi_m+0.5)

    ax.set_xlim(*xlim)

    # add outcome headers
    start = 0
    for o in outcomes:
        n = (dfp["Outcome"] == o).sum()
        if n:
            mid = start + (n-1)/2
            ax.text(-0.04, mid, o, transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold")
            start += n

    fig.tight_layout()
    out_base = os.path.join(out_dir, out_name)
    _save_dual(fig, out_base)
    plt.close(fig)

#!/usr/bin/env python3
"""
Random forest prediction of abuse category.

This script extends the main R analysis workflow by using the derived main
analysis dataset:

  data/analysisData_20260211_tillMar2024_singleType_dedup_with_derived_variables.csv

The target is the four single-type maltreatment categories:
Physical Abuse, Neglect, Emotional Abuse, and Sexual Abuse.

Outputs are written to:

  result/random_forest_abuse_category_YYYYMMDD/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install the analysis dependencies before running this script:\n"
        "  python3 -m pip install pandas numpy scikit-learn matplotlib joblib"
    ) from exc


TARGET_CLASSES = [
    "Physical Abuse",
    "Neglect",
    "Emotional Abuse",
    "Sexual Abuse",
]

TARGET_RECODE = {
    "1": "Physical Abuse",
    "2": "Neglect",
    "3": "Emotional Abuse",
    "4": "Sexual Abuse",
    1: "Physical Abuse",
    2: "Neglect",
    3: "Emotional Abuse",
    4: "Sexual Abuse",
}

DERIVED_ORAL_FEATURES = [
    "Perm_D",
    "Perm_M",
    "Perm_F",
    "Perm_Sound",
    "Perm_C0",
    "Perm_total_teeth",
    "Perm_DMFT",
    "Perm_DMFT_C0",
    "Perm_sound_rate",
    "Baby_d",
    "Baby_m",
    "Baby_f",
    "Baby_sound",
    "Baby_C0",
    "Baby_total_teeth",
    "Baby_DMFT",
    "Baby_DMFT_C0",
    "Baby_sound_rate",
    "DMFT_Index",
    "DMFT_C0",
    "C0_Count",
    "filled_total",
    "decayed_total",
    "missing_total",
    "Care_Index",
    "UTN_Score",
    "total_teeth",
    "Healthy_Rate",
    "Present_Teeth",
    "Present_Perm_Teeth",
    "Present_Baby_Teeth",
    "has_caries",
    "has_untreated_caries",
    "dentition_type",
]

CLINICAL_FEATURES = [
    "needTOBEtreated",
    "emergency",
    "emergencyInMonths",
    "gingivitis",
    "occlusalRelationship",
    "habits",
    "OralCleanStatus",
    "Orthodontics",
    # # "dentists",
    # "dental_hygienist",
]

BEHAVIOR_FEATURES = [
    "wake_up",
    "breakfast",
    "morning_brushing",
    "school",
    "bedtime",
    "night_brushing",
    "TV",
    "game",
    "meal",
    "extra_lesson",
]

DEMOGRAPHIC_FEATURES = [
    "age_year",
    "age_month",
    "sex",
    "CGC",
    "year",
]

TARGET_LEAKAGE_COLUMNS = {
    "abuse",
    "abuse_1",
    "abuse_num",
}

ID_OR_TEXT_COLUMNS = {
    "No_All",
    "date",
    "age",
    "instruction_detail",
    "instruction",
    "memo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict four abuse categories with random forest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input CSV. Defaults to the derived single-type/deduplicated dataset "
            "created by the main R analysis script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to result/random_forest_abuse_category_YYYYMMDD.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["oral_only", "oral_plus_demographics", "both"],
        default="both",
        help="Feature set to run.",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=20260622)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def find_base_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "result").exists():
            return candidate.resolve()
    return script_dir.parent.resolve()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base_dir = find_base_dir()
    input_path = args.input
    if input_path is None:
        input_path = (
            base_dir
            / "data"
            / "analysisData_20260211_tillMar2024_singleType_dedup_with_derived_variables.csv"
        )
    elif not input_path.is_absolute():
        input_path = (base_dir / input_path).resolve()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            base_dir
            / "result"
            / f"random_forest_abuse_category_{datetime.now().strftime('%Y%m%d')}"
        )
    elif not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, input_path, output_dir


def clean_target(series: pd.Series) -> pd.Series:
    target = series.copy()
    target = target.replace(TARGET_RECODE)
    target = target.astype("string").str.strip()
    return target


def load_analysis_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run code/Analysis_code_refactored_REVISED_R_no_custom_functions.R first, "
            "or pass --input to another analysis-ready CSV."
        )

    df = pd.read_csv(input_path)
    if "abuse" not in df.columns:
        raise ValueError("Input data must contain an 'abuse' target column.")

    df = df.copy()
    df["abuse"] = clean_target(df["abuse"])
    df = df[df["abuse"].isin(TARGET_CLASSES)].copy()

    if "abuse_num" in df.columns:
        df = df[df["abuse_num"].isna() | (pd.to_numeric(df["abuse_num"], errors="coerce") == 1)].copy()

    if df.empty:
        raise ValueError("No rows remained after filtering to the four target abuse categories.")

    return df


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    oral_only = existing_columns(
        df,
        DERIVED_ORAL_FEATURES + CLINICAL_FEATURES + BEHAVIOR_FEATURES,
    )
    oral_plus_demographics = existing_columns(
        df,
        oral_only + DEMOGRAPHIC_FEATURES,
    )

    feature_sets = {
        "oral_only": oral_only,
        "oral_plus_demographics": oral_plus_demographics,
    }

    cleaned: dict[str, list[str]] = {}
    for name, columns in feature_sets.items():
        filtered = []
        seen = set()
        for col in columns:
            if col in TARGET_LEAKAGE_COLUMNS or col in ID_OR_TEXT_COLUMNS:
                continue
            if col in seen:
                continue
            if df[col].notna().sum() == 0:
                continue
            filtered.append(col)
            seen.add(col)
        cleaned[name] = filtered

    return cleaned


def split_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols = []
    numeric_cols = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> Pipeline:
    numeric_cols, categorical_cols = split_columns(df, feature_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        bootstrap=True,
        oob_score=True,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def safe_auc(y_true: pd.Series, y_proba: np.ndarray, labels: list[str]) -> tuple[float, float]:
    try:
        macro_auc = roc_auc_score(
            y_true,
            y_proba,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = np.nan

    try:
        weighted_auc = roc_auc_score(
            y_true,
            y_proba,
            labels=labels,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        weighted_auc = np.nan

    return macro_auc, weighted_auc


def get_transformed_feature_names(
    fitted_pipeline: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        names.extend([f"num__{col}" for col in numeric_cols])
        if categorical_cols:
            cat_pipe = preprocessor.named_transformers_.get("cat")
            if cat_pipe is not None:
                encoder = cat_pipe.named_steps["onehot"]
                try:
                    cat_names = encoder.get_feature_names_out(categorical_cols)
                    names.extend([f"cat__{name}" for name in cat_names])
                except Exception:
                    names.extend([f"cat__{col}" for col in categorical_cols])
        return names


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, title: str) -> None:
    top = importance_df.head(30).copy()
    if top.empty:
        return

    top = top.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.25)))
    ax.barh(top["feature"], top["importance"], color="#2f6f73")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_permutation_importance(
    permutation_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    top = permutation_df.head(30).copy()
    if top.empty:
        return

    top = top.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.25)))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#6b5b95")
    ax.set_xlabel("Balanced accuracy decrease after permutation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_cross_validation(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    min_class_count = int(y.value_counts().min())
    n_splits = min(args.cv_folds, min_class_count)
    if n_splits < 2:
        return pd.DataFrame(
            [
                {
                    "metric": "note",
                    "mean": np.nan,
                    "sd": np.nan,
                    "n_splits": n_splits,
                    "detail": "Cross-validation skipped because at least one class has fewer than two rows.",
                }
            ]
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }
    cv_results = cross_validate(
        pipeline,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score=np.nan,
    )

    rows = []
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_splits": n_splits,
                "detail": "",
            }
        )
    return pd.DataFrame(rows)


def run_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    feature_dir = output_dir / feature_set_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    x = df[feature_cols].copy()
    y = df["abuse"].copy()
    labels = [label for label in TARGET_CLASSES if label in set(y)]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = make_pipeline(df, feature_cols, args)

    split_info = {
        "feature_set": feature_set_name,
        "n_total": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "class_counts_total": y.value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
        "class_counts_train": y_train.value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
        "class_counts_test": y_test.value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
        "features": feature_cols,
    }
    write_json(feature_dir / "split_info.json", split_info)
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_dir / "input_features.csv", index=False)

    cv_summary = run_cross_validation(pipeline, x_train, y_train, args)
    cv_summary.insert(0, "feature_set", feature_set_name)
    cv_summary.to_csv(feature_dir / "cross_validation_metrics.csv", index=False)

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)
    fitted_labels = list(pipeline.named_steps["model"].classes_)
    y_proba_for_labels = np.zeros((len(y_test), len(labels)))
    for idx, label in enumerate(labels):
        if label in fitted_labels:
            y_proba_for_labels[:, idx] = y_proba[:, fitted_labels.index(label)]

    macro_auc, weighted_auc = safe_auc(y_test, y_proba_for_labels, labels)

    holdout_metrics = {
        "feature_set": feature_set_name,
        "model": "Random forest",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": macro_auc,
        "roc_auc_ovr_weighted": weighted_auc,
        "oob_score": getattr(pipeline.named_steps["model"], "oob_score_", np.nan),
        "n_features_input": len(feature_cols),
        "n_features_model_matrix": int(pipeline.named_steps["model"].n_features_in_),
    }
    pd.DataFrame([holdout_metrics]).to_csv(feature_dir / "holdout_metrics.csv", index=False)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(feature_dir / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(feature_dir / "confusion_matrix.csv")

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        feature_dir / "confusion_matrix_normalized.csv"
    )

    plot_confusion_matrix(
        cm,
        labels,
        feature_dir / "confusion_matrix.png",
        f"Random forest confusion matrix: {feature_set_name}",
    )
    plot_confusion_matrix(
        cm,
        labels,
        feature_dir / "confusion_matrix_normalized.png",
        f"Random forest normalized confusion matrix: {feature_set_name}",
        normalize=True,
    )

    predictions = pd.DataFrame(
        {
            "row_index": x_test.index,
            "true_abuse": y_test.to_numpy(),
            "predicted_abuse": y_pred,
        }
    )
    for idx, label in enumerate(fitted_labels):
        predictions[f"prob_{label}"] = y_proba[:, idx]
    predictions = predictions.sort_values("row_index")
    predictions.to_csv(feature_dir / "test_predictions.csv", index=False)

    numeric_cols, categorical_cols = split_columns(df, feature_cols)
    transformed_feature_names = get_transformed_feature_names(
        pipeline,
        numeric_cols,
        categorical_cols,
    )
    importances = pipeline.named_steps["model"].feature_importances_
    n_names = min(len(transformed_feature_names), len(importances))
    importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names[:n_names],
            "importance": importances[:n_names],
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(feature_dir / "random_forest_feature_importance.csv", index=False)
    plot_feature_importance(
        importance_df,
        feature_dir / "random_forest_feature_importance_top30.png",
        f"Random forest feature importance: {feature_set_name}",
    )

    if not args.skip_permutation:
        perm = permutation_importance(
            pipeline,
            x_test,
            y_test,
            scoring="balanced_accuracy",
            n_repeats=args.permutation_repeats,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        perm_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_sd": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(feature_dir / "permutation_importance.csv", index=False)
        plot_permutation_importance(
            perm_df,
            feature_dir / "permutation_importance_top30.png",
            f"Permutation importance: {feature_set_name}",
        )

    joblib.dump(pipeline, feature_dir / "random_forest_pipeline.joblib")

    return holdout_metrics


def write_notes(output_dir: Path, input_path: Path, metrics: pd.DataFrame) -> None:
    lines = [
        "Random forest abuse-category prediction",
        "",
        f"Input: {input_path}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Target:",
        "  Four single-type abuse categories: Physical Abuse, Neglect, Emotional Abuse, Sexual Abuse.",
        "",
        "Method:",
        "  Stratified train/test split, class_weight='balanced_subsample', random forest classifier.",
        "  Main holdout metrics emphasize balanced_accuracy and macro F1 because class sizes are uneven.",
        "",
        "Leakage controls:",
        "  Excluded abuse, abuse_1, abuse_num, identifiers, date, and free-text instruction/memo columns.",
        "",
        "Holdout summary:",
    ]
    if not metrics.empty:
        for _, row in metrics.iterrows():
            lines.append(
                "  "
                f"{row['feature_set']}: balanced_accuracy={row['balanced_accuracy']:.3f}, "
                f"f1_macro={row['f1_macro']:.3f}, accuracy={row['accuracy']:.3f}"
            )
    (output_dir / "analysis_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _, input_path, output_dir = resolve_paths(args)

    df = load_analysis_data(input_path)
    feature_sets = build_feature_sets(df)

    if args.feature_set != "both":
        feature_sets = {args.feature_set: feature_sets[args.feature_set]}

    if not feature_sets:
        raise ValueError("No feature sets were selected.")

    for name, features in feature_sets.items():
        if len(features) == 0:
            raise ValueError(f"Feature set '{name}' has no usable columns.")

    class_counts = df["abuse"].value_counts().reindex(TARGET_CLASSES).fillna(0).astype(int)
    class_counts.to_csv(output_dir / "target_class_counts.csv", header=["n"])

    all_metrics = []
    for feature_set_name, feature_cols in feature_sets.items():
        print(f"Running feature set: {feature_set_name} ({len(feature_cols)} input features)")
        metrics = run_feature_set(df, feature_set_name, feature_cols, output_dir, args)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "model_performance_summary.csv", index=False)
    write_notes(output_dir, input_path, metrics_df)

    print(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Random forest prediction of abuse category.

Revised version implementing the following analysis changes:

1. Explicit numeric/categorical feature typing rather than relying only on pandas dtypes.
2. Numeric feature missing values are kept as NaN by default; no median imputation is applied
   unless --numeric-na-policy median is explicitly selected.
3. Strict single-type filtering: only rows with abuse_num == 1 are retained.
4. Stratified split validation before train/test split.
5. DummyClassifier baseline evaluation.
6. Repeated stratified cross-validation support.
7. Optional training-set-only hyperparameter tuning for max_depth/min_samples_leaf and related RF parameters.
8. Optional probability calibration using CalibratedClassifierCV.
9. Missingness reports by feature and by abuse class.
10. Aggregated feature importance for one-hot encoded variables and subgroup performance reports.

The target is the four single-type maltreatment categories:
Physical Abuse, Neglect, Emotional Abuse, and Sexual Abuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        RepeatedStratifiedKFold,
        StratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
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
    "1.0": "Physical Abuse",
    "2": "Neglect",
    "2.0": "Neglect",
    "3": "Emotional Abuse",
    "3.0": "Emotional Abuse",
    "4": "Sexual Abuse",
    "4.0": "Sexual Abuse",
    1: "Physical Abuse",
    1.0: "Physical Abuse",
    2: "Neglect",
    2.0: "Neglect",
    3: "Emotional Abuse",
    3.0: "Emotional Abuse",
    4: "Sexual Abuse",
    4.0: "Sexual Abuse",
}

DERIVED_ORAL_FEATURES = [
    # "Perm_D",
    # "Perm_M",
    # "Perm_F",
    # "Perm_Sound",
    # "Perm_C0",
    # "Perm_total_teeth",
    "Perm_DMFT",
    # "Perm_DMFT_C0",
    "Perm_sound_rate",
    # "Baby_d",
    # "Baby_m",
    # "Baby_f",
    # "Baby_sound",
    # "Baby_C0",
    # "Baby_total_teeth",
    "Baby_DMFT",
    # "Baby_DMFT_C0",
    "Baby_sound_rate",
    "DMFT_Index",
    # "DMFT_C0",
    # "C0_Count",
    # "filled_total",
    # "decayed_total",
    # "missing_total",
    "Care_Index",
    # "UTN_Score",
    # "total_teeth",
    "Healthy_Rate",
    # "Present_Teeth",
    # "Present_Perm_Teeth",
    # "Present_Baby_Teeth",
    "has_caries",
    "has_untreated_caries",
    "dentition_type",
]

CLINICAL_FEATURES = [
    # "needTOBEtreated",
    # "emergency",
    # "emergencyInMonths",
    "gingivitis",
    "occlusalRelationship",
    # "habits",
    "OralCleanStatus",
    # "Orthodontics",
    # "dentists",
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
    # "age_year",
    # "age_month",
    "age",
    "sex",
    "CGC",
    "year",
]

# Explicit feature-type definitions. Unlisted columns fall back to pandas dtype inference.
NUMERIC_FEATURES = {
    # "Perm_D",
    # "Perm_M",
    # "Perm_F",
    # "Perm_Sound",
    # "Perm_C0",
    # "Perm_total_teeth",
    "Perm_DMFT",
    # "Perm_DMFT_C0",
    "Perm_sound_rate",
    # "Baby_d",
    # "Baby_m",
    # "Baby_f",
    # "Baby_sound",
    # "Baby_C0",
    # "Baby_total_teeth",
    "Baby_DMFT",
    # "Baby_DMFT_C0",
    "Baby_sound_rate",
    "DMFT_Index",
    # "DMFT_C0",
    # "C0_Count",
    # "filled_total",
    # "decayed_total",
    # "missing_total",
    "Care_Index",
    "UTN_Score",
    # "total_teeth",
    "Healthy_Rate",
    # "Present_Teeth",
    # "Present_Perm_Teeth",
    # "Present_Baby_Teeth",
    # "age_year",
    # "age_month",
    "age",
}

CATEGORICAL_FEATURES = {
    "has_caries",
    "has_untreated_caries",
    "dentition_type",
    "gingivitis",
    "occlusalRelationship",
    "OralCleanStatus",
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
    "sex",
    "CGC",
    "year",
}

TARGET_LEAKAGE_COLUMNS = {
    "abuse",
    "abuse_1",
    "abuse_num",
}

ID_OR_TEXT_COLUMNS = {
    "No_All",
    "date",
    "instruction_detail",
    "instruction",
    "memo",
}

SUBGROUP_CANDIDATES = ["sex", "age", "year", "CGC"]


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
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=5,
        help="Number of repeats for repeated stratified CV. Use 1 to reproduce ordinary stratified K-fold CV.",
    )
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--numeric-na-policy",
        choices=["keep", "median"],
        default="keep",
        help=(
            "How to handle numeric feature NaNs. 'keep' passes NaN values into RandomForestClassifier; "
            "'median' applies SimpleImputer(strategy='median')."
        ),
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Tune random-forest hyperparameters inside the training set only.",
    )
    parser.add_argument(
        "--tuning-method",
        choices=["grid", "random"],
        default="grid",
        help="Search strategy for hyperparameter tuning.",
    )
    parser.add_argument(
        "--tuning-grid",
        choices=["compact", "extended"],
        default="compact",
        help="Parameter grid size used when --tune-hyperparameters is enabled.",
    )
    parser.add_argument(
        "--tuning-refit",
        choices=["balanced_accuracy", "f1_macro", "roc_auc_ovr_weighted"],
        default="balanced_accuracy",
        help="Metric used to select the best hyperparameter combination.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Number of sampled parameter settings for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--calibrate-probabilities",
        action="store_true",
        help="Calibrate predicted probabilities on the training set using CalibratedClassifierCV.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibration method. Sigmoid is safer for smaller samples; isotonic is more flexible but data-hungry.",
    )
    parser.add_argument(
        "--calibration-cv-folds",
        type=int,
        default=3,
        help="Number of stratified folds for probability calibration.",
    )
    parser.add_argument(
        "--subgroup-min-n",
        type=int,
        default=10,
        help="Flag subgroup rows with n below this threshold as unstable in subgroup_metrics.csv.",
    )
    return parser.parse_args()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    raw = series.copy()
    target = raw.replace(TARGET_RECODE)
    target = target.astype("string").str.strip()

    numeric_codes = pd.to_numeric(raw.astype("string").str.strip(), errors="coerce")
    numeric_mapped = numeric_codes.map(
        {
            1.0: "Physical Abuse",
            2.0: "Neglect",
            3.0: "Emotional Abuse",
            4.0: "Sexual Abuse",
        }
    )
    target = numeric_mapped.fillna(target)
    return target.astype("string").str.strip()


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
    if "abuse_num" not in df.columns:
        raise ValueError(
            "Input data must contain 'abuse_num' because this revised script strictly retains abuse_num == 1."
        )

    df = df.copy()
    df["abuse"] = clean_target(df["abuse"])
    df = df[df["abuse"].isin(TARGET_CLASSES)].copy()

    abuse_num = pd.to_numeric(df["abuse_num"], errors="coerce")
    df = df[abuse_num == 1].copy()

    if df.empty:
        raise ValueError(
            "No rows remained after filtering to the four target abuse categories and abuse_num == 1."
        )

    return df


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    oral_only = existing_columns(
        df,
        DERIVED_ORAL_FEATURES + CLINICAL_FEATURES,
        # + BEHAVIOR_FEATURES
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


def feature_type_rule(df: pd.DataFrame, col: str) -> tuple[str, str]:
    if col in NUMERIC_FEATURES:
        return "numeric", "explicit_numeric"
    if col in CATEGORICAL_FEATURES:
        return "categorical", "explicit_categorical"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric", "fallback_pandas_numeric_dtype"
    return "categorical", "fallback_pandas_non_numeric_dtype"


def split_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        feature_type, _ = feature_type_rule(df, col)
        if feature_type == "numeric":
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_feature_type_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        feature_type, rule = feature_type_rule(df, col)
        rows.append(
            {
                "feature": col,
                "feature_type": feature_type,
                "typing_rule": rule,
                "pandas_dtype": str(df[col].dtype),
                "n_missing": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "n_unique_nonmissing": int(df[col].dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    x = df[feature_cols].copy()

    for col in numeric_cols:
        # Keep missing values as NaN. Non-numeric strings are converted to NaN instead of being imputed.
        x[col] = pd.to_numeric(x[col], errors="coerce")

    for col in categorical_cols:
        # Keep missing values as np.nan for the categorical imputer; cast observed values later to strings.
        x[col] = x[col].astype("object")
        x[col] = x[col].where(pd.notna(x[col]), np.nan)

    return x


def categorical_to_string_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    return arr.astype(str)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    args: argparse.Namespace,
) -> Pipeline:
    transformers = []

    if numeric_cols:
        if args.numeric_na_policy == "keep":
            transformers.append(("num", "passthrough", numeric_cols))
        else:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )
            transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "to_string",
                    FunctionTransformer(
                        categorical_to_string_array,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("onehot", make_one_hot_encoder()),
            ]
        )
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


def validate_stratified_split(y: pd.Series, test_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be a float between 0 and 1.")

    counts = y.value_counts()
    n_classes = int(y.nunique())
    n_total = int(len(y))

    if n_classes < 2:
        raise ValueError("At least two target classes are required for classification.")

    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Stratified train/test split requires at least two rows per class. "
            f"Too-small classes: {too_small.to_dict()}"
        )

    n_test = math.ceil(n_total * test_size)
    n_train = n_total - n_test

    if n_test < n_classes:
        raise ValueError(
            f"test_size={test_size} gives only {n_test} test rows, but there are {n_classes} classes. "
            "Increase --test-size or use a CV-only design."
        )
    if n_train < n_classes:
        raise ValueError(
            f"test_size={test_size} leaves only {n_train} training rows, but there are {n_classes} classes. "
            "Decrease --test-size."
        )


def make_cv_splitter(
    y: pd.Series,
    args: argparse.Namespace,
    *,
    requested_folds: int | None = None,
    requested_repeats: int | None = None,
) -> tuple[Any | None, int, int, str]:
    min_class_count = int(y.value_counts().min())
    n_splits = min(requested_folds or args.cv_folds, min_class_count)
    n_repeats = max(1, requested_repeats if requested_repeats is not None else args.cv_repeats)

    if n_splits < 2:
        return None, n_splits, n_repeats, "skipped"

    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=args.random_state,
        )
        return cv, n_splits, n_repeats, "RepeatedStratifiedKFold"

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    return cv, n_splits, n_repeats, "StratifiedKFold"


def scoring_dict() -> dict[str, str]:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }


def run_cross_validation(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cv, n_splits, n_repeats, cv_type = make_cv_splitter(y, args)
    if cv is None:
        return pd.DataFrame(
            [
                {
                    "metric": "note",
                    "mean": np.nan,
                    "sd": np.nan,
                    "n_splits": n_splits,
                    "n_repeats": n_repeats,
                    "cv_type": cv_type,
                    "detail": "Cross-validation skipped because at least one class has fewer than two rows.",
                }
            ]
        )

    cv_results = cross_validate(
        pipeline,
        x,
        y,
        cv=cv,
        scoring=scoring_dict(),
        n_jobs=1,
        return_train_score=True,
        error_score=np.nan,
    )

    rows = []
    for key, values in cv_results.items():
        if not (key.startswith("test_") or key.startswith("train_")):
            continue
        split, metric = key.split("_", 1)
        rows.append(
            {
                "metric": metric,
                "split": split,
                "mean": float(np.nanmean(values)),
                "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "cv_type": cv_type,
                "detail": "",
            }
        )
    return pd.DataFrame(rows)


def tuning_parameter_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    if args.tuning_grid == "compact":
        return {
            "model__max_depth": [None, 6, 8, 10, 12, 16],
            "model__min_samples_leaf": [1, 2, 3, 5, 8, 10],
        }

    return {
        "model__max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
        "model__min_samples_leaf": [1, 2, 3, 5, 8, 10, 15, 20],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }


def tune_random_forest_hyperparameters(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_dir: Path,
    args: argparse.Namespace,
) -> Pipeline:
    cv, n_splits, n_repeats, cv_type = make_cv_splitter(y_train, args)
    if cv is None:
        pipeline.fit(x_train, y_train)
        write_json(
            feature_dir / "best_hyperparameters.json",
            {
                "tuning_performed": False,
                "reason": "At least one class had fewer than two training rows.",
                "fallback": "Fitted the supplied random-forest parameters without tuning.",
            },
        )
        return pipeline

    param_grid = tuning_parameter_grid(args)
    common_kwargs = dict(
        estimator=pipeline,
        scoring=scoring_dict(),
        refit=args.tuning_refit,
        cv=cv,
        n_jobs=1,
        return_train_score=True,
        error_score=np.nan,
        verbose=0,
    )

    if args.tuning_method == "random":
        search = RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=args.n_iter,
            random_state=args.random_state,
            **common_kwargs,
        )
    else:
        search = GridSearchCV(
            param_grid=param_grid,
            **common_kwargs,
        )

    search.fit(x_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(feature_dir / "hyperparameter_tuning_results.csv", index=False)

    best_summary = {
        "tuning_performed": True,
        "tuning_method": args.tuning_method,
        "tuning_grid": args.tuning_grid,
        "refit_metric": args.tuning_refit,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "cv_type": cv_type,
        "n_candidates": int(len(cv_results)),
        "note": "Tuning used training data only. The holdout test set was not used for model selection.",
    }
    write_json(feature_dir / "best_hyperparameters.json", best_summary)

    return search.best_estimator_


def proba_in_lexicographic_label_order(
    y_proba: np.ndarray,
    labels: list[str],
) -> tuple[list[str], np.ndarray]:
    sorted_labels = sorted(labels)
    column_order = [labels.index(label) for label in sorted_labels]
    return sorted_labels, y_proba[:, column_order]


def safe_balanced_accuracy_value(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(y_true, y_pred))


def safe_auc(y_true: pd.Series, y_proba: np.ndarray, labels: list[str]) -> tuple[float, float]:
    # scikit-learn's probability metrics expect columns to match lexicographically sorted labels.
    auc_labels, auc_proba = proba_in_lexicographic_label_order(y_proba, labels)

    try:
        macro_auc = roc_auc_score(
            y_true,
            auc_proba,
            labels=auc_labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = np.nan

    try:
        weighted_auc = roc_auc_score(
            y_true,
            auc_proba,
            labels=auc_labels,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        weighted_auc = np.nan

    return macro_auc, weighted_auc


def multiclass_brier_score(y_true: pd.Series, y_proba: np.ndarray, labels: list[str]) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_one_hot = np.zeros((len(y_true), len(labels)), dtype=float)
    for row_idx, label in enumerate(y_true):
        if label in label_to_idx:
            y_one_hot[row_idx, label_to_idx[label]] = 1.0
    return float(np.mean(np.sum((y_one_hot - y_proba) ** 2, axis=1)))


def align_predict_proba(
    estimator: Any,
    x: pd.DataFrame,
    labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    y_proba = estimator.predict_proba(x)
    fitted_labels = list(estimator.classes_)
    aligned = np.zeros((len(x), len(labels)))
    for idx, label in enumerate(labels):
        if label in fitted_labels:
            aligned[:, idx] = y_proba[:, fitted_labels.index(label)]
    return aligned, fitted_labels


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba_for_labels: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    macro_auc, weighted_auc = safe_auc(y_true, y_proba_for_labels, labels)

    try:
        ll_labels, ll_proba = proba_in_lexicographic_label_order(y_proba_for_labels, labels)
        ll = log_loss(y_true, ll_proba, labels=ll_labels)
    except ValueError:
        ll = np.nan

    try:
        brier = multiclass_brier_score(y_true, y_proba_for_labels, labels)
    except Exception:
        brier = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_balanced_accuracy_value(y_true, y_pred),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(macro_auc) if not pd.isna(macro_auc) else np.nan,
        "roc_auc_ovr_weighted": float(weighted_auc) if not pd.isna(weighted_auc) else np.nan,
        "log_loss": float(ll) if not pd.isna(ll) else np.nan,
        "brier_score_multiclass_sum": float(brier) if not pd.isna(brier) else np.nan,
    }


def evaluate_dummy_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[str],
    feature_dir: Path,
) -> dict[str, Any]:
    x_train_dummy = np.zeros((len(y_train), 1))
    x_test_dummy = np.zeros((len(y_test), 1))

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train_dummy, y_train)
    y_pred = baseline.predict(x_test_dummy)
    y_proba = baseline.predict_proba(x_test_dummy)

    aligned = np.zeros((len(y_test), len(labels)))
    fitted_labels = list(baseline.classes_)
    for idx, label in enumerate(labels):
        if label in fitted_labels:
            aligned[:, idx] = y_proba[:, fitted_labels.index(label)]

    metrics = evaluate_predictions(y_test, y_pred, aligned, labels)
    metrics.update(
        {
            "model": "DummyClassifier_most_frequent",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
    )
    pd.DataFrame([metrics]).to_csv(feature_dir / "baseline_holdout_metrics.csv", index=False)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(feature_dir / "baseline_classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        feature_dir / "baseline_confusion_matrix.csv"
    )

    return metrics


def fit_calibrated_estimator(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_dir: Path,
    args: argparse.Namespace,
) -> tuple[Any, bool]:
    cv, n_splits, _, cv_type = make_cv_splitter(
        y_train,
        args,
        requested_folds=args.calibration_cv_folds,
        requested_repeats=1,
    )
    if cv is None:
        write_json(
            feature_dir / "calibration_info.json",
            {
                "calibration_performed": False,
                "reason": "At least one class had fewer than two training rows for calibration CV.",
            },
        )
        return estimator, False

    try:
        calibrated = CalibratedClassifierCV(
            estimator=estimator,
            method=args.calibration_method,
            cv=cv,
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=estimator,
            method=args.calibration_method,
            cv=cv,
        )

    calibrated.fit(x_train, y_train)
    write_json(
        feature_dir / "calibration_info.json",
        {
            "calibration_performed": True,
            "method": args.calibration_method,
            "n_splits": n_splits,
            "cv_type": cv_type,
            "note": "Calibration was fitted on the training set only. The holdout test set was not used for calibration.",
        },
    )
    return calibrated, True


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


def transformed_to_original_mapping(
    fitted_pipeline: Pipeline,
    transformed_feature_names: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for col in numeric_cols:
        mapping[f"num__{col}"] = col
        mapping[col] = col

    preprocessor = fitted_pipeline.named_steps["preprocess"]
    if categorical_cols:
        try:
            cat_pipe = preprocessor.named_transformers_.get("cat")
            encoder = cat_pipe.named_steps["onehot"]
            cat_names = list(encoder.get_feature_names_out(categorical_cols))
            start = 0
            for col, categories in zip(categorical_cols, encoder.categories_):
                for name in cat_names[start : start + len(categories)]:
                    mapping[f"cat__{name}"] = col
                    mapping[name] = col
                start += len(categories)
        except Exception:
            pass

    # Fallback parsing for any names not covered above.
    categorical_by_length = sorted(categorical_cols, key=len, reverse=True)
    for name in transformed_feature_names:
        if name in mapping:
            continue
        if name.startswith("num__"):
            mapping[name] = name.replace("num__", "", 1)
        elif name.startswith("cat__"):
            remainder = name.replace("cat__", "", 1)
            matched = None
            for col in categorical_by_length:
                if remainder == col or remainder.startswith(f"{col}_"):
                    matched = col
                    break
            mapping[name] = matched or remainder
        else:
            mapping[name] = name

    return mapping


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
    im = ax.imshow(values)
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
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    value_col: str = "importance",
    xlabel: str = "Importance",
) -> None:
    top = importance_df.head(30).copy()
    if top.empty:
        return

    top = top.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.25)))
    ax.barh(top["feature"], top[value_col])
    ax.set_xlabel(xlabel)
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
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"])
    ax.set_xlabel("Balanced accuracy decrease after permutation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_feature_importance_notes(feature_dir: Path) -> None:
    lines = [
        "Feature importance notes",
        "",
        "Random-forest impurity importance is not a causal effect estimate.",
        "It can favor continuous variables or variables with many split points.",
        "One-hot encoded categorical variables are split into multiple transformed columns;",
        "therefore random_forest_feature_importance_aggregated.csv sums those columns back to the original feature.",
        "Permutation importance is reported at the original input-feature level using balanced_accuracy on the holdout test set.",
        "Correlated predictors can dilute each other's permutation importance.",
    ]
    (feature_dir / "feature_importance_notes.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_feature_importance_outputs(
    fitted_pipeline: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
    feature_dir: Path,
    feature_set_name: str,
) -> None:
    transformed_feature_names = get_transformed_feature_names(
        fitted_pipeline,
        numeric_cols,
        categorical_cols,
    )
    importances = fitted_pipeline.named_steps["model"].feature_importances_
    n_names = min(len(transformed_feature_names), len(importances))
    mapping = transformed_to_original_mapping(
        fitted_pipeline,
        transformed_feature_names[:n_names],
        numeric_cols,
        categorical_cols,
    )

    importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names[:n_names],
            "original_feature": [mapping[name] for name in transformed_feature_names[:n_names]],
            "importance": importances[:n_names],
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(feature_dir / "random_forest_feature_importance.csv", index=False)
    plot_feature_importance(
        importance_df,
        feature_dir / "random_forest_feature_importance_top30.png",
        f"Random forest feature importance: {feature_set_name}",
        value_col="importance",
        xlabel="Mean decrease in impurity",
    )

    aggregated = (
        importance_df.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .rename(columns={"original_feature": "feature"})
    )
    aggregated.to_csv(
        feature_dir / "random_forest_feature_importance_aggregated.csv",
        index=False,
    )
    plot_feature_importance(
        aggregated,
        feature_dir / "random_forest_feature_importance_aggregated_top30.png",
        f"Aggregated RF feature importance: {feature_set_name}",
        value_col="importance",
        xlabel="Summed mean decrease in impurity",
    )
    write_feature_importance_notes(feature_dir)


def write_missingness_reports(
    x: pd.DataFrame,
    y: pd.Series,
    feature_types: pd.DataFrame,
    feature_dir: Path,
) -> None:
    type_map = dict(zip(feature_types["feature"], feature_types["feature_type"]))
    overall_rows = []
    for col in x.columns:
        overall_rows.append(
            {
                "feature": col,
                "feature_type": type_map.get(col, "unknown"),
                "n_missing": int(x[col].isna().sum()),
                "missing_rate": float(x[col].isna().mean()),
                "n_nonmissing": int(x[col].notna().sum()),
            }
        )
    pd.DataFrame(overall_rows).sort_values("missing_rate", ascending=False).to_csv(
        feature_dir / "missingness_by_feature.csv",
        index=False,
    )

    by_class_rows = []
    for abuse_class, idx in y.groupby(y).groups.items():
        subset = x.loc[idx]
        for col in x.columns:
            by_class_rows.append(
                {
                    "abuse": abuse_class,
                    "feature": col,
                    "feature_type": type_map.get(col, "unknown"),
                    "n_class": int(len(subset)),
                    "n_missing": int(subset[col].isna().sum()),
                    "missing_rate": float(subset[col].isna().mean()),
                }
            )
    pd.DataFrame(by_class_rows).sort_values(
        ["feature", "abuse"]
    ).to_csv(feature_dir / "missingness_by_class.csv", index=False)


def save_calibration_curves(
    y_true: pd.Series,
    y_proba_for_labels: np.ndarray,
    labels: list[str],
    feature_dir: Path,
    title: str,
) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_any = False

    for idx, label in enumerate(labels):
        binary_true = (y_true == label).astype(int).to_numpy()
        if np.unique(binary_true).size < 2:
            continue
        try:
            prob_true, prob_pred = calibration_curve(
                binary_true,
                y_proba_for_labels[:, idx],
                n_bins=10,
                strategy="uniform",
            )
        except ValueError:
            continue

        for bin_idx, (pred, true) in enumerate(zip(prob_pred, prob_true), start=1):
            rows.append(
                {
                    "class": label,
                    "bin": bin_idx,
                    "mean_predicted_probability": float(pred),
                    "observed_fraction": float(true),
                }
            )
        ax.plot(prob_pred, prob_true, marker="o", label=label)
        plotted_any = True

    if rows:
        pd.DataFrame(rows).to_csv(feature_dir / "calibration_curves.csv", index=False)

    if plotted_any:
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(feature_dir / "calibration_curves.png", dpi=300)
    plt.close(fig)


def age_group_series(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    grouped = pd.cut(
        age,
        bins=[-np.inf, 5, 11, 17, np.inf],
        labels=["0-5", "6-11", "12-17", "18+"],
    ).astype("object")
    return pd.Series(grouped, index=series.index).where(age.notna(), "Missing").astype(str)


def subgroup_values(df_subset: pd.DataFrame, variable: str) -> pd.Series:
    if variable == "age_year":
        return age_group_series(df_subset[variable])
    return df_subset[variable].astype("object").where(pd.notna(df_subset[variable]), "Missing").astype(str)


def evaluate_subgroups(
    df_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    feature_dir: Path,
    args: argparse.Namespace,
) -> None:
    rows = []
    prediction_series = pd.Series(y_pred, index=y_test.index, name="predicted_abuse")

    for variable in SUBGROUP_CANDIDATES:
        if variable not in df_test.columns:
            continue
        groups = subgroup_values(df_test, variable)
        for value, idx in groups.groupby(groups).groups.items():
            yt = y_test.loc[idx]
            yp = prediction_series.loc[idx]
            n = int(len(yt))
            if n == 0:
                continue

            report = classification_report(
                yt,
                yp,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            row = {
                "subgroup_variable": variable,
                "subgroup_value": value,
                "n": n,
                "below_min_n": bool(n < args.subgroup_min_n),
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": safe_balanced_accuracy_value(yt, yp),
                "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            }
            for label in labels:
                row[f"support_{label}"] = int((yt == label).sum())
                row[f"recall_{label}"] = float(report.get(label, {}).get("recall", np.nan))
            rows.append(row)

    if rows:
        pd.DataFrame(rows).sort_values(["subgroup_variable", "subgroup_value"]).to_csv(
            feature_dir / "subgroup_metrics.csv",
            index=False,
        )


def run_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    feature_dir = output_dir / feature_set_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols, categorical_cols = split_columns(df, feature_cols)
    feature_types = make_feature_type_table(df, feature_cols)
    feature_types.to_csv(feature_dir / "feature_types.csv", index=False)

    x = prepare_feature_matrix(df, feature_cols, numeric_cols, categorical_cols)
    y = df["abuse"].copy()
    labels = [label for label in TARGET_CLASSES if label in set(y)]

    write_missingness_reports(x, y, feature_types, feature_dir)
    validate_stratified_split(y, args.test_size)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

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
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "numeric_na_policy": args.numeric_na_policy,
    }
    write_json(feature_dir / "split_info.json", split_info)
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_dir / "input_features.csv", index=False)

    baseline_metrics = evaluate_dummy_baseline(y_train, y_test, labels, feature_dir)

    pipeline = make_pipeline(feature_cols, numeric_cols, categorical_cols, args)

    if args.tune_hyperparameters:
        fitted_pipeline = tune_random_forest_hyperparameters(
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            feature_dir=feature_dir,
            args=args,
        )
        cv_summary = run_cross_validation(fitted_pipeline, x_train, y_train, args)
        cv_summary["detail"] = cv_summary["detail"].replace(
            "",
            "Best-parameter CV summary after training-set-only tuning; not a nested-CV estimate.",
        )
    else:
        cv_summary = run_cross_validation(pipeline, x_train, y_train, args)
        fitted_pipeline = pipeline.fit(x_train, y_train)
        write_json(
            feature_dir / "best_hyperparameters.json",
            {
                "tuning_performed": False,
                "params_used": {
                    "max_depth": args.max_depth,
                    "min_samples_leaf": args.min_samples_leaf,
                    "max_features": "sqrt",
                    "class_weight": "balanced_subsample",
                },
            },
        )

    cv_summary.insert(0, "feature_set", feature_set_name)
    cv_summary.to_csv(feature_dir / "cross_validation_metrics.csv", index=False)

    eval_estimator: Any = fitted_pipeline
    calibration_performed = False
    if args.calibrate_probabilities:
        eval_estimator, calibration_performed = fit_calibrated_estimator(
            fitted_pipeline,
            x_train,
            y_train,
            feature_dir,
            args,
        )

    y_pred = eval_estimator.predict(x_test)
    y_proba_for_labels, fitted_labels = align_predict_proba(eval_estimator, x_test, labels)

    model_metrics = evaluate_predictions(y_test, y_pred, y_proba_for_labels, labels)
    holdout_metrics: dict[str, Any] = {
        "feature_set": feature_set_name,
        "model": "Random forest" + (" + calibrated probabilities" if calibration_performed else ""),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "numeric_na_policy": args.numeric_na_policy,
        "hyperparameter_tuned": bool(args.tune_hyperparameters),
        "calibrated_probabilities": bool(calibration_performed),
        **model_metrics,
        "baseline_accuracy": baseline_metrics.get("accuracy", np.nan),
        "baseline_balanced_accuracy": baseline_metrics.get("balanced_accuracy", np.nan),
        "baseline_f1_macro": baseline_metrics.get("f1_macro", np.nan),
        "oob_score": getattr(fitted_pipeline.named_steps["model"], "oob_score_", np.nan),
        "n_features_input": int(len(feature_cols)),
        "n_features_model_matrix": int(fitted_pipeline.named_steps["model"].n_features_in_),
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
    for idx, label in enumerate(labels):
        predictions[f"prob_{label}"] = y_proba_for_labels[:, idx]
    predictions = predictions.sort_values("row_index")
    predictions.to_csv(feature_dir / "test_predictions.csv", index=False)

    save_calibration_curves(
        y_test,
        y_proba_for_labels,
        labels,
        feature_dir,
        f"Calibration curves: {feature_set_name}",
    )

    save_feature_importance_outputs(
        fitted_pipeline,
        numeric_cols,
        categorical_cols,
        feature_dir,
        feature_set_name,
    )

    if not args.skip_permutation:
        perm = permutation_importance(
            eval_estimator,
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

    evaluate_subgroups(
        df.loc[x_test.index],
        y_test,
        y_pred,
        labels,
        feature_dir,
        args,
    )

    joblib.dump(eval_estimator, feature_dir / "random_forest_pipeline.joblib")
    if calibration_performed:
        joblib.dump(fitted_pipeline, feature_dir / "random_forest_pipeline_uncalibrated.joblib")
        joblib.dump(eval_estimator, feature_dir / "random_forest_pipeline_calibrated.joblib")

    return holdout_metrics


def write_run_provenance(output_dir: Path, input_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "input_sha256": file_sha256(input_path),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "args": vars(args),
        "note_on_numeric_nan": (
            "With --numeric-na-policy keep, numeric NaNs are passed through to RandomForestClassifier. "
            "This requires a scikit-learn version whose RandomForestClassifier supports missing values natively."
        ),
    }
    write_json(output_dir / "run_provenance.json", payload)


def write_notes(output_dir: Path, input_path: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "Random forest abuse-category prediction",
        "",
        f"Input: {input_path}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Target:",
        "  Four single-type abuse categories: Physical Abuse, Neglect, Emotional Abuse, Sexual Abuse.",
        "  Rows are strictly retained only when abuse_num == 1.",
        "",
        "Method:",
        "  Stratified train/test split, class_weight='balanced_subsample', random forest classifier.",
        f"  Cross-validation: {args.cv_folds}-fold stratified CV repeated {args.cv_repeats} time(s).",
        "  Main holdout metrics emphasize balanced_accuracy and macro F1 because class sizes are uneven.",
        "",
        "Missing values:",
        f"  Numeric feature NaN policy: {args.numeric_na_policy}.",
        "  If policy is 'keep', numeric NaNs are passed to RandomForestClassifier rather than imputed.",
        "  Categorical feature missingness is encoded as a 'Missing' level before one-hot encoding.",
        "  Missingness summaries are saved as missingness_by_feature.csv and missingness_by_class.csv.",
        "",
        "Hyperparameter tuning and calibration:",
        f"  Hyperparameter tuning enabled: {args.tune_hyperparameters}.",
        f"  Probability calibration enabled: {args.calibrate_probabilities}.",
        "  When enabled, tuning and calibration use training data only; the holdout test set is evaluated once.",
        "",
        "Baseline and subgroup analysis:",
        "  A most-frequent-class DummyClassifier baseline is saved for comparison.",
        "  Subgroup performance is saved for sex, age_year groups, year, and CGC when available.",
        "",
        "Leakage controls:",
        "  Excluded abuse, abuse_1, abuse_num, identifiers, date, and free-text instruction/memo columns.",
        "",
        "Feature importance interpretation:",
        "  Random-forest feature importance and permutation importance are exploratory, not causal.",
        "  Aggregated one-hot feature importance is saved to reduce interpretability problems from dummy variables.",
        "",
        "Holdout summary:",
    ]
    if not metrics.empty:
        for _, row in metrics.iterrows():
            lines.append(
                "  "
                f"{row['feature_set']}: balanced_accuracy={row['balanced_accuracy']:.3f}, "
                f"f1_macro={row['f1_macro']:.3f}, accuracy={row['accuracy']:.3f}; "
                f"baseline_balanced_accuracy={row['baseline_balanced_accuracy']:.3f}"
            )
    (output_dir / "analysis_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _, input_path, output_dir = resolve_paths(args)

    df = load_analysis_data(input_path)
    write_run_provenance(output_dir, input_path, args)
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
    write_notes(output_dir, input_path, metrics_df, args)

    print(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

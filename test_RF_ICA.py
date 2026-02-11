# -*- coding: utf-8 -*-
"""
abuse_dental_ml.py

目的:
  1) 歯科健診データから虐待4分類(Physical/Neglect/Emotional/Sexual)をRandom Forestで多クラス分類
  2) 列単位Permutation Importanceで「どの所見が分類に寄与するか」を定量化
  3) 口腔内指標(連続量中心)にFastICAを適用し、潜在パターン(独立成分)を抽出して虐待群で比較

注意:
  - U17等の歯別コード(0,1,2,3,4...)の意味が不明なため、カテゴリとしてone-hot化して扱います。
  - 自由記述(instruction_detail, memo等)はリーク/倫理面の懸念があるため、既定では特徴量に使いません。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from sklearn.decomposition import FastICA

import matplotlib.pyplot as plt

# statsmodelsはICAスコアの群間比較(調整)に使用。なければスキップ可能。
try:
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

timestamp = datetime.now().strftime('%Y%m%d')


ABUSE_LABELS = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]


def read_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 先頭の空列がある場合、pandasは "Unnamed: 0" 等の列名にすることが多い
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def detect_tooth_columns(df: pd.DataFrame):
    """
    歯別列を自動検出:
      - 永久歯: U17..U11, U21..U27, L37..L31, L41..L47 など (大文字)
      - 乳歯: u55..u51, u61..u65, l75..l71, l81..l85 など (小文字)
    """
    perm_cols = [c for c in df.columns if re.fullmatch(r"[UL]\d{2}", str(c))]
    baby_cols = [c for c in df.columns if re.fullmatch(r"[ul]\d{2}", str(c))]
    return perm_cols, baby_cols


def add_basic_binary_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - gingivitis / needTOBEtreated / OralCleanStatus を、可能なら数値化した列を追加
    - 元のカテゴリ列も保持(モデル側でone-hot化するのでどちらでもよい)
    """

    ging_map = {
        "Gingivitis": 1, "No Gingivitis": 0,
        "あり": 1, "なし": 0
    }
    need_map = {
        "Treatment Required": 1, "No Treatment Required": 0,
        "あり": 1, "なし": 0
    }
    oral_clean_map = {
        "Poor": 0, "Fair": 1, "Good": 2,
        "不良": 0, "普通": 1, "良": 2
    }

    if "gingivitis" in df.columns:
        df["gingivitis_bin"] = df["gingivitis"].map(ging_map)

    if "needTOBEtreated" in df.columns:
        df["need_bin"] = df["needTOBEtreated"].map(need_map)

    if "OralCleanStatus" in df.columns:
        df["OralCleanStatus_ord"] = df["OralCleanStatus"].map(oral_clean_map)

    return df


def coerce_tooth_codes_as_category_strings(df: pd.DataFrame, tooth_cols: list[str]) -> pd.DataFrame:
    """
    歯別コード(数値)をカテゴリとして扱うため、文字列に変換しNAも明示。
    例: 0.0 -> "0", 3.0 -> "3", 欠損 -> "NA"
    """
    for c in tooth_cols:
        # もともと 0.0 のようにfloatで入ってくることがあるので、まず数値化を試みる
        # 失敗しても文字列化で吸収
        s = pd.to_numeric(df[c], errors="coerce")
        # pandasの欠損許容整数型にすると "1", "2" 等にしやすい
        df[c] = s.astype("Int64").astype("string").fillna("NA")
    return df


def build_onehot_encoder():
    # sklearnのバージョン差 (sparse_output vs sparse) 吸収
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def prepare_xy(df: pd.DataFrame):
    # 解析対象(虐待4分類)のみ
    df = df.copy()
    df = df[df["abuse"].isin(ABUSE_LABELS)].reset_index(drop=True)

    # 追加の数値特徴
    df = add_basic_binary_ordinal_features(df)

    # 歯別列
    perm_tooth_cols, baby_tooth_cols = detect_tooth_columns(df)
    tooth_cols = perm_tooth_cols + baby_tooth_cols
    df = coerce_tooth_codes_as_category_strings(df, tooth_cols)

    # 目的変数
    y = df["abuse"].astype("string")

    # 特徴量候補（連続指標中心 + 口腔所見カテゴリ + 歯別カテゴリ）
    # まず「存在する列だけ使う」ために候補を列挙して intersection を取ります。
    numeric_candidates = [
        # 年齢
        "age",
        # う蝕指標など
        "Perm_D", "Perm_M", "Perm_F",
        "Baby_d", "Baby_m", "Baby_f",
        "Perm_DMFT", "Baby_DMFT",
        "Perm_DMFT_C0", "Baby_DMFT_C0",
        "DMFT_Index", "DMFT_C0",
        # 歯数/指数
        "Present_Teeth", "Present_Perm_Teeth", "Present_Baby_Teeth",
        "Healthy_Rate", "C0_Count", "Care_Index",
        "Trauma_Count", "RDT_Count", "UTN_Score",
        # 追加した数値化列
        "gingivitis_bin", "need_bin", "OralCleanStatus_ord"
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

    categorical_candidates = [
        "sex"
        "gingivitis", "needTOBEtreated",
        "occlusalRelationship", "habits", "OralCleanStatus"
    ]
    categorical_cols = [c for c in categorical_candidates if c in df.columns]

    # 自由記述は既定で除外（リーク/倫理/再現性の観点）
    text_like_cols = [c for c in ["instruction_detail", "memo"] if c in df.columns]

    # 最終特徴量: 数値 + カテゴリ (歯別列は除外)
    # Tooth-level columns (U17, L36, u55, etc.) are excluded from this analysis
    feature_cols = list(dict.fromkeys(numeric_cols + categorical_cols))
    # 除外
    feature_cols = [c for c in feature_cols if c not in text_like_cols]

    X = df[feature_cols].copy()

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "tooth_cols": tooth_cols,
        "text_excluded": text_like_cols
    }
    return df, X, y, meta


def build_rf_pipeline(numeric_cols: list[str], categorical_cols: list[str]):
    ohe = build_onehot_encoder()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = RandomForestClassifier(
        n_estimators=800,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced",  # 少数クラス(Sexual Abuseなど)の重み付け
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", clf),
    ])
    return pipe


def evaluate_rf_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    scoring = {
        "macro_f1": "f1_macro",
        "balanced_acc": "balanced_accuracy",
        "acc": "accuracy",
    }

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    summary = pd.DataFrame(scores).agg(["mean", "std"]).T
    summary.to_csv(outdir / "cv_metrics_summary.csv")
    print("\n[CV metrics]\n", summary)

    # 交差検証での予測→混同行列
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y, y_pred, labels=ABUSE_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ABUSE_LABELS)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(outdir / "cv_confusion_matrix.png", dpi=200)
    plt.close()

    print("\n[CV classification report]\n", classification_report(y, y_pred, labels=ABUSE_LABELS))


def fit_rf_and_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # ホールドアウトでPermutation Importance（列単位）を算出
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_test)

    print("\n[Holdout macro-F1]:", f1_score(y_test, y_hat, average="macro"))
    print("[Holdout balanced acc]:", balanced_accuracy_score(y_test, y_hat))

    # 列単位Permutation Importance（one-hot後ではなく“元の列”の重要度）
    pfi = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=20,
        random_state=0,
        scoring="f1_macro",
        n_jobs=-1
    )
    imp = pd.Series(pfi.importances_mean, index=X_test.columns).sort_values(ascending=False)
    imp.to_csv(outdir / "permutation_importance_by_column.csv")

    # 上位を図示
    topk = 30 if imp.shape[0] > 30 else imp.shape[0]
    plt.figure(figsize=(10, 6))
    imp.head(topk).sort_values().plot(kind="barh")
    plt.title("Permutation Importance (by original column) - top")
    plt.tight_layout()
    plt.savefig(outdir / "permutation_importance_top.png", dpi=200)
    plt.close()

    print("\n[Saved] permutation importance:", outdir / "permutation_importance_by_column.csv")

    # SHAP（任意：インストールされていない環境もあるためtry/except）
    try:
        import shap

        # 前処理後の特徴名を取得
        prep = pipe.named_steps["prep"]
        rf = pipe.named_steps["rf"]

        # feature names
        try:
            feat_names = prep.get_feature_names_out()
        except Exception:
            feat_names = None

        # SHAP計算用にサンプル（大きいと重い）
        X_shap = X_train.sample(n=min(400, len(X_train)), random_state=0)
        Xt = prep.transform(X_shap)
        # sparseの場合はdenseへ（SHAPが扱いやすい）
        if hasattr(Xt, "toarray"):
            Xt_dense = Xt.toarray()
        else:
            Xt_dense = np.asarray(Xt)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xt_dense)

        # 多クラス: shap_values は [n_classes] のlistになることが多い
        # ここでは各クラスごとにsummary plotを保存
        for i, cls in enumerate(rf.classes_):
            plt.figure()
            shap.summary_plot(shap_values[i], Xt_dense, feature_names=feat_names, show=False, max_display=25)
            plt.title(f"SHAP summary - class: {cls}")
            plt.tight_layout()
            plt.savefig(outdir / f"shap_summary_{cls}.png", dpi=200)
            plt.close()

        print("[Saved] SHAP plots to:", outdir)

    except Exception as e:
        print("\n[Note] SHAP was skipped (not installed or failed). Error:", repr(e))


def run_fastica(df_sub: pd.DataFrame, outdir: Path):
    """
    ICAは連続量中心が望ましいため、口腔内の集約指標(数値列)のみで実施します。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ica_candidates = [
        # 年齢はICAに入れるか悩ましいので、ここでは入れず、後段の回帰で調整する設計にしています。
        "Perm_D", "Perm_M", "Perm_F",
        "Baby_d", "Baby_m", "Baby_f",
        "Perm_DMFT", "Baby_DMFT",
        "Perm_DMFT_C0", "Baby_DMFT_C0",
        "DMFT_Index", "DMFT_C0",
        "Present_Teeth", "Present_Perm_Teeth", "Present_Baby_Teeth",
        "Healthy_Rate", "C0_Count", "Care_Index",
        "Trauma_Count", "RDT_Count", "UTN_Score",
        # 数値化した口腔所見
        "gingivitis_bin", "need_bin", "OralCleanStatus_ord",
    ]
    ica_cols = [c for c in ica_candidates if c in df_sub.columns]

    if len(ica_cols) < 4:
        print("\n[ICA] Not enough numeric columns for ICA. Found:", ica_cols)
        return

    X_ica = df_sub[ica_cols].copy()

    # 欠測補完→標準化
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_ica)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    # 成分数（まずは3で固定。あとで安定性評価を組み込む場合はここを拡張）
    n_components = 3
    
    # Check if we have enough valid samples for the requested components
    if X_std.shape[0] < n_components:
        print(f"\n[ICA] Not enough valid samples ({X_std.shape[0]}) for {n_components} components. Skipping ICA.")
        return
        
    # sklearnのバージョン差 (whiten引数)
    try:
        ica = FastICA(n_components=n_components, random_state=0, max_iter=3000, whiten="unit-variance")
    except TypeError:
        ica = FastICA(n_components=n_components, random_state=0, max_iter=3000, whiten=True)

    S = ica.fit_transform(X_std)  # (n_samples, n_components)
    W = ica.components_          # (n_components, n_features)

    # 負荷量（解釈用）
    loadings = pd.DataFrame(W, columns=ica_cols, index=[f"IC{i+1}" for i in range(n_components)])
    loadings.to_csv(outdir / "ica_components_loadings.csv")

    # スコアをdfへ
    df_ica = df_sub.copy()
    for i in range(n_components):
        df_ica[f"IC{i+1}"] = S[:, i]

    df_ica.to_csv(outdir / "ica_scores_added.csv", index=False)

    # 群別可視化（箱ひげ）
    for i in range(n_components):
        plt.figure(figsize=(8, 5))
        df_ica.boxplot(column=f"IC{i+1}", by="abuse", grid=False)
        plt.title(f"ICA score by abuse group: IC{i+1}")
        plt.suptitle("")
        plt.xlabel("abuse")
        plt.ylabel(f"IC{i+1} score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(outdir / f"ica_boxplot_IC{i+1}.png", dpi=200)
        plt.close()

    print("\n[ICA] Saved loadings:", outdir / "ica_components_loadings.csv")
    print("[ICA] Saved scores:", outdir / "ica_scores_added.csv")

    # 年齢・性別調整した群差（任意）
    if HAVE_STATSMODELS and ("age" in df_ica.columns) and ("sex" in df_ica.columns):
        # baselineは自動で1群になるので、結果解釈は注意（C(abuse)の係数）
        for i in range(n_components):
            formula = f"IC{i+1} ~ C(abuse) + age + C(sex)"
            model = smf.ols(formula, data=df_ica).fit(cov_type="HC3")
            with open(outdir / f"ica_regression_IC{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(model.summary().as_text())
        print("[ICA] Saved regression summaries (age/sex-adjusted).")
    else:
        print("[ICA] statsmodels not available or age/sex missing -> regression skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_OnlyAbuse_N1235.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--outdir",
        default="/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/result",
        help="Output directory"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv_flex(args.csv)

    # 必須列チェック
    if "abuse" not in df.columns:
        raise ValueError("Column 'abuse' not found. Please confirm which column is the abuse label.")

    df_sub, X, y, meta = prepare_xy(df)

    print("[Info] N (after filtering 4 abuse types):", len(df_sub))
    print("[Info] Features:", X.shape[1])
    print("[Info] Numeric cols:", len(meta["numeric_cols"]))
    print("[Info] Categorical cols:", len(meta["categorical_cols"]))
    print("[Info] Tooth cols:", len(meta["tooth_cols"]))
    if meta["text_excluded"]:
        print("[Info] Excluded text columns:", meta["text_excluded"])

    # RF用：カテゴリ列のみ（歯別列は除外）
    cat_cols_for_model = meta["categorical_cols"]
    pipe = build_rf_pipeline(meta["numeric_cols"], cat_cols_for_model)

    # 1) CV評価
    evaluate_rf_cv(pipe, X, y, outdir / "rf_cv")

    # 2) ホールドアウト + Permutation Importance + SHAP(任意)
    fit_rf_and_importance(pipe, X, y, outdir / "rf_explain")

    # 3) ICA（連続量中心）
    run_fastica(df_sub, outdir / "ica")


if __name__ == "__main__":
    main()

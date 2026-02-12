import numpy as np
import pandas as pd

def get_tooth_columns(df: pd.DataFrame):
    perm = [f'U{i}{j}' for i in [1,2] for j in range(1,8)] + [f'L{i}{j}' for i in [3,4] for j in range(1,8)]
    baby = [f'u{i}{j}' for i in [5,6] for j in range(1,6)] + [f'l{i}{j}' for i in [7,8] for j in range(1,6)]
    perm = [c for c in perm if c in df.columns]
    baby = [c for c in baby if c in df.columns]
    return perm, baby

def count_code(arr: np.ndarray, code: int) -> np.ndarray:
    # arr shape: (n, k)
    return (arr == code).sum(axis=1)

def derive_dmft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # age_group
    if "age_year" in df.columns and "age_group" not in df.columns:
        df["age_group"] = pd.cut(
            df["age_year"],
            bins=[0, 6, 12, 18],
            labels=["Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)"],
            right=True
        )

    perm_cols, baby_cols = get_tooth_columns(df)

    # 若没牙位列就直接返回
    if not perm_cols and not baby_cols:
        return df

    # 转 numpy（缺失值先填一个不可能的 code，比如 -999，避免 (NaN==2) 的比较问题）
    if perm_cols:
        perm = df[perm_cols].to_numpy()
    else:
        perm = np.empty((len(df), 0))

    if baby_cols:
        baby = df[baby_cols].to_numpy()
    else:
        baby = np.empty((len(df), 0))

    # 你当前 mapping：0 sound, 2 decayed, 3 filled, 4 missing
    df["Perm_D"] = count_code(perm, 2)
    df["Perm_M"] = count_code(perm, 4)
    df["Perm_F"] = count_code(perm, 3)
    df["Perm_Sound"] = count_code(perm, 0)
    df["Perm_DMFT"] = df["Perm_D"] + df["Perm_M"] + df["Perm_F"]

    df["Baby_d"] = count_code(baby, 2)
    df["Baby_m"] = count_code(baby, 4)
    df["Baby_f"] = count_code(baby, 3)
    df["Baby_sound"] = count_code(baby, 0)
    df["Baby_DMFT"] = df["Baby_d"] + df["Baby_m"] + df["Baby_f"]

    df["DMFT_Index"] = df["Perm_DMFT"] + df["Baby_DMFT"]

    # Care Index / Healthy Rate
    denom_dmft = df["DMFT_Index"].replace(0, np.nan)
    df["Care_Index"] = (df["Perm_F"] + df["Baby_f"]) / denom_dmft * 100

    total_teeth = df["Perm_Sound"] + df["Baby_sound"] + df["DMFT_Index"]
    df["Healthy_Rate"] = (df["Perm_Sound"] + df["Baby_sound"]) / total_teeth.replace(0, np.nan) * 100

    df["has_caries"] = (df["DMFT_Index"] > 0).astype(int)
    df["has_untreated_caries"] = ((df["Perm_D"] + df["Baby_d"]) > 0).astype(int)

    return df
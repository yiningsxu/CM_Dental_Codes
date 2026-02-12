import pandas as pd
import numpy as np

ABUSE_MAP = {
    1: "Physical Abuse", 2: "Neglect", 3: "Emotional Abuse", 4: "Sexual Abuse",
    5: "Delinquency", 6: "Parenting Difficulties", 7: "Others"
}
OCCLUSAL_MAP = {
    1: "Normal Occlusion", 2: "Crowding", 3: "Anterior Crossbite", 4: "Open Bite",
    5: "Maxillary Protrusion", 6: "Crossbite", 7: "Others"
}
NEED_TREATED_MAP = {1: "No Treatment Required", 2: "Treatment Required"}
EMERGENCY_MAP = {1: "Urgent Treatment Required"}
GINGIVITIS_MAP = {1: "No Gingivitis", 2: "Gingivitis"}
ORAL_CLEAN_MAP = {1: "Poor", 2: "Fair", 3: "Good"}
HABITS_MAP = {1: "None", 2: "Digit Sucking", 3: "Nail biting", 4: "Tongue Thrusting", 5: "Smoking", 6: "Others"}

ORDERS = {
    "abuse": ["Physical Abuse","Neglect","Emotional Abuse","Sexual Abuse","Delinquency","Parenting Difficulties","Others"],
    "occlusalRelationship": ["Normal Occlusion","Crowding","Anterior Crossbite","Open Bite","Maxillary Protrusion","Crossbite","Others"],
    "needTOBEtreated": ["No Treatment Required","Treatment Required"],
    "emergency": ["Urgent Treatment Required"],
    "gingivitis": ["No Gingivitis","Gingivitis"],
    "OralCleanStatus": ["Poor","Fair","Good"],
    "habits": ["None","Digit Sucking","Nail biting","Tongue Thrusting","Smoking","Others"],
}

def apply_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 日期
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # replace 映射：保留 NaN/未知值不动
    df["abuse"] = df["abuse"].replace(ABUSE_MAP)
    df["occlusalRelationship"] = df["occlusalRelationship"].replace(OCCLUSAL_MAP)
    df["needTOBEtreated"] = df["needTOBEtreated"].replace(NEED_TREATED_MAP)
    df["emergency"] = df["emergency"].replace(EMERGENCY_MAP)
    df["gingivitis"] = df["gingivitis"].replace(GINGIVITIS_MAP)
    df["OralCleanStatus"] = df["OralCleanStatus"].replace(ORAL_CLEAN_MAP)
    df["habits"] = df["habits"].replace(HABITS_MAP)

    # 分类顺序（ordered categorical）
    for col, order in ORDERS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    # 映射自检：是否仍有数字残留（除了 NaN）
    check_cols = ["abuse","occlusalRelationship","needTOBEtreated","emergency","gingivitis","OralCleanStatus","habits"]
    remain = {}
    for col in check_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique()
        nums = [x for x in vals if isinstance(x, (int, float, np.number)) and not isinstance(x, bool)]
        if nums:
            remain[col] = nums

    return df, remain
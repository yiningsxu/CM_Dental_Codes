from pathlib import Path
import pandas as pd

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def safe_save(df: pd.DataFrame | None, path: Path) -> None:
    if df is None:
        return
    if hasattr(df, "empty") and df.empty:
        return
    save_df(df, path)
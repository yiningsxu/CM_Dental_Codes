import pandas as pd
from src.context import PipelineContext
from src.analysis.mappings import apply_mappings

def run(ctx: PipelineContext) -> PipelineContext:
    log = ctx.logger
    cfg = ctx.cfg

    log.info(f"[step01] read: {cfg.raw_csv}")
    df = pd.read_csv(cfg.raw_csv)
    ctx.data_raw = df

    df, remain = apply_mappings(df)
    if remain:
        log.warning(f"[step01] mapping numeric remnants: {remain}")

    # 过滤：<= cutoff
    if "date" in df.columns:
        df = df[df["date"] <= cfg.data_cutoff].copy()

    # 去掉 abuse_num == 0
    if "abuse_num" in df.columns:
        df = df[df["abuse_num"] != 0].copy()

    # 单一虐待 + 限定类型（你当前逻辑）
    if cfg.require_single_abuse and "abuse_num" in df.columns:
        df = df[(df["abuse_num"] == 1) & (df["abuse"].isin(cfg.keep_abuse_types))].copy()

    ctx.data_clean = df
    ctx.data_analysis = df  # 后续在内存中继续加工
    log.info(f"[step01] done. shape={df.shape}")
    return ctx
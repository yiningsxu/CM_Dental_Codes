import pandas as pd
from src.context import PipelineContext
from src.analysis.io import safe_save, save_df
from src.analysis.tables import generate_summary_report  # 你原 Functions 里已有

def run(ctx: PipelineContext) -> PipelineContext:
    out_tables = ctx.cfg.output_dir / "tables"
    log = ctx.logger
    log.info("[step05] export tables + report")

    # 保存所有 tables
    for name, df in ctx.tables.items():
        safe_save(df, out_tables / f"{name}.csv")

    # consolidate tidy posthoc
    tidy_all = []
    for k in ["table3_tidy_posthoc", "table5_tidy_posthoc", "table5_5_tidy_posthoc"]:
        tidy_all.extend(ctx.misc.get(k, []))

    if tidy_all:
        save_df(pd.DataFrame(tidy_all), out_tables / "posthoc_pairwise_consolidated_summary.csv")

    # summary report（沿用你已有函数；建议将其改为返回字符串/df，再由这里统一写文件）
    if "table3_overall" in ctx.tables:
        generate_summary_report(ctx.data_analysis, ctx.tables["table3_overall"], str(ctx.cfg.output_dir) + "/", timestamp="")

    log.info("[step05] done")
    return ctx
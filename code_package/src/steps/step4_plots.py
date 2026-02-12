from src.context import PipelineContext
from src.analysis.plots import (
    create_forest_plot_vertical,
    create_visualizations,
    plot_boxplot_with_dunn,
    plot_boxplot_by_dentition_type,
)

def run(ctx: PipelineContext) -> PipelineContext:
    df = ctx.data_analysis
    out_fig = ctx.cfg.output_dir / "figures"
    log = ctx.logger
    log.info("[step04] build plots")

    # forest plot uses table4
    if "table4" in ctx.tables:
        create_forest_plot_vertical(ctx.tables["table4"], df, str(out_fig) + "/", timestamp="")  # 建议你在 plots.py 改成 Path + 自动命名

    create_visualizations(df, str(out_fig) + "/")

    for var, yname in [
        ("DMFT_Index","dmft&DMFT Index"),
        ("Baby_DMFT","Baby DMFT"),
        ("Baby_d","Baby d"),
        ("Healthy_Rate","Healthy Rate"),
        ("Care_Index","Care Index"),
    ]:
        plot_boxplot_with_dunn(df, var, group_col="abuse", yaxis_name=yname, output_dir=str(out_fig) + "/")

    plot_boxplot_by_dentition_type(df, output_dir=str(out_fig) + "/")

    log.info("[step04] done")
    return ctx
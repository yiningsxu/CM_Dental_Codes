from src.context import PipelineContext
from src.analysis.tables import (
    create_table1_demographics,
    create_table2_oral_health_descriptive,
    create_table3_statistical_comparisons,
    create_table4_multivariate_analysis,
    create_table5_dmft_by_lifestage_abuse,
    create_table5_5_caries_prevalence_treatment,
    create_table6_dmft_by_dentition_abuse,
    analyze_dmft_by_dentition_with_pairwise,
)

def run(ctx: PipelineContext) -> PipelineContext:
    df = ctx.data_analysis
    log = ctx.logger
    log.info("[step03] build tables")

    ctx.tables["table1"] = create_table1_demographics(df)

    t2_cont, t2_cat = create_table2_oral_health_descriptive(df)
    ctx.tables["table2_cont"] = t2_cont
    ctx.tables["table2_cat"] = t2_cat

    t3_overall, t3_posthoc, t3_pairwise, t3_tidy = create_table3_statistical_comparisons(df)
    ctx.tables["table3_overall"] = t3_overall
    ctx.tables["table3_posthoc"] = t3_posthoc
    ctx.tables["table3_pairwise"] = t3_pairwise
    ctx.misc["table3_tidy_posthoc"] = t3_tidy

    ctx.tables["table4"] = create_table4_multivariate_analysis(df)

    t5, t5_tidy = create_table5_dmft_by_lifestage_abuse(df)
    ctx.tables["table5"] = t5
    ctx.misc["table5_tidy_posthoc"] = t5_tidy

    t55, t55_tidy = create_table5_5_caries_prevalence_treatment(df)
    ctx.tables["table5_5"] = t55
    ctx.misc["table5_5_tidy_posthoc"] = t55_tidy

    ctx.tables["table6"] = create_table6_dmft_by_dentition_abuse(df)
    ctx.tables["table7"] = analyze_dmft_by_dentition_with_pairwise(df)

    log.info("[step03] done")
    return ctx
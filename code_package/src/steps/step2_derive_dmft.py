from src.context import PipelineContext
from src.analysis.teeth import derive_dmft

def run(ctx: PipelineContext) -> PipelineContext:
    log = ctx.logger
    log.info("[step02] derive dmft metrics")
    ctx.data_analysis = derive_dmft(ctx.data_analysis)
    log.info("[step02] done")
    return ctx
from pathlib import Path
from src.config import make_output_dir
from src.context import PipelineConfig, PipelineContext
from src.analysis.logging import setup_logger

from src.steps.step1_load_map_filter import run as s1
from src.steps.step2_derive_dmft import run as s2
from src.steps.step3_tables import run as s3
from src.steps.step4_plots import run as s4
from src.steps.step5_report import run as s5

def main():
    raw_csv = Path("/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/analysisData_20260211.csv")
    out = make_output_dir(Path("outputs"))

    logger = setup_logger(out / "logs", "oral_health")

    cfg = PipelineConfig(
        raw_csv=raw_csv,
        output_dir=out,
        data_cutoff="2024-03-31",
        require_single_abuse=True,
    )
    ctx = PipelineContext(cfg=cfg, logger=logger)

    for step in (s1, s2, s3, s4, s5):
        ctx = step(ctx)

    logger.info(f"DONE. outputs at: {out}")

if __name__ == "__main__":
    main()
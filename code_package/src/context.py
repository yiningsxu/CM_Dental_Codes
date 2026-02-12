from dataclasses import dataclass, field
from pathlib import Path
import logging
import pandas as pd

@dataclass
class PipelineConfig:
    raw_csv: Path
    output_dir: Path
    data_cutoff: str = "2024-03-31"     # 2024-03-31まで抽出
    require_single_abuse: bool = True   # abuse_num == 1
    keep_abuse_types: tuple = ("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")

@dataclass
class PipelineContext:
    cfg: PipelineConfig
    logger: logging.Logger

    # DataFrames in memory
    data_raw: pd.DataFrame | None = None
    data_clean: pd.DataFrame | None = None
    data_analysis: pd.DataFrame | None = None

    # Analysis products (tables, stats, etc.)
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    misc: dict = field(default_factory=dict)
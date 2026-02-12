import logging
from pathlib import Path

def setup_logger(log_dir: Path, name: str="pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
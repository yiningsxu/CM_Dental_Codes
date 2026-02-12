from pathlib import Path
from datetime import datetime

def make_output_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    return out
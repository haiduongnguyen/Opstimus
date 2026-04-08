from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_report(output_dir: str | Path, summary: dict[str, Any], predictions: pd.DataFrame) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    predictions.to_csv(output_path / "predictions.csv", index=False)

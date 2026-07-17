from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.directory / "metrics.jsonl"
        self.csv = self.directory / "summary.csv"
        self._csv_fields: list[str] | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(self.directory / "tensorboard")
        except ImportError:
            self.tensorboard = None
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(), "python": platform.python_version(),
        }
        (self.directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    def log(self, step: int, metrics: dict[str, float]) -> None:
        row = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        if self.tensorboard is not None:
            for key, value in metrics.items():
                self.tensorboard.add_scalar(key, float(value), step)
            self.tensorboard.flush()
        with self.jsonl.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        fields = list(row)
        if self._csv_fields is None:
            self._csv_fields = fields
            with self.csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
        if fields != self._csv_fields:
            return
        with self.csv.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self._csv_fields).writerow(row)

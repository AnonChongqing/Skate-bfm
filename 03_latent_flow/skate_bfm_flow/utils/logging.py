from __future__ import annotations

import csv
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.directory / "metrics.jsonl"
        self.csv = self.directory / "summary.csv"
        self._csv_fields: list[str] | None = None
        self._started = time.perf_counter()
        self._last_report_time = self._started
        self._last_report_step = 0
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
        elif new_fields := [field for field in fields if field not in self._csv_fields]:
            with self.csv.open(newline="") as handle:
                previous_rows = list(csv.DictReader(handle))
            self._csv_fields.extend(new_fields)
            with self.csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_fields)
                writer.writeheader()
                writer.writerows(previous_rows)
        with self.csv.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self._csv_fields).writerow(row)

    def report(self, phase: str, step: int, total: int, metrics: dict[str, float]) -> None:
        self.log(step, metrics)
        now = time.perf_counter()
        elapsed = now - self._started
        interval = max(now - self._last_report_time, 1e-6)
        rate = max(0, step - self._last_report_step) / interval
        remaining = max(0, total - step)
        eta = remaining / rate if rate > 0 else float("inf")
        progress = 100.0 * min(step, total) / max(1, total)
        eta_text = self._duration(eta) if eta != float("inf") else "unknown"
        print("=" * 88)
        print(
            f"{phase} | step {step:,}/{total:,} ({progress:5.1f}%) | "
            f"elapsed {self._duration(elapsed)} | rate {rate:,.1f}/s | ETA {eta_text}"
        )
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for name, value in sorted(metrics.items()):
            group, _, label = name.partition("/")
            groups[group if label else "train"].append((label or group, float(value)))
        for group, values in groups.items():
            body = "  ".join(f"{name}={self._format(value)}" for name, value in values)
            print(f"{group:>10}: {body}")
        print("=" * 88, flush=True)
        self._last_report_time = now
        self._last_report_step = step

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format(value: float) -> str:
        absolute = abs(value)
        if absolute and (absolute >= 10000 or absolute < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"


class MetricAccumulator:
    def __init__(self) -> None:
        self._sum: dict[str, object] = {}
        self._weight: dict[str, float] = defaultdict(float)

    def update(self, metrics: dict[str, object], weight: float = 1.0) -> None:
        for name, value in metrics.items():
            detached = value.detach() if hasattr(value, "detach") else value
            contribution = detached * weight
            self._sum[name] = self._sum[name] + contribution if name in self._sum else contribution
            self._weight[name] += weight

    def mean(self, reset: bool = False) -> dict[str, float]:
        output = {name: float(total / self._weight[name]) for name, total in self._sum.items() if self._weight[name]}
        if reset:
            self._sum.clear()
            self._weight.clear()
        return output

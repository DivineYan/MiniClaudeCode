"""Metrics collection for SWE-bench evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    instance_id: str
    tokens_used: int
    compression_count: int
    messages_dropped: int
    turns: int
    duration_seconds: float
    success: bool | None = None      # None until merged from swebench.com
    error: str | None = None
    patch_generated: str = ""


@dataclass
class EvalRun:
    label: str                       # e.g. "baseline" or "with_disk_offload"
    model: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    results: list[TaskResult] = field(default_factory=list)

    # ── aggregate metrics ────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def task_success_rate(self) -> float:
        scored = [r for r in self.results if r.success is not None]
        if not scored:
            return 0.0
        return sum(1 for r in scored if r.success) / len(scored)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)

    @property
    def avg_tokens_per_task(self) -> float:
        return self.total_tokens / self.n if self.n else 0.0

    @property
    def avg_compressions(self) -> float:
        return sum(r.compression_count for r in self.results) / self.n if self.n else 0.0

    @property
    def avg_messages_dropped(self) -> float:
        return sum(r.messages_dropped for r in self.results) / self.n if self.n else 0.0

    @property
    def avg_turns(self) -> float:
        return sum(r.turns for r in self.results) / self.n if self.n else 0.0

    @property
    def avg_duration(self) -> float:
        return sum(r.duration_seconds for r in self.results) / self.n if self.n else 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "label": self.label,
            "model": self.model,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvalRun":
        data = json.loads(path.read_text(encoding="utf-8"))
        run = cls(label=data["label"], model=data["model"], timestamp=data["timestamp"])
        run.results = [TaskResult(**r) for r in data["results"]]
        return run


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _delta_str(a: float, b: float, fmt: str = ".1f", pct: bool = False) -> str:
    """Format delta between baseline (a) and focus (b)."""
    d = b - a
    if pct and a != 0:
        return f"{d/a*100:+.1f}%"
    prefix = "+" if d > 0 else ""
    return f"{prefix}{d:{fmt}}"


def print_comparison(baseline: EvalRun, focus: EvalRun) -> None:
    rows = [
        ("Metric", baseline.label, focus.label, "Delta"),
        ("-" * 30, "-" * 15, "-" * 15, "-" * 12),
        (
            "Task Success",
            f"{sum(r.success for r in baseline.results)}/{baseline.n} ({baseline.task_success_rate*100:.0f}%)",
            f"{sum(r.success for r in focus.results)}/{focus.n} ({focus.task_success_rate*100:.0f}%)",
            _delta_str(baseline.task_success_rate, focus.task_success_rate, pct=True),
        ),
        (
            "Total Tokens",
            f"{baseline.total_tokens:,}",
            f"{focus.total_tokens:,}",
            _delta_str(baseline.total_tokens, focus.total_tokens, fmt=",", pct=True),
        ),
        (
            "Avg Tokens/Task",
            f"{baseline.avg_tokens_per_task:,.0f}",
            f"{focus.avg_tokens_per_task:,.0f}",
            _delta_str(baseline.avg_tokens_per_task, focus.avg_tokens_per_task, pct=True),
        ),
        (
            "Avg Compressions",
            f"{baseline.avg_compressions:.1f}",
            f"{focus.avg_compressions:.1f}",
            _delta_str(baseline.avg_compressions, focus.avg_compressions),
        ),
        (
            "Avg Msgs Dropped",
            f"{baseline.avg_messages_dropped:.1f}",
            f"{focus.avg_messages_dropped:.1f}",
            _delta_str(baseline.avg_messages_dropped, focus.avg_messages_dropped),
        ),
        (
            "Avg Turns",
            f"{baseline.avg_turns:.1f}",
            f"{focus.avg_turns:.1f}",
            _delta_str(baseline.avg_turns, focus.avg_turns),
        ),
    ]

    col_w = [32, 18, 18, 14]
    print()
    print(f"  A/B COMPARISON: {baseline.label!r} vs {focus.label!r}"
          f"  (model={focus.model}, N={focus.n})")
    print()
    for row in rows:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(row, col_w)))
    print()

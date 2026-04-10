"""Helpers for per-run logging and milestone progress updates."""
from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class _TeeStream:
    """Mirror writes to the active console stream and a run-local log file."""

    def __init__(self, console: TextIO, log_file: TextIO):
        self.console = console
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.console.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())


@contextlib.contextmanager
def tee_run_output(run_dir: str | Path):
    """Write stdout/stderr to run.log while preserving normal console output."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        stdout = _TeeStream(sys.stdout, handle)
        stderr = _TeeStream(sys.stderr, handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield log_path


@dataclass
class MilestoneLogger:
    """Emit 10% progress milestones once per run."""

    label: str
    total_steps: int
    next_pct: int = 10

    def start(self, detail: str = "") -> None:
        suffix = f" | {detail}" if detail else ""
        print(f"[{self.label}] Start | total_steps={self.total_steps}{suffix}")
        if self.total_steps <= 0:
            self.next_pct = 101

    def update(self, completed_steps: int, detail: str = "") -> None:
        if self.total_steps <= 0:
            return
        completed_steps = max(0, min(completed_steps, self.total_steps))
        pct = math.floor((completed_steps * 100) / self.total_steps)
        suffix = f" | {detail}" if detail else ""
        while self.next_pct <= pct and self.next_pct <= 100:
            print(
                f"[{self.label}] Progress {self.next_pct}% "
                f"({completed_steps}/{self.total_steps}){suffix}"
            )
            self.next_pct += 10

    def finish(self, detail: str = "") -> None:
        self.update(self.total_steps, detail=detail)
        suffix = f" | {detail}" if detail else ""
        print(f"[{self.label}] Complete | total_steps={self.total_steps}{suffix}")

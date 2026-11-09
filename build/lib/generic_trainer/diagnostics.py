from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .distributed import Runtime


class RankDiagnosticLogger:
    """Structured, flush-on-write diagnostics emitted independently by each rank."""

    def __init__(self, output_dir: Path, runtime: Runtime, enabled: bool) -> None:
        self.enabled, self.runtime = enabled, runtime
        self.path = output_dir / "diagnostics" / f"rank-{runtime.rank}.jsonl"
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "event": event,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "rank": self.runtime.rank,
            "local_rank": self.runtime.local_rank,
            "world_size": self.runtime.world_size,
            "device": str(self.runtime.device),
            **fields,
        }
        if self.runtime.device.type == "cuda":
            record.update(
                memory_allocated_mib=torch.cuda.memory_allocated(self.runtime.device) / 1024**2,
                memory_reserved_mib=torch.cuda.memory_reserved(self.runtime.device) / 1024**2,
            )
        line = json.dumps(record, sort_keys=True, default=str)
        print(f"DIAGNOSTIC {line}", flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

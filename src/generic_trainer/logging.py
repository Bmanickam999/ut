"""Rank-zero metric logging to JSONL and TensorBoard.

Example:
    ``MetricLogger(Path("artifacts/run"), True).emit({"loss": 0.5})``
    writes both a machine-readable event and TensorBoard scalar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricLogger:
    """Write metrics only from the process designated as main."""
    def __init__(self, output_dir: Path, enabled: bool) -> None:
        """Create output sinks when ``enabled`` is true."""
        self.enabled = enabled
        self.path = output_dir / "metrics.jsonl"
        self.writer = None
        if enabled:
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(output_dir / "tensorboard"))
            except ImportError:
                pass

    def emit(self, event: dict[str, Any]) -> None:
        """Persist and print one event, including numeric TensorBoard fields."""
        if not self.enabled:
            return
        print(json.dumps(event, sort_keys=True), flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if self.writer:
            step = event.get("global_step", event.get("epoch", 0))
            for key, value in event.items():
                if isinstance(value, (int, float)) and key not in {"global_step", "epoch"}:
                    self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer if one was created."""
        if self.writer:
            self.writer.close()

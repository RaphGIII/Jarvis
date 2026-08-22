from __future__ import annotations

import json
from pathlib import Path


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.log_dir / "metrics.jsonl"
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            self.writer = None

    @property
    def command(self) -> str:
        return f"tensorboard --logdir {self.log_dir}"

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
            self.writer.flush()
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tag": tag, "value": float(value), "step": int(step)}, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

import os
import time
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Thin TensorBoard wrapper that recursively flattens metric dictionaries."""

    def __init__(self, log_dir: str = "experiments/logs", run_name: str | None = None):
        """Create a SummaryWriter under a timestamped (or run-named) directory."""
        current_time = time.strftime("%Y%m%d-%H%M%S")
        folder = f"{run_name}_{current_time}" if run_name else current_time
        self.log_dir = os.path.join(log_dir, folder)
        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] Logging to: {self.log_dir}")

    def log_step(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Logs scalars, flattening nested dictionaries on the fly."""
        for key, value in metrics.items():
            tag = f"{prefix}{key}"
            if isinstance(value, dict):
                self.log_step(value, step, prefix=f"{tag}/")
                continue
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception:
                continue

    def close(self):
        """Close the underlying SummaryWriter."""
        self.writer.close()

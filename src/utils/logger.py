from torch.utils.tensorboard import SummaryWriter
import os
import time

class Logger:
    def __init__(self, log_dir="experiments/logs"):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] Logging to: {self.log_dir}")

    def log_step(self, metrics, step):
        """
        Log a dictionary of metrics.
        Example: {'loss/nav': 0.1, 'reward/total': 50}
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()
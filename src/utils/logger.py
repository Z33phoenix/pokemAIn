from torch.utils.tensorboard import SummaryWriter
import os
import time

class Logger:
    def __init__(self, log_dir="experiments/logs"):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] Logging to: {self.log_dir}")

    def log_step(self, metrics, step, prefix=""):
        """
        Logs metrics to TensorBoard. 
        Handles nested dictionaries by flattening them (e.g. 'loss' -> 'loss/actor').
        """
        for key, value in metrics.items():
            # Create the full tag name (e.g., "train/loss")
            tag = f"{prefix}{key}"
            
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                self.log_step(value, step, prefix=f"{tag}/")
            else:
                try:
                    # Log scalar values (floats, ints)
                    self.writer.add_scalar(tag, value, step)
                except Exception as e:
                    # Ignore non-numeric data (like strings in 'info')
                    pass
    def close(self):
        self.writer.close()
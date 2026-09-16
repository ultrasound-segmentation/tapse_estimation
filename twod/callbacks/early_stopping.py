import torch
import os


class EarlyStopping:
    """
    Implements early stopping to halt training when a monitored metric stops improving.

    Args:
        monitor (str): Name of the metric to monitor (e.g., 'val_loss', 'accuracy').
        mode (str): 'min' to stop when the metric stops decreasing (e.g., loss),
                    'max' to stop when the metric stops increasing (e.g., accuracy).
        patience (int): Number of epochs to wait before stopping if no improvement.
        delta (float): Minimum change in the monitored metric to be considered an improvement.
        path (str): Path to save the best model checkpoint.
    """

    def __init__(
        self, monitor="val_loss", mode="min", patience=5, delta=0, path="/checkpoints"
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # Define comparison function based on mode
        if mode == "min":
            self.compare = lambda new, best: new < best - delta  # Improvement if lower
            self.best_score = float("inf")  # Initialize for minimization
        elif mode == "max":
            self.compare = lambda new, best: new > best + delta  # Improvement if higher
            self.best_score = float("-inf")  # Initialize for maximization
        else:
            raise ValueError("mode must be 'min' or 'max'")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __call__(self, metric_value, model):
        """
        Checks if the monitored metric improves; if not, increases the counter and stops training if patience is exceeded.

        Args
            metric_value (float): Current value of the monitored metric.
            model (torch.nn.Module): Model to save if it improves.
        """

        if self.compare(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(self.path, "best_model.pth"),
            )  # Save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

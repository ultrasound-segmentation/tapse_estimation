import os
import torch


class ReduceLROnPlateau:
    """
    Reduces learning rate by a factor when a monitored metric stops improving for a certain number of epochs.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate needs adjustment.
        monitor (str): Name of the metric to monitor (e.g., 'val_loss', 'accuracy').
        mode (str): 'min' to reduce when the metric stops decreasing (e.g., loss),
                    'max' to reduce when the metric stops increasing (e.g., accuracy).
        patience (int): Number of epochs to wait before reducing LR if no improvement.
        factor (float): Factor by which the learning rate will be reduced.
        min_lr (float): Minimum learning rate limit.
        delta (float): Minimum change in the monitored metric to be considered an improvement.
    """

    def __init__(
        self,
        optimizer,
        monitor="val_loss",
        mode="min",
        patience=5,
        factor=0.1,
        min_lr=1e-6,
        delta=0,
        initial_lr=1e-4,
    ):
        self.optimizer = optimizer
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.currentlr = initial_lr

        if mode == "min":
            self.compare = lambda new, best: new < best - delta  # Improvement if lower
            self.best_score = float("inf")  # Initialize for minimization
        elif mode == "max":
            self.compare = lambda new, best: new > best + delta  # Improvement if higher
            self.best_score = float("-inf")  # Initialize for maximization
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, metric_value):
        """
        Checks if the monitored metric improves; if not, increases the counter and reduces LR if patience is exceeded.

        Args:
            metric_value (float): Current value of the monitored metric.
        """
        if self.compare(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0

    def _reduce_lr(self):
        """Reduces the learning rate by the specified factor, ensuring it doesn't go below min_lr."""
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group["lr"] * self.factor, self.min_lr)
            param_group["lr"] = new_lr
            self.currentlr = new_lr
        print(f"Learning rate reduced to {new_lr:.6f}")

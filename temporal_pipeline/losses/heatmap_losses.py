import torch
import torch.nn as nn
import torch.nn.functional as F

""" losses that are used for the heatmap regression type of training. I reality, what worked best was 
a simple MSEloss with radius of the gaussian that decreased at each epoch"""


def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable coordinate extraction via softmax-weighted expectation.

    Args:
        heatmaps:    (B, N_points, N_frames, H, W)
        temperature: lower = sharper / closer to hard argmax

    Returns:
        coords: (B, N_points, N_frames, 2)  — (x, y) in pixel space
    """
    B, N, F, H, W = heatmaps.shape

    flat = heatmaps.view(B, N, F, -1)  # (B, N, F, H*W)
    weights = torch.softmax(flat / temperature, dim=-1)  # (B, N, F, H*W)

    ys = torch.arange(H, device=heatmaps.device, dtype=torch.float32)
    xs = torch.arange(W, device=heatmaps.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

    grid_x = grid_x.reshape(-1)  # (H*W,)
    grid_y = grid_y.reshape(-1)  # (H*W,)

    pred_x = (weights * grid_x).sum(-1)  # (B, N, F)
    pred_y = (weights * grid_y).sum(-1)  # (B, N, F)

    return torch.stack([pred_x, pred_y], dim=-1)  # (B, N, F, 2)


class HeatmapMotionLoss(nn.Module):
    """
    Combined heatmap MSE + frame-to-frame motion loss.

    Args:
        alpha:       weight on MSE;  (1 - alpha) goes to motion loss
        temperature: soft-argmax temperature (tune alongside learning rate)
    """

    def __init__(self, alpha: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse = nn.MSELoss(reduction="mean")

    def _motion_loss(
        self,
        pred_coords: torch.Tensor,  # (B, N_points, N_frames, 2)
        target_coords: torch.Tensor,  # (B, N_points, N_frames, 2)
    ) -> torch.Tensor:
        # Frame-to-frame displacements — (B, N_points, N_frames-1, 2)
        pred_delta = pred_coords[:, :, 1:, :] - pred_coords[:, :, :-1, :]
        target_delta = target_coords[:, :, 1:, :] - target_coords[:, :, :-1, :]

        # L2 error on displacements — (B, N_points, N_frames-1)
        motion_error = torch.linalg.norm(pred_delta - target_delta, ord=2, dim=-1)
        return motion_error.mean()

    def forward(
        self,
        pred_heatmaps: torch.Tensor,  # (B, N_points, N_frames, H, W)
        target_heatmaps: torch.Tensor,  # (B, N_points, N_frames, H, W)
    ) -> tuple[torch.Tensor, dict]:

        loss_mse = self.mse(pred_heatmaps, target_heatmaps)

        pred_coords = soft_argmax_2d(pred_heatmaps, self.temperature)
        target_coords = soft_argmax_2d(target_heatmaps, self.temperature)
        loss_motion = self._motion_loss(pred_coords, target_coords)

        total = self.alpha * loss_mse + (1.0 - self.alpha) * loss_motion

        return total, {
            "mse": self.alpha * loss_mse.item(),
            "motion": (1.0 - self.alpha) * loss_motion.item(),
            "total": total.item(),
        }


class HeatmapBCETopKLoss(nn.Module):
    """
    Binary Cross-Entropy loss with Top-K selection for heatmap regression.

    Computes BCE loss between predicted and target heatmaps, then selects
    the top K% of the highest loss values to focus on challenging voxels.
    This mitigates foreground-background imbalance in sparse landmark heatmaps.

    Args:
        top_k_percent (float): Fraction of highest loss values to keep (default 0.2 for 20%).
        reduction (str): 'mean' or 'sum' for final aggregation (default 'mean').

    Input shapes:
        pred, target: [B, num_keypoints, T, H, W] — heatmaps in [0,1].
    """

    def __init__(self, top_k_percent=0.2, reduction="mean"):
        super().__init__()
        self.top_k_percent = top_k_percent
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred, target):
        """
        Args:
            pred   (Tensor): predicted heatmaps [B, num_keypoints, T, H, W]
            target (Tensor): target heatmaps [B, num_keypoints, T, H, W]

        Returns:
            loss (Tensor): scalar loss value
        """
        # Compute element-wise BCE loss
        loss = self.bce(pred, target)  # [B, num_keypoints, T, H, W]

        # Flatten all losses into a 1D tensor
        loss_flat = loss.flatten()  # [total_elements]

        # Number of elements to select: top K%
        num_elements = loss_flat.numel()
        k = max(1, int(num_elements * self.top_k_percent))  # at least 1

        # Select the top K highest losses
        top_k_losses = torch.topk(loss_flat, k, largest=True).values

        # Aggregate the selected losses
        if self.reduction == "mean":
            return top_k_losses.mean()
        elif self.reduction == "sum":
            return top_k_losses.sum()
        else:
            return top_k_losses  # unreduced

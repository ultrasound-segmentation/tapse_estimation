import torch
import torch.nn as nn

# script with the loss functions used during the training
# OrderedDistanceLoss was the one that got the best results, the other was briefly
# tried but got worse results


class OrderedDistanceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        """
        Custom loss function for ordered keypoints.
        Computes the Euclidean distance between corresponding keypoints in 'pred' and 'target'.
        Handles inputs of shape (B, N, 3, 2) or (B, 3, 2).
        """
        super(OrderedDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        :param pred: Tensor of predicted keypoints. Shape: (B, N, 3, 2) or (B, 3, 2)
        :param target: Tensor of ground-truth keypoints. Same shape as pred.
        :return: Loss value (scalar or per sample)
        """
        # Ensure pred and target are 3D: (batch_size * N, 3, 2)
        if pred.dim() == 4:
            B, N, P, C = pred.shape  # Usually (1, 64, 3, 2)
            pred = pred.view(-1, P, C)  # Shape: (B*N, 3, 2)
            target = target.view(-1, P, C)
        elif pred.dim() == 3:
            # Already in shape (B, 3, 2), nothing to change
            pass
        else:
            raise ValueError(f"Unexpected input shape: {pred.shape}")

        # Compute Euclidean distances
        distances = torch.norm(pred - target, dim=2)  # Shape: (B*N, 3)

        # Reduce
        if self.reduction == "mean":
            return distances.mean()
        elif self.reduction == "sum":
            return distances.sum()
        else:  # 'none'
            return distances


class GaussianKeypointLoss(nn.Module):
    def __init__(self, sigma=1.0, reduction="mean"):
        """
        Loss basata su una gaussiana centrata nel punto reale.
        Loss = 1 - exp(-||pred - target||^2 / (2 * sigma^2))
        """
        super(GaussianKeypointLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.dim() == 4:
            B, N, P, C = pred.shape
            pred = pred.view(-1, P, C)
            target = target.view(-1, P, C)
        elif pred.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {pred.shape}")

        # Calcolo della distanza quadratica (||x - y||^2)
        sq_dist = torch.sum((pred - target) ** 2, dim=2)  # Shape: (B*N, 3)

        # Calcolo del valore della gaussiana
        gaussian = torch.exp(
            -sq_dist / (2 * self.sigma**2)
        )  # max 1, decresce con la distanza

        loss = 1.0 - gaussian  # più lontano => loss più alta

        # Riduzione
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_classes = 2  # Assuming each sample has exactly 2 points

    pred = torch.rand(batch_size, num_classes, 2)  # Random predicted points
    target = torch.rand(batch_size, num_classes, 2)  # Random ground-truth points

    loss_fn = UnorderedMSELoss(reduction="mean")  # Create loss criterion
    loss = loss_fn(pred, target)  # Compute loss
    print("Loss:", loss.item())

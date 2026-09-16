import torch

# function that takes the feature maps that are the output of the Unet and swinunetr,
# and calculates the coordinates of center of mass of the top 1-thresh activated pixels


def center_of_mass(tensor: torch.Tensor, thresh=0.9, device="cpu", normalize=False):
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor must be 2D")

    tensor = tensor.float().to(
        device
    )  # Ensure float type and move to the correct device

    # Shift values to be non-negative without breaking computation graph
    min_val = tensor.min()
    clipped_tensor = tensor - min_val

    # Apply thresholding in a differentiable way
    threshold = clipped_tensor.max() * thresh
    clipped_tensor = torch.where(
        clipped_tensor >= threshold,
        clipped_tensor - threshold,
        torch.tensor(0.0, device=device, dtype=tensor.dtype),
    )

    total_mass = clipped_tensor.sum()

    # If the total mass is zero, return the geometric center
    if total_mass == 0:
        return torch.tensor(
            [tensor.shape[1] / 2, tensor.shape[0] / 2],
            device=device,
            dtype=tensor.dtype,
            requires_grad=True,
        )

    height, width = tensor.shape

    # Create coordinate tensors on the same device
    x_coords = torch.arange(width, device=device, dtype=tensor.dtype).repeat(height, 1)
    y_coords = (
        torch.arange(height, device=device, dtype=tensor.dtype).repeat(width, 1).t()
    )

    # Compute weighted sum of coordinates
    x_center = (clipped_tensor * x_coords).sum() / total_mass
    y_center = (clipped_tensor * y_coords).sum() / total_mass

    if normalize:
        x_center = x_center / tensor.shape[1]
        y_center = y_center / tensor.shape[0]

    # Return a differentiable tensor
    return torch.stack([x_center, y_center])

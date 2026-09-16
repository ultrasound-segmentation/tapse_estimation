import torch

# script with the functions used to retrieve the coordinates of the landmarks from the
# predicted feature maps (only when coordinates are not directly regressed)


def center_of_mass_3d(tensor: torch.Tensor, thresh=0.9, device="cpu", normalize=False):
    """
    Computes the 2D center of mass for each landmark and frame in a batch of tensors.
    The function takes the (1-thresh) most activated pixels in each 2d feature map, and
    calculates its center of mass.

    Args:
        tensor:    Input tensor of shape [N, 3, 32, H, W] (batched)
                   or [3, 32, H, W] / [1, 3, 32, H, W] (single sample).
                   Axis meaning: N=batch, 3=landmarks, 32=frames, H=height, W=width.
        thresh:    Fraction of the per-channel max used as a soft threshold (default 0.9).
                   Pixels below this fraction are zeroed out before computing the CoM.
        device:    Torch device string, e.g. 'cpu' or 'cuda' (default 'cpu').
        normalize: If True, divides x_center by W and y_center by H so coordinates
                   are in [0, 1] (default False).

    Returns:
        coords:    Float tensor of shape [N, C, B, 2], where
                   N = batch size, C = frames (32), B = landmarks (3), 2 = (x, y).
    """

    # ------------------------------------------------------------------ #
    # 1.  Normalise rank: make sure we always work with a 5-D tensor     #
    #     [N, B, C, H, W]  (N=batch, B=landmarks, C=frames)              #
    # ------------------------------------------------------------------ #
    if tensor.ndim == 4:
        # Single sample supplied without a batch dimension → add one
        tensor = tensor.unsqueeze(0)  # [1, B, C, H, W]

    if tensor.ndim != 5:
        raise ValueError(
            f"Expected a 4-D or 5-D input tensor, got shape {tensor.shape}"
        )

    # Move to the requested device and cast to float32 for all arithmetic
    tensor = tensor.float().to(device)

    N, B, C, H, W = tensor.shape  # unpack all five axes for clarity
    # N = batch size          (e.g. 4)
    # B = number of landmarks (e.g. 3)
    # C = number of frames    (e.g. 32)
    # H, W = spatial dims     (e.g. 256 × 256)

    # ------------------------------------------------------------------ #
    # 2.  Per-channel min-subtraction  →  shift every channel so that     #
    #     its minimum value becomes 0.                                     #
    # ------------------------------------------------------------------ #
    # Collapse H and W into a single axis to find the spatial minimum.
    # Result shape: [N, B, C, 1, 1]  (kept for broadcasting)
    min_val = (
        tensor.view(N, B, C, -1)  # [N, B, C, H*W]
        .min(dim=-1)[0]  # [N, B, C]
        .unsqueeze(-1)  # [N, B, C, 1]
        .unsqueeze(-1)
    )  # [N, B, C, 1, 1]

    clipped = tensor - min_val  # shift min → 0; shape unchanged

    # ------------------------------------------------------------------ #
    # 3.  Soft threshold: zero out everything below `thresh × max`.       #
    #     This focuses the centre-of-mass on the brightest region only.   #
    # ------------------------------------------------------------------ #
    # Per-channel maximum after the min-shift; shape: [N, B, C, 1, 1]
    max_val = clipped.view(N, B, C, -1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

    threshold = max_val * thresh  # scalar threshold per channel

    # Subtract the threshold and clamp negatives to zero
    clipped = torch.where(
        clipped >= threshold,
        clipped - threshold,  # keep (and shift) values above threshold
        torch.zeros_like(clipped),  # zero out values below threshold
    )

    # ------------------------------------------------------------------ #
    # 4.  Total mass per channel  →  used as the normalising denominator. #
    # ------------------------------------------------------------------ #
    # Sum over the two spatial axes; shape: [N, B, C, 1, 1]
    total_mass = clipped.sum(dim=(3, 4), keepdim=True)

    # "Dead" channels: channels where all pixels were zeroed out.
    # We'll fall back to the image centre for those.
    dead_mask = total_mass < 1e-6  # bool tensor [N, B, C, 1, 1]

    # ------------------------------------------------------------------ #
    # 5.  Coordinate grids  →  one value per pixel indicating its (x, y). #
    # ------------------------------------------------------------------ #
    # y_coords[..., row, :] == row index;  shape → [1, 1, 1, H, 1]
    y_coords = (
        torch.arange(H, device=device, dtype=tensor.dtype)
        .view(1, 1, 1, H, 1)
        .expand(N, B, C, H, W)
    )

    # x_coords[..., :, col] == col index;  shape → [1, 1, 1, 1, W]
    x_coords = (
        torch.arange(W, device=device, dtype=tensor.dtype)
        .view(1, 1, 1, 1, W)
        .expand(N, B, C, H, W)
    )

    # ------------------------------------------------------------------ #
    # 6.  Weighted average position  =  centre of mass.                   #
    # ------------------------------------------------------------------ #
    # Replace zero-mass denominators with 1 to avoid NaN (result is       #
    # overwritten by the dead_mask fallback below anyway).
    safe_mass = torch.where(dead_mask, torch.ones_like(total_mass), total_mass)

    # Weighted sum over spatial axes; divide by total mass → [N, B, C, 1, 1]
    x_center = (clipped * x_coords).sum(dim=(3, 4), keepdim=True) / safe_mass
    y_center = (clipped * y_coords).sum(dim=(3, 4), keepdim=True) / safe_mass

    # ------------------------------------------------------------------ #
    # 7.  Dead-channel fallback: use the image centre (W/2, H/2).         #
    # ------------------------------------------------------------------ #
    x_center = torch.where(dead_mask, torch.full_like(x_center, W / 2), x_center)
    y_center = torch.where(dead_mask, torch.full_like(y_center, H / 2), y_center)

    # Remove the two trailing size-1 axes → [N, B, C]
    x_center = x_center.squeeze(-1).squeeze(-1)
    y_center = y_center.squeeze(-1).squeeze(-1)

    # ------------------------------------------------------------------ #
    # 8.  Optional normalisation to [0, 1].                               #
    # ------------------------------------------------------------------ #
    if normalize:
        x_center = x_center / W
        y_center = y_center / H

    # ------------------------------------------------------------------ #
    # 9.  Pack (x, y) pairs and reshape to match expected output layout.  #
    # ------------------------------------------------------------------ #
    # Stack along a new last axis → [N, B, C, 2]
    coords = torch.stack([x_center, y_center], dim=-1)

    # Permute to [N, C, B, 2]  (frames first, then landmarks, then xy)
    # This matches the original single-sample convention [1, 32, 3, 2]
    # but now generalised to a full batch.
    coords = coords.permute(0, 2, 1, 3)  # [N, C, B, 2]

    return coords


def center_of_mass_3d_global_threshold(
    tensor: torch.Tensor,
    global_thresh: float = 0.1,
    device: str = "cpu",
    normalize: bool = False,
):
    """
    Computes the 2D center of mass for each landmark and frame in a batch of tensors.
    Uses a GLOBAL threshold: global_thresh × (global max across all channels). All values
    below this threshold are zeroed out.

    This is useful when you want a single threshold applied across all feature maps, rather
    than adapting to each channel's own maximum. If after thresholding a feature map becomes
    entirely zero, no prediction is made for that channel (returns NaN). This way the model
    is allowed not to make a prediction if it's not sure.

    Args:
        tensor:    Input tensor of shape [N, 3, 32, H, W] (batched)
                   or [3, 32, H, W] / [1, 3, 32, H, W] (single sample).
                   Axis meaning: N=batch, 3=landmarks, 32=frames, H=height, W=width.
        global_thresh: Fraction of the global max used as threshold (default 0.1).
                        All pixels with value < (global_thresh × global_max) are zeroed out.
                        Range: [0, 1].
        device:    Torch device string, e.g. 'cpu' or 'cuda' (default 'cpu').
        normalize: If True, divides x_center by W and y_center by H so coordinates
                   are in [0, 1] (default False).

    Returns:
        coords:    Float tensor of shape [N, C, B, 2], where
                   N = batch size, C = frames (32), B = landmarks (3), 2 = (x, y).
                   For channels where all pixels were below the threshold (empty after
                   thresholding), the coordinates are set to float('nan').
    """

    # ------------------------------------------------------------------ #
    # 1.  Normalise rank: make sure we always work with a 5-D tensor     #
    #     [N, B, C, H, W]  (N=batch, B=landmarks, C=frames)              #
    # ------------------------------------------------------------------ #
    if tensor.ndim == 4:
        # Single sample supplied without a batch dimension → add one
        tensor = tensor.unsqueeze(0)  # [1, B, C, H, W]

    if tensor.ndim != 5:
        raise ValueError(
            f"Expected a 4-D or 5-D input tensor, got shape {tensor.shape}"
        )

    # Move to the requested device and cast to float32 for all arithmetic
    tensor = tensor.float().to(device)

    N, B, C, H, W = tensor.shape  # unpack all five axes for clarity
    # N = batch size          (e.g. 4)
    # B = number of landmarks (e.g. 3)
    # C = number of frames    (e.g. 32)
    # H, W = spatial dims     (e.g. 256 × 256)

    # ------------------------------------------------------------------ #
    # 2.  GLOBAL threshold: compute global max, then threshold.           #
    #     Unlike center_of_mass_3d which uses per-channel max × thresh,   #
    #     here we use global max × global_thresh for ALL channels.        #
    # ------------------------------------------------------------------ #
    # Compute global max across all dimensions (N, B, C, H, W)
    global_max = tensor.max()  # scalar

    # Apply global threshold: keep values >= global_thresh × global_max
    threshold = global_max * global_thresh
    clipped = torch.where(tensor >= threshold, tensor, torch.zeros_like(tensor))

    # ------------------------------------------------------------------ #
    # 3.  Total mass per channel  →  used as the normalising denominator. #
    # ------------------------------------------------------------------ #
    # Sum over the two spatial axes; shape: [N, B, C, 1, 1]
    total_mass = clipped.sum(dim=(3, 4), keepdim=True)

    # "Dead" channels: channels where all pixels were zeroed out.
    # We'll return NaN for those (no valid prediction).
    dead_mask = total_mass < 1e-6  # bool tensor [N, B, C, 1, 1]

    # ------------------------------------------------------------------ #
    # 4.  Coordinate grids  →  one value per pixel indicating its (x, y). #
    # ------------------------------------------------------------------ #
    # y_coords[..., row, :] == row index;  shape → [1, 1, 1, H, 1]
    y_coords = (
        torch.arange(H, device=device, dtype=tensor.dtype)
        .view(1, 1, 1, H, 1)
        .expand(N, B, C, H, W)
    )

    # x_coords[..., :, col] == col index;  shape → [1, 1, 1, 1, W]
    x_coords = (
        torch.arange(W, device=device, dtype=tensor.dtype)
        .view(1, 1, 1, 1, W)
        .expand(N, B, C, H, W)
    )

    # ------------------------------------------------------------------ #
    # 5.  Weighted average position  =  centre of mass.                   #
    # ------------------------------------------------------------------ #
    # Replace zero-mass denominators with 1 to avoid NaN (result is       #
    # overwritten by the dead_mask fallback below anyway).
    safe_mass = torch.where(dead_mask, torch.ones_like(total_mass), total_mass)

    # Weighted sum over spatial axes; divide by total mass → [N, B, C, 1, 1]
    x_center = (clipped * x_coords).sum(dim=(3, 4), keepdim=True) / safe_mass
    y_center = (clipped * y_coords).sum(dim=(3, 4), keepdim=True) / safe_mass

    # ------------------------------------------------------------------ #
    # 6.  Dead-channel fallback: use NaN to indicate no prediction.       #
    #     Unlike center_of_mass_3d which uses image centre, here we signal #
    #     that there is NO prediction for this channel.                    #
    # ------------------------------------------------------------------ #
    x_center = torch.where(dead_mask, torch.full_like(x_center, float("nan")), x_center)
    y_center = torch.where(dead_mask, torch.full_like(y_center, float("nan")), y_center)

    # Remove the two trailing size-1 axes → [N, B, C]
    x_center = x_center.squeeze(-1).squeeze(-1)
    y_center = y_center.squeeze(-1).squeeze(-1)

    # ------------------------------------------------------------------ #
    # 7.  Optional normalisation to [0, 1].                               #
    # ------------------------------------------------------------------ #
    if normalize:
        x_center = x_center / W
        y_center = y_center / H

    # ------------------------------------------------------------------ #
    # 8.  Pack (x, y) pairs and reshape to match expected output layout.  #
    # ------------------------------------------------------------------ #
    # Stack along a new last axis → [N, B, C, 2]
    coords = torch.stack([x_center, y_center], dim=-1)

    # Permute to [N, C, B, 2]  (frames first, then landmarks, then xy)
    # This matches the original single-sample convention [1, 32, 3, 2]
    # but now generalised to a full batch.
    coords = coords.permute(0, 2, 1, 3)  # [N, C, B, 2]

    return coords


def argmax_3d(
    tensor: torch.Tensor,
    device="cpu",
    normalize=False,
    thresh_method=False,
    threshold=None,
):
    """
    Computes the argmax coordinates for each landmark and frame in a batch of heatmaps.
    Takes the channel-wise maximum position.
    Args:
        tensor:        Input tensor of shape [N, 3, 32, H, W] (batched)
                       or [3, 32, H, W] / [1, 3, 32, H, W] (single sample).
                       Axis meaning: N=batch, 3=landmarks, 32=frames, H=height, W=width.
        device:        Torch device string, e.g. 'cpu' or 'cuda' (default 'cpu').
        normalize:     If True, divides x by W and y by H so coordinates are in [0, 1].
        threshold:     Raw heatmap value used as the confidence cutoff.
    Returns:
        coords:        Float tensor of shape [N, C, B, 2], where
                       N = batch size, C = frames (32), B = landmarks (3), 2 = (x, y).
        max_values:     Tensor of shape (N, C, B), that contains the maximum of each point heatmap. Udeful to determine the ocnfidence of the model.
    """
    # Normalize rank
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected 4-D or 5-D tensor, got {tensor.shape}")
    tensor = tensor.float().to(device)
    N, B, C, H, W = tensor.shape

    # Flatten spatial dims for argmax
    flat_tensor = tensor.view(N, B, C, -1)  # [N, B, C, H*W]

    # Argmax indices and raw max values (single pass)
    max_values, argmax_indices = flat_tensor.max(dim=-1)  # [N, B, C]

    # Convert flat indices to (x, y)
    y_coords = argmax_indices // W
    x_coords = argmax_indices % W

    # Stack to [N, B, C, 2], then permute to [N, C, B, 2]
    coords = torch.stack([x_coords.float(), y_coords.float()], dim=-1)

    if normalize:
        coords[..., 0] /= W
        coords[..., 1] /= H

    coords = coords.permute(0, 2, 1, 3)

    return coords, max_values

import torch
import torchvision.transforms.functional as TF


def random_h_flip(image, keypoints, p=0.5):
    """Randomly flips the image horizontally and adjusts keypoints."""
    if torch.rand(1).item() < p:
        image = TF.hflip(image)  # Optimized horizontal flip
        keypoints[:, 0] = (
            image.shape[-1] - keypoints[:, 0]
        )  # Correct x-coordinate adjustment
    return image, keypoints


def adjust_brightness_contrast(
    image_sequence, brightness_range=(0.8, 1.2), contrast_range=(0.7, 1.3)
):
    """
    Adjusts brightness and contrast for a batch of grayscale frames on GPU.
    Input shape: [T, H, W] (torch.Tensor on GPU)
    Output shape: [T, H, W]
    """
    T, H, W = image_sequence.shape
    device = image_sequence.device

    # Add channel dim -> [T, 1, H, W]
    image_sequence = image_sequence.unsqueeze(1)

    # Random brightness and contrast factors on GPU
    brightness_factor = torch.empty(1).uniform_(*brightness_range).item()
    contrast_factor = torch.empty(1).uniform_(*contrast_range).item()

    image_sequence = TF.adjust_brightness(image_sequence, brightness_factor)
    image_sequence = TF.adjust_contrast(image_sequence, contrast_factor)

    image_sequence = image_sequence.squeeze(1)

    return torch.clamp(image_sequence, 0.0, 1.0)


def random_rotate(image_sequence, keypoints, degrees=(-30, 30), p=0.5):
    """
    Randomly rotates a volume [T, H, W] and adjusts keypoints accordingly.
    Rotation is applied consistently across all frames.

    - image_sequence: Tensor [T, H, W], grayscale frames on GPU.
    - keypoints: Tensor [N, 2] or compatible shape (x, y) coords.
    - degrees: Rotation angle range.
    - p: Probability of applying rotation.

    Returns:
        rotated_sequence: [T, H, W]
        rotated_keypoints: same shape as input
    """
    if torch.rand(1, device=image_sequence.device) < p:
        angle = torch.empty(1, device=image_sequence.device).uniform_(*degrees).item()
        T, H, W = image_sequence.shape
        center = torch.tensor([W / 2, H / 2], device=image_sequence.device)

        # Rotate each frame with the same angle
        rotated_frames = [
            TF.rotate(image_sequence[i].unsqueeze(0), angle).squeeze(0)
            for i in range(T)
        ]
        rotated_sequence = torch.stack(rotated_frames)

        # Convert angle to radians
        angle_rad = torch.deg2rad(torch.tensor(angle, device=image_sequence.device))

        # Rotation matrix (clockwise for image, so reverse angle for points)
        rot_mat = torch.tensor(
            [
                [torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                [torch.sin(-angle_rad), torch.cos(-angle_rad)],
            ],
            device=image_sequence.device,
        )

        rotated_keypoints = keypoints.view(-1, 2)
        rotated_keypoints = (rotated_keypoints - center) @ rot_mat.T + center

        return rotated_sequence, rotated_keypoints.view_as(keypoints)
    else:
        return image_sequence, keypoints


def random_crop(image, keypoints, crop_size=220, p=0.5):
    """Randomly crops the image, resizes it back, and adjusts keypoints accordingly."""
    if torch.rand(1).item() < p:
        h, w = image.shape[-2], image.shape[-1]
        crop_h = torch.randint(crop_size, image.shape[-2] + 1, (1,)).item()
        crop_w = torch.randint(crop_size, image.shape[-1] + 1, (1,)).item()

        if h <= crop_h or w <= crop_w:
            return image, keypoints  # Skip cropping if image is too small

        # Randomly select the top-left corner of the crop
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()

        # Crop the image
        image = TF.crop(image, top, left, crop_h, crop_w)

        # Resize image back to original size
        image = TF.resize(image, (h, w))

        # Adjust keypoints
        cropped_keypoints = keypoints.view(
            -1, 2
        )  # Ensure keypoints are in (N, 2) format
        cropped_keypoints = cropped_keypoints - torch.tensor(
            [left, top], device=cropped_keypoints.device
        )  # Shift keypoints

        # Scale keypoints to match resized image
        scale_x = w / crop_w
        scale_y = h / crop_h
        cropped_keypoints = cropped_keypoints * torch.tensor(
            [scale_x, scale_y], device=cropped_keypoints.device
        )

        # Clamp to image bounds
        cropped_keypoints[:, 0] = cropped_keypoints[:, 0].clamp(0, 255)
        cropped_keypoints[:, 1] = cropped_keypoints[:, 1].clamp(0, 255)

        # Keep only keypoints that remain within the resized area
        # mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) & \
        #        (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
        # keypoints = keypoints[mask]

        return image, cropped_keypoints.view_as(keypoints)
    else:
        return image, keypoints


def add_gaussian_noise(image, std=0.02, p=0.5):
    """Adds Gaussian noise with given standard deviation, efficiently."""
    if torch.rand(1, device=image.device).item() < p:
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)
    else:
        return image


def time_flip(image, keypoints, p=0.5):
    """Randomly flips the time axis of the image and adjusts keypoints."""
    if torch.rand(1).item() < p:
        image = image.flip(0)  # Flip along the time axis (first dimension)
        keypoints = keypoints.flip(0)  # Flip keypoints accordingly
    return image, keypoints


def vertically_align(image, keypoints):
    keyp_0 = keypoints[0]
    septum = keyp_0[1]
    fw = keyp_0[0]
    apex = keyp_0[2]

    middle = (septum + fw) / 2

    versor = (middle - apex) / torch.norm(middle - apex)
    versor = torch.tensor(
        [versor[1], versor[0]], device=versor.device
    )  # Flip y-component

    # print("versor", versor)

    # Desired direction is up: [0, -1] (negative y)
    vertical = torch.tensor([-1.0, 0.0], device=versor.device)

    # Compute angle between versor and vertical
    dot = torch.dot(versor, vertical).clamp(-1.0, 1.0)
    angle_rad = torch.acos(dot)

    # Determine rotation direction using 2D cross product (sign)
    cross = versor[0] * vertical[1] - versor[1] * vertical[0]
    if cross < 0:
        angle_rad = -angle_rad

    angle_deg = torch.rad2deg(angle_rad).item()

    T, H, W = image.shape
    center = torch.tensor([W / 2, H / 2], device=image.device)

    # Rotate each frame
    aligned_frames = [
        TF.rotate(image[i].unsqueeze(0), angle_deg).squeeze(0) for i in range(T)
    ]
    aligned_sequence = torch.stack(aligned_frames)

    # Inverse rotation for keypoints
    rot_mat = torch.tensor(
        [
            [torch.cos(-angle_rad), -torch.sin(-angle_rad)],
            [torch.sin(-angle_rad), torch.cos(-angle_rad)],
        ],
        device=image.device,
    )

    aligned_keypoints = keypoints.view(-1, 2)
    aligned_keypoints = (aligned_keypoints - center) @ rot_mat.T + center

    aligned_keypoints[:, 0] = aligned_keypoints[:, 0].clamp(0, 255)
    aligned_keypoints[:, 1] = aligned_keypoints[:, 1].clamp(0, 255)

    return aligned_sequence, aligned_keypoints.view_as(keypoints)


def coarse_dropout_3d(
    image_sequence: torch.Tensor,
    p: float = 0.9,
    num_patches: int = 10,
    size_range: tuple[int, int, int] = (6, 30, 30),
    noise_std: float = 0.08,
):
    """
    Randomly masks small 3D blocks in a clip and fills them with noisy values.
    image_sequence: [T, H, W]
    """
    if torch.rand(1, device=image_sequence.device).item() >= p:
        return image_sequence

    T, H, W = image_sequence.shape
    out = image_sequence.clone()

    mean = out.mean()
    std = out.std(unbiased=False).clamp(min=1e-4)

    for _ in range(num_patches):
        dt = torch.randint(
            size_range[0],
            min(size_range[0] + 1, T) if size_range[0] <= T else T + 1,
            (1,),
        ).item()
        dh = torch.randint(
            size_range[1],
            min(size_range[1] + 1, H) if size_range[1] <= H else H + 1,
            (1,),
        ).item()
        dw = torch.randint(
            size_range[2],
            min(size_range[2] + 1, W) if size_range[2] <= W else W + 1,
            (1,),
        ).item()

        t0 = torch.randint(0, max(1, T - dt + 1), (1,)).item()
        y0 = torch.randint(0, max(1, H - dh + 1), (1,)).item()
        x0 = torch.randint(0, max(1, W - dw + 1), (1,)).item()

        noise = torch.randn((dt, dh, dw), device=out.device) * noise_std * std + mean
        out[t0 : t0 + dt, y0 : y0 + dh, x0 : x0 + dw] = torch.clamp(noise, 0.0, 1.0)

    return out


def elastic_deformation(image_sequence, keypoints, alpha=20, sigma=5, p=0.5):
    """
    Applies elastic deformation to simulate non-rigid tissue stretching.
    Uses a random displacement field smoothed with Gaussian blur.

    - image_sequence: [T, H, W] tensor
    - keypoints: [T, 2] or similar, but deformation is spatial only
    - alpha: scaling factor for displacement magnitude
    - sigma: standard deviation for Gaussian smoothing
    - p: probability of applying deformation

    Returns deformed image and adjusted keypoints.
    """
    if torch.rand(1, device=image_sequence.device).item() >= p:
        return image_sequence, keypoints

    T, H, W = image_sequence.shape
    device = image_sequence.device

    # Create displacement field for spatial dimensions
    # Random displacements
    dx = torch.randn(H, W, device=device) * alpha
    dy = torch.randn(H, W, device=device) * alpha

    # Smooth with Gaussian (simple approximation using convolution)
    from torch.nn.functional import conv2d

    kernel_size = int(4 * sigma + 0.5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] @ g[None, :]
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    dx = dx.unsqueeze(0).unsqueeze(0)
    dy = dy.unsqueeze(0).unsqueeze(0)
    dx = conv2d(dx, kernel, padding=kernel_size // 2).squeeze()
    dy = conv2d(dy, kernel, padding=kernel_size // 2).squeeze()

    # Create grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    x_coords = x_coords + dx
    y_coords = y_coords + dy

    # Normalize to [-1, 1] for grid_sample
    x_coords = 2 * x_coords / (W - 1) - 1
    y_coords = 2 * y_coords / (H - 1) - 1

    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Apply to each frame
    deformed_frames = []
    for t in range(T):
        frame = image_sequence[t].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        deformed = torch.nn.functional.grid_sample(
            frame, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        deformed_frames.append(deformed.squeeze(0).squeeze(0))

    deformed_sequence = torch.stack(deformed_frames)

    # Adjust keypoints: apply the same spatial deformation
    # keypoints: [T, num_keypoints, 2]
    kp_deformed = keypoints.clone()
    for t in range(T):
        for k in range(keypoints.shape[1]):
            x, y = keypoints[t, k]
            # Interpolate displacement at keypoint location
            x_idx = int(torch.clamp(x, 0, W - 1))
            y_idx = int(torch.clamp(y, 0, H - 1))
            dx_kp = dx[y_idx, x_idx]
            dy_kp = dy[y_idx, x_idx]
            kp_deformed[t, k] = torch.tensor([x + dx_kp, y + dy_kp], device=device)

    # Clamp keypoints to image bounds
    kp_deformed[..., 0] = kp_deformed[..., 0].clamp(0, W - 1)
    kp_deformed[..., 1] = kp_deformed[..., 1].clamp(0, H - 1)

    return deformed_sequence, kp_deformed


def apply_transform(image: torch.Tensor, keypoints: torch.Tensor, version: str = "0"):
    """Apply transformations to an image and its keypoints."""
    if version == "0":
        pass

    elif version == "1":
        image, keypoints = random_h_flip(image, keypoints, p=0.5)
        image = adjust_brightness_contrast(image, contrast_range=(0.8, 1.2))
        image = add_gaussian_noise(image)

    elif version == "2":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.5, 1.4), contrast_range=(0.5, 1.4)
        )
        image = add_gaussian_noise(image)

    elif version == "3":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.02)
    elif version == "4":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)
        )
        image = add_gaussian_noise(image, std=0.08)
        image, keypoints = random_rotate(image, keypoints, p=1)
    elif version == "5":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, p=0.6)

    elif version == "6":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, p=0.5)
        image, keypoints = random_crop(image, keypoints, crop_size=220, p=0.6)
    elif version == "7":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)
        )
        image = add_gaussian_noise(image, std=0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=0.6)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.6)
    elif version == "8":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)
        )
        image = add_gaussian_noise(image, std=0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=0.6)
        # image, keypoints = random_crop(image, keypoints, crop_size=230, p=.6)
        image, keypoints = time_flip(image, keypoints, p=0.6)
        # TODO: there's a problem: when cropping, the keypoints go out of bounds sometimes
    elif version == "9":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)
        )
        image = add_gaussian_noise(image, std=0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=0.6)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.6)
        image, keypoints = time_flip(image, keypoints, p=0.6)
    elif version == "10":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
    elif version == "11":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
        image = coarse_dropout_3d(image)
    elif version == "12":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
        image = coarse_dropout_3d(image)
        image, keypoints = elastic_deformation(
            image, keypoints, alpha=20, sigma=6, p=0.7
        )
    elif version == "13":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
        image = coarse_dropout_3d(
            image,
            p=0.3,
        )
        image, keypoints = elastic_deformation(
            image, keypoints, alpha=20, sigma=6, p=0.7
        )
    elif version == "14":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
        # image = coarse_dropout_3d(image, p=0.3, )
        image, keypoints = elastic_deformation(
            image, keypoints, alpha=20, sigma=6, p=0.7
        )
    elif version == "15":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image, std=0.06)
        image, keypoints = random_rotate(image, keypoints, degrees=(-20, 20), p=0.9)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.9)
        image, keypoints = time_flip(image, keypoints, p=0.9)
        image = coarse_dropout_3d(image, p=0.3, num_patches=10, size_range=(3, 20, 20))
        image, keypoints = elastic_deformation(
            image, keypoints, alpha=5, sigma=10, p=0.7
        )

    else:
        raise ValueError(f"Unsupported version: {version}")

    return image, keypoints


def apply_transform_val(
    image: torch.Tensor, keypoints: torch.Tensor, version: str = "0"
):
    image, keypoints = vertically_align(image, keypoints)
    return image, keypoints

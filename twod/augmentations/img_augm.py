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
    image, brightness_range=(0.8, 1.2), contrast_range=(0.7, 1.3)
):
    """Adjusts brightness and contrast with given ranges."""
    brightness_factor = torch.empty(1).uniform_(*brightness_range).item()
    contrast_factor = torch.empty(1).uniform_(*contrast_range).item()

    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)

    return image


def random_rotate(image, keypoints, degrees=(-30, 30), p=0.5):
    """Randomly rotates the image and adjusts keypoints."""
    if torch.rand(1).item() < p:
        angle = torch.empty(1).uniform_(*degrees).item()  # Random angle in range
        h, w = image.shape[-2], image.shape[-1]
        center = torch.tensor([w / 2, h / 2])

        # Rotate image
        image = TF.rotate(image, angle)

        # Convert angle to radians
        angle_rad = torch.deg2rad(torch.tensor(angle))

        # Define rotation matrix
        rotation_matrix = torch.tensor(
            [
                [torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                [torch.sin(-angle_rad), torch.cos(-angle_rad)],
            ]
        )

        # Ensure keypoints shape is (N, 2) before transformation
        keypoints = keypoints.view(-1, 2)  # Make sure it's (num_keypoints, 2)

        # Apply rotation: move to origin -> rotate -> move back
        keypoints = (keypoints - center) @ rotation_matrix.T + center

    return image, keypoints.view_as(keypoints)  # Restore original shape


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
        image = TF.resize(image, (h, w), antialias=True)

        # Adjust keypoints
        keypoints = keypoints.view(-1, 2)  # Ensure keypoints are in (N, 2) format
        keypoints = keypoints - torch.tensor([left, top])  # Shift keypoints

        # Scale keypoints to match resized image
        scale_x = w / crop_w
        scale_y = h / crop_h
        keypoints = keypoints * torch.tensor([scale_x, scale_y])

        # Keep only keypoints that remain within the resized area
        # mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) & \
        #        (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
        # keypoints = keypoints[mask]

    return image, keypoints.view(-1, 2)


def add_gaussian_noise(image, std=0.02):
    """Adds Gaussian noise with given standard deviation."""
    noise = torch.normal(0, std, size=image.shape, device=image.device)
    return torch.clamp(image + noise, 0.0, 1.0)


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
            image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)
        )
        image = add_gaussian_noise(image)

    elif version == "3":
        image = adjust_brightness_contrast(
            image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)
        )
        image = add_gaussian_noise(image, std=0.08)
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
    elif version == "7":  # this one was used for AutoRV training
        image = adjust_brightness_contrast(
            image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)
        )
        image = add_gaussian_noise(image, std=0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=0.6)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=0.6)
    else:
        raise ValueError(f"Unsupported version: {version}")

    return image, keypoints

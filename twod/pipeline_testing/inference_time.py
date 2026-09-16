import os
import time
import h5py
import torch
import numpy as np
import argparse
from models.models import (
    Unet,
    ResNet18Regression,
    ResNet34Regression,
    ResNet50Regression,
    ResNeXt50Regression,
    SwinUNETR,
)
from dataloader.preprocessing import (
    preprocess_images,
    apply_lut,
    resize_or_crop_image_np_nokeypoints,
)
from postprocessing.coordinates_calculation_from_masks import center_of_mass
import torch

# code to calculate the inference time for each of the models of the study
# remember to chose the subset of patients that you want to test this on (lines 79-80)


def center_of_mass_three(
    masks: torch.Tensor, thresh=0.9, device="cpu", normalize=False
):
    """
    Compute center of mass for three masks at once.
    Needed for U-Net and Swinunetr to extract the
    coordinates from the predicted feature maps
    Args:
        masks: Tensor of shape (3, H, W)
        thresh: float threshold for mass clipping
        device: device to run on
        normalize: whether to normalize coordinates to [0,1]
    Returns:
        Tensor of shape (3, 2) with (x, y) coordinates for each mask
    """
    if masks.shape[0] != 3 or len(masks.shape) != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    masks = masks.float().to(device)  # ensure float and correct device
    C, H, W = masks.shape

    # Shift to non-negative
    min_val = masks.amin(dim=(1, 2), keepdim=True)  # (3,1,1)
    clipped = masks - min_val

    # Thresholding
    threshold = clipped.amax(dim=(1, 2), keepdim=True) * thresh
    clipped = torch.where(
        clipped >= threshold, clipped - threshold, torch.zeros_like(clipped)
    )

    # Total mass
    total_mass = clipped.sum(dim=(1, 2))  # (3,)

    # If total mass is zero, replace with geometric center
    zero_mask = total_mass == 0
    if zero_mask.any():
        clipped[zero_mask] = 1.0  # avoid division by zero
        total_mass[zero_mask] = H * W

    # Coordinate grids
    y_coords = torch.arange(H, device=device, dtype=masks.dtype).view(
        1, H, 1
    )  # (1,H,1)
    x_coords = torch.arange(W, device=device, dtype=masks.dtype).view(
        1, 1, W
    )  # (1,1,W)

    # Weighted sum
    x_center = (clipped * x_coords).sum(dim=(1, 2)) / total_mass  # (3,)
    y_center = (clipped * y_coords).sum(dim=(1, 2)) / total_mass  # (3,)

    # Replace zero-mass coordinates with geometric center
    x_center[zero_mask] = W / 2
    y_center[zero_mask] = H / 2

    if normalize:
        x_center = x_center / W
        y_center = y_center / H

    # Return (3, 2) tensor
    return torch.stack([x_center, y_center], dim=1)


# --- ARGUMENTS ---
parser = argparse.ArgumentParser(
    description="Measure inference time of the model on test dataset."
)
parser.add_argument(
    "--model",
    type=str,
    default="unet",
    choices=["unet", "resnet18", "resnet34", "resnet50", "resnext50", "swinunetr"],
    help="Model type to use.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG ---
# PATIENTS = ['100', '111', '140', '149', '160', '170', '190', '198', '199', '920'] # test set
PATIENTS = ["106", "135", "141", "184", "990"]  # validation set
if args.model == "unet":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights\best_unet\best_model.pth"
    # --- LOAD MODEL ---
    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    THRESHOLD = 0.875
elif args.model == "resnet18":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights/best_resnet18/best_model.pth"
    model = ResNet18Regression().to(device)
elif args.model == "resnet34":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights/best_resnet34/best_model.pth"
    model = ResNet34Regression().to(device)
elif args.model == "resnet50":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights/best_resnet50/best_model.pth"
    model = ResNet50Regression().to(device)
elif args.model == "resnext50":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights/best_resnext50/best_model.pth"
    model = ResNeXt50Regression().to(device)
elif args.model == "swinunetr":
    MODEL_CKPT = r"c:\Users\vcxr10\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights/best_swin_unetr/best_model.pth"
    model = SwinUNETR(start_filts=12).to(device)
    THRESHOLD = 0.9

TEST_PATH = r"c:\Users\vcxr10\Desktop\final_reviewed_dataset"

model.load_state_dict(torch.load(MODEL_CKPT, map_location=device)["model_state_dict"])
model.eval()

# --- GPU warm-up ---
dummy = torch.zeros((1, 1, 1, 256, 256)).to(device)
for _ in range(5):
    _ = model(dummy)
torch.cuda.synchronize()

# --- TIMER INIT ---
total_frames = 0
total_time = 0.0

# --- PROCESS EACH PATIENT ---
for folder in os.listdir(TEST_PATH):
    if folder not in PATIENTS:
        continue

    folder_path = os.path.join(TEST_PATH, folder)

    for file in os.listdir(folder_path):
        if "interpolated" not in file:
            continue

        file_path = os.path.join(folder_path, file)
        with h5py.File(file_path, "r") as f:
            images = f["tissue"]["data"][()]  # (H, W, N)

        # preprocess same as main code
        images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
        images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
        images = images / images.max()

        N = len(images)
        total_frames += N

        # --- Measure inference time ---
        start = time.time()
        with torch.no_grad():
            for i in range(N):
                img = images[i]
                img = preprocess_images(
                    np.expand_dims(img, axis=0), model_type="U-Net", device=device
                )
                output = model(img.float().unsqueeze(0).to(device))
                # simple inference: no saving, just compute coordinates
                if args.model in ["unet", "swinunetr"]:
                    _ = center_of_mass_three(output.squeeze(), thresh=THRESHOLD)
        end = time.time()

        elapsed = end - start
        total_time += elapsed
        print(
            f"{folder}/{file}: {N} frames in {elapsed:.3f} s -> {N / elapsed:.2f} fps"
        )

# --- FINAL STATS ---
fps = total_frames / total_time if total_time > 0 else 0
print(
    f"\n=== Overall inference speed: {fps:.2f} frames/sec across {total_frames} frames ==="
)

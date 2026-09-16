import numpy as np
import os
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dataloader.preprocessing import (
    preprocess_images,
    apply_lut,
    resize_or_crop_image_np_nokeypoints,
)
from utils.plot import save_image_ill
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.kalman_filter import KalmanFilter
from models.models import Unet


def process_h5_file_single_illustrative(
    file_path,
    model,
    device,
    save_path,
    folder,
    threshold=0.875,
    apply_filter=False,
):
    """
    Processes a .h5 file frame by frame (no batching) and optionally applies a Kalman filter.
    """

    with h5py.File(file_path, "r") as f:
        images = f["tissue"]["data"][()]  # (H, W, N)
        if "tissue" in f and "times" in f["tissue"]:
            times = f["tissue"]["times"][()]
            dt = times[1] - times[0]
        else:
            dt = 1  # fallback if timing info not available

    images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
    images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
    images = images / images.max()

    N = len(images)
    coordinates_array = np.zeros((N, 3, 2))

    file_name = os.path.basename(file_path).replace(".h5", "")
    save_path = os.path.join(save_path, folder, file_name)
    os.makedirs(save_path, exist_ok=True)

    # --- Predict coordinates frame by frame ---
    for i in range(N):
        img = images[i]
        img = preprocess_images(
            np.expand_dims(img, axis=0), model_type="U-Net", device=device
        )
        output = model(img.float().unsqueeze(0).to(device))

        for c in range(3):
            coordinates_array[i, c] = center_of_mass(
                output[0, c].detach(), thresh=threshold
            )

    # --- Apply Kalman filter if requested ---
    if apply_filter:
        filtered_array = coordinates_array.copy()

        # initialize 3 independent Kalman filters
        kfs = [
            KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1)
            for _ in range(3)
        ]
        for j in range(3):
            kfs[j].x = np.matrix(
                [coordinates_array[0, j, 0], coordinates_array[0, j, 1], 0, 0]
            ).T

        for i, coords in enumerate(coordinates_array):
            for j in range(3):
                kfs[j].predict()
                filtered_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1

        coordinates_array = filtered_array

    # --- Save visualization using filtered (or raw) coordinates ---
    for i in range(N):
        img = images[i]
        pred_points = [tuple(coordinates_array[i, k]) for k in range(3)]
        ann_points = None
        bold_flag = False

        save_image_ill(img, points=pred_points, save_folder=save_path, bold=bold_flag)

    return coordinates_array, None, None


def main():
    parser = argparse.ArgumentParser(description="Landmark prediction from .h5 files")
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        default=0.875,
        help="Threshold for center_of_mass",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Flag to save images with predicted keypoints",
    )
    parser.add_argument(
        "--no_sudden_movements",
        action="store_true",
        help="Flag to avoid sudden movements in keypoints",
    )
    parser.add_argument(
        "--threshold_sudden",
        type=int,
        default=20,
        help="Threshold for sudden movement detection",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply Kalman filter to smooth coordinates",
    )
    args = parser.parse_args()

    model_checkpoint = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights\best_unet\best_model.pth"
    test_path = r"C:\Users\User\Desktop\final_reviewed_dataset"
    save_path = r"C:\Users\User\Desktop\illustrative_video"

    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    model.load_state_dict(
        torch.load(model_checkpoint, map_location=device)["model_state_dict"]
    )
    model.eval()

    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        if folder in [
            "100",
            "111",
            "140",
            "149",
            "160",
            "170",
            "190",
            "198",
            "199",
            "920",
        ]:
            for file in os.listdir(folder_path):
                if "interpolated" in file:
                    file_path = os.path.join(folder_path, file)

                    process_h5_file_single_illustrative(
                        file_path=file_path,
                        model=model,
                        device=device,
                        save_path=save_path,
                        folder=folder,
                        threshold=args.threshold,
                        apply_filter=args.filter,
                    )


if __name__ == "__main__":
    main()

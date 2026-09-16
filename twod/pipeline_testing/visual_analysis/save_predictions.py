import numpy as np
import os
import torch
import h5py
import argparse

from dataloader.preprocessing import (
    preprocess_images,
    apply_lut,
    resize_or_crop_image_np_nokeypoints,
)
from utils.plot import save_image, save_image_ann_pred
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet

# code to visualize predictions of the model on converted hdf5 data


def process_h5_file_single(
    file_path,
    model,
    device,
    save_model_path,
    folder,
    threshold=0.875,
):
    """
    Processes a .h5 file frame by frame (no batching).
    """

    # extract and preprocess images
    with h5py.File(file_path, "r") as f:
        images = f["tissue"]["data"][()]  # (H, W, N)
    images = apply_lut(images.transpose(2, 1, 0)[:, :, ::-1])
    images = resize_or_crop_image_np_nokeypoints(images)
    images = images / images.max()

    # get number of images
    N = len(images)

    # create folder to save images in
    file_name = os.path.basename(file_path).replace(".h5", "")
    save_path = os.path.join(save_model_path, folder, file_name)
    os.makedirs(save_path, exist_ok=True)

    # for each image
    for i in range(N):
        img = images[i]
        img = preprocess_images(
            np.expand_dims(img, axis=0), model_type="U-Net", device=device
        )
        output = model(img.float().unsqueeze(0).to(device))

        pred_points = []
        for c in range(3):
            pred_points.append(
                tuple(center_of_mass(output[0, c].detach(), thresh=threshold))
            )

        # Save images with predictions on it
        save_image_ann_pred(
            img[0, 0].cpu().numpy(),
            pred_points=pred_points,
            save_folder=save_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Landmark prediction from .h5 files")
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        default=0.875,
        help="Threshold for center_of_mass",
    )
    args = parser.parse_args()

    model_checkpoint = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights\best_unet\best_model.pth"
    test_path = r"C:\Users\User\Desktop\final_reviewed_dataset_test_set"
    save_model_path = r"C:\Users\User\Desktop\RV_predictions"

    os.makedirs(save_model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    model.load_state_dict(
        torch.load(model_checkpoint, map_location=device)["model_state_dict"]
    )
    model.eval()

    # --- Loop over patients (folders) ---
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            print(file_path)

            # function to save images
            process_h5_file_single(
                file_path=file_path,
                model=model,
                device=device,
                save_model_path=save_model_path,
                folder=folder,
                threshold=args.threshold,
            )


if __name__ == "__main__":
    main()

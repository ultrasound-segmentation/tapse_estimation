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
from utils.plot import save_image, save_image_ann_pred
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet

# code to test model on annotated data, and create bocplots with the error distribution
# the boxplots can be divided by patient or by landmark


def compute_keypoint_distance_stats(predictions, ground_truths):
    if predictions.shape != ground_truths.shape:
        raise ValueError(
            "Shape mismatch: predictions and ground_truths must have the same shape."
        )

    distances = np.linalg.norm(predictions - ground_truths, axis=-1)  # (N, 3)
    distances = distances / 2

    stats = {
        "global": {
            "mean": np.mean(distances),
            "max": np.max(distances),
            "min": np.min(distances),
            "std": np.std(distances),
        },
        "per_point": [],
    }

    for i in range(3):
        point_dists = distances[:, i]
        stats["per_point"].append(
            {
                "mean": np.mean(point_dists),
                "max": np.max(point_dists),
                "min": np.min(point_dists),
                "std": np.std(point_dists),
            }
        )

    return stats, distances


def process_h5_file_single(
    file_path,
    model,
    device,
    save_model_path,
    folder,
    save_images=True,
    save_annotations=False,
    prediction_stats=False,
    threshold=0.875,
    no_sudden_movements=False,
    threshold_sudden=20,
):
    """
    Processes a .h5 file frame by frame (no batching).
    """

    with h5py.File(file_path, "r") as f:
        print(f"\n--- Contents of {file_path} ---")

        # def print_h5_structure(name, obj):
        #     if isinstance(obj, h5py.Dataset):
        #         print(f"Dataset: {name} | shape={obj.shape} | dtype={obj.dtype}")
        #     elif isinstance(obj, h5py.Group):
        #         print(f"Group: {name}")

        # f.visititems(print_h5_structure)
        # print("--- End of file structure ---\n")

        images = f["frames"][()]  # (H, W, N)
        if prediction_stats or save_annotations:
            annotations = f["annotations"][()]
            print(annotations.shape)
            print(images.shape)

    images = images.transpose(2, 0, 1)

    # I'm using the frames that were saved after the annotation, not the original ones. (I saved them since not all the frames were annotated in each acquisition)
    #  this is not needed since preprocessing was already performed to aid annotation
    # images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
    # images = images / images.max()

    N = len(images)
    coordinates_array = np.zeros((N, 3, 2))

    if save_images:
        file_name = os.path.basename(file_path).replace(".h5", "")
        save_path = os.path.join(save_model_path, folder, file_name)
        os.makedirs(save_path, exist_ok=True)

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

        if save_images:
            pred_points = [tuple(coordinates_array[i, k]) for k in range(3)]
            ann_points = None
            bold_flag = False

            if (
                save_annotations and "annotations" in locals()
            ):  # flag to save the annotations and the predictions in the same image
                ann_points = [tuple(annotations[i, k]) for k in range(3)]

            if prediction_stats:
                dists = np.linalg.norm(coordinates_array[i] - annotations[i], axis=-1)
                if np.any(dists > 40):
                    bold_flag = True
                    print(
                        f"[WARNING] Error > 40 in image {i} of file {file_path}, with an error of {dists.max()/2} mm"
                    )

            # Save both pred (red) and ann (green)
            save_image_ann_pred(
                img[0, 0].cpu().numpy(),
                ann_points=ann_points,
                pred_points=pred_points,
                save_folder=save_path,
                bold=True,
            )

    if no_sudden_movements:
        for j in range(3):  # for each keypoint
            for i in range(1, N - 1):
                if (
                    np.linalg.norm(
                        coordinates_array[i, j] - coordinates_array[i - 1, j]
                    )
                    > threshold_sudden
                    and np.linalg.norm(
                        coordinates_array[i, j] - coordinates_array[i + 1, j]
                    )
                    > threshold_sudden
                ):
                    coordinates_array[i, j] = (
                        coordinates_array[i - 1, j] + coordinates_array[i + 1, j]
                    ) / 2

    if prediction_stats:
        print(coordinates_array.shape)
        stats, distances = compute_keypoint_distance_stats(
            coordinates_array, annotations
        )
        return coordinates_array, stats, distances

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
        "--save_annotations",
        action="store_true",
        help="Flag to also draw annotation points on images",
    )
    parser.add_argument(
        "--per_patient",
        action="store_true",
        help="If set, compute boxplots grouped by patient instead of keypoint",
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
    args = parser.parse_args()

    model_checkpoint = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights\best_unet\best_model.pth"
    test_path = r"C:\Users\User\Desktop\RV_followup_converted"
    save_model_path = r"C:\Users\User\Desktop\RV_followup_predictions"

    os.makedirs(save_model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    model.load_state_dict(
        torch.load(model_checkpoint, map_location=device)["model_state_dict"]
    )
    model.eval()

    stats_list = []
    all_distances = []
    keypoint_names = ["FW annular point", "Septal annular point", "Apex"]

    # --- Loop over patients (folders) ---
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
        ]:  # test set
            # if folder in ['106', '135', '141', '184', '190']: # validation set
            for file in os.listdir(folder_path):
                if "interpolated" in file:
                    file_path = os.path.join(folder_path, file)
                    print(file_path)

                    coordinates_array, stats, distances = process_h5_file_single(
                        file_path=file_path,
                        model=model,
                        device=device,
                        save_model_path=save_model_path,
                        folder=folder,
                        save_images=args.save_images,
                        save_annotations=args.save_annotations,  # <-- pass new arg
                        prediction_stats=True,
                        threshold=args.threshold,
                        no_sudden_movements=args.no_sudden_movements,
                        threshold_sudden=args.threshold_sudden,
                    )

                    # Collect distances for each keypoint (default)
                    for i in range(3):
                        for d in distances[:, i]:
                            all_distances.append(
                                {
                                    "patient": folder,  # NEW: store patient ID
                                    "file": f"{folder}/{file}",
                                    "kp": keypoint_names[i],
                                    "distance": d,
                                }
                            )

                    # Store summary stats for Excel output
                    row = {
                        "file": f"{folder}/{file}",
                        "mean_global": stats["global"]["mean"],
                        "max_global": stats["global"]["max"],
                        "min_global": stats["global"]["min"],
                        "std_global": stats["global"]["std"],
                    }

                    for i, kp_stats in enumerate(stats["per_point"]):
                        row.update(
                            {
                                f"mean_kp{i}": kp_stats["mean"],
                                f"max_kp{i}": kp_stats["max"],
                                f"min_kp{i}": kp_stats["min"],
                                f"std_kp{i}": kp_stats["std"],
                            }
                        )

                    stats_list.append(row)

    # --- Save per-file stats into Excel ---
    df = pd.DataFrame(stats_list)
    avg_row = {"file": "AVERAGE"}
    for col in df.columns:
        if col != "file":
            avg_row[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    output_excel_path = os.path.join(save_model_path, "keypoint_stats.xlsx")
    df.to_excel(output_excel_path, index=False)
    print(f"Saved stats to {output_excel_path}")

    # --- Build DataFrame for distances ---
    df_box = pd.DataFrame(all_distances)
    print("\n--- Keypoint Distance Statistics (mm) ---")

    global_mean = df_box["distance"].mean()
    global_median = df_box["distance"].median()
    global_std = df_box["distance"].std()

    print(f"\nGlobal mean distance: {global_mean:.2f} mm")
    print(f"\nGlobal median distance: {global_median:.2f} mm")
    print(f"Global std distance:  {global_std:.2f} mm")

    for kp, group in df_box.groupby("kp")["distance"]:
        mean_val = group.mean()
        std_val = group.std()
        median_val = group.median()
        q1_val = group.quantile(0.25)
        q3_val = group.quantile(0.75)
        print(
            f"{kp:25s}  mean = {mean_val:.2f} mm   std = {std_val:.2f} mm   median = {median_val:.2f} mm   Q1 = {q1_val:.2f} mm   Q3 = {q3_val:.2f} mm"
        )

    # --- Switch plotting based on flag ---
    if args.per_patient:
        # Group by patient ID
        grouped = df_box.groupby("patient")["distance"]
        medians = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)

        plt.figure(figsize=(10, 6))
        sns.boxplot(y="patient", x="distance", data=df_box, orient="h", width=0.4)
        plt.title("Landmark error distribution per patient")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Patient ID")
        plt.grid(True, axis="x", linestyle="--", alpha=0.4)

        boxplot_path = os.path.join(save_model_path, "patient_distance_boxplot.pdf")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        print(f"Saved per-patient boxplot to {boxplot_path}")

    else:
        # Default: group by keypoint
        grouped = df_box.groupby("kp")["distance"]
        medians = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)

        plt.figure(figsize=(13, 3))  # small height -> less vertical spacing

        sns.boxplot(y="kp", x="distance", data=df_box, orient="h", width=0.5)

        plt.title("Landmark error distribution", fontsize=10)
        plt.xlabel("Distance (mm)", fontsize=9)
        plt.ylabel("")
        plt.grid(True, axis="x", linestyle="--", alpha=0.4)

        # plt.tight_layout(pad=0.2)
        plt.subplots_adjust(left=0.17, bottom=0.15)

        # plt.figure(figsize=(8, 6))
        # sns.boxplot(y='kp', x='distance', data=df_box, orient='h', width=0.2, dodge=False)
        # plt.title('Landmark error distribution in the test set')
        # plt.xlabel('Distance (mm)')
        # plt.ylabel('')
        # plt.grid(True)
        # plt.tight_layout(pad=0.5)

        boxplot_path = os.path.join(save_model_path, "keypoint_distance_boxplot.pdf")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        print(f"Saved per-keypoint boxplot to {boxplot_path}")

    # print(medians)
    # print(q1)
    # print(q3)


if __name__ == "__main__":
    main()

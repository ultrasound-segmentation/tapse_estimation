import os
import h5py
import torch
import numpy as np
import pandas as pd
import argparse

from twod.dataloader.preprocessing import (
    preprocess_images,
    apply_lut,
    resize_or_crop_image_np_nokeypoints,
)
from twod.models.models import Unet
from twod.postprocessing.coordinates_calculation_from_masks import center_of_mass
from twod.pipeline_testing.auto_idx_calculation.indices_calculation import (
    find_parallel_direction,
    find_es,
    RVCalculator,
    RVCalculatorBest,
)
from twod.postprocessing.kalman_filter import KalmanFilter
from twod.postprocessing.pan_tompkins import pan_tompkins_detector
from twod.utils.plot import visualize_image, save_image

"""
This script processes cardiac ultrasound sequences stored in HDF5 files to estimate 
right ventricular function indices such as TAPSE, RVLSF, and RVFAC. It uses a trained 
U-Net segmentation model to detect anatomical landmarks, applies optional Kalman filtering 
for smoothing, calculates indices for each heartbeat, and stores the results in an Excel file. 
Model parameters and processing options can be customized via command-line arguments.
"""


def predict_indices(
    model,
    test_path,
    apply_filter="none",
    device="cpu",
    reduction="max",
    threshold=0.875,
    two_dimensional=True,
    area_method="triangle",
    best_combination=False,
    threshold_sudden=4,  # 2 mm
    avg_all=False,
    count_beats=False,
):
    """Compute indices from a cardiac HDF5 sequence using a trained tracking model."""

    # Load ultrasound + ECG data
    with h5py.File(test_path, "r") as f:
        images = f["tissue"]["data"][()]
        images_times = f["tissue"]["times"][()]
        ecg = f["ecg"]["ecg_data"][()]
        ecg_times = f["ecg"]["ecg_times"][()]
        pixelsize = f["tissue"]["pixelsize"][()]

    fs = 1 / (ecg_times[1] - ecg_times[0])  # ECG sampling frequency
    dt = images_times[1] - images_times[0]  # Image frame interval

    # Reorient and normalize images
    if two_dimensional:  # preprocessing for 2d images
        images = apply_lut(images.transpose(2, 1, 0)[:, :, ::-1])
        images = resize_or_crop_image_np_nokeypoints(images)
        images = images / images.max()
    else:  # three dimensional
        images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
        images = images / images.max()

    # Detect R-peaks in ECG
    r_peaks = pan_tompkins_detector(ecg, fs, plot=False)
    beat_start = [np.argmin(np.abs(images_times - ecg_times[r])) for r in r_peaks]

    if count_beats:
        # estract number of frames
        num_frames = len(images)

    # Inference loop
    coordinates_array = np.zeros((len(images), 3, 2))
    for i, im in enumerate(images):
        im = preprocess_images(
            np.expand_dims(im, axis=0), model_type="U-Net", device=device
        )
        output = model(im.float().unsqueeze(0).to(device))

        coords = [
            center_of_mass(output[0, c].detach(), thresh=threshold) for c in range(3)
        ]
        for j in range(3):
            coordinates_array[i, j] = coords[j]

    if best_combination:
        # set threshold sudden to the best
        threshold_sudden = 4

    # since some indexes are better calculated from the kalman filtering, others from the avg.,
    # others from both, and others from none, I need to create the arrays to calculate everything
    # Actually I think it's easier to calculate each of the 4 arrays each time and use the one I want based on the type of filtering.
    # (since also each index needs its own filtering for best performance)
    # It's more ordered and at the end of the day I will be using --best_combination argument most of the time
    coordinates_array_avg = coordinates_array.copy()
    both_filters_array = coordinates_array.copy()
    filtered_array = coordinates_array.copy()

    # only avg to the avg one
    for j in range(3):
        for i in range(1, len(coordinates_array_avg) - 1):
            if (
                np.linalg.norm(
                    coordinates_array_avg[i, j] - coordinates_array_avg[i - 1, j]
                )
                > threshold_sudden
                and np.linalg.norm(
                    coordinates_array_avg[i, j] - coordinates_array_avg[i + 1, j]
                )
                > threshold_sudden
            ):
                coordinates_array_avg[i, j] = (
                    coordinates_array_avg[i - 1, j] + coordinates_array_avg[i + 1, j]
                ) / 2

    # only Kalman to the filtered one
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

    # both to the bpth filters one
    for j in range(3):
        for i in range(1, len(both_filters_array) - 1):
            if (
                np.linalg.norm(both_filters_array[i, j] - both_filters_array[i - 1, j])
                > threshold_sudden
                and np.linalg.norm(
                    both_filters_array[i, j] - both_filters_array[i + 1, j]
                )
                > threshold_sudden
            ):
                both_filters_array[i, j] = (
                    both_filters_array[i - 1, j] + both_filters_array[i + 1, j]
                ) / 2
    kfs = [
        KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1)
        for _ in range(3)
    ]
    for j in range(3):
        kfs[j].x = np.matrix(
            [both_filters_array[0, j, 0], both_filters_array[0, j, 1], 0, 0]
        ).T

    for i, coords in enumerate(both_filters_array):
        for j in range(3):
            kfs[j].predict()
            both_filters_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1

    # so at the end of thsis block I have 4 arrays:
    # coordinates_array = no filtering
    # filtered_array=Kalman filter
    # both_filters_array = both
    # coordinates_array_avg= only avg

    # Rescale to physical units
    filtered_array[:, :, 0] *= pixelsize[0]
    filtered_array[:, :, 1] *= pixelsize[1]

    both_filters_array[:, :, 0] *= pixelsize[0]
    both_filters_array[:, :, 1] *= pixelsize[1]

    coordinates_array_avg[:, :, 0] *= pixelsize[0]
    coordinates_array_avg[:, :, 1] *= pixelsize[1]

    coordinates_array[:, :, 0] *= pixelsize[0]
    coordinates_array[:, :, 1] *= pixelsize[1]

    # find the direction parallel to the fw movement
    direction = find_parallel_direction(coordinates_array[:, 0])
    direction /= np.linalg.norm(direction)

    if len(coordinates_array) - beat_start[-1] < 20:
        index_container = np.zeros((len(beat_start) - 1, 17))
    else:
        index_container = np.zeros((len(beat_start), 17))

    # initialize the heart cycles count
    count = 0
    for i, frame_num in enumerate(beat_start):
        # I need to select a window from the calculated arrays (window = 1 heartbeat)
        if (
            i == len(beat_start) - 1
        ):  # if it's the last R peak in the acquisition I look at how many frames are there left
            if (
                len(coordinates_array) - frame_num < 20
            ):  # if the last r peak is too close to the end, i dont consider the last part
                print("jump heartbeat")
                continue
            else:  # take the window from last r-peak to the end otherwise

                window_kalman = filtered_array[beat_start[i] :]
                window_unfiltered = coordinates_array[beat_start[i] :]
                window_both = both_filters_array[beat_start[i] :]
                window_avg = coordinates_array_avg[beat_start[i] :]
        else:  # take the window from the r-peak to the next one if it's not the las r-peak
            window_kalman = filtered_array[beat_start[i] : beat_start[i + 1]]
            window_unfiltered = coordinates_array[beat_start[i] : beat_start[i + 1]]
            window_both = both_filters_array[beat_start[i] : beat_start[i + 1]]
            window_avg = coordinates_array_avg[beat_start[i] : beat_start[i + 1]]

        # increase the count of heart cycles
        count = count + 1
        # find the es frame based on the the frame in which the fw is the most distant from ed position (aka frame 0)
        es_frame = find_es(window_both)
        ed_frame = 0  # the end-diastole frame is frame 0 of the window since the window starts with the frame closest to the R-peak

        # fold = os.path.join(r"C:\Users\User\Desktop\test_es_ed_direction", patient.split('\\' )[0])
        """
        for j in range(len(window_both)):
            if j == 0:
                save_image(images[count+j], save_folder=fold, cmap='Blues') #ed_pixel
            elif j== es_pixel:
                save_image(images[count+j], save_folder=fold, cmap='Greens')#es pixel
            else:
                save_image(images[count+j], save_folder= fold, cmap='gray')
        count = count + len(window_both)"""

        if not best_combination:  # manually select filter, and reduction
            if apply_filter == "none":
                window = window_unfiltered
            elif apply_filter == "kalman":
                window = window_kalman
            elif apply_filter == "both":
                window = window_both
            else:  # apply_filter == 'avg'
                window = window_avg

            # calculate shape related indexes
            calculator = RVCalculator(
                window[ed_frame], window[es_frame], method=area_method
            )
        else:  # best calculation method
            calculator = RVCalculatorBest(
                window_unfiltered,
                window_kalman,
                window_avg,
                window_both,
                ed_frame,
                es_frame,
            )

        # the array has 17 values (all the indexes)
        index_container[i] = [
            calculator.tapse_fw * 1000,  # 0
            calculator.tapse_sep * 1000,  # 1
            calculator.rvfac,  # 2
            calculator.ed_area * 1e4,  # 3
            calculator.es_area * 1e4,  # 4
            calculator.ed_len_fw * 1000,  # 5
            calculator.ed_len_sep * 1000,  # 6
            calculator.es_len_fw * 1000,  # 7
            calculator.es_len_sep * 1000,  # 8
            calculator.ed_diam * 1000,  # 9
            calculator.es_diam * 1000,  # 10
            calculator.ed_len_mid * 1000,  # 11
            calculator.es_len_mid * 1000,  # 12
            calculator.strain_fw,  # 13
            calculator.strain_global,  # 14
            calculator.strain_sep,  # 15
            calculator.strain_mid,  # 16
        ]

    if not best_combination:
        # pick max mean or min value across heartbeats based on specifications
        if reduction == "mean":
            return index_container.mean(axis=0)
        elif reduction == "max":
            return index_container.max(axis=0)
        elif reduction == "min":
            return index_container.min(axis=0)
        elif reduction == "median":
            return np.median(index_container, axis=0)
    else:
        if not avg_all:
            # pick the maximum, minimum median or mean value based on the index best performance (I pick a reduction method for each index)
            result = []
            for i in range(index_container.shape[1]):
                if i in [5]:
                    result.append(index_container[:, i].mean())
                elif i in [1, 3, 6, 13, 14, 15, 16]:
                    result.append(index_container[:, i].max())
                elif i in [0, 2, 9, 11]:
                    result.append(np.median(index_container[:, i]))
                else:
                    result.append(index_container[:, i].min())
            return np.asarray(result)
        else:
            # print the number oif heart cycles
            print("Number of heart cycles", count)
            return count, num_frames, index_container.mean(axis=0)


import os
import argparse
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Predict RV indices from HDF5 files using a U-Net model."
    )

    parser.add_argument(
        "--h5_dir", type=str, required=True, help="Directory containing the HDF5 files"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to Excel file where to save the data",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--depth", type=int, default=6, help="U-Net depth")
    parser.add_argument(
        "--filters", type=int, default=16, help="Number of filters to start"
    )
    parser.add_argument(
        "--residuals", type=int, default=0, help="Number of residual units"
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["kalman", "avg", "none", "both"],
        help="type of filter to apply",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        choices=["mean", "max", "min", "median"],
        default="max",
        help="Reduction method for multiple beats",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default=None,
        help="Path to save images with keypoints (optional)",
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Save images with keypoints"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.875,
        help="Threshold for center of mass detection",
    )  # default is 0.875, but must be adjusted based on the threshold used during training
    parser.add_argument(
        "--two_dimensional",
        action="store_true",
        help="whether the images are 2d or 3d derived",
    )
    parser.add_argument(
        "--count_beats",
        action="store_true",
        help="Creates an entry in the excel with the number of heartbeats (as calculated by the pipeline) and with the number of frames in the acquisition",
    )
    parser.add_argument(
        "--area_method",
        type=str,
        choices=["triangle", "spline"],
        default="triangle",
        help="how to calculate the area inside the triangle",
    )
    parser.add_argument(
        "--best_combination",
        action="store_true",
        help="Use best combination of parameters found (overrides other parameters)",
    )
    parser.add_argument(
        "--threshold_avg",
        type=int,
        default=4,
        help="Threshold for avg filter application",
    )
    parser.add_argument(
        "--avg_all",
        action="store_true",
        help="if called together with --best combination, "
        'it uses an averaging reduction method instead of "cherry picking" the best method for each index. Ideally,'
        " with a perfect model, this would be the best way to proceed",
    )
    args = parser.parse_args()

    columns = [
        "tapsefw",
        "tapsesep",
        "rvfac",
        "rvad",
        "rvas",
        "rvldfw",
        "rvldsep",
        "rvlsfw",
        "rvlssep",
        "tadd",
        "tasd",
        "rvldmid",
        "rvlsmid",
        "rvlsffw",
        "rvlsfglobal",
        "rvlsfsep",
        "rvlsfmid",
    ]

    df = pd.read_excel(args.excel_path)

    # Ensure metric columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # Initialize beat/frame count columns if the --count_beats flag is active
    if args.count_beats:
        if "N heartbeats" not in df.columns:
            df["N heartbeats"] = None
        if "N frames" not in df.columns:
            df["N frames"] = None

    if "path" not in df.columns:
        raise ValueError("Excel file must contain a 'path' column.")

    paths = df["path"].dropna().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(
        depth=args.depth, start_filts=args.filters, num_residuals=args.residuals
    ).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device)["model_state_dict"]
    )
    model.eval()

    for path in paths:
        test_path = os.path.join(args.h5_dir, str(path) + ".h5")
        print(f"Processing: {test_path}")

        try:
            # function that predicts indexes froom the h5 file
            n_heartbeats, n_frames, indexes = predict_indices(
                model,
                test_path,
                apply_filter=args.filter,
                device=device,
                reduction=args.reduction,
                threshold=args.threshold,
                two_dimensional=args.two_dimensional,
                area_method=args.area_method,
                best_combination=args.best_combination,
                threshold_sudden=args.threshold_avg,
                avg_all=args.avg_all,
                count_beats=args.count_beats,
            )

            row_idx = df.index[df["path"] == path].tolist()
            if not row_idx:
                print(f"Path {path} not found in DataFrame.")
                continue

            row = row_idx[0]

            # Fill existing metric columns
            for i, col in enumerate(columns):
                df.at[row, col] = indexes[i]

            # Write heartbeats and frames if the --count_beats flag is active
            if args.count_beats:
                df.at[row, "N heartbeats"] = n_heartbeats
                df.at[row, "N frames"] = n_frames

        except Exception as e:
            print(f"Error processing {path}: {e}")

    df.to_excel(args.excel_path, index=False)
    print("Excel file updated.")


if __name__ == "__main__":
    main()

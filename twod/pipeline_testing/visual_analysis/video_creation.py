import cv2
import os
import re
import h5py
import numpy as np

from dataloader.preprocessing import (
    apply_lut,
    preprocess_images,
    resize_or_crop_image_np_nokeypoints,
)

# utiliy functions to create videos from an image folder or a hdf5 file


def create_video_from_images(folder_path, output_video, fps=30):
    """
    Creates a video from a folder of PNG images.

    :param folder_path: Path to the folder containing PNG images.
    :param output_video: Path to the output video file (e.g., 'output.mp4').
    :param fps: Frames per second for the output video.

    Image Naming Convention:
    - Images should be named in a way that ensures correct ordering when sorted alphabetically.
    - Example: 'frame_001.png', 'frame_002.png', ..., 'frame_100.png'
    """

    # Get a list of all PNG images in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    # Sort the images using a regex to extract the numeric part
    images.sort(key=lambda img: int(re.search(r"\d+", img).group()))

    # Check if there are any images in the folder
    if not images:
        print("No PNG images found in the folder.")
        return

    # Read the first image to determine the video resolution
    first_image = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, _ = first_image.shape  # Get image dimensions

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through each image, read it, and write it to the video file
    for image in images:
        frame = cv2.imread(os.path.join(folder_path, image))
        video_writer.write(frame)  # Write the frame to the video

    # Release the video writer to finalize the file
    video_writer.release()
    print(f"Video saved as {output_video}")


def create_video_from_h5(
    file_path, output_video_path, num_landmarks=3, fps=30, landmarks=True
):
    """
    Reads ultrasound frames and annotations from an HDF5 file and creates a video with annotated landmarks.
    """
    if landmarks:
        with h5py.File(file_path, "r") as h5_file:
            frames = h5_file["frames"][()]
            frames = np.array(frames)  # Ensure numpy array

            if "annotations" in h5_file:
                ref_coord = h5_file["annotations"][()]
            else:
                num_frames = frames.shape[2]
                ref_coord = np.zeros((num_frames, num_landmarks, 2))

        if len(frames.shape) < 3:
            raise ValueError(
                "Frames data has an unexpected shape. Expected 3D array (height, width, num_frames)."
            )

        height, width = frames.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (width, height), isColor=True
        )

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue

        for i in range(frames.shape[2]):
            frame = frames[:, :, i].astype(np.uint8)
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for j in range(num_landmarks):
                x, y = ref_coord[i][j]
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        frame_color, (int(x), int(y)), 2, colors[j % len(colors)], -1
                    )

            out.write(frame_color)

        out.release()
        print(f"Video saved to {output_video_path}")

    else:
        with h5py.File(file_path, "r") as h5_file:
            frames = h5_file["tissue"]["data"][()]
            frames = np.array(frames)  # Ensure numpy array

            frames = apply_lut(frames.transpose(1, 0, 2)[:, ::-1, :])

        if len(frames.shape) < 3:
            raise ValueError(
                "Frames data has an unexpected shape. Expected 3D array (height, width, num_frames)."
            )

        height, width = frames.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (width, height), isColor=True
        )

        for i in range(frames.shape[2]):
            frame = frames[:, :, i].astype(np.uint8)
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            out.write(frame_color)

        out.release()
        print(f"Video saved to {output_video_path}")


import h5py


def explore_h5(name, obj):
    print(name)


if __name__ == "__main__":
    prediction_folder = r"C:\Users\User\Desktop\test_set_with_predictions"  # path to the folders with the single frames
    folder = r"C:\Users\User\Desktop\final_reviewed_dataset"  # path to the h5 files with requency info
    output_folder = r"C:\Users\User\Desktop\test_set_with_predictions\videos"
    patients = []
    frequencies = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for file in os.listdir(subfolder_path):
            if "interpolated" in file:
                file_path = os.path.join(subfolder_path, file)
                with h5py.File(file_path, "r") as f:
                    times = f["tissue"]["times"][()]
                    frequency = 1 / (times[1] - times[0])
                patients.append(subfolder)
                frequencies.append(frequency)
    print(f"Patients: {patients}")
    print(f"Frequencies: {frequencies}")

    for i, patient in enumerate(patients):
        patient_folder = os.path.join(prediction_folder, patient)
        if not os.path.exists(patient_folder):
            print(f"Skipping {patient}: folder does not exist.")
            continue

        output_video_path = os.path.join(output_folder, f"{patient}.mp4")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(patient_folder, output_video_path)
        for fold in os.listdir(patient_folder):
            create_video_from_images(
                os.path.join(patient_folder, fold),
                output_video_path,
                fps=frequencies[i],
            )  # useful to create videos at the same frequency as the original images
        # folder = r'2d/illustrative_video/100/P4297P80_interpolated'
        # output_video_path = r'2d/illustrative_video/video_100_1.mp4'
        # create_video_from_images(folder, output_video_path, fps=30)

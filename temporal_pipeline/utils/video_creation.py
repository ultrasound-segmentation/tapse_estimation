import cv2
import os
import re
import h5py
import numpy as np


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


def create_video_from_h5(file_path, output_video_path, num_landmarks=3, fps=30):
    """
    Reads ultrasound frames and annotations from an HDF5 file and creates a video with annotated landmarks.
    """
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
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

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


if __name__ == "__main__":

    folder = r"runs/exp24"
    output = r"D:\mmissana\results_video.mp4"
    create_video_from_images(folder, output, fps=5)

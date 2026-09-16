import cupy as cp
import numpy as np
import cv2


def create_video(array, output_path: str, fps: int = 1):
    """
    Creates a grayscale video from a NumPy or CuPy array where the third axis (index 2) is the time dimension.

    Parameters:
        array (np.ndarray or cp.ndarray): Input array of shape (H, W, T) [Height, Width, Time].
        output_path (str): Path to save the output video (e.g., "output.mp4").
        fps (int): Frames per second (default: 1).
    """
    # Check if the input is CuPy, and convert if necessary
    if isinstance(array, cp.ndarray):
        array = cp.asnumpy(array)  # Move to CPU for OpenCV processing

    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy or CuPy array.")

    # Get dimensions
    height, width, time_frames = array.shape

    # Define video writer (MP4 format, grayscale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    for t in range(time_frames):
        frame = array[:, :, t]  # Extract single frame

        # Normalize pixel values to 0-255 for proper visualization
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)  # Convert to 8-bit grayscale

        # Convert grayscale to BGR (since OpenCV requires 3-channel input)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        print(f"Writing frame {t+1}/{time_frames}")
        # Write frame to video
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import h5py

"""image visualization with annotations superimposed"""


def load_and_plot_annotations(file_path, annotation_path):
    """
    Load and display an image with annotations.

    Parameters:
    -----------
    file_path : str
        Path to the `.npz` file containing the images.
    annotation_path : str
        Path to the `.npz` file containing the annotations.
    """
    # Load images
    data = np.load(file_path)
    frames = data["frames"]  # Assume images are stored under the key 'video'

    # Load annotations
    annotations = np.load(annotation_path)["annotations"]

    num_frames = frames.shape[2]

    plt.ion()  # Interactive mode

    for idx in range(num_frames):
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap="gray")  # Display the image

        # Plot annotations
        for j in range(3):
            plt.scatter(
                annotations[idx][j][0],
                annotations[idx][j][1],
                color="r",
                marker="*",
                s=100,
            )  # Annotations

        plt.title(f"Frame {idx + 1}/{num_frames}")
        plt.pause(0.5)  # Pause to visualize the frame

    plt.ioff()
    plt.show()


def load_and_plot_annotations_h5(file_path, save_folder):
    """
    Load an HDF5 file and save each frame with its annotations as an image.

    Parameters:
    -----------
    file_path : str
        Path to the `.h5` file containing the frames and annotations.
    save_folder : str
        Folder where the annotated frames will be saved.
    """
    # Load images and annotations from the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        frames = h5_file["frames"][()]  # shape: (H, W, N)
        annotations = h5_file["annotations"][()]  # shape: (N, M, 2)

    os.makedirs(save_folder, exist_ok=True)
    num_frames = frames.shape[2]
    colors = ["r", "g", "b"]

    for idx in range(num_frames):
        plt.figure(figsize=(6, 6))
        plt.imshow(frames[:, :, idx], cmap="gray")

        # Plot circles at annotation positions
        for j in range(annotations.shape[1]):
            x, y = annotations[idx, j, 0], annotations[idx, j, 1]
            if x > 0 and y > 0:
                circle = plt.Circle(
                    (x, y), radius=4, color=colors[j % len(colors)], fill=True
                )
                plt.gca().add_patch(circle)

        # plt.title(f"Frame {idx + 1}/{num_frames}")
        plt.axis("off")

        output_path = os.path.join(save_folder, f"frame_{idx+1:03d}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"Saved {num_frames} annotated frames to '{save_folder}'")


def visualize_dataset(dataset_path):
    """
    Visualize images and keypoints from a dataset.

    Parameters:
    -----------
    dataset_path : str
        Path to the `.npz` file containing the dataset.
    """
    # Load the dataset
    dataset = np.load(dataset_path)
    images = dataset["images"]
    keypoints = dataset["keypoints"]

    num_images = images.shape[0]
    print(images.shape)
    print(images.max(), images.min())

    plt.ion()
    plt.ion()  # Interactive mode

    for idx in range(num_images):
        plt.clf()
        plt.imshow(images[idx, :, :], cmap="gray")  # Display the image

        # Plot annotations
        for j in range(3):
            plt.scatter(
                keypoints[idx][j][0], keypoints[idx][j][1], color="r", marker="*", s=100
            )  # Annotations

        plt.title(f"image {idx + 1}/{num_images}")
        plt.pause(0.5)  # Pause to visualize the frame

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    ann_images = r"d:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed\111\P429AL06_interpolated.h5"
    save_folder = r"d:\mmissana\data\images_paper\images_input_3_colors"
    load_and_plot_annotations_h5(ann_images, save_folder=save_folder)

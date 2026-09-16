# import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import os


def visualize_image(image, points=None):
    """
    Displays a 2D image with optional points highlighted.
    If the image is a CuPy array, it is converted to a NumPy array.

    :param image: 2D array representing the image
    :param points: list of tuples (x, y) for the points to highlight in red
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)

    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.title("2D Image Visualization")
    plt.axis("off")

    # If points are provided, plot them in red
    if isinstance(points, np.ndarray):
        plt.scatter(points[:, 0], points[:, 1], c="red", marker="x")
    elif points:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], c="red", marker="x")
    plt.show()


def save_image(
    image, points=None, save_folder="visualizations", bold=False, cmap="gray"
):
    """
    Saves a 2D image with optional points highlighted to a specified folder.
    If the image is a CuPy array, it is converted to a NumPy array.

    :param image: 2D array representing the image
    :param points: list of tuples (x, y) for the points to highlight in red
    :param save_folder: folder where the image will be saved, default is 'visualizations'
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)

    # Make sure the folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate an incremental file name
    existing_files = os.listdir(save_folder)
    num_images = len(
        [f for f in existing_files if f.endswith(".png")]
    )  # Count current PNG files
    save_path = os.path.join(save_folder, f"image_{num_images + 1}.png")

    # Create the plot
    plt.imshow(image, cmap=cmap)
    # plt.colorbar()
    # plt.title("2D Image Visualization")
    plt.axis("off")

    # If points are provided, plot them in red
    if points:
        points = np.array(points)
        if bold:
            for x, y in points:
                circle = plt.Circle(
                    (x, y), radius=4, color="red", fill=True
                )  # fill=True per cerchi pieni
                plt.gca().add_patch(circle)
        else:
            plt.scatter(points[:, 0], points[:, 1], c="red", marker="x")

    # Save the image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the plot to avoid displaying it in an interactive session


def save_image_ann_pred(
    image, ann_points=None, pred_points=None, save_folder="visualizations", bold=False
):
    """
    Saves a 2D image with optional annotation and prediction points highlighted.

    Parameters:
    -----------
    image : 2D array
        Image to save (can be NumPy or CuPy array).
    ann_points : list of (x, y), optional
        Annotation points to plot (green circles if bold, green '+' otherwise).
    pred_points : list of (x, y), optional
        Prediction points to plot (red circles if bold, red 'x' otherwise).
    save_folder : str
        Folder where the image will be saved (default: 'visualizations').
    bold : bool
        If True, use filled circles; otherwise, use simple markers.
    """
    # Convert CuPy arrays to NumPy if needed
    if "cp" in globals() and isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)

    os.makedirs(save_folder, exist_ok=True)

    # Auto-increment file name
    existing_files = [f for f in os.listdir(save_folder) if f.endswith(".png")]
    save_path = os.path.join(save_folder, f"image_{len(existing_files) + 1}.png")

    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # Plot annotation points
    if ann_points:
        ann_points = np.array(ann_points)
        if bold:
            for x, y in ann_points:
                circle = plt.Circle((x, y), radius=4, color="blue", fill=True)
                plt.gca().add_patch(circle)
        else:
            plt.scatter(
                ann_points[:, 0],
                ann_points[:, 1],
                c="#05fa22",
                marker="o",
                label="Annotation",
                s=50,
            )

    # Plot prediction points
    if pred_points:
        pred_points = np.array(pred_points)
        if bold:
            colors = ["red", "green", "blue"]
            for i, (x, y) in enumerate(pred_points):
                current_color = colors[i]
                # circle = plt.Circle((x, y), radius=4, color=current_color, fill=True)
                # plt.gca().add_patch(circle)
        else:
            plt.scatter(
                pred_points[:, 0],
                pred_points[:, 1],
                c="red",
                marker="o",
                label="Prediction",
                s=50,
            )

    # Add legend only if both are present
    # if ann_points is not None and len(ann_points) > 0 and pred_points is not None and len(pred_points) > 0:
    #     leg = plt.legend(loc='upper left', frameon=False)

    #     colors = ['blue', 'red']
    #     for text, color in zip(leg.get_texts(), colors):
    #         text.set_color(color)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_image_ill(image, points=None, save_folder="visualizations", bold=False):
    """
    Saves a 2D image with optional points highlighted to a specified folder and lines connecting them.
    If the image is a CuPy array, it is converted to a NumPy array.

    :param image: 2D array representing the image
    :param points: list of tuples (x, y) for the points to highlight in red
    :param save_folder: folder where the image will be saved, default is 'visualizations'
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)

    # Make sure the folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate an incremental file name
    existing_files = os.listdir(save_folder)
    num_images = len(
        [f for f in existing_files if f.endswith(".png")]
    )  # Count current PNG files
    save_path = os.path.join(save_folder, f"image_{num_images + 1}.png")

    # Create the plot
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # If points are provided, plot them
    if points:
        points = np.array(points)

        mid_point = (
            (points[0, 0] + points[1, 0]) / 2,
            (points[0, 1] + points[1, 1]) / 2,
        )

        # Draw lines connecting the points
        x_values = [points[0][0], points[1][0]]
        y_values = [points[0][1], points[1][1]]
        plt.plot(x_values, y_values, color="#fcfc0a", linewidth=2)

        x_values = [points[1][0], points[2][0]]
        y_values = [points[1][1], points[2][1]]
        plt.plot(x_values, y_values, color="#fcfc0a", linewidth=2)

        x_values = [points[2][0], points[0][0]]
        y_values = [points[2][1], points[0][1]]
        plt.plot(x_values, y_values, color="#fcfc0a", linewidth=2)

        # x_values = [points[2][0], mid_point[0]]
        # y_values = [points[2][1], mid_point[1]]
        # plt.plot(x_values, y_values, color='#18d9cc', linewidth=2)

        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=["red", "blue", "green"],
            edgecolors="black",
            s=80,
            linewidths=1.5,
            zorder=3,
        )
        # plt.scatter(mid_point[0], mid_point[1], c='green', edgecolors='black',
        #     s=80, linewidths=1.5, zorder=3)

    # Save the image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the plot to avoid displaying it in an interactive session

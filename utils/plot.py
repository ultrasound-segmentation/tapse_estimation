import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import os


class VolumeViewer:
    def __init__(self, volume, click_radius=5):
        if not isinstance(volume, cp.ndarray):
            self.volume = cp.asarray(volume)
        else:
            self.volume = volume

        self.slice_idx = 0
        self.clicked_points = []
        self.click_radius = click_radius
        self.view_mode = "XY"
        self.unit_vectors = []
        self.img = None

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        self.update_image()

        self.cbar = self.fig.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("Intensity")

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax_slider,
            "Slice",
            0,
            self.get_max_slices() - 1,
            valinit=self.slice_idx,
            valstep=1,
        )
        self.slider.on_changed(self.update_slice)

        ax_button_xy = plt.axes([0.1, 0.2, 0.15, 0.05])
        ax_button_zy = plt.axes([0.4, 0.2, 0.15, 0.05])
        ax_button_xz = plt.axes([0.7, 0.2, 0.15, 0.05])

        self.button_xy = Button(ax_button_xy, "XY Plane")
        self.button_zy = Button(ax_button_zy, "ZY Plane")
        self.button_xz = Button(ax_button_xz, "XZ Plane")

        self.button_xy.on_clicked(lambda event: self.change_view("XY"))
        self.button_zy.on_clicked(lambda event: self.change_view("ZY"))
        self.button_xz.on_clicked(lambda event: self.change_view("XZ"))

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )  # Aggiunto listener per la tastiera

    def get_max_slices(self):
        if self.view_mode == "XY":
            return self.volume.shape[2]
        elif self.view_mode == "ZY":
            return self.volume.shape[0]
        elif self.view_mode == "XZ":
            return self.volume.shape[1]

    def update_image(self):
        self.ax.clear()
        if self.view_mode == "XY":
            img_data = cp.asnumpy(self.volume[:, :, self.slice_idx])
        elif self.view_mode == "ZY":
            img_data = cp.asnumpy(self.volume[self.slice_idx, :, :].T)
        elif self.view_mode == "XZ":
            img_data = cp.asnumpy(self.volume[:, self.slice_idx, :])

        self.img = self.ax.imshow(img_data, cmap="gray", vmin=0, vmax=255)
        self.ax.set_title(f"Slice: {self.slice_idx} ({self.view_mode} plane)")

        self.redraw_points()
        self.fig.canvas.draw_idle()

    def update_slice(self, val):
        self.slice_idx = int(self.slider.val)
        self.update_image()

    def onclick(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)

            if self.view_mode == "XY":
                z = self.slice_idx
            elif self.view_mode == "ZY":
                z = y
                y = x
                x = self.slice_idx
            elif self.view_mode == "XZ":
                z = x
                x = y
                y = self.slice_idx

            for i, (px, py, pz) in enumerate(self.clicked_points):
                if (
                    abs(pz - z) <= self.click_radius
                    and abs(px - x) <= self.click_radius
                    and abs(py - y) <= self.click_radius
                ):
                    del self.clicked_points[i]
                    self.redraw_points()
                    return

            self.clicked_points.append((x, y, z))
            self.redraw_points()
            if len(self.clicked_points) == 2:
                self.calculate_unit_vectors()

    def redraw_points(self):
        points_in_slice = []
        for x, y, z in self.clicked_points:
            if self.view_mode == "XY" and z == self.slice_idx:
                points_in_slice.append((x, y))
            elif self.view_mode == "ZY" and x == self.slice_idx:
                points_in_slice.append((y, z))
            elif self.view_mode == "XZ" and y == self.slice_idx:
                points_in_slice.append((z, x))

        if points_in_slice:
            x_vals, y_vals = zip(*points_in_slice)
            self.ax.scatter(x_vals, y_vals, c="red", marker="x", s=50)

        self.fig.canvas.draw_idle()

    def change_view(self, mode):
        self.view_mode = mode
        self.slice_idx = 0
        self.slider.valmax = self.get_max_slices() - 1
        self.slider.set_val(0)
        self.update_image()

    def get_selected_points(self):
        return self.clicked_points

    def reset_points(self):
        self.clicked_points = []
        self.redraw_points()

    def show(self):
        plt.show()

    def calculate_unit_vectors(self):
        self.unit_vectors = []
        if len(self.clicked_points) < 2:
            print("Sono necessari almeno due punti per calcolare il vettore.")
            return

        for i in range(0, len(self.clicked_points) - 1, 2):
            p1 = cp.array(self.clicked_points[i])
            p2 = cp.array(self.clicked_points[i + 1])
            vector = p2 - p1
            norm = cp.linalg.norm(vector)
            if norm != 0:
                vector = vector / norm
            else:
                print(
                    "I punti sono identici; impossibile calcolare il vettore unitario."
                )
                continue
            self.unit_vectors.append(vector)

    def on_key_press(self, event):
        """Gestisce l'input da tastiera per navigare tra le slices."""
        if event.key == "right":  # Freccia destra
            if self.slice_idx < self.get_max_slices() - 1:
                self.slice_idx += 1
        elif event.key == "left":  # Freccia sinistra
            if self.slice_idx > 0:
                self.slice_idx -= 1

        self.slider.set_val(self.slice_idx)  # Aggiorna lo slider
        self.update_image()


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
            for x, y in pred_points:
                circle = plt.Circle((x, y), radius=4, color="red", fill=True)
                plt.gca().add_patch(circle)
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

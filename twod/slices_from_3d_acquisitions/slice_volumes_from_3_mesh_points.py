import h5py
from utils.plot import VolumeViewer
import numpy as np
import cupy as cp
import os
import json
from utils.extract_slices_new import extract_from_hdf5

"""code that i used to extract the 2d slices from the 3d volume. See the description of 
slice_volumes_from_2_mesh_points.py. Only thing that changes is that you insert 3 points 
belonging to the tric valve. Then the view is aligned to this plane and you can choose from the
image the midpoint of the tric. valve. After this the ventricle is sliced with planes perpendicular 
to the tric valve and passing through the center, and you can choose the best views."""

folder = r"D:\mmissana\data\4DRVQ_Jinyang\voxels"
save_folder = r"D:\mmissana\data\processed_imgs_2"
checkpoint_file = r"D:\mmissana\data\checkpoint_3.json"


def load_checkpoint():
    """Load checkpoint file if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_data):
    """Save checkpoint data."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=4)


degrees = np.linspace(np.pi, 2 * np.pi, 18)
checkpoint = load_checkpoint()  # Load progress

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    print(f"Processing {file}...")

    # Skip already processed files
    if file in checkpoint and checkpoint[file] == "complete":
        print(f"Skipping {file}, already processed.")
        continue

    # Manually enter 3 points on the triscuspid valve, in order to define that plane
    first = input(f"Enter 3 values for first point (space-separated) for {file}: ")
    second = input(f"Enter 3 values for second point (space-separated) for {file}: ")
    third = input(f"Enter 3 values for third point (space-separated) for {file}: ")

    try:
        first = np.array([float(x) for x in first.split()])
        second = np.array([float(x) for x in second.split()])
        third = np.array([float(x) for x in third.split()])
        # center = np.array([float(x) for x in center.split()])

        if first.shape != (3,) or second.shape != (3,) or third.shape != (3,):
            raise ValueError("Tric and Apex must each have exactly 3 values.")

    except ValueError as e:
        print(f"Error in input values: {e}")
        continue

    # Extract slices from the volume
    save_subfolder = os.path.join(save_folder, file.replace(".h5", ""))
    os.makedirs(save_subfolder, exist_ok=True)
    extract_from_hdf5(
        file_path, save_subfolder, degrees, first=first, second=second, third=third
    )

    # Save the checkpoint
    checkpoint[file] = "complete"
    save_checkpoint(checkpoint)
    print(f"File {file} processed successfully!")

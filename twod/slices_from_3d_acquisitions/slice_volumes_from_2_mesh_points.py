import h5py
from utils.plot import VolumeViewer
import numpy as np
import os
import json
from utils.extract_slices import extract_slices

# code that was used to extract the 2d slices from 3D acquisitions.
# I started from the 3d mash (segmentation), made by an expert clinician on top of
# the acquisition, then I used 3Dslicer to manually select the center of the tricuspid
# valve and the apex of the ventricle. Using this line, the ventricle is oriented. Then you cna
# slice it with planes passing through this line, and obtain 2d views of the ventricle.

folder = r"D:\mmissana\data\4DRVQ_Jinyang\voxels"
save_folder = r"D:\mmissana\data\processed_imgs_2"
checkpoint_file = r"D:\mmissana\data\checkpoint.json"


def load_checkpoint():
    """Load checkpoint file if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_data):
    """Save checkpoint data."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=4)


def print_structure(name, obj):
    print(name, obj)


degrees = np.linspace(np.pi, 2 * np.pi, 20)
checkpoint = load_checkpoint()  # Load progress

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    # Skip processed files
    if file in checkpoint and checkpoint[file] == "complete":
        print(f"Skipping {file}, already processed.")
        continue

    with h5py.File(file_path, "r") as h5_file:
        h5_file.visititems(print_structure)
        grids = list(h5_file["Input"].keys())

        for grid in grids:
            # Skip processed grids
            if file in checkpoint and grid in checkpoint[file]:
                print(f"Skipping {grid} in {file}, already processed.")
                continue

            while True:  # Repeat until user decides to save
                input_data = h5_file["Input"][grid][:]
                ground_truth = h5_file["GroundTruth"][grid][:]
                imgs = extract_slices(input_data, ground_truth, degrees)

                viewer = VolumeViewer(imgs)
                viewer.show()

                user_choice = ""
                while user_choice not in ["y", "n"]:
                    user_choice = (
                        input(f"Save images for {grid}? (y/n): ").strip().lower()
                    )

                if user_choice == "y":
                    save_path = os.path.join(save_folder, file, f"{grid}.npz")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.savez_compressed(save_path, imgs)
                    print(f"Saved images to {save_path}")

                    # Update checkpoint
                    if file not in checkpoint:
                        checkpoint[file] = {}
                    checkpoint[file][grid] = "done"
                    save_checkpoint(checkpoint)  # Save progress

                    break  # Move to next grid
                else:
                    print("Repeating visualization...")

        # Mark file as fully processed
        checkpoint[file] = "complete"
        save_checkpoint(checkpoint)

print("Processing completed for all files.")

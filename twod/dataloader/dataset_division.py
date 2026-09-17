import os
import numpy as np
import h5py
import torch
from preprocessing import resize_or_crop_image_np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# old script to create numpy dataset from files and txt file with the division
# Function to divide dataset based on a text file containing train/val/test splits
"""
The txtfile should be something like this:
Validation:
name.h5
Test:
another_name.h5
Training:
etc.
"""


def dataset_division_from_txt_h5(
    txt_path, save_path=os.getenv("CONVERTED_DATASET_PATH", "")
):
    data_path = Path(os.getenv("DATASET_PATH", ""))

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        print(f"No directory detected. Creating directory {save_path}")
        os.makedirs(save_path)

    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Division file not found: {txt_path}")

    # Read the dataset division file
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
            is_empty = 1 / len(lines)
    except FileNotFoundError:
        print("ERROR! File not found!")
        raise FileNotFoundError
    except ZeroDivisionError:
        print(f"ERROR! {txt_path} is an empty file!")
        # TODO: Add custom exceptions
        raise FileExistsError

    # Lists to store video names for validation and test sets
    test = []
    val = []
    training = []

    # Flags to track which section is being read
    flag_val = False
    flag_test = False
    flag_train = False

    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces and newline characters

        if line == "Validation:":
            flag_val = True
            flag_test = False
            flag_train = False
            continue
        if line == "Test:":
            flag_test = True
            flag_val = False
            flag_train = False
            continue
        if line == "Training:":
            flag_val = False
            flag_test = False
            flag_train = True
            continue

        # Add video names to respective lists
        if flag_val:
            val.append(line)
        if flag_test:
            test.append(line)
        if flag_train:
            training.append(line)
    for name, split in [("training", training), ("val", val), ("test", test)]:
        try:
            1 / len(split)
        except ZeroDivisionError:
            print(f"ERROR! {name} is empty")
            # TODO: Add custom exception
            raise ZeroDivisionError
    # Lists to store images and keypoints for each dataset split
    img_val_list = []
    img_test_list = []
    img_train_list = []
    keypoint_val_list = []
    keypoint_test_list = []
    keypoint_train_list = []

    for split in [training, val, test]:
        for relative_path in split:
            relative_path = relative_path.strip()

            # Allow absolute paths, such as /cluster/...
            path = Path(relative_path)

            # Otherwise interpret the path relative to DATASET_PATH
            if not path.is_absolute():
                path = data_path / path

            if not path.is_file():
                raise FileNotFoundError(f"H5 file not found: {path}")

            print(f"Reading: {path}")

            with h5py.File(path, "r") as h5_file:
                images = h5_file["frames"][()]
                annotations = h5_file["annotations"][()]
                images = images.transpose(2, 0, 1)
                print(images.shape)
                images, annotations = resize_or_crop_image_np(images, annotations)
                print(images.shape)
                print(annotations.shape)

                if split == val:
                    img_val_list.append(images)
                    keypoint_val_list.append(annotations)
                elif split == test:
                    img_test_list.append(images)
                    keypoint_test_list.append(annotations)
                else:
                    img_train_list.append(images)
                    keypoint_train_list.append(annotations)

    # Convert lists to NumPy arrays using np.concatenate (better than np.stack for variable N)
    img_train_np = np.concatenate(img_train_list, axis=0) if img_train_list else None
    img_val_np = np.concatenate(img_val_list, axis=0) if img_val_list else None
    img_test_np = np.concatenate(img_test_list, axis=0) if img_test_list else None

    keypoint_train_np = (
        np.concatenate(keypoint_train_list, axis=0) if keypoint_train_list else None
    )
    keypoint_val_np = (
        np.concatenate(keypoint_val_list, axis=0) if keypoint_val_list else None
    )
    keypoint_test_np = (
        np.concatenate(keypoint_test_list, axis=0) if keypoint_test_list else None
    )

    print("training", img_train_np.shape)
    print("test", img_test_np.shape)
    print("val", img_val_np.shape)

    # Save the divided datasets into compressed .npz files
    np.savez_compressed(
        os.path.join(save_path, "train.npz"),
        images=img_train_np,
        keypoints=keypoint_train_np,
    )
    np.savez_compressed(
        os.path.join(save_path, "test.npz"),
        images=img_test_np,
        keypoints=keypoint_test_np,
    )
    np.savez_compressed(
        os.path.join(save_path, "val.npz"), images=img_val_np, keypoints=keypoint_val_np
    )


# Main execution
if __name__ == "__main__":
    txt_path = os.getenv("DIVISION_TXT_PATH")  # Path to the dataset division text file
    dataset_division_from_txt_h5(txt_path)  # Call the function to process the dataset

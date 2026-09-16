import os
import numpy as np
import h5py
import torch
from dataloader.preprocessing import resize_or_crop_image_np

# old script to create numpy dataset from files and txt file with the division


# Function to divide dataset based on a text file containing train/val/test splits
"""
The txtfile should be something like this:
Validation:
400001 
660001
730001  
135001 
Test:
106001 
157001 
184001 
661001 
800001 
920001 
112001 
Training:
other
"""


def dataset_division_from_txt(
    txt_path,
    data_separated_by_video_path=r"D:\mmissana\data\dataset_separated_by_video_256",
    save_path=r"D:\mmissana\data\dataset_256",
):

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read the dataset division file
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Lists to store video names for validation and test sets
    test = []
    val = []

    # Flags to track which section is being read
    flag_val = False
    flag_test = False

    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces and newline characters

        if line == "Validation:":
            flag_val = True
            continue
        if line == "Test:":
            flag_test = True
            flag_val = False
            continue
        if line == "Training:":
            flag_val = False
            flag_test = False
            continue

        # Add video names to respective lists
        if flag_val:
            val.append(line)
        if flag_test:
            test.append(line)

    # Lists to store images and keypoints for each dataset split
    img_val_list = []
    img_test_list = []
    img_train_list = []
    keypoint_val_list = []
    keypoint_test_list = []
    keypoint_train_list = []

    # Process each file in the dataset folder
    for file in os.listdir(data_separated_by_video_path):
        file_name = file.split(".")[0]  # Extract video name without extension
        file_path = os.path.join(data_separated_by_video_path, file)

        # Load the .npz file containing images and keypoints
        data = np.load(file_path)

        if file_name in val:  # If the file belongs to the validation set
            for i in range(data["images"].shape[0]):
                img_val_list.append(data["images"][i])
                keypoint_val_list.append(data["keypoints"][i])
        elif file_name in test:  # If the file belongs to the test set
            for i in range(data["images"].shape[0]):
                img_test_list.append(data["images"][i])
                keypoint_test_list.append(data["keypoints"][i])
        else:  # If the file belongs to the training set (default case)
            for i in range(data["images"].shape[0]):
                img_train_list.append(data["images"][i])
                keypoint_train_list.append(data["keypoints"][i])

    # Save the divided datasets into compressed .npz files
    np.savez_compressed(
        os.path.join(save_path, "val.npz"),
        images=np.array(img_val_list),
        keypoints=np.array(keypoint_val_list),
    )
    np.savez_compressed(
        os.path.join(save_path, "test.npz"),
        images=np.array(img_test_list),
        keypoints=np.array(keypoint_test_list),
    )
    np.savez_compressed(
        os.path.join(save_path, "train.npz"),
        images=np.array(img_train_list),
        keypoints=np.array(keypoint_train_list),
    )


def dataset_division_from_txt_h5(
    txt_path, save_path=r"D:\mmissana\data\RV_PATIENTS\dataset_256"
):

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read the dataset division file
    with open(txt_path, "r") as f:
        lines = f.readlines()

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

        if line == "validation:":
            flag_val = True
            flag_test = False
            flag_train = False
            continue
        if line == "test:":
            flag_test = True
            flag_val = False
            flag_train = False
            continue
        if line == "training:":
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

    # Lists to store images and keypoints for each dataset split
    img_val_list = []
    img_test_list = []
    img_train_list = []
    keypoint_val_list = []
    keypoint_test_list = []
    keypoint_train_list = []

    for split in [training, val, test]:
        for path in split:
            print(path)
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
    txt_path = r"c:\Users\vcxr10\Desktop\dataset_division_by_patient.txt"  # Path to the dataset division text file
    dataset_division_from_txt_h5(txt_path)  # Call the function to process the dataset

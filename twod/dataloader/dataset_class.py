import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms as T
from torchvision.transforms import v2 as T
import h5py
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


from twod.dataloader.preprocessing import (
    preprocess_images,
    resize_or_crop_image_np,
)
from twod.augmentations.img_augm import apply_transform
from utils.plot import visualize_image

# this file defines a custom PyTorch Dataset class called KeypointDataset for
# loading and preprocessing image-keypoint pairs from a NumPy .npz file. It's designed
# for keypoint detection, so it returns the images, coupled with the corresponding keypoints.
# It also performs image augmentations


class KeypointDataset(Dataset):
    def __init__(
        self,
        images=None,
        keypoints=None,
        transform=None,
        filter=False,
        preprocessing=False,
        device="cpu",
        model_type="U-Net",
    ):
        """
        Args:
            images (list of np.array): List of grayscale images as numpy arrays.
            keypoints (list of lists): List of keypoint coordinates [[x1, y1, x2, y2], ...].
            transform (callable, optional): Image transformations.
            filter (bool, optional): If True, removes images with keypoints (0,0,0,0).
        """
        path = os.getenv("DATASET_PATH")
        total_files = 0
        target_path = Path(path)
        all_images = []
        all_keypoints = []
        skipped_files = []

        for root, _, files in os.walk(target_path):
            for filename in files:
                if filename.endswith(".h5"):
                    total_files += 1
                    filepath = Path(root) / filename

                    with h5py.File(filepath, "r") as f:
                        required = {"frames", "annotations"}

                        if not required.issubset(f.keys()):
                            missing = required - set(f.keys())
                            # Add incase some files were to miss some annotations
                            print(
                                f"SKIPPING: {filepath}\n"
                                f"  Missing: {sorted(missing)}\n"
                                f"  Available: {list(f.keys())}"
                            )

                            skipped_files.append((filepath, missing))
                            continue
                        frames = f["frames"][:]
                        annotations = f["annotations"][:]

                        if frames.ndim == 3 and frames.shape[0] != annotations.shape[0]:
                            frames = frames.transpose(2, 0, 1)

                        print("Frames before preprocessing:", frames.shape)
                        print("Annotations:", annotations.shape)

                        # It is expected that the images are 256,256, so they must be reshapes/cropped
                        frames, annotations = resize_or_crop_image_np(
                            frames,
                            annotations,
                            target_size=(256, 256),
                        )

                        all_images.append(frames)
                        all_keypoints.append(annotations)

        print(f"Found {total_files} .h5 files in path.")
        print(f"Loaded {len(all_images)} .h5 files.")
        print(f"Skipped {len(skipped_files)} .h5 files.")

        if not all_images:
            raise RuntimeError("No valid .h5 files were found.")

        images = np.concatenate(all_images, axis=0)
        keypoints = np.concatenate(all_keypoints, axis=0)

        print(f"Total images: {len(images)}")
        print(f"Images shape: {images.shape}")
        print(f"Keypoints shape: {keypoints.shape}")

        if filter:
            unannotated_indices = np.where(np.all(keypoints == 0, axis=1))[0]

            # Finding out-of-bounds keypoints
            out_of_bounds_indices = np.where(
                (keypoints[:, 0] < 0)
                | (keypoints[:, 0] > 256)
                | (keypoints[:, 1] < 0)
                | (keypoints[:, 1] > 256)
            )[0]

            # Combining both cases
            invalid_indices = np.unique(
                np.concatenate((unannotated_indices, out_of_bounds_indices))
            )

            print(f"Removing {len(invalid_indices)} invalid samples.")

            images = np.delete(images, invalid_indices, axis=0)
            keypoints = np.delete(keypoints, invalid_indices, axis=0)

        self.images = images
        self.keypoints = keypoints
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        img = preprocess_images(img, model_type=self.model_type, device=self.device)

        keypoint = self.keypoints[idx]

        keypoint = torch.tensor(keypoint, dtype=torch.float32).to(self.device)

        # Apply any transformations
        if self.transform:
            img, keypoint = apply_transform(img, keypoint, version=self.transform)

        img = img - img.min()
        img = img / img.max()

        return img, keypoint


if __name__ == "__main__":
    path = os.getenv("DATASET_PATH")
    with h5py.File(path, "r") as data:
        images = data["frames"][()]
        keypoints = data["annotations"][()]
    keypoint_dataset = KeypointDataset(images=images, keypoints=keypoints, filter=True)

    print(f"Number of images: {len(keypoint_dataset)}")
    print(f"Image shape: {keypoint_dataset[0][0].shape}")
    print(f"Keypoint shape: {keypoint_dataset[0][1].shape}")
    print(f"Keypoint coordinates: {keypoint_dataset[0][1]}")

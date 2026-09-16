import torch
from torch.utils.data import Dataset
import random
import os
import h5py
import numpy as np
from temporal_pipeline.augmentations.augm_3d import apply_transform
from temporal_pipeline.utils.plot import visualize_image, VolumeViewer
from temporal_pipeline.dataloader.preprocessing import (
    resize_or_crop_image_torch,
    apply_lut,
)


class RandomClipDataset(Dataset):
    """this dataset clas should select sequences of clip_length frames, so as to
    train the model to learn movement patterns in the data you feed it.

    dataset_path: path to the dataset of hdf5 files organized like this:
     -dataset
        -id1
            -hdf5 file with acquisitions and annotations in ["frames"] and
            ["annotations"] datasets
        -id2
            -...
        id3
        etc.
    clip_length: length of the clips to feed to the model
    transform: version of the augmentation to apply: see temporal_pipeline/augmentations/augm_3d.py
    """

    def __init__(
        self,
        dataset_path,
        clip_length=64,
        transform=None,
        smooth_annotations=False,
        smooth_window=3,
    ):

        # initialize
        self.clip_length = clip_length
        self.transform = transform
        self.smooth_annotations = smooth_annotations
        self.smooth_window = smooth_window

        # initialize the paths of the video files
        self.video_files = []

        # initialize video lengths
        self.video_lengths = []

        # extract the paths of each sequence
        for subfolder in os.listdir(dataset_path):
            if not subfolder.startswith("."):
                # TODO: see if this makes sense for a publication or if it's better just to have always the same videos
                # now they are being selected based on if one video has the right amount of frames
                for file in os.listdir(os.path.join(dataset_path, subfolder)):
                    if not file.startswith("."):
                        file_path = os.path.join(dataset_path, subfolder, file)

                        # exctract the number of frames in that acquisition
                        with h5py.File(file_path, "r") as h5_file:
                            T = h5_file["annotations"].shape[0]

                        # check if the acquisition has enough frames
                        if T >= self.clip_length:
                            # if so, then append the path and the length in two lists,
                            # to use them in the __getitem__ function
                            self.video_lengths.append(T)
                            self.video_files.append(file_path)

    def _apply_moving_average(self, annotations, window_size):
        """Apply moving average smoothing to annotations across time.

        annotations: [T, num_keypoints, 2] or [T, 3, 2] tensor
        window_size: size of the moving average window

        Returns: smoothed annotations of same shape
        """
        if window_size < 2:
            return annotations

        T = annotations.shape[0]
        smoothed = annotations.clone()

        # Apply 1D moving average for each coordinate independently
        for i in range(T):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(T, i + window_size // 2 + 1)
            smoothed[i] = annotations[start_idx:end_idx].mean(dim=0)

        return smoothed

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # extract the number of frames
        T = self.video_lengths[idx]

        # temporal cropping of the acquisition selection of a random index,
        # accounting for the length of the clip
        start = random.randint(0, T - self.clip_length - 1)
        end = start + self.clip_length

        # load acquisition and annotations
        with h5py.File(self.video_files[idx], "r") as h5_file:

            # clip the acquisitions and the annotations. I'm doing it like this so I don't load
            # the entire acquisition in RAM but just the clip I need
            clip_acq = h5_file["frames"][:, :, start:end]
            clip_ann = h5_file["annotations"][start:end]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()
        clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq - clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        # apply moving average smoothing to annotations if requested
        if self.smooth_annotations:
            clip_ann = self._apply_moving_average(clip_ann, self.smooth_window)

        # apply the image augmentation if requested
        if self.transform:
            clip_acq, clip_ann = apply_transform(
                clip_acq, clip_ann, version=self.transform
            )

        # unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann


class RandomClipDatasetForActivationMethod(Dataset):
    """this dataset class selects sequences of clip_length frames and returns
    activation-style regression targets for each landmark.

    dataset_path: path to the dataset of hdf5 files organized like this:
     -dataset
        -id1
            -hdf5 file with acquisitions and annotations in ["frames"] and
            ["annotations"] datasets
        -id2
            -...
        id3
        etc.
    clip_length: length of the clips to feed to the model
    transform: version of the augmentation to apply: see temporal_pipeline/augmentations/augm_3d.py
    activation_radius: radius in pixels used to build the EDT-style heatmaps
    """

    def __init__(
        self,
        dataset_path,
        clip_length=64,
        transform=None,
        smooth_annotations=False,
        smooth_window=3,
        activation_radius=100,
        peak_value=100.0,
        normalize_activation_maps=False,
    ):

        # initialize
        self.clip_length = clip_length
        self.transform = transform
        self.smooth_annotations = smooth_annotations
        self.smooth_window = smooth_window
        self.activation_radius = activation_radius
        self.peak_value = float(peak_value)
        self.normalize_activation_maps = normalize_activation_maps

        # initialize the paths of the video files
        self.video_files = []

        # initialize video lengths
        self.video_lengths = []

        # extract the paths of each sequence
        for subfolder in os.listdir(dataset_path):
            if not subfolder.startswith("."):
                # TODO: see if this makes sense for a publication or if it's better just to have always the same videos
                # now they are being selected based on if one video has the right amount of frames
                for file in os.listdir(os.path.join(dataset_path, subfolder)):
                    if not file.startswith("."):
                        file_path = os.path.join(dataset_path, subfolder, file)

                        # exctract the number of frames in that acquisition
                        with h5py.File(file_path, "r") as h5_file:
                            T = h5_file["annotations"].shape[0]

                        # check if the acquisition has enough frames
                        if T >= self.clip_length:
                            # if so, then append the path and the length in two lists,
                            # to use them in the __getitem__ function
                            self.video_lengths.append(T)
                            self.video_files.append(file_path)

    def _apply_moving_average(self, annotations, window_size):
        """Apply moving average smoothing to annotations across time.

        annotations: [T, num_keypoints, 2] tensor
        window_size: size of the moving average window

        Returns: smoothed annotations of same shape
        """
        if window_size < 2:
            return annotations

        T = annotations.shape[0]
        smoothed = annotations.clone()

        # Apply 1D moving average for each coordinate independently
        for i in range(T):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(T, i + window_size // 2 + 1)
            smoothed[i] = annotations[start_idx:end_idx].mean(dim=0)

        return smoothed

    def set_activation_radius(self, radius):
        self.activation_radius = int(radius)

    def set_peak_value(self, peak_value):
        self.peak_value = float(peak_value)

    def _annotations_to_activation_maps(self, annotations, image_size, radius):
        """Create per-landmark EDT-style activation maps from point annotations.

        annotations: [T, num_keypoints, 2]
        image_size: tuple (H, W)
        radius: maximum radius in pixels

        Returns: [num_keypoints, T, H, W]
        """
        T, num_keypoints, _ = annotations.shape
        H, W = image_size
        device = annotations.device
        dtype = annotations.dtype

        # Create a broadcastable coordinate grid [H, W, 1, 1]
        y_coords = torch.arange(H, device=device, dtype=dtype).view(H, 1, 1, 1)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, W, 1, 1)

        # Keypoint coordinates: [T, num_keypoints]
        # Keep the original time/keypoint order so each channel remains tied to the same landmark.
        kp_x = annotations[..., 0].reshape(1, 1, T, num_keypoints)
        kp_y = annotations[..., 1].reshape(1, 1, T, num_keypoints)

        dist = torch.sqrt((x_coords - kp_x) ** 2 + (y_coords - kp_y) ** 2)
        activation_maps = torch.clamp(radius - dist, min=0.0)

        # Scale so the center is always the peak value.
        activation_maps = activation_maps * (
            float(self.peak_value) / float(max(radius, 1))
        )

        # Output shape: [num_keypoints, T, H, W]
        return activation_maps.permute(3, 2, 0, 1).contiguous()

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # extract the number of frames
        T = self.video_lengths[idx]

        # temporal cropping of the acquisition selection of a random index,
        # accounting for the length of the clip
        start = random.randint(0, T - self.clip_length)
        end = start + self.clip_length

        # load acquisition and annotations
        with h5py.File(self.video_files[idx], "r") as h5_file:

            # clip the acquisitions and the annotations. I'm doing it like this so I don't load
            # the entire acquisition in RAM but just the clip I need
            clip_acq = h5_file["frames"][:, :, start:end]
            clip_ann = h5_file["annotations"][start:end]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()
        clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq - clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        # apply moving average smoothing to annotations if requested
        if self.smooth_annotations:
            clip_ann = self._apply_moving_average(clip_ann, self.smooth_window)

        # apply the image augmentation if requested
        if self.transform:
            clip_acq, clip_ann = apply_transform(
                clip_acq, clip_ann, version=self.transform
            )

        # convert coordinates to activation-style heatmaps
        clip_ann = self._annotations_to_activation_maps(
            clip_ann,
            image_size=clip_acq.shape[1:],
            radius=self.activation_radius,
        )

        # unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann


class ValidationDataset(Dataset):
    """
    Validation dataset that returns all possible non-overlapping clips of length clip_length from each video.
    Precomputes the mapping from dataset index to (video, start_frame) for efficient access.
    dataset_path: path to the dataset of hdf5 files organized like this:
     -dataset
        -id1
            -hdf5 file with acquisitions and annotations in ["frames"] and
            ["annotations"] datasets
        -id2
            -...
        id3
        etc.
    clip_length: length of the clips to feed to the model
    transform: to be added, apply augmentations
    return_heatmaps: if True, convert annotations to EDT heatmaps
    activation_radius: radius for heatmaps
    """

    def __init__(
        self,
        dataset_path,
        clip_length=64,
        return_heatmaps=False,
        activation_radius=5,
        peak_value=100.0,
        normalize_activation_maps=True,
    ):
        self.clip_length = clip_length
        self.return_heatmaps = return_heatmaps
        self.activation_radius = activation_radius
        self.peak_value = float(peak_value)
        self.normalize_activation_maps = normalize_activation_maps

        # list of paths of the files
        self.video_files = []

        self.clip_indices = []  # List of (video_idx, start_frame)

        # Gather all video files
        for subfolder in os.listdir(dataset_path):
            if not subfolder.startswith("."):
                for file in os.listdir(os.path.join(dataset_path, subfolder)):
                    if not file.startswith("."):
                        self.video_files.append(
                            os.path.join(dataset_path, subfolder, file)
                        )

        # Precompute all possible (video_idx, start_frame) pairs: I have to track the idx of the
        # video and the number of clips i can obtain from that video
        for vid_idx, video_path in enumerate(self.video_files):
            with h5py.File(video_path, "r") as h5_file:
                # extract numberof frame in the acquisition
                T = h5_file["annotations"].shape[0]
            # number of popossible clips
            n_clips = T // self.clip_length
            # track the video they belong to and the starting frame of the clip
            for j in range(n_clips):
                start = j * self.clip_length
                self.clip_indices.append((vid_idx, start))

    def set_activation_radius(self, radius):
        self.activation_radius = int(radius)

    def set_peak_value(self, peak_value):
        self.peak_value = float(peak_value)

    def _annotations_to_activation_maps(self, annotations, image_size, radius):
        """Create per-landmark EDT-style activation maps from point annotations.

        annotations: [T, num_keypoints, 2]
        image_size: tuple (H, W)
        radius: maximum radius in pixels

        Returns: [num_keypoints, T, H, W]
        """
        T, num_keypoints, _ = annotations.shape
        H, W = image_size
        device = annotations.device
        dtype = annotations.dtype

        # Create a broadcastable coordinate grid [H, W, 1, 1]
        y_coords = torch.arange(H, device=device, dtype=dtype).view(H, 1, 1, 1)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, W, 1, 1)

        # Keypoint coordinates: [T, num_keypoints]
        # Keep the original time/keypoint order so each channel remains tied to the same landmark.
        kp_x = annotations[..., 0].reshape(1, 1, T, num_keypoints)
        kp_y = annotations[..., 1].reshape(1, 1, T, num_keypoints)

        dist = torch.sqrt((x_coords - kp_x) ** 2 + (y_coords - kp_y) ** 2)
        activation_maps = torch.clamp(radius - dist, min=0.0)

        # Scale so the center is always the peak value.
        activation_maps = activation_maps * (
            float(self.peak_value) / float(max(radius, 1))
        )

        # Output shape: [num_keypoints, T, H, W]
        return activation_maps.permute(3, 2, 0, 1).contiguous()

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        vid_idx, start = self.clip_indices[idx]
        video_path = self.video_files[vid_idx]
        with h5py.File(video_path, "r") as h5_file:
            clip_acq = h5_file["frames"][:, :, start : start + self.clip_length]
            clip_ann = h5_file["annotations"][start : start + self.clip_length]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()
        clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq - clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        if self.return_heatmaps:
            clip_ann = self._annotations_to_activation_maps(
                clip_ann,
                image_size=clip_acq.shape[1:],
                radius=self.activation_radius,
            )

        # unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann


class SingleFileClipDataset(Dataset):
    """
    Dataset for testing/inference on a single file.
    Returns all possible non-overlapping clips of length clip_length from the video,
    starting from a specified frame.

    file_path: path to a single hdf5 file with acquisitions and annotations
    clip_length: length of the clips to feed to the model
    start_frame: frame index to start from (default 0)
    return_heatmaps: if True, convert annotations to EDT heatmaps
    activation_radius: radius for heatmaps
    peak_value: peak value for heatmaps
    normalize_activation_maps: whether to normalize heatmaps
    """

    def __init__(
        self,
        file_path,
        clip_length=32,
        start_frame=0,
        return_heatmaps=False,
        activation_radius=5,
        peak_value=100.0,
        normalize_activation_maps=True,
        load_from_annotations=False,  # use this when you need to compare the annotations to the predictions. Otherwise the images are loaded as new, and you loose the origin of the coordinates
    ):
        self.file_path = file_path
        self.clip_length = clip_length
        self.start_frame = int(start_frame)
        self.return_heatmaps = return_heatmaps
        self.activation_radius = activation_radius
        self.peak_value = float(peak_value)
        self.normalize_activation_maps = normalize_activation_maps
        self.load_from_annotations = load_from_annotations

        # Get total number of frames
        with h5py.File(file_path, "r") as h5_file:
            self.total_frames = h5_file["annotations"].shape[0]

        # Calculate number of clips from start_frame to end
        available_frames = self.total_frames - self.start_frame
        self.n_clips = available_frames // self.clip_length

    def set_activation_radius(self, radius):
        self.activation_radius = int(radius)

    def set_peak_value(self, peak_value):
        self.peak_value = float(peak_value)

    def _annotations_to_activation_maps(self, annotations, image_size, radius):
        """Create per-landmark EDT-style activation maps from point annotations.

        annotations: [T, num_keypoints, 2]
        image_size: tuple (H, W)
        radius: maximum radius in pixels

        Returns: [num_keypoints, T, H, W]
        """
        T, num_keypoints, _ = annotations.shape
        H, W = image_size
        device = annotations.device
        dtype = annotations.dtype

        # Create a broadcastable coordinate grid [H, W, 1, 1]
        y_coords = torch.arange(H, device=device, dtype=dtype).view(H, 1, 1, 1)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, W, 1, 1)

        # Keypoint coordinates: [T, num_keypoints]
        kp_x = annotations[..., 0].reshape(1, 1, T, num_keypoints)
        kp_y = annotations[..., 1].reshape(1, 1, T, num_keypoints)

        dist = torch.sqrt((x_coords - kp_x) ** 2 + (y_coords - kp_y) ** 2)
        activation_maps = torch.clamp(radius - dist, min=0.0)

        # Scale so the center is always the peak value.
        activation_maps = activation_maps * (
            float(self.peak_value) / float(max(radius, 1))
        )

        # Output shape: [num_keypoints, T, H, W]
        return activation_maps.permute(3, 2, 0, 1).contiguous()

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx):
        # Calculate the actual frame range for this clip
        clip_start = self.start_frame + idx * self.clip_length
        clip_end = clip_start + self.clip_length

        with h5py.File(self.file_path, "r") as h5_file:
            if self.load_from_annotations:
                clip_acq = h5_file["frames"][:, :, clip_start:clip_end]
            else:
                clip_acq = h5_file["tissue"]["data"][:, :, clip_start:clip_end]
                # preprocessing that I also did when annotating and saving the images for the training set
                clip_acq = clip_acq.transpose(2, 1, 0)[:, :, ::-1]
                clip_acq = apply_lut(clip_acq)
            clip_ann = h5_file["annotations"][clip_start:clip_end]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()

        if self.load_from_annotations:
            clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq - clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        if self.return_heatmaps:
            clip_ann = self._annotations_to_activation_maps(
                clip_ann,
                image_size=clip_acq.shape[1:],
                radius=self.activation_radius,
            )

        # unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    data_path = "data/final_reviewed_dataset/train"

    dataset = RandomClipDatasetForActivationMethod(data_path, clip_length=64)
    print(f"Number of clips: {len(dataset)}")
    print(f"Clip shape: {dataset[0][0].shape}")
    print(f"Keypoint shape: {dataset[0][1].shape}")

    print(dataset)

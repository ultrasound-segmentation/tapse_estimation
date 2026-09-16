import h5py
import os
import numpy as np

path = r"d:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed"
save_path = r"d:\mmissana\data\RV_PATIENTS\dataset_after_review"

test = [100, 111, 140, 149, 160, 170, 190, 198, 199, 920]
val = [106, 135, 141, 184, 990]


def resize_or_crop_image_np(imgs, keypoints, target_size=(256, 256)):
    """
    Resizes or crops a batch of images and updates corresponding keypoints.

    Args:
        imgs (np.ndarray): Array di shape (N, H, W) contenente N immagini in scala di grigi.
        keypoints (np.ndarray): Array di shape (N, K, 2) con le coordinate dei keypoints.
        target_size (tuple): Dimensione target (H, W).

    Returns:
        np.ndarray: Batch di immagini ridimensionate/croppate.
        np.ndarray: Keypoints aggiornati.
    """
    N, H, W = imgs.shape  # Numero di immagini e dimensioni originali
    new_imgs = np.zeros((N, target_size[0], target_size[1]), dtype=imgs.dtype)
    new_keypoints = keypoints.copy()

    for i in range(N):
        img = imgs[i]
        kp = keypoints[i]

        h, w = img.shape

        # Se l'immagine è più grande, la crop
        if h > target_size[0] and w > target_size[1]:
            crop_h = (h - target_size[0]) // 2
            crop_w = (w - target_size[1]) // 2
            img = img[
                crop_h : crop_h + target_size[0], crop_w : crop_w + target_size[1]
            ]
            kp[:, 0] -= crop_w
            kp[:, 1] -= crop_h

        # Se l'immagine è più alta che larga, crop + padding
        elif h > target_size[0] and w < target_size[1]:
            crop_h = (h - target_size[0]) // 2
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            img = img[crop_h : crop_h + target_size[0], :]
            img = np.pad(
                img, ((0, 0), (pad_w1, pad_w2)), mode="constant", constant_values=0
            )
            kp[:, 1] -= crop_h
            kp[:, 0] += pad_w1

        # Se è più larga che alta, padding + crop
        elif h < target_size[0] and w > target_size[1]:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1
            crop_w = (w - target_size[1]) // 2
            img = img[:, crop_w : crop_w + target_size[1]]
            img = np.pad(
                img, ((pad_h1, pad_h2), (0, 0)), mode="constant", constant_values=0
            )
            kp[:, 0] -= crop_w
            kp[:, 1] += pad_h1

        # Se l'immagine è più piccola, padding simmetrico
        else:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1  # Bilancia l'eventuale pixel extra
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            img = np.pad(
                img,
                ((pad_h1, pad_h2), (pad_w1, pad_w2)),
                mode="constant",
                constant_values=0,
            )
            kp[:, 0] += pad_w1
            kp[:, 1] += pad_h1

        new_imgs[i] = img
        new_keypoints[i] = kp

    return new_imgs, new_keypoints


for set in ["val", "test", "train"]:
    annotations = []
    images = []
    save = os.path.join(save_path, set)
    print(f"Processing {set} set in {save}")

    for subfolder in os.listdir(path):
        sub_path = os.path.join(path, subfolder)
        patient = int(sub_path.split(os.sep)[-1])

        if set == "val" and patient in val:
            print(f"patient: {patient}")
            for file in os.listdir(sub_path):
                if "interpolated" in file:
                    file_path = os.path.join(sub_path, file)
                    with h5py.File(file_path, "r") as h5_file:
                        an = h5_file["annotations"][()]
                        im = h5_file["frames"][()].transpose((2, 0, 1))

                        im, an = resize_or_crop_image_np(im, an, target_size=(256, 256))

                        images.append(im)  # Transpose to (num_frames, height, width)
                        annotations.append(an)

        elif set == "test" and patient in test:
            print(f"patient: {patient}")
            for file in os.listdir(sub_path):
                if "interpolated" in file:
                    file_path = os.path.join(sub_path, file)
                    with h5py.File(file_path, "r") as h5_file:
                        an = h5_file["annotations"][()]
                        im = h5_file["frames"][()].transpose((2, 0, 1))

                        im, an = resize_or_crop_image_np(im, an, target_size=(256, 256))

                        images.append(im)  # Transpose to (num_frames, height, width)
                        annotations.append(an)

        elif set == "train" and patient not in test and patient not in val:
            print(f"patient: {patient}")
            for file in os.listdir(sub_path):
                if "interpolated" in file:
                    file_path = os.path.join(sub_path, file)
                    with h5py.File(file_path, "r") as h5_file:
                        an = h5_file["annotations"][()]
                        im = h5_file["frames"][()].transpose((2, 0, 1))

                        im, an = resize_or_crop_image_np(im, an, target_size=(256, 256))

                        images.append(im)  # Transpose to (num_frames, height, width)
                        annotations.append(an)

    images_np = np.concatenate(images, axis=0)
    annotations_np = np.concatenate(annotations, axis=0)
    print(images_np.shape, annotations_np.shape)

    np.savez_compressed(save + ".npz", images=images_np, keypoints=annotations_np)

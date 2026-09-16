from PIL import Image
import os
import numpy as np
from cleanvision import Imagelab

# This code is thought to clean the dataset from near duplicate images.
# It can be used, for example, to avouìid useless annotation by running this before
# annotationg, so that most of the information is extracted while doing the least
# amount of manual work. IUf the dataset is limited, as in our case, this strategy
# is not beneficial, while instead it is for bigger datasets


def create_png_folder(npz_file, png_folder):
    os.makedirs(png_folder, exist_ok=True)
    images = np.load(npz_file)["images"]

    # Save the images as .png
    for i, img_array in enumerate(images):
        img = Image.fromarray(
            (img_array).astype("uint8")
        )  # Ensuring the range is [0, 255]
        img.save(os.path.join(png_folder, f"image_{i}.png"))


def cleanvision(png_folder):
    # List all image file paths in the directory
    files = sorted(
        os.listdir(png_folder), key=lambda x: (int(x.split("_")[1].split(".")[0]))
    )
    filepaths = [
        os.path.join(png_folder, f)
        for f in files
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Initialize CleanVision with filepaths directly
    imagelab = Imagelab(filepaths=filepaths)

    # Find issues
    issues = imagelab.find_issues()
    print(imagelab.issue_summary)

    sets = imagelab.info["near_duplicates"]["sets"]

    indices_to_remove = []
    for duplicate_pair in sets:
        for i, img_path in enumerate(duplicate_pair):
            if i != 0:
                # Extract the name of the file from the image
                img_name = img_path.split("/")[-1]
                indices_to_remove.append(int(img_name.split("_")[1].split(".")[0]))

    return sorted(indices_to_remove)


if __name__ == "__main__":
    npz_file = r"data/2d_focused_rv/dataset_256/train.npz"
    png_folder = r"D:\mmissana\data\2d_focused_rv/dataset_png/training"
    output_folder = r"data/2d_focused_rv/cleaned_dataset"

    indices_to_remove = cleanvision(png_folder)
    print(len(indices_to_remove))
    training = np.load(npz_file)
    images = training["images"]
    annotations = training["keypoints"]

    new_images = np.delete(images, indices_to_remove, axis=0)
    new_annotations = np.delete(annotations, indices_to_remove, axis=0)
    np.savez_compressed(
        os.path.join(output_folder, "training.npz"),
        images=new_images,
        keypoints=new_annotations,
    )

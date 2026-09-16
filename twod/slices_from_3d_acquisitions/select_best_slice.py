import os
import numpy as np  # For file I/O (npz loading/saving)
import cupy as cp  # For GPU acceleration
from utils.plot import VolumeViewer
from utils.extract_slices import crop_black_borders

# code that was used to select the best slice out of those outputted from
# "slice_volume" scripts. Useful since during the slicing you can choose to
# retain multiple views of the ventricle, from different angles.

folder_path = r"D:\mmissana\data\processed_imgs_2"
save_path = r"D:\mmissana\data\best_slices_2"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for subfolder in os.listdir(folder_path):
    folder = os.path.join(folder_path, subfolder)
    txt_path = os.path.join(folder, "best_slice.txt")

    # **Skip folders that already have best_slice.txt**
    if os.path.exists(txt_path):
        print(f"Skipping {subfolder}, annotation already exists.")
        continue

    file = "grid00.npz"
    npz_path = os.path.join(folder, file)

    # **Check if file exists before loading**
    if not os.path.exists(npz_path):
        print(f"Skipping {subfolder}, {file} not found.")
        continue

    # **Load using NumPy, then convert to CuPy**
    imgs_file = np.load(npz_path)
    print(imgs_file.files)
    imgs = cp.asarray(imgs_file["arr_0"])  # Move data to GPU

    save_folder = os.path.join(save_path, subfolder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    viewer = VolumeViewer(imgs)  # Convert back to NumPy for visualization
    viewer.show()

    if len(viewer.clicked_points) != 0:
        print(viewer.clicked_points)
        for point in viewer.clicked_points:
            best_slice = point[2]

            # **Filter only .npz files**
            npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]

            # **Initialize video array on GPU**
            video = cp.zeros(
                (len(npz_files), imgs.shape[0], imgs.shape[1]), dtype=cp.float32
            )
            for i, grid in enumerate(npz_files):
                imgs_file = np.load(os.path.join(folder, grid))
                video[i] = cp.asarray(
                    imgs_file["arr_0"][:, :, best_slice]
                )  # Move to GPU

            # crop the black borders
            video = video.transpose(1, 2, 0)
            video = crop_black_borders(video)
            # **Save back to CPU (CuPy -> NumPy) before writing to disk**
            np.savez_compressed(
                os.path.join(save_folder, f"video_{best_slice}.npz"),
                video=cp.asnumpy(video),
            )

            # **Save the best slice index**
            with open(txt_path, "w") as f:
                f.write(str(best_slice))
    else:
        print(f"No slice selected for {subfolder}")
        with open(txt_path, "w") as f:
            f.write("No slice selected")

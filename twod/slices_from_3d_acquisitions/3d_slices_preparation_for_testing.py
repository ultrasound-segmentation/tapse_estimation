folder = r"D:\mmissana\data\best_slices_2"
folder_data = r"D:\mmissana\data\4DRVQ_Jinyang\voxels"
new_folder = r"D:\mmissana\data\test_set_slices_from_3d"
import os
import numpy as np
from utils.plot import VolumeViewer
import h5py

# takes the slices that were obtained from the 3d acquisitions and outs them in the
# format the model needs to function properly

flip = False

for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    subfolder_data_path = os.path.join(folder_data, subfolder + ".h5")
    if subfolder == "190001":
        for file in os.listdir(subfolder_path):
            if "video_17" in file:
                file_path = os.path.join(subfolder_path, file)
                np_file = np.load(
                    file_path
                )  # Load the file to ensure it's a valid .npy file
                print(file_path)
                if flip:
                    volume = np_file["video"][
                        :, ::-1, :
                    ]  # Access the 'video' key in the .npy file
                else:
                    volume = np_file["video"]
                viewer = VolumeViewer(volume)
                viewer.show()  # Display the volume in the viewer

                with h5py.File(subfolder_data_path, "r") as f:
                    resolution = f["VolumeInfo"]["resolution"][()]
                    res = resolution[:2]
                    ecg_samples = f["ECG"]["samples"][()]
                    ecg_times = f["ECG"]["times"][()]
                    frame_times = f["FrameInfo"]["frameTimes"][()]
                    print(
                        f"Resolution: {res}, ECG Samples: {ecg_samples.shape}, ECG Times: {ecg_times.shape}, Frame Times: {frame_times.shape}"
                    )
                print("volume_shape", volume.shape, res)

                with h5py.File(os.path.join(new_folder, subfolder + ".h5"), "w") as f:
                    f.create_dataset("tissue/data", data=volume)
                    f.create_dataset("tissue/times", data=frame_times)
                    f.create_dataset("ecg/ecg_data", data=ecg_samples)
                    f.create_dataset("ecg/ecg_times", data=ecg_times)
                    f.create_dataset("tissue/pixelsize", data=res)

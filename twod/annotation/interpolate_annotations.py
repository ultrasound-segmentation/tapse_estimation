import os
import numpy as np
import h5py

"""code that was used to interpolate the annotations. To avoid wasting time, 
the acquisitions were annotated a frame every two apart from at end-systole and 
end-diastole. Then this script linearly interpolated the annotations to have a 
full dataset"""


def interpolate_annotations(annotation_path, save_path):
    annotations = np.load(annotation_path)["annotations"]
    for i, image in enumerate(annotations):
        for p, point in enumerate(image):
            if np.all(point == 0):
                annotations[i][p][0] = (
                    annotations[i - 1][p][0] + annotations[i + 1][p][0]
                ) / 2
                annotations[i][p][1] = (
                    annotations[i - 1][p][1] + annotations[i + 1][p][1]
                ) / 2

    images = np.load(annotation_path)["frames"]
    np.savez(save_path, annotations=annotations, frames=images)


def interpolate_annotations_h5(annotation_path, save_path):
    with h5py.File(annotation_path, "r") as h5_file:
        if "annotations" in h5_file:
            annotations = h5_file["annotations"][()]
            print(f"Loaded annotations with shape: {annotations.shape}")
        else:
            raise ValueError("Annotations not found in the HDF5 file.")

    for i, image in enumerate(annotations):
        for p, point in enumerate(image):
            if np.all(point == 0):
                if i == 0:
                    annotations[i][p][0] = (
                        annotations[i + 1][p][0] + annotations[-1][p][0]
                    ) / 2
                    annotations[i][p][1] = (
                        annotations[i + 1][p][1] + annotations[-1][p][1]
                    ) / 2
                elif i == len(annotations) - 1:
                    annotations[i][p][0] = (
                        annotations[0][p][0] + annotations[i - 1][p][0]
                    ) / 2
                    annotations[i][p][1] = (
                        annotations[0][p][1] + annotations[i - 1][p][1]
                    ) / 2
                else:
                    annotations[i][p][0] = (
                        annotations[i - 1][p][0] + annotations[i + 1][p][0]
                    ) / 2
                    annotations[i][p][1] = (
                        annotations[i - 1][p][1] + annotations[i + 1][p][1]
                    ) / 2

    with h5py.File(annotation_path, "r") as h5_file, h5py.File(
        save_path, "w"
    ) as new_h5_file:
        for key in h5_file.keys():
            if key != "annotations":
                h5_file.copy(key, new_h5_file)
        new_h5_file.create_dataset("annotations", data=annotations)


if __name__ == "__main__":
    # folder = r'D:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed'
    # for subfolder in os.listdir(folder):
    #     if subfolder == 'readme.txt':
    #         continue
    #     sub_path = os.path.join(folder, subfolder)
    #     flag_done = False
    #     flag_annotated = False
    #     for file in os.listdir(sub_path):
    #         # if 'interpolated' in file:
    #         #     flag_done = True
    #         if 'corrected' in file:
    #             flag_annotated = True
    #     if not flag_done and flag_annotated:
    #         for file in os.listdir(sub_path):
    #             if 'corrected' not in file:
    #                 continue
    #             else:
    #                 file_path = os.path.join(sub_path, file)
    #                 save_path = os.path.join(sub_path, file.replace('_corrected.h5', '_interpolated.h5'))
    #                 interpolate_annotations_h5(file_path, save_path)
    #                 print(f'Interpolated annotations for {file_path} and saved to {save_path}')
    interpolate_annotations_h5(
        r"d:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed\190\P429G08O_corrected.h5",
        r"d:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed\190\P429G08O_interpolated.h5",
    )

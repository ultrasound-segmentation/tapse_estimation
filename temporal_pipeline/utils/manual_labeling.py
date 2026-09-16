import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2
import shutil

""" how to annotate the images:
1. run the code
2. click on the image to select the points
3. press enter to reselect the points
4. press c to close the window
5. press left or right to move to the next or previous image
6. press up or down to jump 10 frames forward or backward

very important: selexct always the left tricuspid annulus first, then the right tricuspid annulus, then the apex"""


def apply_clahe_to_stack(image_stack, clip_limit=10.0, tile_grid_size=(2, 2)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each image in a 3D numpy array.

    Parameters:
    - image_stack: numpy array of shape (H, W, N), where N is the number of images.
    - clip_limit: CLAHE clip limit (default: 2.0).
    - tile_grid_size: Grid size for CLAHE (default: (8,8)).

    Returns:
    - numpy array of the same shape with CLAHE applied to each image.
    """
    h, w, n = image_stack.shape  # Get dimensions
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    equalized_stack = np.zeros_like(
        image_stack
    )  # Create empty array with the same shape

    for i in range(n):
        img = image_stack[:, :, i]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )  # Normalize to 8-bit
        equalized_stack[:, :, i] = clahe.apply(img)

    return equalized_stack


def press(event):
    global user_input
    user_input = event.key


def annotate_2_points_2d_video(file_path, num_landmarks=2):
    """
    Displays frames from a 2D ultrasound video stored in an `.npz` file and allows the user
    to manually annotate two landmark points per frame. The annotated points are then saved
    as a new `.npz` file.

    Parameters:
    -----------
    file_path : str
        Path to the `.npz` file containing the ultrasound frames.
        The `.npz` file should contain a numpy array with key 'arr_0'
        representing a grayscale video of shape (height, width, num_frames).

    Functionality:
    --------------
    - Iterates through the frames of the ultrasound video.
    - Allows the user to manually select **two landmark points** per frame using mouse clicks.
    - Provides keyboard controls for navigation:
        - `Enter` → Select and save 2 points for the current frame.
        - `Left/Right arrow` → Move to the previous/next frame.
        - `Up/Down arrow` → Jump 10 frames forward/backward.
        - `C` → Save and close the annotation session.
    - Saves the annotated points as a new `.npz` file with the same name,
      appending `_annotations.npz`.

    Output:
    -------
    - A new `.npz` file is created with two arrays:
        - `'arr_0'`: Original video frames.
        - `'annotations'`: An array of shape (num_frames, 2, 2) storing
          the (x, y) coordinates of the two selected landmarks for each frame.
    """
    # Load the numpy array from `.npz`
    data = np.load(file_path)
    frames = data["video"]  # Shape: (dim1, dim2, frame_index)

    # Output directory to save annotations
    save_path = file_path.replace(".npz", "_annotations.npz")

    # Initialize landmark coordinates (2 landmarks per frame)
    num_frames = frames.shape[2]
    ref_coord = np.zeros(
        (num_frames, num_landmarks, 2)
    )  # Shape: (frames, landmarks, (x, y))

    plt.ion()  # Enable interactive mode

    idx = 0  # Start at first frame
    idx_max = num_frames - 1  # Last frame index

    while True:
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap="gray")  # Display current frame
        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        # Plot previous frame's landmarks in blue, current frame's in red
        prev_idx = idx_max if idx == 0 else idx - 1
        for j in range(num_landmarks):
            plt.scatter(
                ref_coord[idx][j][0], ref_coord[idx][j][1], color="r", marker="*", s=100
            )  # Current frame
            plt.scatter(
                ref_coord[prev_idx][j][0],
                ref_coord[prev_idx][j][1],
                color="b",
                marker="*",
                s=100,
            )  # Previous frame

        plt.gcf().canvas.mpl_connect("key_press_event", press)

        print(f"Frame {idx + 1}/{num_frames}")
        print("Enter = correct landmarks, c = close\n")

        while not plt.waitforbuttonpress(100000):
            pass

        if user_input == "enter":
            coordinates = plt.ginput(
                n=3, timeout=0, show_clicks=True
            )  # Only 2 landmarks
            ref_coord[idx] = np.array(coordinates)
        elif user_input == "c":
            print("Saving annotations and closing...")
            np.savez(save_path, frames=frames, annotations=ref_coord)
            break
        elif user_input == "left":
            idx = idx_max if idx == 0 else idx - 1
        elif user_input == "right":
            idx = 0 if idx == idx_max else idx + 1
        elif user_input == "down":
            idx = max(0, idx - 10)
        elif user_input == "up":
            idx = min(idx_max, idx + 10)

    plt.ioff()
    plt.close()


import h5py
import numpy as np
import matplotlib.pyplot as plt


def annotate_single_point_per_video_h5(file_path, save_path, num_landmarks=3, LUT=None):
    """
    Displays frames from a 2D ultrasound video stored in an `h5` file and allows the user
    to manually annotate one landmark point per frame at a time. The annotated points are then
    saved as a new `.h5` file along with the original data.
    """

    # Open the original HDF5 file and read data
    with h5py.File(file_path, "r") as h5_file:
        frames = h5_file["tissue"]["data"][()]
        frames = frames.transpose(1, 0, 2)
        frames = frames[:, ::-1, :]
        frames = frames.astype(np.uint8)

        # for i in range(frames.shape[2]):
        #     for j in range(frames.shape[0]):
        #         for k in range(frames.shape[1]):
        #             frames[j, k, i] = LUT[frames[j, k, i]]

        frames = np.take(LUT, frames)

    # Initialize landmark coordinates (one landmark per frame for each point)
    num_frames = frames.shape[2]
    ref_coord = np.zeros(
        (num_frames, num_landmarks, 2)
    )  # Shape: (frames, landmarks, (x, y))

    plt.ion()
    idx = 0  # Start at first frame
    idx_max = num_frames - 1
    current_landmark = 0

    while True:
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap="gray")
        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        for j in range(num_landmarks):
            if ref_coord[idx][j][0] != 0 and ref_coord[idx][j][1] != 0:
                color = ["r", "g", "b"][j]
                plt.scatter(
                    ref_coord[idx][j][0],
                    ref_coord[idx][j][1],
                    color=color,
                    marker="*",
                    s=100,
                )

        plt.gcf().canvas.mpl_connect("key_press_event", press)

        print(f"Frame {idx + 1}/{num_frames}")
        print("Enter = select landmark, C = close, Arrow keys = navigate frames")

        while not plt.waitforbuttonpress(100000):
            pass

        if user_input == "enter":
            coordinates = plt.ginput(n=1, timeout=0, show_clicks=True)
            if len(coordinates) == 0:
                print("No point selected. Try again.")
                continue
            ref_coord[idx][current_landmark] = np.array(coordinates)
            print(f"Landmark {current_landmark + 1} saved at: {coordinates}")

        elif user_input == "c":
            print("Saving annotations and closing...")
            break

        elif user_input == "left":
            idx = idx_max if idx == 0 else idx - 1
        elif user_input == "right":
            idx = 0 if idx == idx_max else idx + 1
        elif user_input == "down":
            idx = max(0, idx - 10)
        elif user_input == "up":
            idx = min(idx_max, idx + 10)

        if user_input == "g":
            current_landmark += 1
            if current_landmark == num_landmarks:
                print("All landmarks annotated. Closing session.")
                break

    plt.ioff()
    plt.close()

    # Save the new HDF5 file with annotations
    with h5py.File(file_path, "r") as h5_file, h5py.File(save_path, "w") as new_h5_file:
        # Copy all original data
        for key in h5_file.keys():
            h5_file.copy(key, new_h5_file)

        # Store frames and annotations
        new_h5_file.create_dataset("frames", data=frames)
        new_h5_file.create_dataset("annotations", data=ref_coord)

    print(f"Annotations saved successfully in {save_path}")


def reannotate_points_in_h5(file_path, save_path, num_landmarks=3, LUT=None):
    """
    Loads an HDF5 file containing ultrasound frames and previous annotations,
    allowing the user to re-annotate points interactively.
    """

    # Open the HDF5 file and load data
    with h5py.File(file_path, "r") as h5_file:
        frames = h5_file["frames"][()]
        print("frames shape", frames.shape)
        if "annotations" in h5_file:
            ref_coord = h5_file["annotations"][()]
            print("annotations shape", ref_coord.shape)
        else:
            num_frames = frames.shape[2]
            ref_coord = np.zeros((num_frames, num_landmarks, 2))

    plt.ion()
    idx = 0  # Start at first frame
    idx_max = frames.shape[2] - 1
    current_landmark = 0

    while True:
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap="gray")
        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        for j in range(num_landmarks):
            if ref_coord[idx][j][0] != 0 and ref_coord[idx][j][1] != 0:
                color = ["r", "g", "b"][j]
                plt.scatter(
                    ref_coord[idx][j][0],
                    ref_coord[idx][j][1],
                    color=color,
                    marker="*",
                    s=100,
                )

        plt.gcf().canvas.mpl_connect("key_press_event", press)

        print(f"Frame {idx + 1}/{frames.shape[2]}")
        print("Enter = select landmark, C = close, Arrow keys = navigate frames")

        while not plt.waitforbuttonpress(100000):
            pass

        if user_input == "enter":
            coordinates = plt.ginput(n=1, timeout=0, show_clicks=True)
            if len(coordinates) == 0:
                print("No point selected. Try again.")
                continue
            ref_coord[idx][current_landmark] = np.array(coordinates)
            print(f"Landmark {current_landmark + 1} saved at: {coordinates}")

        elif user_input == "c":
            print("Saving annotations and closing...")
            break

        elif user_input == "left":
            idx = idx_max if idx == 0 else idx - 1
        elif user_input == "right":
            idx = 0 if idx == idx_max else idx + 1
        elif user_input == "down":
            idx = max(0, idx - 10)
        elif user_input == "up":
            idx = min(idx_max, idx + 10)

        elif user_input == "g":
            current_landmark += 1
            # if current_landmark == num_landmarks:
            #     print("All landmarks annotated. Closing session.")
            #     break

        elif user_input == "1":
            for i in range(1, len(ref_coord), 2):  # loop through odd indices only
                ref_coord[i][current_landmark] = np.zeros((2,))
            print(ref_coord[idx][current_landmark])

    plt.ioff()
    plt.close()

    # Save the updated HDF5 file with modified annotations
    with h5py.File(file_path, "r") as h5_file, h5py.File(save_path, "w") as new_h5_file:
        for key in h5_file.keys():
            if key != "annotations":
                h5_file.copy(key, new_h5_file)
        new_h5_file.create_dataset("annotations", data=ref_coord)

    print(f"Updated annotations saved successfully in {save_path}")


def find_already_annotated(
    unannotated_file, annotation_folder, new_annotation_path, LUT
):
    """
    given a file, check if the same exact vieedo has already been annotated, if so, uses the same annotation
    """

    # Open the original HDF5 file and read data
    with h5py.File(unannotated_file, "r") as h5_file:
        frames = h5_file["tissue"]["data"][()]
        frames = frames.transpose(1, 0, 2)
        frames = frames[:, ::-1, :]
        frames = frames.astype(np.uint8)

        for i in range(frames.shape[2]):
            for j in range(frames.shape[0]):
                for k in range(frames.shape[1]):
                    frames[j, k, i] = LUT[frames[j, k, i]]

    for subfolder in os.listdir(annotation_folder):
        if subfolder != "readme.txt":
            folder = os.path.join(annotation_folder, subfolder)
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                print("checking", file)
                with h5py.File(file_path, "r") as h5_file_2:
                    frames_annotated = h5_file_2["frames"][()]
                    if np.array_equal(frames, frames_annotated):
                        print(f"Found matching annotation in {file_path}")
                        shutil.copy(file_path, new_annotation_path)


if __name__ == "__main__":
    # Default lookup table for GE
    LUT = [
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        5.416839879697600921e-109,
        1.472449741265426833e-108,
        1.472449741265426833e-108,
        4.002533375001032603e-108,
        4.002533375001032603e-108,
        4.002533375001032603e-108,
        1.088001374106615945e-107,
        2.957494364572485672e-107,
        2.957494364572485672e-107,
        8.039303188987418261e-107,
        8.039303188987418261e-107,
        2.185309177209735204e-106,
        2.185309177209735204e-106,
        5.940286225974011103e-106,
        1.614737210391071792e-105,
        4.389310816742700680e-105,
        4.389310816742700680e-105,
        1.193138383261041334e-104,
        3.243286386055492711e-104,
        8.816166447703253065e-104,
        2.396482505146208087e-103,
        6.514314845958948177e-103,
        6.514314845958948177e-103,
        1.770774367063119509e-102,
        4.813463784288744541e-102,
        1.308435113677780416e-101,
        3.556695393228056346e-101,
        9.668100456775821155e-101,
        2.628062178737031233e-100,
        7.143813664521359541e-100,
        1.941889887016583139e-99,
        1.434873331311163542e-98,
        3.900390102643631095e-98,
        1.060235953991769413e-97,
        2.882020127614766811e-97,
        7.834142942148438469e-97,
        5.788692168657643875e-96,
        1.573529673260525569e-95,
        1.162689902905403913e-94,
        3.160518835200571108e-94,
        2.335325097502397273e-93,
        6.348071776085115550e-93,
        1.725584815468590525e-92,
        4.690625847353124236e-92,
        3.465929752518635897e-91,
        9.421373864986962783e-91,
        6.961506001738774640e-90,
        5.143895837989021708e-89,
        1.398255858389167097e-88,
        1.033179097829598747e-87,
        2.808471967173908365e-87,
        7.634218314105463291e-87,
        2.075195691772212795e-86,
        1.533373738276407791e-85,
        1.133018457275138174e-84,
        3.079863483719709203e-84,
        2.275728405825291715e-83,
        6.186071172062962685e-83,
        4.570922692235090021e-82,
        1.242505609359374486e-81,
        9.180943650792429284e-81,
        6.783850767682641964e-80,
        5.012625388918081340e-79,
        3.703857020163976607e-78,
        1.006812723312220456e-77,
        7.439395693671140224e-77,
        2.022237412882273316e-76,
        1.494242568914349970e-75,
        4.061772422389840816e-75,
        3.001266429012797238e-74,
        8.158287996349656898e-74,
        6.028204767626013778e-73,
        4.454274320382982031e-72,
        3.291288283333605643e-71,
        2.431951376330516621e-70,
        6.610729233975208078e-70,
        4.884704916468365468e-69,
        1.327800461182051828e-68,
        9.811192095860170043e-68,
        2.666958518969771289e-67,
        1.970630611018864524e-66,
        5.356729380537723324e-66,
        3.958117389958326745e-65,
        1.075927857603146641e-64,
        7.950091298231918329e-64,
        2.161058871057420474e-63,
        1.596818523133501619e-62,
        4.340602774780607539e-62,
        1.179898164724503386e-61,
        8.718333730174673753e-61,
        2.369888815317537995e-60,
        1.751124140460958445e-59,
        4.760048930390988699e-59,
        3.517226858031385466e-58,
        9.560813854754816412e-58,
        7.064538992421674337e-57,
        1.920340796954020926e-56,
        1.418950587775845334e-55,
        3.857107598232361988e-55,
        2.850038442295058349e-54,
        7.747207708100381429e-54,
        2.105909393422711263e-53,
        5.724455236522166123e-53,
        1.149787570183009674e-51,
        3.125446658616554427e-51,
        2.309410069473307261e-50,
        6.277627426309632450e-50,
        4.638574122118750361e-49,
        1.260895174611576965e-48,
        9.316825180075900245e-48,
        2.532575658592998838e-47,
        1.871334361612990863e-46,
        5.086814190143600111e-46,
        1.382739457781496258e-45,
        3.758675541580754053e-45,
        1.021713942375242704e-44,
        7.549501637270265095e-44,
        5.578369111675877908e-43,
        4.121888230671500676e-42,
        1.120445387637354474e-41,
        8.279033825040511196e-41,
        2.250474720380540399e-40,
        1.662888395811708180e-39,
        4.520199309090378025e-39,
        3.340000627321637406e-38,
        9.079063012290218145e-38,
        2.467945200574313964e-37,
        6.708570592353872135e-37,
        4.957000445053917868e-36,
        1.347452423345346453e-35,
        9.956401546738817165e-35,
        2.706430540134165694e-34,
        7.356840957233302509e-34,
        1.999796708891053013e-33,
        5.436011054390753364e-33,
        4.016699063530042531e-32,
        1.091852007478217846e-31,
        2.967961471294569291e-31,
        8.067755734986599734e-31,
        2.193043381086032512e-30,
        1.620452057023324203e-29,
        4.404845380495582738e-29,
        1.197361115497291082e-28,
        3.254764962359738827e-28,
        8.847368453087666257e-28,
        2.404964089571001357e-27,
        6.537370182777404428e-27,
        1.777041457375380675e-26,
        4.830499502001877147e-26,
        1.313065901867216908e-25,
        3.569283180614843283e-25,
        2.637363365453269652e-24,
        7.169096911355215233e-24,
        1.948762586059874693e-23,
        5.297285925667413630e-23,
        1.439951607189358438e-22,
        3.914194287683230482e-22,
        1.063988320526752206e-21,
        2.892220117380528479e-21,
        7.861869388979176538e-21,
        7.861869388979176538e-21,
        2.137077669778051444e-20,
        5.809179395763277227e-20,
        5.809179395763277227e-20,
        4.292435244404691073e-19,
        1.166804872470243210e-18,
        3.171704482193336260e-18,
        8.621586659188251145e-18,
        2.343590234815634879e-17,
        6.370538748653406648e-17,
        6.370538748653406648e-17,
        1.731691971795878072e-16,
        4.707226819421149006e-16,
        1.279556912566757742e-15,
        3.478196303909377189e-15,
        3.478196303909377189e-15,
        9.454717808730274267e-15,
        6.986144028826667324e-14,
        1.899030836455719177e-13,
        5.162101014420962350e-13,
        6.986144028826667324e-14,
        1.899030836455719177e-13,
        5.162101014420962350e-13,
        1.403204538417050386e-12,
        1.403204538417050386e-12,
        1.403204538417050386e-12,
        3.814305398390331470e-12,
        1.036835705263767578e-11,
        1.036835705263767578e-11,
        2.818411656716017816e-11,
        7.661237191568304403e-11,
        7.661237191568304403e-11,
        2.082540184135473035e-10,
        1.538800624885437650e-9,
        1.538800624885437650e-9,
        4.182893776247508078e-9,
        1.137028414234803735e-8,
        1.137028414234803735e-8,
        3.090763676856071573e-8,
        8.401566738859123968e-8,
        8.401566738859123968e-8,
        2.283782619682667372e-7,
        2.283782619682667372e-7,
        6.207964795233989027e-7,
        6.207964795233989027e-7,
        1.687499789459803066e-6,
        1.687499789459803066e-6,
        4.587100013217047569e-6,
        4.587100013217047569e-6,
        3.389433932906628784e-5,
        3.389433932906628784e-5,
        9.213436668582564723e-5,
        9.213436668582564723e-5,
        2.504471747386622835e-4,
        2.504471747386622835e-4,
        2.504471747386622835e-4,
        2.504471747386622835e-4,
        2.504471747386622835e-4,
        6.807860040810129361e-4,
        6.807860040810129361e-4,
        1.850568223962662786e-3,
        1.850568223962662786e-3,
        1.850568223962662786e-3,
        5.030365975521435516e-3,
        5.030365975521435516e-3,
        5.030365975521435516e-3,
        1.367395242175857561e-2,
        1.367395242175857561e-2,
        1.367395242175857561e-2,
        3.716965639127988580e-2,
        3.716965639127988580e-2,
        3.716965639127988580e-2,
        2.746486762531433201e-1,
        2.746486762531433201e-1,
        2.746486762531433201e-1,
    ]

    LUT = np.log(np.array(LUT)).astype(int) + 249

    folder_path = r"D:\mmissana\data\RV_PATIENTS\RV_patients_converted"
    save_folder = r"D:\mmissana\data\RV_PATIENTS\RV_patients_annotated"
    # for subfolder in os.listdir(folder_path):
    #     folder = os.path.join(folder_path, subfolder)
    #     folder_save = os.path.join(save_folder, subfolder)
    #     for file in os.listdir(folder):
    #         os.makedirs(folder_save, exist_ok=True)
    #         file_path = os.path.join(folder, file)
    #         new_file_path = os.path.join(folder_save, file)

    #     # **Check if file exists before loading**
    #     if not os.path.exists(file_path) or os.path.exists(new_file_path):
    #         print(f"Skipping {subfolder}, video_best_slice.npz not found or annotation has already been done.")
    #         continue

    #     print(f"Processing {subfolder}...")
    #     print(file_path)
    #     # **Load using NumPy**
    #     annotate_single_point_per_video_h5(file_path, new_file_path, num_landmarks=3, LUT=LUT)
    # find_already_annotated(file_path, r'data/2d_focused_rv/RV_focused_TEE_images_annotated', new_file_path, LUT)

    reannotate_points_in_h5(
        r"data/RV_PATIENTS/RV_patients_annotated/_9101815/P42A93R4.h5",
        r"data/RV_PATIENTS/RV_patients_annotated/_9101815/P42A93R4_corrected.h5",
        num_landmarks=3,
        LUT=LUT,
    )

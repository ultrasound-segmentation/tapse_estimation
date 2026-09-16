import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn as nn

# functions to perform preprocessing and resize of images


def resize_or_crop_image(img, keypoints, target_size=(256, 256)):
    """
    Resizes or crops the input image to match the target size.
    If the image is larger, it is cropped centrally.
    If the image is smaller, it is padded symmetrically.
    """
    h, w = img.shape[-2:]
    # print(img.shape)

    # If image is larger, crop it centrally
    if h > target_size[0] and w > target_size[1]:
        crop_h = max(0, h - target_size[0]) // 2
        additional_h = abs(h - target_size[0]) % 2
        crop_w = max(0, w - target_size[1]) // 2
        additional_w = abs(w - target_size[1]) % 2
        img = img[
            crop_h : h - crop_h - additional_h, crop_w : w - crop_w - additional_w
        ]
        keypoints[:, 0] -= crop_w
        keypoints[:, 1] -= crop_h

    elif h > target_size[0] and w < target_size[1]:
        crop_h = max(0, h - target_size[0]) // 2
        additional_h = abs(h - target_size[0]) % 2
        pad_w = (target_size[1] - w) // 2
        keypoints[:, 1] -= crop_h
        keypoints[:, 0] += pad_w

        img = img[crop_h : h - crop_h - additional_h]
        img = F.pad(img, (pad_w, 0, pad_w + (target_size[1] - w) % 2, 0), fill=0)

    elif h < target_size[0] and w > target_size[1]:
        pad_h = (target_size[0] - h) // 2
        crop_w = max(0, w - target_size[1]) // 2
        additional_w = abs(w - target_size[1]) % 2
        keypoints[:, 0] -= crop_w
        keypoints[:, 1] += pad_h

        img = img[:, crop_w : w - crop_w - additional_w]
        img = F.pad(img, (0, pad_h, 0, pad_h + (target_size[0] - h) % 2), fill=0)

    # If image is smaller, pad it
    else:
        pad_h = (target_size[0] - h) // 2
        pad_w = (target_size[1] - w) // 2
        keypoints[:, 0] += pad_w
        keypoints[:, 1] += pad_h

        img = F.pad(
            img,
            (
                pad_w,
                pad_h,
                pad_w + (target_size[1] - w) % 2,
                pad_h + (target_size[0] - h) % 2,
            ),
            fill=0,
        )

    print(img.shape, keypoints.shape)
    return img, keypoints


def create_numpy_dataset(
    folder=r"D:\mmissana\data\best_slices",
    save_path=r"D:\mmissana\data\dataset_separated_by_video_256",
):
    keypoints_np = {}
    images_np = {}

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for patient in os.listdir(folder):
        if patient == "readme.txt":
            continue

        image_list = []
        keypoint_list = []
        patient_folder = os.path.join(folder, patient)

        for doc in os.listdir(patient_folder):
            if doc == "video_best_slice_annotations.npz":
                file_path = os.path.join(patient_folder, doc)
                file = np.load(file_path)

                images = file["frames"].transpose(
                    2, 0, 1
                )  # (H, W, frames) → (frames, H, W)
                keypoints = file["annotations"]

                print(patient, "number of images:", len(images))

                for i in range(len(images)):
                    image_processed, keypoints_processed = resize_or_crop_image(
                        torch.tensor(images[i], dtype=torch.float32),
                        torch.tensor(keypoints[i], dtype=torch.float32),
                    )
                    image_list.append(image_processed.numpy())
                    keypoint_list.append(keypoints_processed.numpy())

                keypoints_np[patient] = np.array(keypoint_list)
                images_np[patient] = np.array(image_list)

                patient_save_path = os.path.join(save_path, f"{patient}.npz")
                np.savez_compressed(
                    patient_save_path,
                    images=images_np[patient],
                    keypoints=keypoints_np[patient],
                )

    print("Dataset creation complete.")


def preprocess_images(images_array, model_type="EchoCoder", device="cpu"):
    """
    images_array: NumPy array con shape (N, 256, 256), valori tra 0-255 o normalizzati 0-1
    model_type: Specifica il modello per la corretta formattazione dell'input
    """

    # Assicuriamoci che il tipo di dato sia float32 e normalizziamo se necessario
    if images_array.max() > 1:
        images_array = images_array.astype(np.float32) / 255.0

    # Modifichiamo il formato in base al modello
    if model_type == "U-Net":
        # Aggiungiamo le dimensioni richieste per PyTorch: (N, 1, 256, 256)
        images_tensor = (
            torch.tensor(images_array).unsqueeze(1).to(device)
        )  # Shape diventa (N, 1, 256, 256)
        # images_tensor = images_tensor.repeat(1, 3, 1, 1)  # Shape diventa (N, 3, 256, 256)
    elif model_type == "improved_unet":
        images_tensor = torch.tensor(images_array)
    elif model_type == "monai_U-Net":
        images_tensor = (
            torch.tensor(images_array).unsqueeze(1).to(device)
        )  # Shape diventa (N, 1, 256, 256)
    elif model_type == "swinunetr":
        images_tensor = torch.tensor(images_array).unsqueeze(1).to(device)
    elif "ResNet" or "resnext" in model_type:
        images_tensor = (
            torch.tensor(images_array).unsqueeze(1).to(device)
        )  # Shape diventa (N, 1, 256, 256)
    return images_tensor


"""
Imported from https://github.com/mailys-hau/echovox
"""


def apply_lut(array):
    """function cthat applies the lookup table to the input array"""
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

    array = array.astype(np.uint8)
    array = np.take(LUT, array)

    return array


def resize_or_crop_image_np(imgs, keypoints, target_size=(256, 256)):
    """
    Resizes or crops a batch of images and updates corresponding keypoints.

    Args:
        imgs (np.ndarray): Array of shape (N, H, W) containing N grayscale images.
        keypoints (np.ndarray): Array of shape (N, K, 2) with keypoint coordinates.
        target_size (tuple): Target size (H, W).

    Returns:
        np.ndarray: Batch of resized/cropped images.
        np.ndarray: Updated keypoints.
    """
    N, H, W = imgs.shape  # Number of images and original dimensions
    new_imgs = np.zeros((N, target_size[0], target_size[1]), dtype=imgs.dtype)
    new_keypoints = keypoints.copy()

    for i in range(N):
        img = imgs[i]
        kp = keypoints[i]

        h, w = img.shape

        # If the image is larger, crop it
        if h > target_size[0] and w > target_size[1]:
            crop_h = (h - target_size[0]) // 2
            crop_w = (w - target_size[1]) // 2
            img = img[
                crop_h : crop_h + target_size[0], crop_w : crop_w + target_size[1]
            ]
            kp[:, 0] -= crop_w
            kp[:, 1] -= crop_h

        # If the image is taller than it is wide, crop + padding
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

        # If it is wider than it is tall, padding + crop
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

        # If the image is smaller, symmetric padding
        else:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1  # Balances any extra pixel
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


def resize_or_crop_image_np_nokeypoints(imgs, target_size=(256, 256)):
    """
    Resizes or crops a batch of images.

    Args:
        imgs (np.ndarray): Array of shape (N, H, W) containing N grayscale images.
        target_size (tuple): Target size (H, W).

    Returns:
        np.ndarray: Batch of resized/cropped images.
    """
    N, H, W = imgs.shape  # Number of images and original dimensions
    new_imgs = np.zeros((N, target_size[0], target_size[1]), dtype=imgs.dtype)

    for i in range(N):
        img = imgs[i]

        h, w = img.shape

        # If the image is larger, crop it
        if h > target_size[0] and w > target_size[1]:
            crop_h = (h - target_size[0]) // 2
            crop_w = (w - target_size[1]) // 2
            img = img[
                crop_h : crop_h + target_size[0], crop_w : crop_w + target_size[1]
            ]

        # If the image is taller than it is wide, crop + padding
        elif h > target_size[0] and w < target_size[1]:
            crop_h = (h - target_size[0]) // 2
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            img = img[crop_h : crop_h + target_size[0], :]
            img = np.pad(
                img, ((0, 0), (pad_w1, pad_w2)), mode="constant", constant_values=0
            )

        # If it is wider than it is tall, padding + crop
        elif h < target_size[0] and w > target_size[1]:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1
            crop_w = (w - target_size[1]) // 2
            img = img[:, crop_w : crop_w + target_size[1]]
            img = np.pad(
                img, ((pad_h1, pad_h2), (0, 0)), mode="constant", constant_values=0
            )

        # If the image is smaller, symmetric padding
        else:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1  # Balances any extra pixel
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            img = np.pad(
                img,
                ((pad_h1, pad_h2), (pad_w1, pad_w2)),
                mode="constant",
                constant_values=0,
            )

        new_imgs[i] = img

    return new_imgs


def resize_or_crop_image_torch(imgs, keypoints, target_size=(256, 256)):
    """
    Ridimensiona o croppa un batch di immagini torch e aggiorna i keypoints corrispondenti.

    Args:
        imgs (torch.Tensor): Tensore di shape (N, H, W) contenente N immagini in scala di grigi.
        keypoints (torch.Tensor): Tensore di shape (N, K, 2) con le coordinate dei keypoints.
        target_size (tuple): Dimensione target (H, W).

    Returns:
        torch.Tensor: Batch di immagini ridimensionate/croppate.
        torch.Tensor: Keypoints aggiornati.
    """
    N, H, W = imgs.shape
    new_imgs = torch.zeros(
        (N, target_size[0], target_size[1]), dtype=imgs.dtype, device=imgs.device
    )
    new_keypoints = keypoints.clone()

    for i in range(N):
        img = imgs[i]
        kp = keypoints[i].clone()

        h, w = img.shape

        if h > target_size[0] and w > target_size[1]:
            crop_h = (h - target_size[0]) // 2
            crop_w = (w - target_size[1]) // 2
            img = img[
                crop_h : crop_h + target_size[0], crop_w : crop_w + target_size[1]
            ]
            kp[:, 0] -= crop_w
            kp[:, 1] -= crop_h

        elif h > target_size[0] and w < target_size[1]:
            crop_h = (h - target_size[0]) // 2
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            img = img[crop_h : crop_h + target_size[0], :]
            pad = nn.ConstantPad2d((pad_w1, pad_w2, 0, 0), 0)
            img = pad(img)
            kp[:, 1] -= crop_h
            kp[:, 0] += pad_w1

        elif h < target_size[0] and w > target_size[1]:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1
            crop_w = (w - target_size[1]) // 2
            img = img[:, crop_w : crop_w + target_size[1]]
            pad = nn.ConstantPad2d((0, 0, pad_h1, pad_h2), 0)
            img = pad(img)
            kp[:, 0] -= crop_w
            kp[:, 1] += pad_h1

        else:
            pad_h1 = (target_size[0] - h) // 2
            pad_h2 = target_size[0] - h - pad_h1
            pad_w1 = (target_size[1] - w) // 2
            pad_w2 = target_size[1] - w - pad_w1
            pad = nn.ConstantPad2d((pad_w1, pad_w2, pad_h1, pad_h2), 0)
            img = pad(img)
            kp[:, 0] += pad_w1
            kp[:, 1] += pad_h1

        new_imgs[i] = img
        new_keypoints[i] = kp

    return new_imgs, new_keypoints


if __name__ == "__main__":
    create_numpy_dataset()

import os
import cupy as cp
import h5py
import numpy as np
from utils.plot import visualize_image, VolumeViewer
from utils.extract_slices import (
    slice_volume_z,
    signed_angle_between_vectors_gpu,
    center_volume,
)
from cupyx.scipy.ndimage import rotate


def plane_from_points(p1, p2, p3):
    """Trova l'equazione del piano ax + by + cz + d = 0 passante per tre punti."""
    v1, v2 = cp.array(p2) - cp.array(p1), cp.array(p3) - cp.array(p1)
    normal = cp.cross(v1, v2)  # Normale al piano

    # Verifica che la normale non sia nulla
    if cp.linalg.norm(normal) == 0:
        raise ValueError(
            "I tre punti sono collineari e non definiscono un piano valido."
        )

    normal = normal / cp.linalg.norm(normal)  # Normalizziamo la normale
    d = -cp.dot(normal, p1)  # Termine noto
    return normal, d


def extract_from_hdf5(
    file_path,
    save_path,
    degrees,
    first,
    second,
    third,
    center=np.array([-0.019, -0.077, -0.002]),
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(file_path, "r") as h5_file:
        grids = list(h5_file["Input"].keys())
        input_data = h5_file["Input"]["grid00"][:]
        ground_truth = h5_file["GroundTruth"]["grid00"][:]
        vres = h5_file["VolumeInfo"]["resolution"][()]
        origin = h5_file["VolumeInfo"]["origin"][()]
        print("origin", origin)
        directions = h5_file["VolumeInfo"]["directions"][()]
        print("directions", directions)
        shape = h5_file["VolumeInfo"]["shape"][()]
        print(shape)

        delta = vres * directions / np.linalg.norm(directions, axis=0)

        def _get_coord(p):
            return np.round(np.linalg.inv(delta) @ (p - origin)).astype(float)

        coord_first = _get_coord(first)
        coord_first = abs(coord_first)
        coord_first[0], coord_first[1] = coord_first[1], coord_first[0]
        coord_first[1] = shape[1] - coord_first[1]

        coord_second = _get_coord(second)
        coord_second = abs(coord_second)
        coord_second[0], coord_second[1] = coord_second[1], coord_second[0]
        coord_second[1] = shape[1] - coord_second[1]

        coord_third = _get_coord(third)
        coord_third = abs(coord_third)
        coord_third[0], coord_third[1] = coord_third[1], coord_third[0]
        coord_third[1] = shape[1] - coord_third[1]

        coord_center = _get_coord(center)
        coord_center = abs(coord_center)
        coord_center[0], coord_center[1] = coord_center[1], coord_center[0]
        coord_center[1] = shape[1] - coord_center[1]

        volume_superimposed = input_data + ground_truth * 100

        volume_superimposed = cp.array(volume_superimposed)
        coord_first = cp.array(coord_first)
        coord_second = cp.array(coord_second)
        coord_third = cp.array(coord_third)
        coord_center = cp.array(coord_center)

        normal, d = plane_from_points(coord_first, coord_second, coord_third)
        print(f"Normal: {normal}, d: {d}")

        def rotate_vol_given_axis(axis, volume, center):
            # Calcolo degli angoli di rotazione in base alle proiezioni dell'asse
            axis_xz = axis.copy()
            axis_xz[1] = 0
            axis_xz = axis_xz / cp.linalg.norm(axis_xz)

            axis_yz = axis.copy()
            axis_yz[0] = 0
            axis_yz = axis_yz / cp.linalg.norm(axis_yz)

            alpha_xz = signed_angle_between_vectors_gpu(axis_xz)
            alpha_yz = signed_angle_between_vectors_gpu(axis_yz)

            # --- Prima rotazione: piano (1,2) ---
            volume_rotatedx = rotate(
                volume,
                -alpha_xz,
                axes=(1, 2),
                reshape=True,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )

            # Calcolo del centro di rotazione per il piano (1,2)
            orig_shape = volume.shape
            new_shape_x = volume_rotatedx.shape

            # Centro geometrico del piano (1,2) nell'array originale e ruotato
            orig_center_yz = cp.array(
                [(orig_shape[1] - 1) / 2.0, (orig_shape[2] - 1) / 2.0]
            )
            new_center_yz = cp.array(
                [(new_shape_x[1] - 1) / 2.0, (new_shape_x[2] - 1) / 2.0]
            )

            # Estrae le coordinate (y,z) del centro di interesse
            pt_yz = center[1:3]

            # Matrice di rotazione 2D per l'angolo alpha_xz
            cos1 = cp.cos(-alpha_xz)
            sin1 = cp.sin(-alpha_xz)
            R1 = cp.array([[cos1, -sin1], [sin1, cos1]])

            # Calcolo delle nuove coordinate nel piano (1,2)
            pt_yz_rot = R1 @ (pt_yz - orig_center_yz) + new_center_yz

            # La coordinata lungo l'asse 0 non cambia in questa rotazione.
            center_after_first = cp.array([center[0], pt_yz_rot[0], pt_yz_rot[1]])

            # --- Seconda rotazione: piano (0,2) ---
            volume_rotatedy = rotate(
                volume_rotatedx,
                -alpha_yz,
                axes=(0, 2),
                reshape=True,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )

            # Calcolo del centro per il piano (0,2)
            shape_after_first = volume_rotatedx.shape
            shape_final = volume_rotatedy.shape

            orig_center_0_2 = cp.array(
                [(shape_after_first[0] - 1) / 2.0, (shape_after_first[2] - 1) / 2.0]
            )
            new_center_0_2 = cp.array(
                [(shape_final[0] - 1) / 2.0, (shape_final[2] - 1) / 2.0]
            )

            # Prendi le coordinate (0,2) dal centro dopo la prima rotazione
            pt_0_2 = cp.array([center_after_first[0], center_after_first[2]])

            # Matrice di rotazione 2D per l'angolo -alpha_yz
            cos2 = cp.cos(-alpha_yz)
            sin2 = cp.sin(-alpha_yz)
            R2 = cp.array([[cos2, -sin2], [sin2, cos2]])

            # Calcolo delle nuove coordinate nel piano (0,2)
            pt_0_2_rot = R2 @ (pt_0_2 - orig_center_0_2) + new_center_0_2

            # La coordinata lungo l'asse 1 rimane invariata in questa rotazione
            center_final = cp.array(
                [pt_0_2_rot[0], center_after_first[1], pt_0_2_rot[1]]
            )

            return volume_rotatedy, center_final

        volume_rotated, center_rotated = rotate_vol_given_axis(
            normal, volume_superimposed, coord_center
        )

        print(f"Center after rotation: {center_rotated}")

        viewer = VolumeViewer(volume_rotated)
        viewer.show()

        target_x = viewer.clicked_points[0][1]
        target_y = viewer.clicked_points[0][0]

        flag = True
        for grid in grids:
            print(f"Processing grid {grid}")
            input_data = h5_file["Input"][grid][:]

            input_data = cp.array(input_data)

            volume_rotated = rotate_vol_given_axis(normal, input_data, coord_center)[0]
            volume_centered = center_volume(volume_rotated, target_x, target_y)
            volume_centered = volume_centered.transpose(2, 0, 1)

            # Step 4: Extract slices passing through the selected point, aligned with the z-axis
            first_slice = slice_volume_z(volume_centered, degrees[0])
            height, width = first_slice.shape  # Get dimensions of a single slice

            # Initialize an empty CuPy array for the slices
            imgs = cp.zeros((len(degrees), height, width), dtype=first_slice.dtype)

            # Extract and store each slice
            for i, angle in enumerate(degrees):
                imgs[i] = slice_volume_z(volume_centered, angle)

            imgs = imgs.transpose(1, 2, 0)

            if flag:
                viewer = VolumeViewer(imgs)
                viewer.show()
                flag = False

            save_path_img = os.path.join(save_path, f"{grid}.npz")
            np.savez_compressed(save_path_img, imgs)
        print(f"Saved images to {save_path}")

import numpy as np
from scipy.ndimage import map_coordinates
import cupy as cp
from utils.plot import VolumeViewer, visualize_image
from cupyx.scipy.ndimage import rotate, affine_transform
import math as m
import h5py
import os
import scipy

"""these are all functions used to extract the slices from the 3d images.
the last one that i used is extract_from_hdf5. It requires the coordinates of 2 points, rotates the volume to align z axis to that 
axis, then asks to select the centre of the tricuspid valve after the rotation and slices the volume with planes parallel to that axis"""


def signed_angle_between_vectors(vec, target=np.array([0, 0, 1]), ref_axis=None):
    """
    Compute the signed angle (in degrees) between `vec` and `target`.

    :param vec: Source vector (does not need to be normalized)
    :param target: Target vector (default: [0, 0, 1]), does not need to be normalized
    :param ref_axis: Optional reference axis to determine sign of rotation (default: cross product of vec and target)
    :return: Signed angle in degrees
    """
    vec = vec / np.linalg.norm(vec)  # Normalize vector
    target = target / np.linalg.norm(target)

    v_cross = np.cross(vec, target)  # Axis of rotation
    dot_product = np.dot(vec, target)  # Cosine of angle
    angle_rad = np.arctan2(np.linalg.norm(v_cross), dot_product)  # Angle in radians

    if ref_axis is None:
        ref_axis = v_cross  # Use cross product to determine sign if no reference axis is provided

    # Determine sign of the angle using dot product with reference axis
    sign = np.sign(np.dot(ref_axis, v_cross))  # +1 or -1
    angle_deg = np.degrees(angle_rad) * sign  # Convert to degrees with sign

    return angle_deg


def signed_angle_between_vectors_gpu(vec, target=cp.array([0, 0, 1]), ref_axis=None):
    """
    Compute the signed angle (in degrees) between `vec` and `target`.

    :param vec: Source vector (does not need to be normalized)
    :param target: Target vector (default: [0, 0, 1]), does not need to be normalized
    :param ref_axis: Optional reference axis to determine sign of rotation (default: cross product of vec and target)
    :return: Signed angle in degrees
    """
    vec = vec / cp.linalg.norm(vec)  # Normalize vector
    target = target / cp.linalg.norm(target)

    v_cross = cp.cross(vec, target)  # Axis of rotation
    dot_product = cp.dot(vec, target)  # Cosine of angle
    angle_rad = cp.arctan2(cp.linalg.norm(v_cross), dot_product)  # Angle in radians

    if ref_axis is None:
        ref_axis = v_cross  # Use cross product to determine sign if no reference axis is provided

    # Determine sign of the angle using dot product with reference axis
    sign = cp.sign(cp.dot(ref_axis, v_cross))  # +1 or -1
    angle_deg = cp.degrees(angle_rad) * sign  # Convert to degrees with sign

    return angle_deg


def rotate_volume(volume, R):
    """
    Apply a 3D rotation matrix to a volume.

    :param volume: 3D NumPy array (H, W, D)
    :param R: 3x3 rotation matrix
    :return: Rotated volume
    """
    coords = np.array(
        np.meshgrid(
            np.arange(volume.shape[0]),
            np.arange(volume.shape[1]),
            np.arange(volume.shape[2]),
            indexing="ij",
        )
    ).reshape(
        3, -1
    )  # (3, N) coordinate grid

    # Rotate coordinates
    new_coords = (
        R @ (coords - np.array(volume.shape)[:, None] / 2)
        + np.array(volume.shape)[:, None] / 2
    )

    # Interpolate rotated volume
    rotated_volume = map_coordinates(volume, new_coords, order=1, mode="nearest")

    return rotated_volume.reshape(volume.shape)


def slice_volume_x(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xz = [[np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]

    center = cp.array([0, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xz = cp.array(e_xz)
    e_xz = e_xz / cp.linalg.norm(e_xz, axis=1)[:, cp.newaxis]

    # R: rotation matrix: data_coords = center + r @ slice_coords
    ey = cp.cross(e_xz[0], e_xz[1])
    r = cp.array([e_xz[0], ey, e_xz[1]], dtype=cp.float32).T

    # free gpu mem
    del e_xz, ey

    # setup slice points P with coordinates (X, Y, 0)
    mx, mz = int(vol.shape[0]), int(vol.shape[2])
    xs = cp.arange(0.5, 0.5 + mx)
    zs = cp.arange(0.5 - mz / 2, 0.5 + mz / 2)
    pp = cp.zeros((3, mx, mz), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[2, :, :] = zs.reshape(1, mz)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum("il,ljk->ijk", r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, zs, pp, center

    # computation of indices without einstein summation
    # idx = cp.zeros((3, mx, mz), dtype=cp.float32)
    # for i in range(r.shape[0]):
    #     for j in range(pp.shape[1]):
    #         for k in range(pp.shape[2]):
    #             idx[i, j, k] = r[i, 0]*pp[0, j, k] + r[i, 1]*pp[1, j, k] + r[i, 2]*pp[2, j, k]
    # idx += (0.5 + center.reshape(3, 1, 1))

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(
        cp.array(
            [
                idx[0, :, :] < 0,
                idx[0, :, :] >= vol.shape[0],
                idx[1, :, :] < 0,
                idx[1, :, :] >= vol.shape[1],
                idx[2, :, :] < 0,
                idx[2, :, :] >= vol.shape[2],
            ]
        ),
        axis=0,
    )

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return img


def slice_volume_y(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xy = [[np.sin(theta), 0.0, np.cos(theta)], [0.0, 1.0, 0.0]]

    center = cp.array([0, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xy = cp.array(e_xy)
    e_xy = e_xy / cp.linalg.norm(e_xy, axis=1)[:, cp.newaxis]

    # R: rotation matrix: data_coords = center + r @ slice_coords
    ez = cp.cross(e_xy[0], e_xy[1])
    r = cp.array([e_xy[0], e_xy[1], ez], dtype=cp.float32).T

    # free gpu mem
    del e_xy, ez

    # setup slice points P with coordinates (X, Y, 0)
    mx, my = int(vol.shape[0]), int(vol.shape[1])
    xs = cp.arange(0.5, 0.5 + mx)
    ys = cp.arange(0.5 - my / 2, 0.5 + my / 2)
    pp = cp.zeros((3, mx, my), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[1, :, :] = ys.reshape(1, my)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum("il,ljk->ijk", r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, ys, pp, center

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(
        cp.array(
            [
                idx[0, :, :] < 0,
                idx[0, :, :] >= vol.shape[0],
                idx[1, :, :] < 0,
                idx[1, :, :] >= vol.shape[1],
                idx[2, :, :] < 0,
                idx[2, :, :] >= vol.shape[2],
            ]
        ),
        axis=0,
    )

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return img


def slice_volume_z(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xy = [[1.0, 0.0, 0.0], [0.0, np.cos(theta), np.sin(theta)]]

    center = cp.array(
        [vol.shape[0] / 2, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float
    )

    e_xy = cp.array(e_xy)
    e_xy = e_xy / cp.linalg.norm(e_xy, axis=1)[:, cp.newaxis]

    # free gpu mem
    # R: rotation matrix: data_coords = center + r @ slice_coords
    ez = cp.cross(e_xy[0], e_xy[1])
    r = cp.array([e_xy[0], e_xy[1], ez], dtype=cp.float32).T

    # free gpu mem
    del e_xy, ez

    # setup slice points P with coordinates (X, Y, 0)
    mx, my = int(vol.shape[0]), int(vol.shape[1])
    xs = cp.arange(0.5 - mx / 2, 0.5 + mx / 2)
    ys = cp.arange(0.5 - my / 2, 0.5 + my / 2)

    pp = cp.zeros((3, mx, my), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[1, :, :] = ys.reshape(1, my)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum("il,ljk->ijk", r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, ys, pp, center

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(
        cp.array(
            [
                idx[0, :, :] < 0,
                idx[0, :, :] >= vol.shape[0],
                idx[1, :, :] < 0,
                idx[1, :, :] >= vol.shape[1],
                idx[2, :, :] < 0,
                idx[2, :, :] >= vol.shape[2],
            ]
        ),
        axis=0,
    )

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return img


def center_volume(volume, target_x, target_y):
    """
    pads the volume by adding 0 along the x and y axes so that the target x and y are in the center
    volume: a cupy volume
    int: target x, target y: the coordinates you want to center"""
    X, Y, Z = volume.shape

    pad_x = 2 * int((target_x - X / 2))
    pad_y = 2 * int((target_y - Y / 2))

    pad_x_before = max(0, 0 - pad_x)
    pad_x_after = max(0, pad_x)

    pad_y_before = max(0, 0 - pad_y)
    pad_y_after = max(0, pad_y)

    padded_volume = cp.pad(
        volume,
        ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return padded_volume


def crop_black_borders(volume):
    """
    Crops the black (zero-valued) borders from the volume along the x and y dimensions using CuPy.

    Parameters:
        volume (cupy.ndarray): A 3D CuPy array of shape (x, y, z).

    Returns:
        cupy.ndarray: The cropped volume with borders removed along x and y.
    """
    # Compute a 2D mask to check where at least one voxel along the z-axis is non-zero
    mask = volume.any(axis=2)

    # Get the non-zero indices along x and y
    x_nonzero = cp.where(mask.any(axis=1))[0]
    y_nonzero = cp.where(mask.any(axis=0))[0]

    # If the entire volume is zero, return an empty volume
    if x_nonzero.size == 0 or y_nonzero.size == 0:
        return volume[0:0, 0:0, :]

    # Get min and max bounds for x and y
    x_min, x_max = x_nonzero[0], x_nonzero[-1]
    y_min, y_max = y_nonzero[0], y_nonzero[-1]

    # Crop the volume using the computed bounds
    cropped_volume = volume[x_min : x_max + 1, y_min : y_max + 1, :]

    return cropped_volume


def extract_slices(input, ground_truth, degrees=np.linspace(0, 2 * np.pi, 10)):
    """
    Extracts 2D slices from a 3D medical volume after aligning the right ventricle along the z-axis.

    This function performs three main steps:
    1. **Alignment**: The volume is rotated in the YZ and XZ planes to ensure the ventricle is aligned with the z-axis.
    2. **Centering**: The user selects the tricuspid valve center in the XY-plane, ensuring slices pass through this point.
    3. **Slice Extraction**: Slices are taken with planes parallel to the z-axis, rotated at the specified degrees.

    Parameters:
    - input (numpy array): The 3D volume containing the medical image.
    - ground_truth (numpy array): The 3D segmentation of the right ventricle.
    - degrees (numpy array): An array containing rotation angles for extracting slices.

    Returns:
    - A 3D CuPy array containing the extracted slices.
    """

    # Superimpose the segmentation onto the input volume for visualization
    volume_superimposed = input + ground_truth * 50

    # Convert to CuPy arrays for GPU processing
    volume_superimposed = cp.asarray(volume_superimposed)
    volume = cp.asarray(input)

    # Step 1: Align the volume in the YZ-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()

    alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])

    # Rotate to align along the z-axis
    if viewer.unit_vectors[0][0] == 0:
        volume_superimposed = rotate(
            volume_superimposed,
            alpha,
            axes=(1, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        volume = rotate(
            volume,
            alpha,
            axes=(1, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    elif viewer.unit_vectors[0][1] == 0:
        volume_superimposed = rotate(
            volume_superimposed,
            -alpha,
            axes=(0, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        volume = rotate(
            volume,
            -alpha,
            axes=(0, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    # Step 2: Align the volume in the XZ-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()
    alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])

    # Rotate to finalize alignment
    if viewer.unit_vectors[0][0] == 0:
        volume_superimposed = rotate(
            volume_superimposed,
            alpha,
            axes=(1, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        volume = rotate(
            volume,
            alpha,
            axes=(1, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    elif viewer.unit_vectors[0][1] == 0:
        volume_superimposed = rotate(
            volume_superimposed,
            -alpha,
            axes=(0, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        volume = rotate(
            volume,
            -alpha,
            axes=(0, 2),
            reshape=True,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    # Step 3: Select the tricuspid valve center in the XY-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()
    target_x = viewer.clicked_points[0][1]
    target_y = viewer.clicked_points[0][0]

    # Center the volume based on the selected point
    volume = center_volume(volume_superimposed, target_x, target_y)
    volume = volume.transpose(2, 0, 1)

    # Step 4: Extract slices passing through the selected point, aligned with the z-axis
    first_slice = slice_volume_z(volume, degrees[0])
    height, width = first_slice.shape  # Get dimensions of a single slice

    # Initialize an empty CuPy array for the slices
    imgs = cp.zeros((len(degrees), height, width), dtype=first_slice.dtype)

    # Extract and store each slice
    for i, angle in enumerate(degrees):
        imgs[i] = slice_volume_z(volume, angle)

    imgs = imgs.transpose(1, 2, 0)
    # imgs = crop_black_borders(imgs)

    return imgs  # Return the extracted slices


def rz(theta):  # true rx, but first dim in US volume is z-axis
    return np.array(
        [
            [1, 0, 0, 0],
            [0, m.cos(theta), -m.sin(theta), 0],
            [0, m.sin(theta), m.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def ry(theta):
    return np.array(
        [
            [m.cos(theta), 0, m.sin(theta), 0],
            [0, 1, 0, 0],
            [-m.sin(theta), 0, m.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def rx(theta):  # true rz, but first dim in US volume is z-axis
    return np.array(
        [
            [m.cos(theta), -m.sin(theta), 0, 0],
            [m.sin(theta), m.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def t(tx, ty, tz):
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])


def rotate_about_axis(axis, sequence, rot_angles):
    print("Rotating volume about " + axis + " axis.")
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # translate to probe center, to rotate about centerline
    tm_center = t(sequence.shape[1] / 2, sequence.shape[2] / 2, sequence.shape[3] / 2)
    inv_tm_center = np.linalg.inv(tm_center)

    if axis == "x":
        rot_imgs = np.ndarray(
            (len(rot_angles), sequence.shape[0], sequence.shape[1], sequence.shape[3])
        )
    else:
        rot_imgs = np.ndarray(
            (len(rot_angles), sequence.shape[0], sequence.shape[1], sequence.shape[2])
        )
    trf_mats = np.ndarray((len(rot_angles), 4, 4))

    for frame in range(sequence.shape[0]):

        print(
            "Processing frame " + str(frame + 1) + " of " + str(sequence.shape[0]) + "."
        )

        # rotate volume about z axis
        for idx, angle in enumerate(rot_angles):
            theta = m.radians(angle % 360)
            vol_gpu = cp.array(sequence[frame][:][:][:])

            # extract image
            if axis == "x":
                rot_img = slice_volume_x(vol_gpu, theta)
            elif axis == "y":
                rot_img = slice_volume_y(vol_gpu, theta)
            elif axis == "z":
                rot_img = slice_volume_z(vol_gpu, theta)

            # rot_imgs[idx, frame, :, :] = np.flipud(cp.asnumpy(rot_img))
            rot_imgs[idx, frame, :, :] = cp.asnumpy(rot_img)

            # free gpu mem
            del vol_gpu, rot_img

            # save transformation matrix for inverse transformation of landmarks
            if frame == 0:
                rm_z = rz(theta)
                trf_mats[idx, :, :] = np.linalg.multi_dot(
                    [tm_center, rm_z, inv_tm_center]
                )

            # VisualizeVolume.display_rgb_image(rot_imgs[idx, frame, :, :], "extracted image after " + str(angle) +
            # " degrees rotation about z axis")

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return rot_imgs, trf_mats


def extract_axis_from_points(p1, p2):
    """
    Compute the unit vector (axis) that passes through two points.

    Parameters:
        p1 (cp.ndarray): First point in 3D space [x1, y1, z1].
        p2 (cp.ndarray): Second point in 3D space [x2, y2, z2].

    Returns:
        cp.ndarray: A unit vector representing the axis direction.
    """

    # Compute direction vector
    axis = p2 - p1

    # Normalize the vector (avoid division by zero)
    norm = cp.linalg.norm(axis)
    if norm > 0:
        axis /= norm

    return axis


def get_rotation_matrix_gpu(axis, angle_deg):
    """
    Compute a 3x3 rotation matrix given an axis and an angle using CuPy.

    Parameters:
        axis (array-like): Rotation axis [x, y, z] (should be a unit vector).
        angle_deg (float): Rotation angle in degrees.

    Returns:
        cp.ndarray: 3x3 rotation matrix (GPU array).
    """
    axis = axis / cp.linalg.norm(axis)  # Normalize the axis
    angle_rad = cp.radians(angle_deg)  # Convert angle to radians

    # Rodrigues' rotation formula (axis-angle to rotation matrix)
    cos_a = cp.cos(angle_rad)
    sin_a = cp.sin(angle_rad)
    one_minus_cos = 1 - cos_a

    x, y, z = axis
    rotation_matrix = cp.array(
        [
            [
                cos_a + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_a,
                x * z * one_minus_cos + y * sin_a,
            ],
            [
                y * x * one_minus_cos + z * sin_a,
                cos_a + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_a,
            ],
            [
                z * x * one_minus_cos - y * sin_a,
                z * y * one_minus_cos + x * sin_a,
                cos_a + z * z * one_minus_cos,
            ],
        ],
        dtype=cp.float32,
    )
    return rotation_matrix


# TODO: verifica le coordinate, e' possibile che x e y siano scambiati all'inizio quando passo gli array alla funzione
def extract_slices_from_points(
    input, ground_truth, tric_valve, apex, degrees=np.linspace(0, 2 * np.pi, 10)
):
    """
    Extracts 2D slices from a 3D medical volume after aligning the right ventricle along the z-axis.

    This function performs three main steps:
    1. **Alignment**: The volume is rotated in the YZ and XZ planes to ensure the ventricle is aligned with the z-axis.
    2. **Centering**: The user selects the tricuspid valve center in the XY-plane, ensuring slices pass through this point.
    3. **Slice Extraction**: Slices are taken with planes parallel to the z-axis, rotated at the specified degrees.

    Parameters:
    - input (numpy array): The 3D volume containing the medical image.
    - ground_truth (numpy array): The 3D segmentation of the right ventricle.
    - degrees (numpy array): An array containing rotation angles for extracting slices.

    Returns:
    - A 3D CuPy array containing the extracted slices.
    """

    # Superimpose the segmentation onto the input volume for visualization
    volume_superimposed = input + ground_truth * 50

    # Convert to CuPy arrays for GPU processing
    volume_superimposed = cp.asarray(volume_superimposed)
    volume = cp.asarray(input)

    axis = extract_axis_from_points(tric_valve, apex)

    axis_xz = axis.copy()
    axis_xz[1] = 0
    axis_xz = axis_xz / cp.linalg.norm(axis_xz)

    axis_yz = axis.copy()
    axis_yz[0] = 0
    axis_yz = axis_yz / cp.linalg.norm(axis_yz)

    alpha_xz = signed_angle_between_vectors_gpu(axis_xz)
    alpha_yz = signed_angle_between_vectors_gpu(axis_yz)
    print("alpha_xz:", alpha_xz)
    print("alpha_yz:", alpha_yz)

    volume_rotatedx = rotate(
        volume_superimposed,
        alpha_xz,
        axes=(1, 2),
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )

    print(volume_rotatedx.shape)
    print("ruoto attorno a x")
    print("nuove coordinate")
    # For the first rotation (e.g. around the x-axis, affecting Y and Z):
    new_tric_valve = rotate_pointx(
        tric_valve,
        angle_deg=alpha_xz,
        orig_shape=volume_superimposed.shape,
        new_shape=volume_rotatedx.shape,
    )

    print("new_tric_valve:", new_tric_valve)

    viewer = VolumeViewer(volume_rotatedx)
    viewer.show()

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

    print("ruoto attorno a y")
    print("nuove coordinate")
    # For the second rotation (e.g. around the y-axis, affecting X and Z):
    new_tric_valve = rotate_pointy(
        new_tric_valve,
        angle_deg=-alpha_yz,
        orig_shape=volume_rotatedx.shape,
        new_shape=volume_rotatedy.shape,
    )

    print("new_tric_valve:", new_tric_valve)
    volume_rotated = volume_rotatedy

    # Step 1: Align the volume in the YZ-plane
    viewer = VolumeViewer(volume_rotated)
    viewer.show()

    # target_x = viewer.clicked_points[0][1]
    # target_y = viewer.clicked_points[0][0]

    target_x = new_tric_valve[0]
    target_y = new_tric_valve[1]
    print("target_x:", target_x)
    print("target_y:", target_y)

    # Center the volume based on the selected point
    volume_rotated = center_volume(volume_rotated, target_x, target_y)
    volume_rotated = volume_rotated.transpose(2, 0, 1)

    # Step 4: Extract slices passing through the selected point, aligned with the z-axis
    first_slice = slice_volume_z(volume_rotated, degrees[0])
    height, width = first_slice.shape  # Get dimensions of a single slice

    # Initialize an empty CuPy array for the slices
    imgs = cp.zeros((len(degrees), height, width), dtype=first_slice.dtype)

    # Extract and store each slice
    for i, angle in enumerate(degrees):
        imgs[i] = slice_volume_z(volume_rotated, angle)

    imgs = imgs.transpose(1, 2, 0)
    # imgs = crop_black_borders(imgs)

    return imgs  # Return the extracted slices


def rotate_pointx(point, angle_deg, orig_shape, new_shape):
    # Compute centers for the original and new volumes
    orig_center = (cp.array(orig_shape) - 1) / 2.0
    new_center = (cp.array(new_shape) - 1) / 2.0
    # Translate point relative to original center
    point_rel = point - orig_center
    # Create rotation matrix for the given axes and angle
    angle = cp.asarray(angle_deg)
    angle = cp.radians(angle)
    c, s = cp.cos(angle), cp.sin(angle)
    rot_matrix = cp.array([[1, 0, 0], [0, c.get(), -s.get()], [0, s.get(), c.get()]])
    # Apply rotation only to the coordinates in the specified axes
    # coords = point_rel[list(axes)]
    rotated_coords = rot_matrix @ point_rel
    # Replace the rotated coordinates
    point_rel = rotated_coords
    # Translate back with the new center offset
    return point_rel + new_center


def rotate_pointy(point, angle_deg, orig_shape, new_shape):
    # Compute centers for the original and new volumes
    orig_center = (cp.array(orig_shape) - 1) / 2.0
    new_center = (cp.array(new_shape) - 1) / 2.0
    # Translate point relative to original center
    point_rel = point - orig_center
    # Create rotation matrix for the given axes and angle
    angle = cp.asarray(angle_deg)
    angle = cp.radians(angle)
    c, s = cp.cos(angle), cp.sin(angle)
    rot_matrix = cp.array([[c.get(), 0, s.get()], [0, 1, 0], [-s.get(), 0, c.get()]])
    # Apply rotation only to the coordinates in the specified axes
    # coords = point_rel[list(axes)]
    rotated_coords = rot_matrix @ point_rel
    # Replace the rotated coordinates
    point_rel = rotated_coords
    # Translate back with the new center offset
    return point_rel + new_center


def extract_from_hdf5(file_path, save_path, degrees, tric, apex):
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

        coord_tric = _get_coord(tric)
        coord_tric = abs(coord_tric)
        coord_tric[0], coord_tric[1] = coord_tric[1], coord_tric[0]
        coord_tric[1] = shape[1] - coord_tric[1]

        coord_apex = _get_coord(apex)
        coord_apex = abs(coord_apex)
        coord_apex[0], coord_apex[1] = coord_apex[1], coord_apex[0]
        coord_apex[1] = shape[1] - coord_apex[1]
        volume_superimposed = input_data + ground_truth * 50

        def rotate_vol_given_2_points(coord_tric, coord_apex, volume):
            axis = extract_axis_from_points(coord_tric, coord_apex)
            axis_xz = axis.copy()
            axis_xz[1] = 0
            axis_xz = axis_xz / cp.linalg.norm(axis_xz)

            axis_yz = axis.copy()
            axis_yz[0] = 0
            axis_yz = axis_yz / cp.linalg.norm(axis_yz)

            alpha_xz = signed_angle_between_vectors_gpu(axis_xz)
            alpha_yz = signed_angle_between_vectors_gpu(axis_yz)

            volume_rotatedx = rotate(
                volume,
                alpha_xz,
                axes=(1, 2),
                reshape=True,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )

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
            return volume_rotatedy

        coord_tric = cp.array(coord_tric)
        coord_apex = cp.array(coord_apex)
        volume_superimposed = cp.array(volume_superimposed)

        volume_rotated = rotate_vol_given_2_points(
            coord_tric, coord_apex, volume_superimposed
        )

        viewer = VolumeViewer(volume_rotated)
        viewer.show()

        target_x = viewer.clicked_points[0][1]
        target_y = viewer.clicked_points[0][0]

        for grid in grids:
            print(f"Processing grid {grid}")
            input_data = h5_file["Input"][grid][:]

            input_data = cp.array(input_data)

            volume_rotated = rotate_vol_given_2_points(
                coord_tric, coord_apex, input_data
            )
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

            save_path_img = os.path.join(save_path, f"{grid}.npz")
            np.savez_compressed(save_path_img, imgs)
        print(f"Saved images to {save_path}")


def plane_from_points(p1, p2, p3):
    """Trova l'equazione del piano ax + by + cz + d = 0 passante per tre punti."""
    v1, v2 = cp.array(p2) - cp.array(p1), cp.array(p3) - cp.array(p1)
    normal = cp.cross(v1, v2)  # Normale al piano
    d = -cp.dot(normal, p1)  # Termine noto
    return normal, d


def generate_plane_grid(p1, p2, p3, shape, spacing):
    """Genera una griglia regolare di punti su un piano definito da tre punti."""
    # Definiamo due vettori per il piano
    v1 = cp.array(p2) - cp.array(p1)
    v2 = cp.array(p3) - cp.array(p1)

    # Normalizziamo per ottenere direzioni ortogonali
    v1 = v1 / cp.linalg.norm(v1)
    v2 = v2 / cp.linalg.norm(v2)

    # Creiamo una griglia 2D nel piano
    grid_x, grid_y = cp.meshgrid(
        cp.linspace(-1, 1, shape[0]) * 300,  # * cp.linalg.norm(p2 - p1),
        cp.linspace(-1, 1, shape[1]) * 300,  # * cp.linalg.norm(p3 - p1)
    )

    # Convertiamo la griglia 2D in coordinate 3D nel volume
    grid_3d = (
        cp.array(p1)[:, None, None]
        + v1[:, None, None] * grid_x
        + v2[:, None, None] * grid_y
    )
    return grid_3d


def extract_slice_3_points(volume, spacing, p1, p2, p3, slice_shape=(600, 600)):
    """Estrae una slice 2D dal volume lungo il piano definito da tre punti."""
    normal, d = plane_from_points(p1, p2, p3)

    # Generiamo la griglia del piano
    grid_3d = generate_plane_grid(p1, p2, p3, slice_shape, spacing)

    # Convertiamo le coordinate in voxel space
    grid_3d_voxel = grid_3d / spacing[:, None, None]

    # Interpoliamo i valori sulla slice usando SciPy
    slice_2d = scipy.ndimage.map_coordinates(
        cp.asnumpy(volume),
        [cp.asnumpy(grid_3d_voxel[i]) for i in range(3)],
        order=1,
        mode="nearest",
    )

    return cp.array(slice_2d)


def slice_extraction_three_points(file_path, save_path, degrees, first, second, third):
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

        volume_superimposed = input_data + ground_truth

        volume_superimposed = cp.array(volume_superimposed)
        coord_first = cp.array(coord_first)
        coord_second = cp.array(coord_second)
        coord_third = cp.array(coord_third)

        spacing = cp.array([1.0, 1.0, 1.0])
        slice_2d = extract_slice_3_points(
            volume_superimposed, spacing, coord_first, coord_second, coord_third
        )

        visualize_image(slice_2d)

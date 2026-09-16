import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def systole_diatole_detection(apex_distances, window_size=40, plot=False):
    """
    Detects the systole and diastole phases based on the apex distances.

    Args:
        apex_distances (list or array): List of apex distances.
        window_size (int): Size of the moving average filter window.
        plot (bool): Whether to plot the smoothed signal and detected points.

    Returns:
        tuple of arrays: Indices of systole (minima) and diastole (maxima).
    """
    # Convert to CuPy array
    apex_distances = cp.array(apex_distances)

    # Apply moving average filter
    kernel = cp.ones(window_size) / window_size
    smoothed = cp.convolve(apex_distances, kernel, mode="same")

    # Compute first derivative
    diff = cp.diff(smoothed)
    diff = cp.concatenate((cp.array([0]), diff))

    # Find zero crossings: maxima (from + to -), minima (from - to +)
    sign_changes = cp.sign(diff)
    sign_diff = cp.diff(sign_changes)

    diastole_idx = cp.where(sign_diff < 0)[0]  # Maxima
    systole_idx = cp.where(sign_diff > 0)[0]  # Minima

    if plot:
        x = np.arange(len(apex_distances))
        original = cp.asnumpy(apex_distances)
        smoothed_np = cp.asnumpy(smoothed)
        systole_np = cp.asnumpy(systole_idx)
        diastole_np = cp.asnumpy(diastole_idx)

        plt.figure(figsize=(12, 6))
        plt.plot(x, original, label="Original Signal", alpha=0.5)
        plt.plot(x, smoothed_np, label="Smoothed Signal", linewidth=2)

        plt.scatter(
            systole_np,
            smoothed_np[systole_np],
            color="red",
            marker="o",
            label="Systole (Minima)",
        )
        plt.scatter(
            diastole_np,
            smoothed_np[diastole_np],
            color="green",
            marker="^",
            label="Diastole (Maxima)",
        )

        plt.title("Smoothed Apex Distance Signal with Systole and Diastole Points")
        plt.xlabel("Time")
        plt.ylabel("Apex Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return systole_idx, diastole_idx

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import h5py

from temporal_pipeline.pipeline_testing.indices_prediction.prediction_utils import (
    Predictor,
)

LANDMARK_NAMES = ["TAfw", "TAsep", "Apex"]
THRESHOLDS = list(range(0, 101))  # 0 to 100 inclusive

"""code to perform the statistical analysis. Based on the arguments it can do:
- threshold based or not threshold based inference
- threshold sweep to get the mean distance vs threshold plot
- boxplots of mean dist per patient and per landmark
"""


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the trained model, save predictions, and compute error statistics."
    )
    parser.add_argument(
        "--excel_path", type=str, required=True, help="Path where to save the excel."
    )
    parser.add_argument(
        "--heatmap_method",
        action="store_true",
        help="If the model was trained with the heatmap method.",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        required=True,
        help="Path to the checkpoints of the trained model.",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        required=True,
        help="Path of the folder with the hdf5/dicom files.",
    )
    parser.add_argument(
        "--unet_initial_channels",
        type=int,
        default=16,
        help="Number of filters in the first layer of the UNet.",
    )
    parser.add_argument(
        "--unet_res_units",
        type=int,
        default=2,
        help="Number of residual units the UNet.",
    )
    parser.add_argument(
        "--window_len",
        type=int,
        default=32,
        help="Number of frames the model receives in input.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/boxplots",
        help="Directory where plots and stats will be saved.",
    )
    parser.add_argument(
        "--thresh_method",
        action="store_true",
        help="Enable confidence threshold analysis.",
    )
    parser.add_argument(
        "--thresh",
        type=str,
        default=None,
        help="if passed without --thresh_method, it produces the boxplots at that threshold of confidence",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "numpy") else np.array(x)


def compute_euclidean_errors(pred, gt):
    """
    pred, gt : np.ndarray  (1, N, 3, 2)
    Returns  : np.ndarray  (N, 3)
    """
    pred = pred.squeeze(0)
    gt = gt.squeeze(0)
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1))


def stats_summary(values, label=""):
    return {
        "label": label,
        "n": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "median": np.median(values),
        "p25": np.percentile(values, 25),
        "p75": np.percentile(values, 75),
    }


def boxplot_panel(data_dict, title, ylabel, output_path):
    labels = list(data_dict.keys())
    values = [data_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    bp = ax.boxplot(
        values,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {output_path}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def run_threshold_sweep(
    all_errors_flat, all_maxima_flat, errors_lm, maxima_lm, n_total, output_dir
):
    """
    Sweeps thresholds 0–100. For each threshold, points whose heatmap max is
    below the threshold are treated as NaN (excluded from mean error; counted
    separately). If all points are NaN at a given threshold, mean error = 0.

    all_errors_flat : np.ndarray (M,)       – all errors across patients/frames/landmarks
    all_maxima_flat : np.ndarray (M,)       – corresponding heatmap max values
    errors_lm       : list of np.ndarray    – one (M_lm,) array per landmark
    maxima_lm       : list of np.ndarray    – one (M_lm,) array per landmark
    n_total         : int                   – total number of points (for NaN %)
    """
    thresholds = THRESHOLDS

    # --- overall curve ---
    mean_errors_total = []
    nan_pct_total = []
    for t in thresholds:
        valid = all_maxima_flat >= t
        nan_pct_total.append(100.0 * (~valid).sum() / n_total)
        mean_errors_total.append(all_errors_flat[valid].mean() if valid.any() else 0.0)

    # --- per-landmark curves ---
    mean_errors_per_lm = []
    nan_pct_per_lm = []
    for lm in range(len(LANDMARK_NAMES)):
        errs = errors_lm[lm]
        maxv = maxima_lm[lm]
        n_lm = len(errs)
        me, np_ = [], []
        for t in thresholds:
            valid = maxv >= t
            np_.append(100.0 * (~valid).sum() / n_lm)
            me.append(errs[valid].mean() if valid.any() else 0.0)
        mean_errors_per_lm.append(me)
        nan_pct_per_lm.append(np_)

    # --- plot 1: overall ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nan_pct_total, mean_errors_total, color="steelblue", linewidth=2)
    ax.set_xlabel("% points below threshold (NaN)")
    ax.set_ylabel("Mean Euclidean error (px)")
    ax.set_title("Mean Error vs % NaN — Overall")
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(output_dir, "threshold_sweep_overall.png")
    plt.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {p}")

    # --- plot 2: per-landmark (subplots) ---
    fig, axes = plt.subplots(
        1, len(LANDMARK_NAMES), figsize=(5 * len(LANDMARK_NAMES), 5), sharey=True
    )
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(LANDMARK_NAMES)))
    for lm, ax in enumerate(axes):
        ax.plot(
            nan_pct_per_lm[lm], mean_errors_per_lm[lm], color=colors[lm], linewidth=2
        )
        ax.set_xlabel("% NaN")
        ax.set_title(LANDMARK_NAMES[lm])
        ax.grid(linestyle="--", alpha=0.5)
        if lm == 0:
            ax.set_ylabel("Mean Euclidean error (px)")
    fig.suptitle("Mean Error vs % NaN — Per Landmark", fontsize=13)
    plt.tight_layout()
    p = os.path.join(output_dir, "threshold_sweep_per_landmark.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {p}")

    # --- save sweep table ---
    sweep_rows = []
    for i, t in enumerate(thresholds):
        row = {
            "threshold": t,
            "nan_pct_overall": nan_pct_total[i],
            "mean_error_overall": mean_errors_total[i],
        }
        for lm in range(len(LANDMARK_NAMES)):
            row[f"nan_pct_{LANDMARK_NAMES[lm]}"] = nan_pct_per_lm[lm][i]
            row[f"mean_error_{LANDMARK_NAMES[lm]}"] = mean_errors_per_lm[lm][i]
        sweep_rows.append(row)
    p = os.path.join(output_dir, "threshold_sweep.xlsx")
    pd.DataFrame(sweep_rows).to_excel(p, index=False)
    print(f"  → Saved: {p}")


# ---------------------------------------------------------------------------
# NaN report
# ---------------------------------------------------------------------------
def report_nan_counts(
    all_maxima_flat, maxima_lm, maxima_per_patient_per_landmark, n_total, output_dir
):
    """
    Reports counts of points with heatmap_maxima == 0 (true blanks) as a
    baseline NaN reference. Saves to nan_counts.xlsx and prints to stdout.
    """
    print("\nNaN POINT COUNTS (heatmap max == 0)")
    print("-" * 60)
    nan_rows = []

    n_nan = int((all_maxima_flat == 0).sum())
    pct = 100.0 * n_nan / n_total
    print(f"  Overall: {n_nan} / {n_total}  ({pct:.1f}%)")
    nan_rows.append(
        {"group": "Overall", "n_total": n_total, "n_nan": n_nan, "pct_nan": pct}
    )

    print("  Per landmark:")
    for lm in range(len(LANDMARK_NAMES)):
        mv = maxima_lm[lm]
        n_nan = int((mv == 0).sum())
        pct = 100.0 * n_nan / len(mv)
        print(f"    {LANDMARK_NAMES[lm]:15s}  {n_nan} / {len(mv)}  ({pct:.1f}%)")
        nan_rows.append(
            {
                "group": LANDMARK_NAMES[lm],
                "n_total": len(mv),
                "n_nan": n_nan,
                "pct_nan": pct,
            }
        )

    print("  Per patient:")
    for patient_id in sorted(maxima_per_patient_per_landmark.keys()):
        all_mv = np.array(
            [
                v
                for lm in range(len(LANDMARK_NAMES))
                for v in maxima_per_patient_per_landmark[patient_id][lm]
            ]
        )
        n_nan = int((all_mv == 0).sum())
        pct = 100.0 * n_nan / len(all_mv)
        print(f"    Patient {patient_id:10s}  {n_nan} / {len(all_mv)}  ({pct:.1f}%)")
        nan_rows.append(
            {
                "group": f"Patient {patient_id}",
                "n_total": len(all_mv),
                "n_nan": n_nan,
                "pct_nan": pct,
            }
        )

    p = os.path.join(output_dir, "nan_counts.xlsx")
    pd.DataFrame(nan_rows).to_excel(p, index=False)
    print(f"\n  NaN counts saved to: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = Predictor(args)
    predictor.create_prediction_excel()
    paths = predictor.df["path"].dropna().tolist()

    n_landmarks = len(LANDMARK_NAMES)

    # Error/maxima storage
    per_patient_per_landmark = defaultdict(lambda: defaultdict(list))
    maxima_per_patient_per_landmark = defaultdict(lambda: defaultdict(list))
    all_errors = []
    all_errors_flat = []
    all_maxima_flat = []
    errors_lm = [[] for _ in range(n_landmarks)]
    maxima_lm = [[] for _ in range(n_landmarks)]

    # ------------------------------------------------------------------
    # Loop over every video
    # ------------------------------------------------------------------
    for path in paths:
        file_path = os.path.join(
            args.test_set_path, "test_hdf5", str(path) + "_interpolated.h5"
        )

        # Load ultrasound + ECG data
        with h5py.File(file_path, "r") as f:
            pixelsize = f["tissue"]["pixelsize"][()]

        pred, maxima, gt = predictor.compute_coordinates_annotations(file_path)

        pred = to_numpy(pred)  # (1, N, 3, 2)
        gt = to_numpy(gt)  # (1, N, 3, 2)
        maxima = to_numpy(maxima).squeeze(0)  # (N, 3)

        pred[:, :, :, 0] *= pixelsize[0] * 1000
        pred[:, :, :, 1] *= pixelsize[1] * 1000
        gt[:, :, :, 0] *= pixelsize[0] * 1000
        gt[:, :, :, 1] *= pixelsize[1] * 1000

        errors = compute_euclidean_errors(pred, gt)  # (N, 3)
        patient_id = str(path)

        for lm in range(n_landmarks):
            lm_errors = errors[:, lm]
            lm_maxima = maxima[:, lm]

            per_patient_per_landmark[patient_id][lm].extend(lm_errors.tolist())
            maxima_per_patient_per_landmark[patient_id][lm].extend(lm_maxima.tolist())
            all_errors.extend(lm_errors.tolist())
            all_errors_flat.extend(lm_errors.tolist())
            all_maxima_flat.extend(lm_maxima.tolist())
            errors_lm[lm].extend(lm_errors.tolist())
            maxima_lm[lm].extend(lm_maxima.tolist())

    all_errors = np.array(all_errors)
    all_errors_flat = np.array(all_errors_flat)
    all_maxima_flat = np.array(all_maxima_flat)
    for lm in range(n_landmarks):
        errors_lm[lm] = np.array(errors_lm[lm])
        maxima_lm[lm] = np.array(maxima_lm[lm])

    n_total = len(all_errors_flat)

    # ------------------------------------------------------------------
    # 1. Overall statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL ERROR")
    print("=" * 60)
    s = stats_summary(all_errors, "Overall")
    print(
        f"  N={s['n']}   Mean={s['mean']:.4f}   Std={s['std']:.4f}   "
        f"Median={s['median']:.4f}"
    )
    rows = [s]

    # ------------------------------------------------------------------
    # 2. Per-landmark statistics
    # ------------------------------------------------------------------
    print("\nPER-LANDMARK ERROR")
    print("-" * 40)
    per_landmark_errors = {}
    for lm in range(n_landmarks):
        lm_vals = np.array(
            [
                e
                for pid in per_patient_per_landmark
                for e in per_patient_per_landmark[pid][lm]
            ]
        )
        per_landmark_errors[LANDMARK_NAMES[lm]] = lm_vals
        s = stats_summary(lm_vals, LANDMARK_NAMES[lm])
        rows.append(s)
        print(
            f"  {LANDMARK_NAMES[lm]:15s}  N={s['n']}  Mean={s['mean']:.4f}  "
            f"Std={s['std']:.4f}  Median={s['median']:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Per-patient statistics
    # ------------------------------------------------------------------
    print("\nPER-PATIENT ERROR")
    print("-" * 40)
    per_patient_errors = {}
    for patient_id in sorted(per_patient_per_landmark.keys()):
        p_vals = np.array(
            [
                e
                for lm in range(n_landmarks)
                for e in per_patient_per_landmark[patient_id][lm]
            ]
        )
        per_patient_errors[patient_id] = p_vals
        s = stats_summary(p_vals, patient_id)
        rows.append(s)
        print(
            f"  Patient {patient_id:10s}  N={s['n']}  Mean={s['mean']:.4f}  "
            f"Std={s['std']:.4f}  Median={s['median']:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Save error statistics
    # ------------------------------------------------------------------
    stats_path = os.path.join(args.output_dir, "error_statistics.xlsx")
    pd.DataFrame(rows).to_excel(stats_path, index=False)
    print(f"\nStatistics saved to: {stats_path}")

    # ------------------------------------------------------------------
    # 5. Box plots  (unfiltered, and optionally at fixed threshold)
    # ------------------------------------------------------------------
    def generate_boxplots(
        bp_errors,
        bp_per_landmark,
        bp_per_patient,
        bp_per_patient_per_landmark,
        out_dir,
        title_suffix="",
    ):
        """Generates the full set of 4 box plots into out_dir."""
        os.makedirs(out_dir, exist_ok=True)

        boxplot_panel(
            {"All videos": bp_errors},
            title=f"Overall Euclidean Error{title_suffix}",
            ylabel="Euclidean error (mm)",
            output_path=os.path.join(out_dir, "boxplot_overall.png"),
        )

        boxplot_panel(
            bp_per_landmark,
            title=f"Error per Landmark{title_suffix}",
            ylabel="Euclidean error (mm)",
            output_path=os.path.join(out_dir, "boxplot_per_landmark.png"),
        )

        boxplot_panel(
            bp_per_patient,
            title=f"Error per Patient{title_suffix}",
            ylabel="Euclidean error (mm)",
            output_path=os.path.join(out_dir, "boxplot_per_patient.png"),
        )

        fig, axes = plt.subplots(
            1, n_landmarks, figsize=(5 * n_landmarks, 5), sharey=True
        )
        patient_ids = sorted(bp_per_patient_per_landmark.keys())
        for lm, ax in enumerate(axes):
            data = [
                np.array(bp_per_patient_per_landmark[pid][lm]) for pid in patient_ids
            ]
            bp = ax.boxplot(
                data, patch_artist=True, medianprops=dict(color="black", linewidth=2)
            )
            colors = plt.cm.Set2(np.linspace(0, 1, len(patient_ids)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_xticks(range(1, len(patient_ids) + 1))
            ax.set_xticklabels(patient_ids, rotation=35, ha="right", fontsize=8)
            ax.set_title(LANDMARK_NAMES[lm])
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            if lm == 0:
                ax.set_ylabel("Euclidean error (mm)")
        fig.suptitle(f"Error per Patient × Landmark{title_suffix}", fontsize=13, y=1.02)
        plt.tight_layout()
        p = os.path.join(out_dir, "boxplot_patient_x_landmark.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {p}")

    print("\nGenerating box plots…")
    generate_boxplots(
        all_errors,
        per_landmark_errors,
        per_patient_errors,
        per_patient_per_landmark,
        args.output_dir,
    )

    # --- Fixed-threshold box plots ---
    if args.thresh is not None:
        t = float(args.thresh)
        print(f"\nGenerating box plots at threshold={t}…")

        thresh_per_patient_per_lm = defaultdict(lambda: defaultdict(list))
        for pid in per_patient_per_landmark:
            for lm in range(n_landmarks):
                errs = np.array(per_patient_per_landmark[pid][lm])
                maxv = np.array(maxima_per_patient_per_landmark[pid][lm])
                thresh_per_patient_per_lm[pid][lm] = errs[maxv >= t].tolist()

        thresh_all_errors = np.concatenate(
            [
                np.array(thresh_per_patient_per_lm[pid][lm])
                for pid in thresh_per_patient_per_lm
                for lm in range(n_landmarks)
            ]
        )
        thresh_per_landmark = {
            LANDMARK_NAMES[lm]: np.concatenate(
                [
                    np.array(thresh_per_patient_per_lm[pid][lm])
                    for pid in thresh_per_patient_per_lm
                ]
            )
            for lm in range(n_landmarks)
        }
        thresh_per_patient = {
            pid: np.concatenate(
                [
                    np.array(thresh_per_patient_per_lm[pid][lm])
                    for lm in range(n_landmarks)
                ]
            )
            for pid in sorted(thresh_per_patient_per_lm)
        }

        n_after = len(thresh_all_errors)
        print(
            f"  Points retained: {n_after} / {n_total}  "
            f"({100.0 * (n_total - n_after) / n_total:.1f}% discarded)"
        )

        thresh_dir = os.path.join(args.output_dir, f"thresh_{args.thresh}")
        generate_boxplots(
            thresh_all_errors,
            thresh_per_landmark,
            thresh_per_patient,
            thresh_per_patient_per_lm,
            thresh_dir,
            title_suffix=f" (threshold={t})",
        )

    # ------------------------------------------------------------------
    # 6. Threshold-specific analyses (--thresh_method only)
    # ------------------------------------------------------------------
    if args.thresh_method:
        report_nan_counts(
            all_maxima_flat,
            maxima_lm,
            maxima_per_patient_per_landmark,
            n_total,
            args.output_dir,
        )

        print("\nRunning threshold sweep (0–100)…")
        run_threshold_sweep(
            all_errors_flat,
            all_maxima_flat,
            errors_lm,
            maxima_lm,
            n_total,
            args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import h5py
import pandas as pd
from PIL import ImageFont, ImageDraw, Image


def create_video(frames_dir, output_path, df, id, black_width=400, fps=5, scale=2.0):
    frame_files = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith((".png", ".jpg"))
        ],
        key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]),
    )

    things_to_write = [
        ("RVEDA = ", "RVEDA", "cm²", "rvad"),
        ("RVESA = ", "RVESA", "cm²", "rvas"),
        ("RVFAC = ", "RVFAC", "%", "rvfac"),
        ("TAPSEfw = ", "TAPSEfw", "mm", "tapsefw"),
        ("RV Strain\n(fw) = ", "RV strain (fw)", "%", "rvlsffw"),
        ("RVLSF = ", "RVLSF", "%", "rvlsfmid"),
        ("\u00a0RV Strain\n(global) = ", "RV strain (global)", "%", "rvlsfglobal"),
    ]

    first_frame = cv2.imread(frame_files[0])
    h, w, _ = first_frame.shape
    h2, w2 = int(h * scale), int(w * scale)
    black_width2 = int(black_width * scale)
    frame_size = (w2 + black_width2, h2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    font_path = "arial.ttf"  # Change if missing on your system
    font_size = 27
    font = ImageFont.truetype(font_path, font_size)

    total_loops = 5
    up_margin = 30

    for loop_idx in range(total_loops):
        for fpath in frame_files:
            frame = cv2.imread(fpath)
            frame = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_CUBIC)

            extended = np.zeros((h2, w2 + black_width2, 3), dtype=np.uint8)
            extended[:, :w2] = frame

            pil_img = Image.fromarray(cv2.cvtColor(extended, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Draw table parameters
            num_rows = len(things_to_write) + 1
            num_cols = 4
            col_width = black_width2 // num_cols
            row_height = int(h2 / (num_rows + 2))
            x0 = w2
            y0 = 70

            headers = ["Index", "Auto", "Manual", "Bias"]

            # Header row
            draw.rectangle(
                [x0, y0, x0 + col_width * num_cols, y0 + row_height], fill=(40, 40, 40)
            )
            for i, header in enumerate(headers):
                bbox = draw.textbbox((0, 0), header, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x_centered = x0 + i * col_width + (col_width - text_w) / 2
                y_centered = y0 + (row_height - text_h) / 2
                draw.text(
                    (x_centered, y_centered), header, font=font, fill=(255, 255, 255)
                )

            # Data rows
            for j, (label, col, unit, column) in enumerate(things_to_write):
                y = y0 + (j + 1) * row_height
                auto_val = df.at[id, column + "_auto"]
                manual_val = df.at[id, column + "_manual"]
                diff_val = auto_val - manual_val

                cells = [
                    f"{label.strip('= ')} ({unit})",
                    f"{auto_val:.1f}",
                    f"{manual_val:.1f}",
                    f"{diff_val:+.1f}",
                ]

                bg_color = (20, 20, 20) if j % 2 == 0 else (0, 0, 0)
                draw.rectangle(
                    [x0, y, x0 + col_width * num_cols, y + row_height], fill=bg_color
                )

                for i, cell_text in enumerate(cells):
                    bbox = draw.textbbox((0, 0), cell_text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    x_centered = x0 + i * col_width + (col_width - text_w) / 2
                    y_centered = y + (row_height - text_h) / 2
                    draw.text(
                        (x_centered, y_centered),
                        cell_text,
                        font=font,
                        fill=(255, 255, 255),
                    )

            extended = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(extended)

    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    prediction_excel = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\results\results_2_frames_method\best_combination\predictions.xlsx"
    manual_excel = r"c:\Users\User\Desktop\maesurements_jinyang.xlsx"

    prediction_folder = r"C:\Users\User\Desktop\illustrative_video"
    folder = r"C:\Users\User\Desktop\final_reviewed_dataset"
    output_folder = r"C:\Users\User\Desktop\illustrative_video\videos"

    # --- Load and merge auto + manual ---
    auto_df = pd.read_excel(prediction_excel)
    manual_df = pd.read_excel(manual_excel)

    print("Auto IDs:", auto_df["id"].unique())
    print("Manual IDs:", manual_df["id"].unique())

    # --- Clean and convert IDs safely ---
    def clean_id_column(df):
        if "id" not in df.columns:
            raise ValueError("Missing 'id' column in one of the Excel files.")
        df = df[df["id"].notna()]  # drop NaNs
        df["id"] = (
            df["id"].astype(str).str.extract(r"(\d+)")[0]
        )  # keep only numeric part
        df = df[df["id"].notna()]  # drop rows where no numeric id found
        df["id"] = df["id"].astype(int)
        return df

    auto_df = clean_id_column(auto_df)
    manual_df = clean_id_column(manual_df)

    merged = pd.merge(auto_df, manual_df, on="id", suffixes=("_auto", "_manual"))
    merged = merged.set_index("id")

    # --- Extract patient frequencies from HDF5 ---
    patients = []
    frequencies = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for file in os.listdir(subfolder_path):
            if "interpolated" in file:
                file_path = os.path.join(subfolder_path, file)
                with h5py.File(file_path, "r") as f:
                    times = f["tissue"]["times"][()]
                    frequency = 1 / (times[1] - times[0])
                patients.append(subfolder)
                frequencies.append(frequency)

    print(f"Patients found: {patients}")

    # --- Create videos ---
    for i, patient in enumerate(patients):
        patient_folder = os.path.join(prediction_folder, patient)
        if not os.path.exists(patient_folder):
            print(f"Skipping {patient}: folder not found.")
            continue
        for sub in os.listdir(patient_folder):
            sub_folder = os.path.join(patient_folder, sub)
            output_video_path = os.path.join(output_folder, f"{patient}.mp4")
            os.makedirs(output_folder, exist_ok=True)

            if int(patient) not in merged.index:
                print(f"Skipping {patient}: ID not in merged DataFrame.")
                continue

            create_video(
                sub_folder,
                output_video_path,
                df=merged,
                id=int(patient),
                fps=frequencies[i],
            )

import os
import datetime
import pandas as pd
from pydicom.filereader import dcmread
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
from torch.utils.data import DataLoader
from scipy import interpolate

from temporal_pipeline.dataloader.data_prep import SingleFileClipDataset
from temporal_pipeline.models.models import UNet3D
from temporal_pipeline.postprocessing.coordinates_from_heatmaps import argmax_3d
from temporal_pipeline.utils.plot import visualize_image

# This is a script containing all the utils to support the prediction of the model, both with and without thresholding.
# Useful for indices prediction, boxplot analysis, and BA plots


class RVCalculator:
    """Compute RV geometry indices from predicted ED/ES landmarks.

    If any landmark needed for a given index is NaN, that index is left as NaN.
    """

    INDEX_NAMES = [
        "tapse_fw",
        "tapse_sep",
        "rvfac",
        "ed_area",
        "es_area",
        "ed_len_fw",
        "ed_len_sep",
        "es_len_fw",
        "es_len_sep",
        "ed_diam",
        "es_diam",
        "ed_len_mid",
        "es_len_mid",
        "strain_fw",
        "strain_global",
        "strain_sep",
        "strain_mid",
    ]

    def __init__(self, ed_frame, es_frame, method="triangle"):
        self.ed_free_wall = torch.as_tensor(ed_frame[0])
        self.ed_septum = torch.as_tensor(ed_frame[1])
        self.ed_apex = torch.as_tensor(ed_frame[2])

        self.es_free_wall = torch.as_tensor(es_frame[0])
        self.es_septum = torch.as_tensor(es_frame[1])
        self.es_apex = torch.as_tensor(es_frame[2])

        self.method = method
        self._compute()

    @staticmethod
    def _has_nan(*tensors):
        return any(torch.isnan(t).any() for t in tensors)

    @staticmethod
    def _safe_norm(a, b, dim=-1):
        if RVCalculator._has_nan(a, b):
            return torch.tensor(float("nan"), device=a.device)
        return torch.linalg.norm(a - b, dim=dim)

    @staticmethod
    def _safe_percent_change(base, compare):
        if torch.isnan(base) or torch.isnan(compare) or base == 0:
            return torch.tensor(float("nan"), device=base.device)
        return (base - compare) / base * 100

    def _compute(self):
        ed_mid = (self.ed_free_wall + self.ed_septum) / 2
        es_mid = (self.es_free_wall + self.es_septum) / 2

        self.tapse_fw = self._safe_norm(self.ed_free_wall, self.es_free_wall)
        self.tapse_sep = self._safe_norm(self.ed_septum, self.es_septum)

        self.ed_len_fw = self._safe_norm(self.ed_free_wall, self.ed_apex)
        self.ed_len_sep = self._safe_norm(self.ed_septum, self.ed_apex)
        self.ed_len_mid = self._safe_norm(ed_mid, self.ed_apex)

        self.es_len_fw = self._safe_norm(self.es_free_wall, self.es_apex)
        self.es_len_sep = self._safe_norm(self.es_septum, self.es_apex)
        self.es_len_mid = self._safe_norm(es_mid, self.es_apex)

        self.ed_diam = self._safe_norm(self.ed_free_wall, self.ed_septum)
        self.es_diam = self._safe_norm(self.es_free_wall, self.es_septum)

        self.ed_area = self._compute_area(
            self.ed_free_wall,
            self.ed_apex,
            self.ed_septum,
            self.ed_len_fw,
            self.ed_len_sep,
            self.ed_diam,
        )
        self.es_area = self._compute_area(
            self.es_free_wall,
            self.es_apex,
            self.es_septum,
            self.es_len_fw,
            self.es_len_sep,
            self.es_diam,
        )

        self.rvfac = self._safe_percent_change(self.ed_area, self.es_area)

        self.strain_fw = self._safe_percent_change(self.ed_len_fw, self.es_len_fw)
        self.strain_sep = self._safe_percent_change(self.ed_len_sep, self.es_len_sep)
        self.strain_mid = self._safe_percent_change(self.ed_len_mid, self.es_len_mid)
        self.strain_global = self._safe_percent_change(
            self.ed_len_fw + self.ed_len_sep,
            self.es_len_fw + self.es_len_sep,
        )

    def _compute_area(self, free_wall, apex, septum, len_fw, len_sep, diam):
        if self.method == "triangle":
            return self._triangle_area(len_fw, len_sep, diam)
        if self.method == "spline":
            return self._spline_area(free_wall, apex, septum)
        raise ValueError("method must be 'triangle' or 'spline'")

    @staticmethod
    def _triangle_area(len_fw, len_sep, diam):
        if torch.isnan(len_fw) or torch.isnan(len_sep) or torch.isnan(diam):
            return torch.tensor(float("nan"), device=len_fw.device)

        s = (len_fw + len_sep + diam) / 2
        area = s * (s - len_fw) * (s - len_sep) * (s - diam)
        if torch.any(area < 0):
            return torch.tensor(float("nan"), device=len_fw.device)
        return torch.sqrt(area)

    @staticmethod
    def _spline_area(p1, apex, p2, n_points=6):
        if RVCalculator._has_nan(p1, apex, p2):
            return torch.tensor(float("nan"), device=p1.device)

        pts = torch.vstack([p1, apex, p2, p1]).cpu().numpy()
        tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = torch.from_numpy(np.column_stack((x_new, y_new))).to(
            dtype=torch.float32, device=p1.device
        )
        return 0.5 * torch.abs(
            torch.dot(poly[:, 0], torch.roll(poly[:, 1], -1))
            - torch.dot(poly[:, 1], torch.roll(poly[:, 0], -1))
        )


class Predictor:
    def __init__(self, args):
        self.args = args

        # set device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print("Using device:", self.device)

        # load the model and weights
        self.model = UNet3D(
            device=self.device,
            initial_channels=self.args.unet_initial_channels,
            num_res_units=self.args.unet_res_units,
        )
        self.model.load_state_dict(
            torch.load(self.args.model_checkpoints, map_location=self.device)[
                "model_state_dict"
            ]
        )

        self.model.eval()

    def create_prediction_excel(self):
        """this function takes as imput the arguments passed with predict_indices.py, and
        returns a blank pd.dataframe, with a column for each index, and some columns to
        recognise the acquisition: (Patient id, path to the hdf5 file, and datetime of the acquisition).
        The path is needed for clarity and for the prediction step, that comes after this.
        """
        # create an excel file with empty fields for each index
        self.df = pd.DataFrame(
            {
                "path": pd.Series(dtype="str"),
                "Date": pd.Series(dtype="str"),
                "Time": pd.Series(dtype="str"),
                "id": pd.Series(dtype="str"),
                "tapsefw": pd.Series(dtype="float64"),
                "tapsesep": pd.Series(dtype="float64"),
                "rvfac": pd.Series(dtype="float64"),
                "rvad": pd.Series(dtype="float64"),
                "rvas": pd.Series(dtype="float64"),
                "rvldfw": pd.Series(dtype="float64"),
                "rvldsep": pd.Series(dtype="float64"),
                "rvlsfw": pd.Series(dtype="float64"),
                "rvlssep": pd.Series(dtype="float64"),
                "tadd": pd.Series(dtype="float64"),
                "tasd": pd.Series(dtype="float64"),
                "rvldmid": pd.Series(dtype="float64"),
                "rvlsmid": pd.Series(dtype="float64"),
                "rvlsffw": pd.Series(dtype="float64"),
                "rvlsfglobal": pd.Series(dtype="float64"),
                "rvlsfsep": pd.Series(dtype="float64"),
                "rvlsfmid": pd.Series(dtype="float64"),
            }
        )

        dcm_path = os.path.join(self.args.test_set_path, "test_dicom")

        # initialize the line number, that is increased in each cycle of the loop to put data in the correct line
        line = 1

        for folder in os.listdir(dcm_path):
            if folder.startswith("."):
                continue

            folder_path = os.path.join(dcm_path, folder)
            for file in os.listdir(
                folder_path
            ):  # for each file (each one is in a separate folder)

                if file.startswith("."):
                    continue

                dcm = os.path.join(folder, file)
                # append the path of the file in the path column
                self.df.loc[line, "path"] = dcm

                # read the dicom file to get the patient id
                ds = dcmread(os.path.join(dcm_path, dcm))

                # write the patient id on the excel
                self.df.loc[line, "id"] = ds.PatientID

                # write the patient's AcquisitionDateTime (optional to recognize the patient if that's not anonymized)
                dt = str(ds.AcquisitionDateTime)
                self.df.loc[line, "Date"] = f"{dt[0:4]}/{dt[4:6]}/{dt[6:8]}"
                self.df.loc[line, "Time"] = f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}"

                # increment the line of the excel
                line = line + 1

    def pan_tompkins_detector(self, plot=False):
        def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0, order=1):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype="band")
            return lfilter(b, a, signal)

        ecg_signal = self.ecg
        fs = self.fs

        # 1. Bandpass filter (5–15 Hz)
        filtered = bandpass_filter(ecg_signal, fs, lowcut=5.0, highcut=15.0)

        # 2. Derivative
        derivative = np.diff(filtered)
        derivative = np.append(derivative, 0)

        # 3. Squaring
        squared = derivative**2

        # 4. Moving window integration (~150 ms window)
        window_size = int(0.150 * fs)
        mwa = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

        # 5. Peak detection with adaptive thresholding
        distance = int(0.25 * fs)
        threshold = 0.5 * np.max(mwa)
        peaks, _ = find_peaks(mwa, height=threshold, distance=distance)

        r_peaks = []
        search_window = int(0.1 * fs)
        for peak in peaks:
            start = max(peak - search_window, 0)
            end = min(peak + search_window, len(ecg_signal))
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)

        self.r_peaks = np.array(r_peaks)

        if plot:
            time = np.arange(len(ecg_signal)) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(time, ecg_signal, label="Original ECG")
            plt.plot(time[r_peaks], ecg_signal[r_peaks], "ro", label="Detected R Peaks")
            plt.title("ECG Signal with Detected R Peaks")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def inference(self):
        """predicts the coordinates of the landmarks in the
        specified hdf5 file"""

        # load the file in 3d images of 32 frames
        test_dataset = SingleFileClipDataset(
            self.file_path, clip_length=self.args.window_len, return_heatmaps=False
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        coordinates_list = []
        max_values_list = []

        for images, masks in test_loader:
            # inference
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            if self.args.heatmap_method:
                # extract the maximum activated pixel from the heatmaps
                com_tensor, max_values = argmax_3d(outputs, device=self.device)
                coordinates_list.append(com_tensor)
                max_values_list.append(max_values.permute(0, 2, 1))

        # Stack along dimension 1
        coordinates_array = torch.cat(coordinates_list, dim=1)
        max_values_array = torch.cat(max_values_list, dim=1)

        return coordinates_array, max_values_array

    def threshold_coords(
        self,
        coords: torch.Tensor,  # (N, C, B, 2)
        max_values: torch.Tensor,  # (N, C, B)
        threshold: float,
    ) -> torch.Tensor:  # (N, C, B, 2)
        mask = max_values >= threshold  # (N, C, B)
        mask = mask.unsqueeze(-1)  # (N, C, B, 1)  — broadcast over xy
        return torch.where(mask, coords, torch.full_like(coords, float("nan")))

    def find_es(self):
        """
        function that finds the es and ed frames from a window of predictions spanning from one r peak to the next one
        args:
        window: np.ndarray with shape (n, 3, 2) containing the coordinates of the 3 landmarks for each of the frames in an heartbeat

        returns:
        self.es: index of the frame where the fw annular point is the farthest from its ED position (frame 0).
        The index is relative to the beginning of the window.
        """
        fw = self.window[:, 0]
        distances_from_ed = torch.linalg.norm(fw - fw[0], axis=1)
        es = distances_from_ed.argmax()
        return es

    def _get_heartbeat_window(self, i, coordinates_array, beat_start):
        """Get the window of frames for heartbeat i, or None if invalid."""
        idx = beat_start[i]
        if idx >= len(coordinates_array):
            return None

        # Find valid ED idx
        if torch.isnan(coordinates_array[idx, 0]).any():
            if idx > 0 and not torch.isnan(coordinates_array[idx - 1, 0]).any():
                idx -= 1
            elif (
                idx < len(coordinates_array) - 1
                and not torch.isnan(coordinates_array[idx + 1, 0]).any()
            ):
                idx += 1
            else:
                print("skipping heart cycle because ED TAfw is NaN")
                return None

        # Determine window
        if i == len(beat_start) - 1:
            if len(coordinates_array) - idx <= 20:
                print("skip heart cycle")
                return None
            return coordinates_array[idx:]
        else:
            next_idx = beat_start[i + 1]
            if next_idx < len(coordinates_array):
                return coordinates_array[idx:next_idx]
            elif len(coordinates_array) - idx <= 20:
                print("skip heart cycle")
                return None
            else:
                return coordinates_array[idx:]

    def calculate_indices(self, coordinates_array):
        self.index_container = []
        for i in range(len(self.beat_start)):
            # extract the window of frames corresponding to one heartbeat
            # so here the problem is that since the model predicts clip_start
            # of n frames, there's the possibility that n-1 frames are left out
            # of the prediction, I need to check if the next r peak is in the
            # predicted coordinates and act consequently.
            # if it is then I consider a heartbeat the window of frames
            # between this beat and the next one. If not then the beat is valid if
            # there are at least 20 frames after my r peak. and I use those
            # frames as my window.

            self.window = self._get_heartbeat_window(
                i, coordinates_array, self.beat_start
            )
            if self.window is None:
                continue

            # in the window, find the index of the ED and ES frame
            ed_idx = 0  # since the window always starts from R-peak
            es_idx = self.find_es()

            # Extract frames
            self.ed = self.window[ed_idx]
            self.es = self.window[es_idx]

            # Scale coordinates
            pixelsize_tensor = torch.tensor(
                self.pixelsize, dtype=torch.float32, device=self.ed.device
            )  # TODO: this line is only to make it work on MPS, needs to be changed
            self.ed = self.ed * pixelsize_tensor
            self.es = self.es * pixelsize_tensor

            calculator = RVCalculator(
                ed_frame=self.ed, es_frame=self.es, method="spline"
            )
            self.index_container.append(
                torch.tensor(
                    [
                        calculator.tapse_fw * 1000,  # 0
                        calculator.tapse_sep * 1000,  # 1
                        calculator.rvfac,  # 2
                        calculator.ed_area * 1e4,  # 3
                        calculator.es_area * 1e4,  # 4
                        calculator.ed_len_fw * 1000,  # 5
                        calculator.ed_len_sep * 1000,  # 6
                        calculator.es_len_fw * 1000,  # 7
                        calculator.es_len_sep * 1000,  # 8
                        calculator.ed_diam * 1000,  # 9
                        calculator.es_diam * 1000,  # 10
                        calculator.ed_len_mid * 1000,  # 11
                        calculator.es_len_mid * 1000,  # 12
                        calculator.strain_fw,  # 13
                        calculator.strain_global,  # 14
                        calculator.strain_sep,  # 15
                        calculator.strain_mid,  # 16
                    ]
                )
            )

        if self.index_container:
            self.index_array = torch.stack(self.index_container, dim=0)
        else:
            self.index_array = torch.full(
                (0, len(RVCalculator.INDEX_NAMES)), float("nan")
            )

    def write_df(self, path):
        """
        Writes the mean of the calculated indices across all heart cycles for the current file
        into the corresponding row of the dataframe.

        Args:
            path (str): The path identifier for the current file, used to locate the correct row in the dataframe.

        The indices are averaged across heart cycles and assigned to the dataframe columns in the following order:
        - 'tapsefw': Mean TAPSE free wall (mm)
        - 'tapsesep': Mean TAPSE septum (mm)
        - 'rvfac': Mean RV fractional area change (%)
        - 'rvad': Mean RV area at end-diastole (cm²)
        - 'rvas': Mean RV area at end-systole (cm²)
        - 'rvldfw': Mean RV length diastolic free wall (mm)
        - 'rvldsep': Mean RV length diastolic septum (mm)
        - 'rvlsfw': Mean RV length systolic free wall (mm)
        - 'rvlssep': Mean RV length systolic septum (mm)
        - 'tadd': Mean TAPSE diameter diastolic (mm)
        - 'tasd': Mean TAPSE diameter systolic (mm)
        - 'rvldmid': Mean RV length diastolic mid (mm)
        - 'rvlsmid': Mean RV length systolic mid (mm)
        - 'rvlsffw': Mean RV longitudinal strain free wall (%)
        - 'rvlsfglobal': Mean RV longitudinal strain global (%)
        - 'rvlsfsep': Mean RV longitudinal strain septum (%)
        - 'rvlsfmid': Mean RV longitudinal strain mid (%)
        """
        # Find the row index in the dataframe corresponding to the current path
        row_idx = self.df[self.df["path"] == path].index[0]

        # Compute the mean across heart cycles using NaN-safe averaging.
        mean_indices = torch.nanmean(self.index_array, dim=0).cpu().numpy()
        mean_indices = np.where(np.isnan(mean_indices), -1000.0, mean_indices)

        # Assign the mean values to the dataframe columns
        self.df.loc[row_idx, "tapsefw"] = mean_indices[0]
        self.df.loc[row_idx, "tapsesep"] = mean_indices[1]
        self.df.loc[row_idx, "rvfac"] = mean_indices[2]
        self.df.loc[row_idx, "rvad"] = mean_indices[3]
        self.df.loc[row_idx, "rvas"] = mean_indices[4]
        self.df.loc[row_idx, "rvldfw"] = mean_indices[5]
        self.df.loc[row_idx, "rvldsep"] = mean_indices[6]
        self.df.loc[row_idx, "rvlsfw"] = mean_indices[7]
        self.df.loc[row_idx, "rvlssep"] = mean_indices[8]
        self.df.loc[row_idx, "tadd"] = mean_indices[9]
        self.df.loc[row_idx, "tasd"] = mean_indices[10]
        self.df.loc[row_idx, "rvldmid"] = mean_indices[11]
        self.df.loc[row_idx, "rvlsmid"] = mean_indices[12]
        self.df.loc[row_idx, "rvlsffw"] = mean_indices[13]
        self.df.loc[row_idx, "rvlsfglobal"] = mean_indices[14]
        self.df.loc[row_idx, "rvlsfsep"] = mean_indices[15]
        self.df.loc[row_idx, "rvlsfmid"] = mean_indices[16]

    def predict(self):
        """
        # TODO
        """

        # ensure path column exists
        if "path" not in self.df.columns:
            raise ValueError("Excel file must contain a 'path' column.")

        # extract the file paths
        paths = self.df["path"].dropna().tolist()

        for path in paths:
            # compose file path
            self.file_path = os.path.join(
                self.args.test_set_path,
                "test_hdf5",
                str(path)
                + "_interpolated.h5",  # TODO: exception if the file does not have interpolated in the name
            )

            print(f"Processing: {self.file_path}")

            # Load ultrasound + ECG data
            with h5py.File(self.file_path, "r") as f:
                self.images = f["tissue"]["data"][()]
                images_times = f["tissue"]["times"][()]
                self.ecg = f["ecg"]["ecg_data"][()]
                ecg_times = f["ecg"]["ecg_times"][()]
                self.pixelsize = f["tissue"]["pixelsize"][()]

            self.fs = 1 / (ecg_times[1] - ecg_times[0])  # ECG sampling frequency
            dt = images_times[1] - images_times[0]  # Image frame interval

            # Detect R-peaks in ECG
            self.pan_tompkins_detector()

            # for each R-peak, extract the closest frame
            self.beat_start = [
                np.argmin(np.abs(images_times - ecg_times[r])) for r in self.r_peaks
            ]

            # run the inference to get the coordinates_array and max value for heatmap
            coordinates_array, max_values_array = self.inference()

            thresholded_coordinates_array = self.threshold_coords(
                coordinates_array, max_values_array, self.args.thresh
            )

            thresholded_coordinates_array = thresholded_coordinates_array.squeeze(0)

            # calculate the indices from the predicted coordinates
            self.calculate_indices(thresholded_coordinates_array)

            # put the predicted indices in the dataframe and return it
            self.write_df(path)

        # save dataframe in excel
        self.df.to_excel(self.args.excel_path, index=False)

        # If ground truth provided, perform Bland-Altman analysis
        if hasattr(self.args, "gt_excel_path") and self.args.gt_excel_path:
            self.analyze_gt(self.args.gt_excel_path, self.args.thresh)

    def compute_coordinates_annotations(self, file_path):
        """predicts the coordinates of the landmarks in the
        specified hdf5 file, and returns them together with the manual annotations, and array of maximum values per point.
        Very useful for testing"""
        # load the file in 3d images of 32 frames
        test_dataset = SingleFileClipDataset(
            file_path,
            clip_length=self.args.window_len,
            return_heatmaps=False,
            load_from_annotations=True,
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        coordinates_list = []
        maxima_list = []
        gt_list = []

        for images, masks in test_loader:
            # inference
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            if self.args.heatmap_method:
                # extract the maximum activated pixel from the heatmaps
                com_tensor, maxima = argmax_3d(outputs, device=self.device)

                # lists in batches of 32 and then concatenate them
                coordinates_list.append(com_tensor)
                maxima_list.append(maxima.permute(0, 2, 1))
                gt_list.append(masks)

        # Stack along dimension 1
        coordinates_array = torch.cat(coordinates_list, dim=1)
        maxima_array = torch.cat(maxima_list, dim=1)
        gt_array = torch.cat(gt_list, dim=1)

        return coordinates_array, maxima_array, gt_array

    def analyze_gt(self, gt_path, thresh):
        """Perform Bland-Altman analysis against ground truth Excel."""
        gt_df = pd.read_excel(gt_path)

        # Create matching key
        self.df["key"] = (
            self.df["id"].astype(str) + "_" + self.df["Date"] + "_" + self.df["Time"]
        )
        gt_df["key"] = (
            gt_df["id"].astype("Int64").astype(str)
            + "_"
            + gt_df["Date"]
            + "_"
            + gt_df["Time"]
        )

        # Merge on key
        merged = pd.merge(self.df, gt_df, on="key", suffixes=("_pred", "_gt"))

        # Columns to analyze
        index_cols = [
            "tapsefw",
            "tapsesep",
            "rvfac",
            "rvad",
            "rvas",
            "rvldfw",
            "rvldsep",
            "rvlsfw",
            "rvlssep",
            "tadd",
            "tasd",
            "rvldmid",
            "rvlsmid",
            "rvlsffw",
            "rvlsfglobal",
            "rvlsfsep",
            "rvlsfmid",
        ]

        # Create plots folder
        plots_dir = os.path.join(
            os.path.dirname(self.args.excel_path), "bland_altman_" + str(thresh)
        )
        os.makedirs(plots_dir, exist_ok=True)

        # Error df
        error_df = merged[["id_pred", "Date_pred", "Time_pred"]].copy()
        error_df.rename(
            columns={"id_pred": "id", "Date_pred": "Date", "Time_pred": "Time"},
            inplace=True,
        )

        for col in index_cols:
            pred_col = f"{col}_pred"
            gt_col = f"{col}_gt"

            # Filter valid values
            valid = (merged[pred_col] != -1000) & (merged[gt_col] != -1000)
            pred = merged.loc[valid, pred_col]
            gt = merged.loc[valid, gt_col]

            if len(pred) == 0:
                continue

            diff = pred - gt
            mean_val = (pred + gt) / 2

            # Bland-Altman plot
            plt.figure(figsize=(8, 6))
            plt.scatter(mean_val, diff, alpha=0.6)
            plt.axhline(
                np.mean(diff),
                color="red",
                linestyle="--",
                label=f"Mean diff: {np.mean(diff):.2f}",
            )
            plt.axhline(
                np.mean(diff) + 1.96 * np.std(diff),
                color="blue",
                linestyle="--",
                label="+1.96 SD",
            )
            plt.axhline(
                np.mean(diff) - 1.96 * np.std(diff),
                color="blue",
                linestyle="--",
                label="-1.96 SD",
            )
            plt.xlabel("Mean of predicted and ground truth")
            plt.ylabel("Difference (predicted - ground truth)")
            plt.title(f"Bland-Altman Plot for {col}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"{col}_bland_altman.png"))
            plt.close()

            # Errors
            errors = pd.Series(np.nan, index=merged.index)
            errors.loc[valid] = diff
            error_df[f"{col}_error"] = errors.values

        # Save error Excel
        error_path = os.path.join(os.path.dirname(self.args.excel_path), "errors.xlsx")
        error_df.to_excel(error_path, index=False)

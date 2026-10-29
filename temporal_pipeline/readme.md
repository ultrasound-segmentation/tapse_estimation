# Temporal pipeline 

This README explains the purpose and layout of the temporal pipeline, the expected
dataset format, and how to use the main training, prediction, and analysis scripts.
The pipeline processes short sequences of ultrasound frames as 3D tensors
(`time × height × width`) and predicts three landmarks in every frame:

1. free-wall annular point (`TAfw`)
2. septal annular point (`TAsep`)
3. apex

The predicted landmarks are used to calculate TAPSE, RVFAC, ventricular lengths and
diameters, areas, and longitudinal strain measures.

## Main scripts

- [`training/temporal_train.py`](training/temporal_train.py) — train a temporal
  landmark model.
- [`pipeline_testing/indices_prediction/predict_indices.py`](pipeline_testing/indices_prediction/predict_indices.py)
  — run inference on a test set and calculate clinical indices.
- [`pipeline_testing/visual_analysis/save_predictions.py`](pipeline_testing/visual_analysis/save_predictions.py)
  — save test frames with predicted landmarks overlaid.
- [`pipeline_testing/visual_analysis/test_boxplots_sweep.py`](pipeline_testing/visual_analysis/test_boxplots_sweep.py)
  — calculate landmark errors and create boxplots or confidence-threshold sweeps.

Other useful modules:

- Data loading and preprocessing:
  [`dataloader/data_prep.py`](dataloader/data_prep.py) and
  [`dataloader/preprocessing.py`](dataloader/preprocessing.py)
- Models:
  [`models/models.py`](models/models.py) and
  [`models/resnet3d.py`](models/resnet3d.py)
- Augmentations:
  [`augmentations/augm_3d.py`](augmentations/augm_3d.py)
- Losses:
  [`losses/distances.py`](losses/distances.py) and
  [`losses/heatmap_losses.py`](losses/heatmap_losses.py)
- Landmark extraction:
  [`postprocessing/coordinates_from_heatmaps.py`](postprocessing/coordinates_from_heatmaps.py)
- Index calculation and prediction helpers:
  [`pipeline_testing/indices_prediction/prediction_utils.py`](pipeline_testing/indices_prediction/prediction_utils.py)

---

## Overview

The temporal pipeline consists of these stages:

1. **Prepare temporal clips.** The dataset classes read HDF5 acquisitions, select
   clips of a fixed length, normalize the frames, resize or crop them to the model
   input size, and optionally apply temporal/spatial augmentations.
2. **Train a 3D model.** [`temporal_train.py`](training/temporal_train.py) trains
   either a MONAI 3D U-Net (`3D_UNet`) or the `echocoder` ResNet-based encoder-decoder.
3. **Extract landmarks.** Model outputs are converted to coordinates with either
   center-of-mass extraction or heatmap argmax extraction.
4. **Find cardiac phases and calculate indices.** The prediction utilities use the
   ECG signal to find R-peaks, identify end-diastolic and end-systolic frames, and
   calculate indices from the corresponding landmark positions.
5. **Evaluate predictions.** Predictions can be exported to Excel, visualized as
   images, or compared with annotations through error statistics and boxplots.

## Dataset format

### Training and validation data

The temporal trainer uses the same split-file approach as the 2D pipeline. Set
`DATASET_PATH` to the directory containing the HDF5 files and
`DIVISION_TXT_PATH` to the split file (both can be placed in `.env`). The split
file contains `Validation:`, `Test:`, and `Training:` sections with HDF5 paths
relative to `DATASET_PATH`, or absolute paths:

```text
data/division.txt
DATASET_PATH/
├── patient_001/
│   └── acquisition.h5
└── ...
```

Run training with `--dataset_path` and `--division_path` to override the
environment values. Each selected training HDF5 file must contain:

- `frames`: ultrasound frames stored as `height × width × time`
- `annotations`: landmark coordinates stored as `time × 3 × 2`

Coordinates use `(x, y)` order. Acquisitions shorter than the requested
`--window_len` are skipped by the random training dataset. The validation dataset
iterates over non-overlapping clips.

### Prediction data

[`predict_indices.py`](pipeline_testing/indices_prediction/predict_indices.py)
expects a test-set directory containing matching `test_dicom/` and `test_hdf5/`
directories. Patient/acquisition subdirectories should have corresponding DICOM and
HDF5 files:

```text
test_set/
├── test_dicom/
│   └── patient_or_acquisition_id/
│       └── acquisition.dcm
└── test_hdf5/
    └── patient_or_acquisition_id/
        └── acquisition.h5
```

The HDF5 files used for prediction must contain the tissue frames, ECG, and pixel
size in the structure expected by `prediction_utils.py`:

- `tissue/data`
- `tissue/pixelsize`
- `ecg/ecg_data`
- corresponding time arrays where present (`tissue/times` and `ecg/ecg_times`)

The DICOM metadata supplies the patient ID and acquisition date/time used to match
rows in the output Excel file.

---

## How to use the main scripts

Run commands from the repository root. The scripts use the Python environment
configured for the project; install the dependencies from [`requirements.txt`](../requirements.txt)
before running them.

### 1. Training

[`training/temporal_train.py`](training/temporal_train.py) supports two target
representations:

- **Coordinate training** (default): predicts landmark positions and uses
  [`CombinedLandmarkLoss`](losses/distances.py).
- **Heatmap training** (`--heatmap_training`): creates activation maps around each
  landmark and uses the heatmap/motion losses in
  [`losses/heatmap_losses.py`](losses/heatmap_losses.py).

Example using the default 3D U-Net:

```bash
python temporal_pipeline/training/temporal_train.py \
  --division_path data/division.txt \
  --model 3D_UNet \
  --window_len 32 \
  --batch_size 4 \
  --epochs 300 \
  --save_model_path temporal_pipeline/runs/my_experiment
```

For heatmap targets, add `--heatmap_training`. The heatmap curriculum can be
controlled with `--heatmap_initial_radius`, `--heatmap_radius_step`,
`--heatmap_radius_step_epochs`, `--heatmap_radius_min`, and
`--heatmap_peak_value`.

Useful options include:

- `--model {3D_UNet,echocoder}` — choose the model architecture.
- `--window_len` — number of frames processed at once; the default is 32.
- `--augm_version` — select an augmentation version from
  [`augmentations/augm_3d.py`](augmentations/augm_3d.py).
- `--smooth_annotations --smooth_window N` — smooth annotations before training.
- `--from_scratch` — explicitly indicate a new training run.
- `--wandb` — enable Weights & Biases logging.
- `--seed` — set the Python and PyTorch random seed.

Checkpoints contain `model_state_dict`. The architecture options used during
inference must match those used during training, especially
`--unet_initial_channels`, `--unet_res_units`, and `--window_len`.

### 2. Prediction and clinical indices

[`pipeline_testing/indices_prediction/predict_indices.py`](pipeline_testing/indices_prediction/predict_indices.py)
loads a trained 3D U-Net, predicts landmarks over each test acquisition, detects
heartbeats from the ECG, and writes one row per acquisition to an Excel file.
The output includes TAPSE, RVFAC, areas, lengths, diameters, and strain-related
indices.

Example:

```bash
python temporal_pipeline/pipeline_testing/indices_prediction/predict_indices.py \
  --excel_path data/temporal_predictions.xlsx \
  --heatmap_method \
  --model_checkpoints checkpoints/best_model.pth \
  --test_set_path data/temporal_test_set \
  --window_len 32 \
  --unet_initial_channels 16 \
  --unet_res_units 2
```

`--heatmap_method` must be supplied for checkpoints trained to output heatmaps.
The `--thresh` argument is a confidence threshold for predicted landmark maxima;
landmarks below it are treated as missing when indices are calculated. If a
ground-truth Excel file is supplied with `--gt_excel_path`, the predictor also
produces an error workbook for matched acquisitions.

The prediction code scales coordinates using `tissue/pixelsize` before calculating
physical distances and areas. If the checkpoint was trained with different U-Net
settings, pass the same values to the prediction script.

### 3. Save visual predictions

To inspect predictions frame by frame, use
[`pipeline_testing/visual_analysis/save_predictions.py`](pipeline_testing/visual_analysis/save_predictions.py):

```bash
python temporal_pipeline/pipeline_testing/visual_analysis/save_predictions.py \
  --checkpoints checkpoints/best_model.pth \
  --heatmap_method \
  --test_set_path data/final_reviewed_dataset_for_3d/test \
  --save_path data/temporal_prediction_images \
  --window_len 32
```

This script loads validation-style HDF5 files, extracts landmark coordinates, and
saves images with the three predictions overlaid. Use `--heatmap_method` for a
heatmap checkpoint; otherwise it uses center-of-mass extraction for model outputs.

### 4. Error analysis and threshold sweeps

[`pipeline_testing/visual_analysis/test_boxplots_sweep.py`](pipeline_testing/visual_analysis/test_boxplots_sweep.py)
compares predictions with annotations and can either create boxplots at a selected
threshold or sweep thresholds from 0 to 100:

```bash
python temporal_pipeline/pipeline_testing/visual_analysis/test_boxplots_sweep.py \
  --excel_path data/temporal_predictions.xlsx \
  --model_checkpoints checkpoints/best_model.pth \
  --test_set_path data/final_reviewed_dataset_for_3d/test \
  --output_dir data/temporal_boxplots \
  --heatmap_method \
  --thresh_method
```

Without `--thresh_method`, pass `--thresh VALUE` to generate boxplots for one
confidence threshold. The output directory contains summary statistics and plots.

---

## Important implementation details

- A model input has shape `batch × 1 × time × height × width`; the default
  temporal window is 32 frames.
- Images are normalized per clip and resized/cropped to the preprocessing target
  size before inference.
- Landmark order is consistently free-wall annular point, septal annular point,
  apex.
- Heatmap outputs can be converted to coordinates with
  [`argmax_3d`](postprocessing/coordinates_from_heatmaps.py). Continuous outputs
  can be converted with [`center_of_mass_3d`](postprocessing/coordinates_from_heatmaps.py).
- Invalid or below-threshold landmarks propagate as `NaN` into index calculations
  rather than being silently replaced with a measurement.
- The prediction code requires the model architecture and checkpoint format to
  agree. A checkpoint saved by the training `Trainer` is expected to contain
  `model_state_dict`.
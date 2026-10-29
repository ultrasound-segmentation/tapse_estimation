# RV Health Indices Prediction from TEE Images

This repository provides a pipeline for predicting clinically relevant indices that assess the health status of the **right ventricle (RV)** from transesophageal echocardiography (TEE) images.

## Target Indices

| Index | Full Name |
|-------|-----------|
| **TAPSE** | Tricuspid Annular Plane Systolic Excursion |
| **RVLFS** | RV Linear Strain |
| **RVFAC** | RV Fractional Area Change |
| **RV Diameter** | Right Ventricular Diameter |

## Approach

Indices are derived by tracking three anatomical landmarks across TEE frames:

1. Free wall tricuspid annulus point
2. Septal wall tricuspid annulus point
3. Apex of the RV

## Pipelines

### `twod` ✅
Individual 2D TEE frames are fed into a model that predicts and tracks the three target landmarks per frame, from which clinical indices are subsequently calculated.

For a detailed explanation of how this pipeline works, see [`/twod/readme.md`](/twod/readme.md). 

### `2D+T` ✅ (Tracking, prototype)
In this pipeline, a 3d model is fed with sequences of N frames, and it tracks the landmarks in the whole time window at the same time. 
The model is trained to reproduce heatmaps containing a "gaussian" centered on the ground truth coordinates. The coordinates are then extracted as the coordinates of the center of mass or of the max value of the output heatmaps. In order for the training to be stable, the radius of the gaussian starts at high values, and then is reduced gradually during the training. The obtained results are similar to the twod pipeline, but it has not been explored much.

## Getting Started

```bash
git clone https://github.com/MatteoMissana/tapse_estimation
cd tapse_estimation
. setup.sh
```

Then refer to [`/twod/readme.md`](/twod/readme.md) for pipeline-specific instructions.

### .env file
To start, you should add an `.env` file. 

The 2D training data is converted from annotated H5 files into three compressed
NPZ files: `train.npz`, `val.npz`, and `test.npz`.

Configure the paths in `.env`:

```env
DATASET_PATH="path to directory with all subdirectories with .h5 files"
CONVERTED_DATASET_PATH="path where you want to store the converted datasets"
DIVISION_TXT_PATH="path to the directory that defines what is training-, validation- and test-data"
GROUND_TRUTH_EXCEL="path to ground truth excel file"
```

Regarding the `division.txt`, the file should follow this structure: 
```txt
Validation:
<path_to_h5_file_from_the_.env_DATASET_PATH-file>
Test:
<path_to_h5_file_from_the_.env_DATASET_PATH-file>
Training:
<path_to_h5_file_from_the_.env_DATASET_PATH-file>
```

# Running the pipeline 
We have created multiple `sbatch`-scripts to make the project run easier on Idun (NTNU's super computer). The intended order is:

### 1. Set up the environment
. setup.sh

### Optional: only for creating/splitting the datasets
This is in order to actually do the division (defined in `division.txt`), simply run 
. setup.sh --division

### 2. Train the model
sbatch training.sh

### 3. Test the trained model on annotated data
sbatch testing.sh

### 4. Run inference and calculate clinical indices
sbatch inference.sh

### 5. Compare predictions with manual annotations
sbatch validation.sh

### 6. Optional: create illustrative videos
sbatch video_creation.sh


## Formatting

After running the [`./requirements.txt-file`](./requirements.txt):

Locate the project-directory and format the project with:

```bash
black .
```

## Demo (click on it for the video)

[![Demo Video](https://img.youtube.com/vi/IUViyJUNPxE/0.jpg)](https://youtu.be/IUViyJUNPxE)

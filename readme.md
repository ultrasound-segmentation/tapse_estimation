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

For a detailed explanation of how this pipeline works, see [`./twod/README.md`](./twod/README.md).

### `2D+T` ✅ (Tracking, prototype)
In this pipeline, a 3d model is fed with sequences of N frames, and it tracks the landmarks in the whole time window at the same time. 
The model is trained to reproduce heatmaps containing a "gaussian" centered on the ground truth coordinates. The coordinates are then extracted as the coordinates of the center of mass or of the max value of the output heatmaps. In order for the training to be stable, the radius of the gaussian starts at high values, and then is reduced gradually during the training. The obtained results are similar to the twod pipeline, but it has not been explored much.

## Getting Started

```bash
git clone https://github.com/MatteoMissana/tapse_estimation
cd tapse_estimation
pip install -r requirements.txt
pip install -e .
```

Then refer to [`./twod/README.md`](./twod/README.md) for pipeline-specific instructions.

### .env file
To start, you should add a `.env` file. Here you should add
```
DATASET_PATH=<path to directory with all subdirectories with .h5 files>
```

## Demo (click on it for the video)

[![Demo Video](https://img.youtube.com/vi/IUViyJUNPxE/0.jpg)](https://youtu.be/IUViyJUNPxE)

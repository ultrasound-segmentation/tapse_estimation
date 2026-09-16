import torch
from torch.utils.data import DataLoader
import argparse

from temporal_pipeline.pipeline_testing.indices_prediction.prediction_utils import (
    Predictor,
)

"""THis code is used to predict the clinical indices from a folder of test data.
* Since I can't predict directly from the dicom data, but some of the information I need 
* is on the dicom, the dataset should contain both files
* the folder should look lije this:

    |--test_dicom
    |--|--100
    |--|--111
    |--|--140
    |--|--149
    |--|--160
    |--|--170
    |--|--190
    |--|--198
    |--|--199
    |--|--920
    |--test_hdf5
    |--|--100
    |--|--111
    |--|--140
    |--|--149
    |--|--160
    |--|--170
    |--|--190
    |--|--198
    |--|--199
    |--|--920
 
 * with each subfolder containing the relative hdf5 or dicom (without extension)
 * files.

 example usage:

python3 temporal_pipeline/pipeline_testing/indices_prediction/predict_indices.py 
    --excel_path "/Users/mmissana/Desktop/tapse_estimation/data/ba_plots/excel_with_indices.xlsx" 
    --heatmap_method --model_checkpoints "/Users/mmissana/Desktop/Best_model_Unet_6_mag/best.pt" 
    --test_set_path "/Users/mmissana/Desktop/tapse_estimation/data/test_set_for_prediction" 
    --thresh 50 
    --gt_excel_path "/Users/mmissana/Desktop/tapse_estimation/data/ba_plots/maesurements_jinyang_modified_for_analysis.xlsx"

 """


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="test the trained model and save the images with predictions"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="path where to save the excel. Example: ",
    )
    parser.add_argument(
        "--heatmap_method",
        action="store_true",
        help="if the  model was trained with the heatmap method",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        required=True,
        help="Path to the checkpoints of the trained model",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        required=True,
        help="path of the folder with the hdf5/dicom files",
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=0,
        help="threshold of confidence for the model predictions",
    )
    parser.add_argument(
        "--unet_initial_channels",
        type=int,
        default=16,
        help="number of filters in the first layer of the UNet",
    )
    parser.add_argument(
        "--unet_res_units",
        type=int,
        default=2,
        help="number of residual units the UNet",
    )
    parser.add_argument(
        "--window_len",
        type=int,
        default=32,
        help="number of frames the model receives in input",
    )
    parser.add_argument(
        "--gt_excel_path",
        type=str,
        help="path to ground truth excel for Bland-Altman analysis",
    )
    return parser.parse_args()


args = parse_args()

predictor = Predictor(args)
predictor.create_prediction_excel()
predictor.predict()

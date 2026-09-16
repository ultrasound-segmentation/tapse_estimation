import os
import h5py
import numpy as np
import pandas as pd
from pydicom import dcmread

"""this file uses a folder of dicom files to create the excel file that needs to be give as input to 
2d/indices_prediction.py. It extracts both the path and the patient id and writes it in the specific field in the excel 
file"""

dcm_path = r"/Users/mmissana/Desktop/Postop RV images test data N = 10 x 3"
excel_path = r"/Users/mmissana/Desktop/best_unet.xlsx"

# create an excel file with empty fields for each index
df = pd.DataFrame(
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
        df.loc[line, "path"] = dcm

        # read the dicom file to get the patient id
        ds = dcmread(os.path.join(dcm_path, dcm))

        # write the patient id on the excel
        df.loc[line, "id"] = ds.PatientID

        # write the patient's AcquisitionDateTime (optional to recognize the patient if that's not anonymized)
        dt = str(ds.AcquisitionDateTime)
        df.loc[line, "Date"] = f"{dt[0:4]}/{dt[4:6]}/{dt[6:8]}"
        df.loc[line, "Time"] = f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}"

        # increment the line of the excel
        line = line + 1

df.to_excel(excel_path, index=False)
print("excel saved successfully")

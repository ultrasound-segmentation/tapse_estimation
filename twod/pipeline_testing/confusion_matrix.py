import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Code to create a confusion matrix from excel file with the predicted and annotated value for indices


# === Load data ===
file_path = r"2d/confusion_matrix_material.xlsx"
df = pd.read_excel(file_path)


# === Define thresholds ===
def classify_rv_function(tapse, strain, rvfac):
    """Return 1 for Normal, 0 for Reduced"""
    if tapse > 18 and strain > 16.5 and rvfac > 35:
        return 1  # Normal
    else:
        return 0  # Reduced


# === Apply classification to true and predicted data ===
df["true_class"] = df.apply(
    lambda row: classify_rv_function(
        row["TAPSEfw true value"],
        row["RV linear strain (global) true value"],
        row["RVFAC true value"],
    ),
    axis=1,
)

df["pred_class"] = df.apply(
    lambda row: classify_rv_function(
        row["TAPSEfw prediction"],
        row["RV linear strain (global) prediction"],
        row["RVFAC prediction"],
    ),
    axis=1,
)

# === Compute confusion matrix ===
cm = confusion_matrix(df["true_class"], df["pred_class"], labels=[1, 0])

# === Plot confusion matrix ===
labels = ["Normal RV function", "Reduced RV function"]

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)

plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.title(
    "Confusion Matrix – Combined TAPSEfw + RV linear Strain (global) + RVFAC analysis",
    fontsize=12,
    pad=20,
)

plt.tight_layout()

# === Save to PDF ===
plt.savefig(
    r"D:\mmissana\tapse_estimation\2d\results\confusion_matrices\confusion_matrix_combined_extracted.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()

# Optionally print counts for reference
print("Confusion matrix (rows = true, columns = predicted):\n", cm)

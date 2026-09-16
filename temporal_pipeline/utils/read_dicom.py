import pydicom
import os


def read_dicom_name(file_path):
    """
    Reads and prints all fields from a DICOM file.
    :param file_path: Path to the DICOM file.
    """

    dicom_data = pydicom.dcmread(file_path)

    # List of tags related to image data
    image_related_tags = [
        (0x7FE0, 0x0010),  # Pixel Data
        (0x0028, 0x0004),  # Photometric Interpretation
        (0x0028, 0x0010),  # Rows
        (0x0028, 0x0011),  # Columns
        (0x0028, 0x0100),  # Bits Allocated
        (0x0028, 0x0101),  # Bits Stored
        (0x0028, 0x0102),  # High Bit
        (0x0028, 0x0103),  # Pixel Representation
    ]

    # Print non-image fields
    for elem in dicom_data:
        if elem.tag not in image_related_tags:
            print(f"{elem.name}: {elem.value}")

    # Print patient name separately
    # for attr in dicom_data.dir():
    #     value = getattr(dicom_data, attr, None)  # Get the value safely
    #     print(f"{attr}")

    # if hasattr(dicom_data, 'StudyID'):
    #     print(f"Study id: {dicom_data.StudyID}")

    if hasattr(dicom_data, "SOPClassUID"):
        return (dicom_data.StudyInstanceUID, dicom_data.SeriesInstanceUID)
    else:
        return None


if __name__ == "__main__":
    read_dicom_name(
        r"data/2d_focused_rv/RV focused TEE images_complete/__103151/P3KAFSO2"
    )
    # folder = r"data/2d_focused_rv/RV focused TEE images_complete"
    # patients = []

    # # Prima iterazione per raccogliere tutti i pazienti unici
    # for subf in os.listdir(folder):
    #     subf_path = os.path.join(folder, subf)
    #     for file in os.listdir(subf_path):
    #         file_path = os.path.join(subf_path, file)
    #         patient = read_dicom_name(file_path)

    #         if patient:
    #             patients.append(patient)

    # # Creazione di un dizionario con liste vuote per ogni paziente unico
    # unique_patients_dict = {patient: [] for patient in set(patients)}

    # # Seconda iterazione per riempire le liste con i percorsi dei file
    # for subf in os.listdir(folder):
    #     subf_path = os.path.join(folder, subf)
    #     for file in os.listdir(subf_path):
    #         file_path = os.path.join(subf_path, file)
    #         patient = read_dicom_name(file_path)
    #         if patient:
    #             unique_patients_dict[patient].append(file_path)

    # # Stampa del risultato
    # for patient, files in unique_patients_dict.items():
    #     print(f"Patient: {patient}")
    #     for f in files:
    #         print(f"  - {f}")

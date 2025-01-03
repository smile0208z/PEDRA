import os
import numpy as np
import torch
import gc
import pydicom
from tqdm import tqdm

def process_enhance_data(input_path, status_dict, day_dict):
    def gather_data(root_path, status_ref, day_ref):
        dicom_files = []
        nii_files = []
        status_collected = []
        patient_ids = []
        day_collected = []
        first_level = os.listdir(root_path)
        for level1 in first_level:  # Machine Layer
            sub_path1 = os.path.join(root_path, level1)
            second_level = os.listdir(sub_path1)
            for level2 in second_level:  # Patient Layer
                patient_id = int(level2)
                sub_path2 = os.path.join(sub_path1, level2)
                third_level = os.listdir(sub_path2)
                for level3 in third_level:
                    if level3 == 'enhance':
                        sub_path3 = os.path.join(sub_path2, level3)
                        fourth_level = os.listdir(sub_path3)
                        for file in fourth_level:
                            if file.endswith('.dcm'):
                                dicom_file_path = os.path.join(sub_path3, file)
                                dicom_files.append(dicom_file_path)
                                patient_ids.append(patient_id)
                                if patient_id in status_ref:
                                    status_collected.append(status_ref[patient_id])
                                if patient_id in day_ref:
                                    day_collected.append(day_ref[patient_id])
                            if file.endswith('.nii'):
                                nii_file_path = os.path.join(sub_path3, file)
                                nii_files.append(nii_file_path)
        return dicom_files, nii_files, status_collected, patient_ids, day_collected

    # Gather enhancement data paths
    dicom_paths, nii_paths, statuses, patients, days = gather_data(input_path, status_dict, day_dict)

    # DICOM Image Generator
    def dicom_image_generator(file_paths):
        for path in file_paths:
            dicom_image = pydicom.dcmread(path)
            image_data = dicom_image.pixel_array
            # Normalization
            image_max = image_data.max()
            image_min = image_data.min()
            normalized_image = (image_data - image_min) / (image_max - image_min)
            yield normalized_image

    # Load enhancement DICOM data
    dicom_data_gen = dicom_image_generator(dicom_paths)
    processed_dicom_data = []
    for dicom_img in tqdm(dicom_data_gen, desc='Loading enhancement DICOM data', total=len(dicom_paths)):
        processed_dicom_data.append(dicom_img)

    # Convert to NumPy array and clear list to free memory
    dicom_data_np = np.array(processed_dicom_data)
    del processed_dicom_data
    gc.collect()

    # Convert to tensors
    def data_to_tensor(data_array):
        tensor_data = torch.tensor(data_array, dtype=torch.float32).unsqueeze(1)
        return tensor_data

    dicom_tensors = data_to_tensor(dicom_data_np)

    # Free memory by deleting NumPy array
    del dicom_data_np
    gc.collect()

    status_tensors = data_to_tensor(np.array(statuses))

    # Return processed enhancement data
    return dicom_tensors, status_tensors, patients, days

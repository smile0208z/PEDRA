import os
import numpy as np
import torch
import gc
import pydicom
from tqdm import tqdm

def process_plain_data(input_directory, status_dict, day_dict):
    def gather_plain_data(root_dir, status_map, day_map):
        dicom_paths = []
        nii_paths = []
        status_records = []
        patient_ids = []
        day_records = []
        top_level = os.listdir(root_dir)
        for level1 in top_level:  # Machine Layer
            sub_path1 = os.path.join(root_dir, level1)
            second_level = os.listdir(sub_path1)
            for level2 in second_level:  # Patient Layer
                patient_id = int(level2)
                sub_path2 = os.path.join(sub_path1, level2)
                third_level = os.listdir(sub_path2)
                for level3 in third_level:
                    if level3 == 'plain':
                        sub_path3 = os.path.join(sub_path2, level3)
                        fourth_level = os.listdir(sub_path3)
                        for file in fourth_level:
                            if file.endswith('.dcm'):
                                dicom_file_path = os.path.join(sub_path3, file)
                                dicom_paths.append(dicom_file_path)
                                patient_ids.append(patient_id)
                                if patient_id in status_map:
                                    status_records.append(status_map[patient_id])
                                if patient_id in day_map:
                                    day_records.append(day_map[patient_id])
                            if file.endswith('.nii'):
                                nii_file_path = os.path.join(sub_path3, file)
                                nii_paths.append(nii_file_path)
        return dicom_paths, nii_paths, status_records, patient_ids, day_records

    # Gather plain data paths
    dicom_paths, nii_paths, status_data, patients, days = gather_plain_data(input_directory, status_dict, day_dict)

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

    # Load plain DICOM data
    dicom_data_gen = dicom_image_generator(dicom_paths)
    processed_dicom_data = []
    for dicom_img in tqdm(dicom_data_gen, desc='Loading plain DICOM data', total=len(dicom_paths)):
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

    status_tensors = data_to_tensor(np.array(status_data))

    # Return plain data
    return dicom_tensors, status_tensors, patients, days

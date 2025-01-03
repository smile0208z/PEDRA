from data.data_split import generate_data_pipelines
from data.e_data_load import process_enhance_data
from model.model_64 import EfficientNetB0
from train.train_model import train_and_evaluate_model
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

# Data Preparation
data_path = r'/home/guest/zhao/ai'
labels_data = pd.read_csv(r'/home/guest/zhao/status.csv', encoding='gbk')
name_status = dict(zip(labels_data['放射科号'], labels_data['status']))
name_days = dict(zip(labels_data['放射科号'], labels_data['days']))
dicom_tensors, status_tensors, patients, days = process_enhance_data(data_path, name_status, name_days)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("Available GPU:", gpu_name)

# Model Initialization with Dynamic Assignment
model = EfficientNetB0().to(device)
model = nn.DataParallel(model)

# Loss and Optimizer Setup
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Data Loaders
train_loader, val_loader, test_loader, dataloader = generate_data_pipelines(dicom_tensors, status_tensors)

# Training Function Invocation
start_time = time.time()

# Dynamic Function for Training and Evaluation
train_and_evaluate = partial(
    train_and_evaluate_model,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    device=device,
    patience=10,
    output_dir=r'/home/guest/zhao',
    name='64p.csv'
)

metrics = train_and_evaluate()

# Optional: Measure Training Duration
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time / 60:.2f} minutes")

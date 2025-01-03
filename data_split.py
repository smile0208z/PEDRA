import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def generate_data_pipelines(data_tensor, label_tensor, b_size=32, t_split=0.2, r_seed=42):
    """
    Generate data loaders for training, validation, testing, and full dataset.
    """
    # First Split: Training vs Temp (Validation + Test)
    primary_data, residual_data, primary_labels, residual_labels = train_test_split(
        data_tensor, label_tensor, test_size=t_split, random_state=r_seed
    )

    # Second Split: Validation vs Test
    secondary_data, tertiary_data, secondary_labels, tertiary_labels = train_test_split(
        residual_data, residual_labels, test_size=0.5, random_state=r_seed
    )

    # Dataset Wrapping
    dataset_mapping = {
        'alpha': TensorDataset(primary_data, primary_labels),  # Training Dataset
        'beta': TensorDataset(secondary_data, secondary_labels),  # Validation Dataset
        'gamma': TensorDataset(tertiary_data, tertiary_labels),  # Test Dataset
        'omega': TensorDataset(data_tensor, label_tensor)  # Full Dataset
    }

    # DataLoader Construction
    loader_config = {
        'alpha': {'batch_size': b_size, 'shuffle': True},
        'beta': {'batch_size': b_size, 'shuffle': False},
        'gamma': {'batch_size': b_size, 'shuffle': False},
        'omega': {'batch_size': b_size, 'shuffle': False},
    }

    loaders = {
        key: DataLoader(dataset_mapping[key], **loader_config[key])
        for key in dataset_mapping
    }

    # Return DataLoaders
    return (loaders['alpha'], loaders['beta'], loaders['gamma'], loaders['omega'])


import pandas as pd
from openpyxl import load_workbook
from sklearn.preprocessing import StandardScaler
import numpy as np


def data_process():
    """
    Data Processing Module for Dataset 1 (Young Adults Spatial Cognitive EEG).

    This function loads pre-decomposed EEG frequency band data, extracts the
    optimal mixed low-high three-band combination (delta, beta2, gamma),
    applies Z-score standardization, and formats the data for the MMFDAHNet model.

    Returns:
        features (np.ndarray): Fused multi-band EEG features of shape (N_samples, 16, 256, 3).
        labels (np.ndarray): Binary spatial cognitive ability labels (high/low).
        groups (np.ndarray): Subject IDs used for Leave-One-Subject-Out (LOSO) cross-validation.
    """
    # File path for Dataset 1 (16 young adult subjects)
    data_path = '../LiuDataset/青年-16人/时域信号4096编号.xlsx'

    # Sheet names corresponding to the optimal three-band combination
    # delta (1-4 Hz), beta2 (20-30 Hz), gamma (30-40 Hz)
    delta_name = 'pindai0'
    beta2_name = 'pindai5'
    gamma_name = 'pindai6'

    # Load workbook in read-only and data-only mode for performance
    wb = load_workbook(data_path, read_only=True, data_only=True)
    ws_delta = wb[delta_name]
    ws_beta2 = wb[beta2_name]
    ws_gamma = wb[gamma_name]

    # Generate iterators to yield rows from the worksheets
    data_delta = ws_delta.values
    data_beta2 = ws_beta2.values
    data_gamma = ws_gamma.values

    # Extract headers (advances the iterators to the second row)
    # Assuming all three sheets share the identical column structure
    cols = next(data_delta)
    cols = next(data_beta2)
    cols = next(data_gamma)

    # Convert the remaining data rows into pandas DataFrames
    df_delta = pd.DataFrame(list(data_delta), columns=cols)
    df_beta2 = pd.DataFrame(list(data_beta2), columns=cols)
    df_gamma = pd.DataFrame(list(data_gamma), columns=cols)

    # Extract labels and subject group identifiers
    # 'Class variable': Spatial cognitive ability label (0 or 1)
    # 'bianhao': Subject ID (used for cross-subject domain splitting)
    labels = df_delta['Class variable'].values
    groups = df_delta['bianhao'].values

    # Isolate the raw EEG features by dropping the label and group columns
    features_delta = df_delta.drop(columns=['Class variable', 'bianhao']).values
    features_beta2 = df_beta2.drop(columns=['Class variable', 'bianhao']).values
    features_gamma = df_gamma.drop(columns=['Class variable', 'bianhao']).values

    # Initialize the Standard Scaler for Z-score normalization
    scaler = StandardScaler()

    # Normalize features and reshape to match MMFDAHNet input requirements:
    # (Samples, 16 Electrodes, 256 Time Steps)
    features_delta = scaler.fit_transform(features_delta).reshape(-1, 16, 256)
    features_beta2 = scaler.fit_transform(features_beta2).reshape(-1, 16, 256)
    features_gamma = scaler.fit_transform(features_gamma).reshape(-1, 16, 256)

    # Stack the three frequency bands along the last dimension (channel/band axis)
    # Final shape: (Samples, 16 channels, 256 time steps, 3 frequency bands)
    features = np.stack([features_delta, features_beta2, features_gamma], axis=-1)

    return features, labels, groups
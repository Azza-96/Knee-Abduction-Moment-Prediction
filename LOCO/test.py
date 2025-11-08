# test.py  (LOCO – test on the left-out trial)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import joblib

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the AdvancedLSTMModel class (same as in training)
class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return out

# Function to create sequences for a single trial
def create_sequences(features, targets, seq_length, step=1):
    sequences = []
    target_seq = []
    for i in range(0, len(features) - seq_length + 1, step):
        sequences.append(features[i:i + seq_length])
        target_seq.append(targets[i + seq_length - 1])
    return np.array(sequences), np.array(target_seq)

# Fit scalers on all data *except* the left-out trial


def load_loco_test_data(trial_to_leave_out, data_dir):
    '''Loads test data for a specific trial from the pre-split data directory.'''
    test_trials = []
    
    # Define the path to the specific split's data folder
    split_folder = f'LOCO_trial{trial_to_leave_out}'
    split_path = os.path.join(data_dir, split_folder)

    if not os.path.isdir(split_path):
        print(f"Error: Directory not found for trial {trial_to_leave_out}. Expected: {split_path}")
        return [], [], []

    try:
        # --- Load TEST Data (ONLY the left-out trial) ---
        # Files are named like: trial4_LLML_Acc_axis1.csv
        test_prefix = f'trial{trial_to_leave_out}'
        l_knee_x_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_LLML_Acc_axis1.csv'))
        l_knee_y_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_LLML_Acc_axis2.csv'))
        l_knee_z_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_LLML_Acc_axis3.csv'))
        r_knee_x_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_RLML_Acc_axis1.csv'))
        r_knee_y_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_RLML_Acc_axis2.csv'))
        r_knee_z_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_RLML_Acc_axis3.csv'))
        l_moment_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_left_knee_moment_axis2.csv'))
        r_moment_test = pd.read_csv(os.path.join(split_path, f'{test_prefix}_right_knee_moment_axis2.csv'))

    except FileNotFoundError as e:
        print(f'Skipping trial {trial_to_leave_out} due to missing file: {e}')
        return [], [], []

    # Process TEST trials (columns in the trialX files)
    n_test_trials = l_knee_x_test.shape[1]
    acc_data_list = []
    moment_data_list = []
    for trial_col in range(n_test_trials):
        acc_data = np.stack([
            l_knee_x_test.iloc[:, trial_col], l_knee_y_test.iloc[:, trial_col], l_knee_z_test.iloc[:, trial_col],
            r_knee_x_test.iloc[:, trial_col], r_knee_y_test.iloc[:, trial_col], r_knee_z_test.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 6)
        moment_data = np.stack([
            l_moment_test.iloc[:, trial_col], r_moment_test.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 2)
        acc_data_list.append(acc_data)
        moment_data_list.append(moment_data)

    X_test = np.concatenate(acc_data_list, axis=0)
    y_test = np.concatenate(moment_data_list, axis=0)
    # subject_ids are not available in this data loading scheme
    return X_test, y_test, list(range(n_test_trials))
    # Model hyperparameters (must match training)
    INPUT_SIZE = 6
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2
    DROPOUT_PROB = 0.3
    print(f"Model: INPUT={INPUT_SIZE}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}, OUTPUT={OUTPUT_SIZE}, DROPOUT={DROPOUT_PROB}")

    # Load model
    model_path = f'model_without_trial_{trial_to_leave_out}.pth'
    model = AdvancedLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        return
    model.eval()

    # Load scalers
    try:
        feature_scaler = joblib.load(f'feature_scaler_trial_{trial_to_leave_out}.pkl')
        target_scaler = joblib.load(f'target_scaler_trial_{trial_to_leave_out}.pkl')
        print(f"Scalers loaded for trial {trial_to_leave_out}")
    except FileNotFoundError:
        print(f"Scalers not found for trial {trial_to_leave_out}. Make sure to train the model first.")
        return

    # Load test trial
    X_raw, y_raw, subj_ids = load_loco_test_data(trial_to_leave_out, data_dir)

    # Scale and sequence
    X_scaled = feature_scaler.transform(X_raw)
    y_scaled = target_scaler.transform(y_raw)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length, step)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

    # Predict
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(X_tensor.shape[0]), desc=f"Predicting Trial {trial_to_leave_out}"):
            seq = X_tensor[i:i+1]
            pred = model(seq)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)

    # Inverse transform
    pred_rescaled = target_scaler.inverse_transform(predictions)
    gt_rescaled = target_scaler.inverse_transform(y_seq)

    # Plot
    plt.figure(figsize=(16, 8))
    time = np.arange(len(pred_rescaled))

    plt.subplot(2, 1, 1)
    plt.plot(time, gt_rescaled[:, 0], label='Actual LY', color='blue')
    plt.plot(time, pred_rescaled[:, 0], label='Pred LY', color='red', linestyle='--')
    plt.title(f'Trial {trial_to_leave_out} - Left Knee Moment (All Subjects)')
    plt.ylabel('KAM (Nm/kg)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, gt_rescaled[:, 1], label='Actual RY', color='green')
    plt.plot(time, pred_rescaled[:, 1], label='Pred RY', color='orange', linestyle='--')
    plt.title(f'Trial {trial_to_leave_out} - Right Knee Moment (All Subjects)')
    plt.xlabel('Time Step')
    plt.ylabel('KAM (Nm/kg)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'trial_{trial_to_leave_out}_loco_predictions.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    TRIAL_TO_LEAVE_OUT = 27  # Change this
    DATA_DIR = 'data'
    test_model_on_trial(TRIAL_TO_LEAVE_OUT, DATA_DIR, seq_length=100, step=10)
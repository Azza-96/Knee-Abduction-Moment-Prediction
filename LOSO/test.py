import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the AdvancedLSTMModel class (same as in the notebook)
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

# Function to fit scalers on training data (all subjects except the left-out one)
def fit_scalers_from_train(subject_to_leave_out, data_dir):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    all_train_features = []
    all_train_targets = []
    all_subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('subj')]
    
    for subj_folder in tqdm(all_subjects, desc='Fitting Scalers - Loading Training Subjects'):
        subj_id = int(subj_folder.replace('subj', ''))
        if subj_id == subject_to_leave_out:
            continue
        subj_path = os.path.join(data_dir, subj_folder)
        
        try:
            # Inputs (Accelerations)
            l_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis1.csv'))
            l_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis2.csv'))
            l_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis3.csv'))
            r_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis1.csv'))
            r_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis2.csv'))
            r_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis3.csv'))
            # Outputs (Moments)
            l_moment = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_left_knee_moment_axis2.csv'))
            r_moment = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_right_knee_moment_axis2.csv'))
        except FileNotFoundError as e:
            print(f'Skipping subject {subj_id} due to missing file: {e}')
            continue
        
        n_trials = l_knee_x.shape[1]
        for trial in range(n_trials):
            acc_data = np.stack([
                l_knee_x.iloc[:, trial], l_knee_y.iloc[:, trial], l_knee_z.iloc[:, trial],
                r_knee_x.iloc[:, trial], r_knee_y.iloc[:, trial], r_knee_z.iloc[:, trial]
            ], axis=1)  # Shape: (timesteps, 6)
            moment_data = np.stack([
                l_moment.iloc[:, trial], r_moment.iloc[:, trial]
            ], axis=1)  # Shape: (timesteps, 2)
            all_train_features.append(acc_data)
            all_train_targets.append(moment_data)
    
    if len(all_train_features) == 0:
        raise ValueError("No training data found.")
    
    all_train_features = np.concatenate(all_train_features, axis=0)
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    
    feature_scaler.fit(all_train_features)
    target_scaler.fit(all_train_targets)
    
    print("Scalers fitted on training data (all subjects except the left-out one).")
    return feature_scaler, target_scaler

# Function to load test subject's data frames
def load_test_data_frames(subject_to_leave_out, data_dir):
    subj_folder = f'subj{subject_to_leave_out}'
    subj_path = os.path.join(data_dir, subj_folder)
    
    try:
        # Inputs (Accelerations)
        l_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis1.csv'))
        l_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis2.csv'))
        l_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis3.csv'))
        r_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis1.csv'))
        r_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis2.csv'))
        r_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis3.csv'))
        # Outputs (Moments)
        l_moment = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_left_knee_moment_axis2.csv'))
        r_moment = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_right_knee_moment_axis2.csv'))
        print(f"Test subject {subject_to_leave_out} data frames loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading test subject files: {e}")
        exit()
    
    return l_knee_x, l_knee_y, l_knee_z, r_knee_x, r_knee_y, r_knee_z, l_moment, r_moment

# Function to preprocess a specific trial
def preprocess_trial(trial_idx, data_frames, feature_scaler, target_scaler, seq_length=100, step=1):
    l_knee_x, l_knee_y, l_knee_z, r_knee_x, r_knee_y, r_knee_z, l_moment, r_moment = data_frames

    # Stack acceleration and moment data for the specified trial
    acc_data = np.stack([
        l_knee_x.iloc[:, trial_idx], l_knee_y.iloc[:, trial_idx], l_knee_z.iloc[:, trial_idx],
        r_knee_x.iloc[:, trial_idx], r_knee_y.iloc[:, trial_idx], r_knee_z.iloc[:, trial_idx]
    ], axis=1)  # Shape: (timesteps, 6)
    moment_data = np.stack([
        l_moment.iloc[:, trial_idx], r_moment.iloc[:, trial_idx]
    ], axis=1)  # Shape: (timesteps, 2)

    # Apply feature and target scaling
    acc_data_scaled = feature_scaler.transform(acc_data)
    moment_data_scaled = target_scaler.transform(moment_data)

    # Create sequences
    X_seq, y_seq = create_sequences(acc_data_scaled, moment_data_scaled, seq_length, step)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

    return X_tensor, y_tensor, moment_data.shape[0]

# Main function to test the model on trials from the test subject
def test_model_on_trials(subject_to_leave_out, data_dir, seq_length=100, step=1):
    # Model hyperparameters (must match training)
    INPUT_SIZE = 6  # 6 acceleration channels
    HIDDEN_SIZE = 64  # Reduced for memory efficiency
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2  # LY and RY moments
    DROPOUT_PROB = 0.3
    print(f"Model hyperparameters: INPUT_SIZE={INPUT_SIZE}, HIDDEN_SIZE={HIDDEN_SIZE}, NUM_LAYERS={NUM_LAYERS}, OUTPUT_SIZE={OUTPUT_SIZE}, DROPOUT_PROB={DROPOUT_PROB}")

    # Construct model path
    model_path = f'model_without_subj_{subject_to_leave_out}.pth'

    # Fit scalers on training data
    feature_scaler, target_scaler = fit_scalers_from_train(subject_to_leave_out, data_dir)

    # Load test subject's data frames
    data_frames = load_test_data_frames(subject_to_leave_out, data_dir)
    l_knee_x, l_knee_y, l_knee_z, r_knee_x, r_knee_y, r_knee_z, l_moment, r_moment = data_frames

    # Determine number of trials
    n_trials = l_knee_x.shape[1]
    print(f"Test subject has {n_trials} trials.")

    # Initialize and load the trained model
    model = AdvancedLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please ensure the model is saved.")
        exit()
    model.eval()

    # Select random trials (up to 5)
    random.seed(42)
    num_trials_to_test = min(5, n_trials)
    trial_indices = random.sample(range(n_trials), num_trials_to_test)

    # Process each trial
    for trial_idx in trial_indices:
        print(f"\nProcessing Trial {trial_idx + 1}")
        X_tensor, y_tensor, num_timesteps = preprocess_trial(
            trial_idx, data_frames,
            feature_scaler, target_scaler, seq_length, step
        )

        # Simulate real-time prediction
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, X_tensor.shape[0], 1), desc=f"Predicting Trial {trial_idx + 1}"):
                seq = X_tensor[i:i+1]  # Process one sequence at a time
                pred = model(seq)
                predictions.append(pred.cpu().numpy())
        
        # Rescale predictions and ground truth
        predictions = np.concatenate(predictions, axis=0)
        predictions_rescaled = target_scaler.inverse_transform(predictions)
        ground_truth_rescaled = target_scaler.inverse_transform(y_tensor.cpu().numpy())

        # Plot predictions vs ground truth
        plt.figure(figsize=(15, 10))
        time_steps = np.arange(len(predictions_rescaled))
        
        # Plot LY Moment
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, ground_truth_rescaled[:, 0], label='Actual LY Moment', color='blue', alpha=0.8)
        plt.plot(time_steps, predictions_rescaled[:, 0], label='Predicted LY Moment', color='red', linestyle='--')
        plt.title(f'Trial {trial_idx + 1}: LY KAM Prediction (Subject {subject_to_leave_out})')
        plt.ylabel('KAM (Nm/Kg)')
        plt.legend()
        plt.grid(True)

        # Plot RY Moment
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, ground_truth_rescaled[:, 1], label='Actual RY Moment', color='green', alpha=0.8)
        plt.plot(time_steps, predictions_rescaled[:, 1], label='Predicted RY Moment', color='orange', linestyle='--')
        plt.title(f'Trial {trial_idx + 1}: RY KAM Prediction (Subject {subject_to_leave_out})')
        plt.xlabel('Time Step')
        plt.ylabel('KAM (Nm/Kg)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'trial_{trial_idx + 1}_subj_{subject_to_leave_out}_predictions.png')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Set the subject to test on (left-out subject)
    SUBJECT_TO_LEAVE_OUT = 9  # Change this to the desired test subject
    DATA_DIR = 'data'  # Path to the data directory with subject folders

    # Run the test
    test_model_on_trials(SUBJECT_TO_LEAVE_OUT, DATA_DIR, seq_length=100, step=10)
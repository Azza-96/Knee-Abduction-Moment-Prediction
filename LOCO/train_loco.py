# train_loco.py
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import gc  # For garbage collection
from tqdm import tqdm
import os
import random
import warnings
import joblib


def load_loco_data(trial_to_leave_out, data_dir):
    '''Loads data based on the Leave-One-Condition-Out principle
    from a pre-split data directory structure.'''
    train_trials = []
    test_trials = []
    
    # Define the path to the specific split's data folder
    split_folder = f'LOCO_trial{trial_to_leave_out}'
    split_path = os.path.join(data_dir, split_folder)

    if not os.path.isdir(split_path):
        print(f"Error: Directory not found for trial {trial_to_leave_out}. Expected: {split_path}")
        return train_trials, test_trials

    try:
        # --- Load TRAINING Data (All trials EXCEPT the left-out one) ---
        # Files are named like: LOCO_trial4_LLML_Acc_axis1.csv
        train_prefix = f'LOCO_trial{trial_to_leave_out}'
        l_knee_x_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis1.csv'))
        l_knee_y_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis2.csv'))
        l_knee_z_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis3.csv'))
        r_knee_x_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis1.csv'))
        r_knee_y_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis2.csv'))
        r_knee_z_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis3.csv'))
        l_moment_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_left_knee_moment_axis2.csv'))
        r_moment_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_right_knee_moment_axis2.csv'))

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
        return train_trials, test_trials

    # Process TRAINING trials (columns in the LOCO_trialX files)
    n_train_trials = l_knee_x_train.shape[1]
    for trial_col in tqdm(range(n_train_trials), desc="Loading Train Trials"):
        acc_data = np.stack([
            l_knee_x_train.iloc[:, trial_col], l_knee_y_train.iloc[:, trial_col], l_knee_z_train.iloc[:, trial_col],
            r_knee_x_train.iloc[:, trial_col], r_knee_y_train.iloc[:, trial_col], r_knee_z_train.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 6)
        moment_data = np.stack([
            l_moment_train.iloc[:, trial_col], r_moment_train.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 2)
        train_trials.append((acc_data, moment_data))

    # Process TEST trials (columns in the trialX files)
    n_test_trials = l_knee_x_test.shape[1]
    for trial_col in tqdm(range(n_test_trials), desc="Loading Test Trials"):
        acc_data = np.stack([
            l_knee_x_test.iloc[:, trial_col], l_knee_y_test.iloc[:, trial_col], l_knee_z_test.iloc[:, trial_col],
            r_knee_x_test.iloc[:, trial_col], r_knee_y_test.iloc[:, trial_col], r_knee_z_test.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 6)
        moment_data = np.stack([
            l_moment_test.iloc[:, trial_col], r_moment_test.iloc[:, trial_col]
        ], axis=1) # Shape: (timesteps, 2)
        test_trials.append((acc_data, moment_data))

    print(f'Data loaded.')
    print(f'{len(train_trials)} trials in the training set.')
    print(f'{len(test_trials)} trials in the test set.')
    return train_trials, test_trials


def create_sequences(features, targets, seq_length, step=1):
    """Creates overlapping sequences and corresponding targets for a single trial."""
    sequences = []
    target_seq = []
    for i in range(0, len(features) - seq_length + 1, step):
        sequences.append(features[i:i + seq_length])
        target_seq.append(targets[i + seq_length - 1])  # Target is the last timestep
    return np.array(sequences), np.array(target_seq)


def preprocess_loco_data(train_trials, test_trials, seq_length=100, step=1):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Step 1: Fit scalers on TRAINING data only
    print('Fitting scalers on training data...')
    all_train_features = np.concatenate([d[0] for d in train_trials])
    all_train_targets = np.concatenate([d[1] for d in train_trials])
    feature_scaler.fit(all_train_features)
    target_scaler.fit(all_train_targets)
    del all_train_features, all_train_targets  # Free memory

    # Step 2: Process training data
    all_train_feature_sequences = []
    all_train_target_sequences = []
    for acc_data, moment_data in tqdm(train_trials, desc="Processing Train Trials"):
        features_scaled = feature_scaler.transform(acc_data)
        targets_scaled = target_scaler.transform(moment_data)
        X_seq, y_seq = create_sequences(features_scaled, targets_scaled, seq_length, step)
        all_train_feature_sequences.append(X_seq)
        all_train_target_sequences.append(y_seq)

    X_train_val = np.concatenate(all_train_feature_sequences, axis=0)  # Shape: (n_sequences, seq_length, 6)
    y_train_val = np.concatenate(all_train_target_sequences, axis=0)  # Shape: (n_sequences, 2)
    del all_train_feature_sequences, all_train_target_sequences  # Free memory

    # Step 3: Process test data (prepared but not used)
    all_test_feature_sequences = []
    all_test_target_sequences = []
    for acc_data, moment_data in tqdm(test_trials, desc="Processing Test Trials"):
        features_scaled = feature_scaler.transform(acc_data)
        targets_scaled = target_scaler.transform(moment_data)
        X_seq, y_seq = create_sequences(features_scaled, targets_scaled, seq_length, step)
        all_test_feature_sequences.append(X_seq)
        all_test_target_sequences.append(y_seq)

    X_test = np.concatenate(all_test_feature_sequences, axis=0)
    y_test = np.concatenate(all_test_target_sequences, axis=0)
    del all_test_feature_sequences, all_test_target_sequences  # Free memory

    # Step 4: Convert to PyTorch Tensors and create Datasets
    train_val_dataset = TensorDataset(torch.tensor(X_train_val, dtype=torch.float32), torch.tensor(y_train_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # Step 5: Split training data into train and validation
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f'Training sequences: {len(train_dataset)}')
    print(f'Validation sequences: {len(val_dataset)}')
    print(f'Test sequences: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler


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


def train_for_trial(trial_to_leave_out, data_dir, seq_length, INPUT_SIZE):
    '''Train a model leaving out one trial (condition).'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f'Leaving out trial: {trial_to_leave_out}')

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load data
    train_trial_data, test_trial_data = load_loco_data(trial_to_leave_out, data_dir)

    # Preprocess data
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = preprocess_loco_data(
        train_trial_data, test_trial_data, seq_length, step=1
    )

    # Create DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    print(f'Created DataLoaders with batch size {batch_size}')

    # Model hyperparameters
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2  # LY and RY moments
    DROPOUT_PROB = 0.3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize the model
    model = AdvancedLSTMModel(
        INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB
    ).to(device)

    print(f'Model Architecture Initialized: {model}')

    # Training with Early Stopping
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Added L2 regularization
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for seq, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [T]', leave=False):
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        history['train_loss'].append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        for seq, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [V]', leave=False):
            seq, labels = seq.to(device), labels.to(device)
            with torch.no_grad():
                y_val_pred = model(seq)
                val_loss = criterion(y_val_pred, labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        history['val_loss'].append(avg_val_loss)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    # Save the Best Model (.pth)
    if best_model_state is not None:
        model_path_pth = f'model_without_trial_{trial_to_leave_out}.pth'
        torch.save(best_model_state, model_path_pth)
        print(f'Best model saved to {model_path_pth}')
        # Load the best performing model for ONNX export and evaluation
        model.load_state_dict(best_model_state)
    else:
        print('Training did not improve, no model saved.')
        # Still free memory even if no model saved
        free_memory(model, train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, train_loader, val_loader, test_loader, train_trial_data, test_trial_data, device)
        return

    # Export Model to ONNX
    model.eval()

    # Create a dummy input with the correct shape for the model
    # Shape: (batch_size, sequence_length, input_size)
    dummy_input = torch.randn(1, seq_length, INPUT_SIZE).to(device)
    model_path_onnx = f'model_without_trial_{trial_to_leave_out}.onnx'

    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input (or a tuple for multiple inputs)
        model_path_onnx,             # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=11,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the model's input names
        output_names=['output'],     # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # dynamic batch size
    )
    print(f'Model successfully exported to {model_path_onnx}')

    # Save the scalers
    joblib.dump(feature_scaler, f'feature_scaler_trial_{trial_to_leave_out}.pkl')
    joblib.dump(target_scaler, f'target_scaler_trial_{trial_to_leave_out}.pkl')
    print(f'Scalers saved for trial {trial_to_leave_out}')


    # Final Evaluation
    model.eval()
    all_predictions = []
    all_ground_truth = []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating on Test Set")
        for seq, labels in test_pbar:
            seq, labels = seq.to(device), labels.to(device)
            predictions = model(seq)
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Rescale predictions and ground truth back to original units
    predictions_rescaled = target_scaler.inverse_transform(all_predictions)
    ground_truth_rescaled = target_scaler.inverse_transform(all_ground_truth)

    # Calculate performance metrics
    mse = mean_squared_error(ground_truth_rescaled, predictions_rescaled)
    r2 = r2_score(ground_truth_rescaled, predictions_rescaled)
    print(f'--- Test Set Performance (Trial {trial_to_leave_out}) ---')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R-squared (R²): {r2:.4f}')

    print(f'Completed training for trial {trial_to_leave_out}\n')

    # Free memory after training and saving
    free_memory(model, train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, train_loader, val_loader, test_loader, train_trial_data, test_trial_data, device)


def free_memory(model, train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, train_loader, val_loader, test_loader, train_trial_data, test_trial_data, device):
    """Explicitly free memory after each trial's training."""
    del model
    del train_dataset
    del val_dataset
    del test_dataset
    del feature_scaler
    del target_scaler
    del train_loader
    del val_loader
    del test_loader
    del train_trial_data
    del test_trial_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Memory freed after trial training.")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model with LOCO cross-validation for a range of trials')
    parser.add_argument('--start_trial', type=int, default=1, help='Starting trial to leave out (default: 1)')
    parser.add_argument('--end_trial', type=int, default=33, help='Ending trial to leave out (default: 33)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    args = parser.parse_args()

    start_trial = args.start_trial
    end_trial = args.end_trial
    data_dir = args.data_dir

    # Fixed hyperparameters
    seq_length = 100
    INPUT_SIZE = 6

    for trial_to_leave_out in range(start_trial, end_trial + 1):
        print(f'Starting training for trial {trial_to_leave_out} (left out)...')
        train_for_trial(trial_to_leave_out, data_dir, seq_length, INPUT_SIZE)


if __name__ == '__main__':
    main()
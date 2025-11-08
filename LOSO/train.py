import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
# Lightweight percentage progress-bar printer that uses '#' characters.
def progress_bar_iter(iterable, desc=None, width=30):
    """Yield items from iterable while printing a compact '#' progress bar with percentage.

    - If the iterable has a length (len(iterable)), the function prints a single-line
      updating progress bar: "desc [#####-----] 42.3%".
    - If length is unknown, it falls back to printing a single '#' per item.
    """
    try:
        total = len(iterable)
    except Exception:
        total = None

    if desc and total:
        # Print the description once; the bar will update on the same line
        print(desc)

    for i, item in enumerate(iterable, start=1):
        if total:
            pct = i / total
            num_hash = int(pct * width)
            bar = '#' * num_hash + '-' * (width - num_hash)
            # Use carriage return to update the same line
            print(f'\r{desc} [{bar}] {pct*100:5.1f}% ', end='', flush=True)
        else:
            print('#', end='', flush=True)
        yield item

    if total:
        print()  # finish the progress line with a newline
import os
import random

def load_loso_data(subject_to_leave_out, data_dir):
    '''Loads data based on the Leave-One-Subject-Out principle.'''
    train_trials = []
    test_trials = []  # Loaded but not used for training
    all_subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('subj')]

    for subj_folder in progress_bar_iter(all_subjects, desc='Loading Subject Data'):
        subj_id = int(subj_folder.replace('subj', ''))
        subj_path = os.path.join(data_dir, subj_folder)

        try:
            # Inputs (Accelerations)
            l_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis1.csv'))
            l_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis2.csv'))
            l_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis3.csv'))
            r_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis1.csv'))
            r_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis2.csv'))
            r_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis3.csv'))
            # Outputs (Moments) - Using axis2 as in the original notebook
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

            # Allocate to train or test set
            if subj_id == subject_to_leave_out:
                test_trials.append((acc_data, moment_data))
            else:
                train_trials.append((acc_data, moment_data))

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

def preprocess_loso_data(train_trials, test_trials, seq_length=100, step=1):
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
    for acc_data, moment_data in progress_bar_iter(train_trials, desc="Processing Train Trials"):
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
    for acc_data, moment_data in progress_bar_iter(test_trials, desc="Processing Test Trials"):
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

    # Step 5: Split training data into train and validation sets (85/15 split)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    print(f'Total training sequences: {len(train_dataset)}')
    print(f'Total validation sequences: {len(val_dataset)}')
    print(f'Total test sequences: {len(test_dataset)}')

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

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model with LOSO cross-validation')
    parser.add_argument('--subject', type=int, required=True, help='Subject to leave out (e.g., 2)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    args = parser.parse_args()

    subject_to_leave_out = args.subject
    data_dir = args.data_dir

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f'Leaving out subject: {subject_to_leave_out}')

    # Load data
    train_trial_data, test_trial_data = load_loso_data(subject_to_leave_out, data_dir)

    # Preprocess data
    seq_length = 100
    step = 1
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = preprocess_loso_data(
        train_trial_data, test_trial_data, seq_length, step
    )

    # Create DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    # test_loader not created since not used

    print(f'Created DataLoaders with batch size {batch_size}')

    # Model hyperparameters
    INPUT_SIZE = 6  # 6 acceleration channels
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
        for seq, labels in progress_bar_iter(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [T]'):
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
        for seq, labels in progress_bar_iter(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [V]'):
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
        model_path_pth = f'model_without_subj_{subject_to_leave_out}.pth'
        torch.save(best_model_state, model_path_pth)
        print(f'Best model saved to {model_path_pth}')
        # Load the best performing model for ONNX export
        model.load_state_dict(best_model_state)
    else:
        print('Training did not improve, no model saved.')
        return

    # Export Model to ONNX
    model.eval()

    # Create a dummy input with the correct shape for the model
    # Shape: (batch_size, sequence_length, input_size)
    dummy_input = torch.randn(1, seq_length, INPUT_SIZE).to(device)
    model_path_onnx = f'model_without_subj_{subject_to_leave_out}.onnx'

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
# End of training/export block

if __name__ == '__main__':
    main()
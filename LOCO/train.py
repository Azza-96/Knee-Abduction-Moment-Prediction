# train.py  (LOCO version – leave one trial/condition out)
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import random
from tqdm import tqdm


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
    del all_train_features, all_train_targets # Free memory

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
    del all_train_feature_sequences, all_train_target_sequences # Free memory

    # Step 3: Process test data
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
    del all_test_feature_sequences, all_test_target_sequences # Free memory

    # Step 4: Convert to PyTorch Tensors and create Datasets
    train_val_dataset = TensorDataset(torch.tensor(X_train_val, dtype=torch.float32), torch.tensor(y_train_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # Step 5: Split training data into train and validation sets (85/15 split)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    print(f'Total training sequences: {len(train_dataset)}')
    print(f'Total validation sequences: {len(val_dataset)}')
    print(f'Total test sequences: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train LSTM with LOCO (Leave-One-Trial-Out)')
    parser.add_argument('--trial', type=int, required=True,
                        help='Trial number to leave out (e.g., 27)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    args = parser.parse_args()

    trial_to_leave_out = args.trial
    data_dir = args.data_dir

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Leaving out trial: {trial_to_leave_out}")

    # Load + preprocess
    train_trials, test_trials = load_loco_data(trial_to_leave_out, data_dir)
    seq_length = 100
    step = 1
    train_ds, val_ds, test_ds, feat_scaler, targ_scaler = preprocess_loco_data(
        train_trials, test_trials, seq_length, step
    )

    # DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    # Hyperparameters
    INPUT_SIZE = 6
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2
    DROPOUT_PROB = 0.3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AdvancedLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Early stopping
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for seq, lbl in progress_bar_iter(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [T]'):
            seq, lbl = seq.to(device), lbl.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        for seq, lbl in progress_bar_iter(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [V]'):
            seq, lbl = seq.to(device), lbl.to(device)
            with torch.no_grad():
                pred = model(seq)
                loss = criterion(pred, lbl)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} -> Train: {avg_train:.6f}, Val: {avg_val:.6f}')

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Save best model
    if best_state is not None:
        pth_path = f'model_without_trial_{trial_to_leave_out}.pth'
        torch.save(best_state, pth_path)
        print(f'Best model saved: {pth_path}')
        model.load_state_dict(best_state)
    else:
        print("No improvement during training.")
        return

    # Export to ONNX
    model.eval()
    dummy = torch.randn(1, seq_length, INPUT_SIZE).to(device)
    onnx_path = f'model_without_trial_{trial_to_leave_out}.onnx'
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f'ONNX exported: {onnx_path}')

    print("Training complete.")


if __name__ == '__main__':
    main()
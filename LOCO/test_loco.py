# test_loco.py
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
    
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

    # Metric helper functions removed (DTW, lag, sMAPE).

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

def create_sequences(features, targets, seq_length, step=1):
    sequences = []
    target_seq = []
    pred_indices = []
    for i in range(0, len(features) - seq_length + 1, step):
        sequences.append(features[i:i + seq_length])
        target_seq.append(targets[i + seq_length - 1])
        pred_indices.append(i + seq_length - 1)
    return np.array(sequences), np.array(target_seq), np.array(pred_indices)

def fit_loco_scalers_from_train(trial_to_leave_out, data_dir):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    all_train_features = []
    all_train_targets = []
    
    split_folder = f'LOCO_trial{trial_to_leave_out}'
    split_path = os.path.join(data_dir, split_folder)
    train_prefix = f'LOCO_trial{trial_to_leave_out}'

    try:
        l_knee_x_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis1.csv'))
        l_knee_y_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis2.csv'))
        l_knee_z_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_LLML_Acc_axis3.csv'))
        r_knee_x_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis1.csv'))
        r_knee_y_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis2.csv'))
        r_knee_z_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_RLML_Acc_axis3.csv'))
        l_moment_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_left_knee_moment_axis2.csv'))
        r_moment_train = pd.read_csv(os.path.join(split_path, f'{train_prefix}_right_knee_moment_axis2.csv'))
    except FileNotFoundError as e:
        print(f'Error: Could not load training data to fit scalers for trial {trial_to_leave_out}. File not found: {e.filename}')
        return None, None

    n_train_trials = l_knee_x_train.shape[1]
    for trial_col in tqdm(range(n_train_trials), desc="Fitting Scalers"):
        acc_data = np.stack([
            l_knee_x_train.iloc[:, trial_col], l_knee_y_train.iloc[:, trial_col], l_knee_z_train.iloc[:, trial_col],
            r_knee_x_train.iloc[:, trial_col], r_knee_y_train.iloc[:, trial_col], r_knee_z_train.iloc[:, trial_col]
        ], axis=1)
        moment_data = np.stack([
            l_moment_train.iloc[:, trial_col], r_moment_train.iloc[:, trial_col]
        ], axis=1)
        all_train_features.append(acc_data)
        all_train_targets.append(moment_data)

    if not all_train_features:
        print(f"Warning: No training data found for trial {trial_to_leave_out}.")
        return None, None
    
    all_train_features = np.concatenate(all_train_features, axis=0)
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    feature_scaler.fit(all_train_features)
    target_scaler.fit(all_train_targets)
    print("Scalers fitted successfully.")
    return feature_scaler, target_scaler

def load_loco_test_data(trial_to_leave_out, data_dir):
    '''Loads all test instances for a specific left-out trial.'''
    test_instances = []
    
    split_folder = f'LOCO_trial{trial_to_leave_out}'
    split_path = os.path.join(data_dir, split_folder)

    if not os.path.isdir(split_path):
        # This check is preliminary. The main check is in test_for_trial.
        return test_instances

    try:
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
        print(f'Error: Could not load test data for trial {trial_to_leave_out}. File not found: {e.filename}')
        return []

    n_test_instances = l_knee_x_test.shape[1]
    for inst_col in tqdm(range(n_test_instances), desc="Loading Test Instances"):
        acc_data = np.stack([
            l_knee_x_test.iloc[:, inst_col], l_knee_y_test.iloc[:, inst_col], l_knee_z_test.iloc[:, inst_col],
            r_knee_x_test.iloc[:, inst_col], r_knee_y_test.iloc[:, inst_col], r_knee_z_test.iloc[:, inst_col]
        ], axis=1)
        moment_data = np.stack([
            l_moment_test.iloc[:, inst_col], r_moment_test.iloc[:, inst_col]
        ], axis=1)
        # In LOCO, each column of the test file is a separate instance (e.g., a subject's trial)
        # We use the column index as the instance_id.
        test_instances.append((acc_data, moment_data, inst_col))

    print(f'Data loaded: {len(test_instances)} instances in the test set.')
    return test_instances

def test_for_trial(trial_to_leave_out, data_dir, seq_length, step, debug_single_instance=False):
    print(f"\n{'='*15} Testing Trial {trial_to_leave_out:02d} (Left Out) {'='*15}")

    # 1. Fit scalers from training data
    feature_scaler, target_scaler = fit_loco_scalers_from_train(trial_to_leave_out, data_dir)
    if feature_scaler is None:
        print(f"Aborting test for trial {trial_to_leave_out} due to scaler fitting failure.")
        return None

    # 2. Load model
    model_path = f'models/model_without_trial_{trial_to_leave_out}.pth'
    model = AdvancedLSTMModel(6, 64, 2, 2, 0.3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model not found: {model_path}. Aborting test for this trial.")
        return None
    model.eval()

    # 3. Load test instances
    test_instances = load_loco_test_data(trial_to_leave_out, data_dir)
    if not test_instances:
        print(f"No test instances found for trial {trial_to_leave_out}. Aborting test.")
        return None

    # 4. Create output folders (only if data and model are present)
    pred_dir = f'predictions/trial_{trial_to_leave_out}'
    plot_dir = f'plots/trial_{trial_to_leave_out}'
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Output will be saved in {pred_dir} and {plot_dir}")

    # Metrics collection removed. We only save per-instance prediction CSVs and plots.
    all_instance_predictions = []

    for idx, (acc_raw, moment_raw, instance_id) in enumerate(test_instances):
        if debug_single_instance and idx > 0:
            break
        
        print(f"\nProcessing instance {idx + 1}/{len(test_instances)} (Column ID: {instance_id})")

        X_scaled = feature_scaler.transform(acc_raw)
        y_scaled = target_scaler.transform(moment_raw)
        X_seq, y_seq, indices = create_sequences(X_scaled, y_scaled, seq_length, step)

        if X_seq.shape[0] == 0:
            print(f"Instance {idx+1} is too short to create sequences. Skipping.")
            continue

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X_tensor), 128), desc="Predicting"):
                batch = X_tensor[i:i + 128]
                pred = model(batch)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        pred_rescaled = target_scaler.inverse_transform(predictions)
        gt_rescaled = target_scaler.inverse_transform(y_seq)

        df = pd.DataFrame({
            'time_index': indices,
            'LY_actual': gt_rescaled[:, 0],
            'LY_pred': pred_rescaled[:, 0],
            'RY_actual': gt_rescaled[:, 1],
            'RY_pred': pred_rescaled[:, 1],
            'instance_col_id': instance_id
        })
        csv_path = os.path.join(pred_dir, f'instance_{idx+1}_col_{instance_id}.csv')
        df.to_csv(csv_path, index=False)
        all_instance_predictions.append(df)

        # Metric calculations removed by request. If needed later, reintroduce
        # computations here and append to an instance-level list.

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(indices, gt_rescaled[:, 0], label='Actual LY Moment', color='blue', alpha=0.8)
        plt.plot(indices, pred_rescaled[:, 0], label='Predicted LY Moment', color='red', linestyle='--')
        plt.title(f'Trial {trial_to_leave_out} - Instance {idx+1} (Col {instance_id}): LY Moment')
        plt.ylabel('Moment (Nm/kg)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(indices, gt_rescaled[:, 1], label='Actual RY Moment', color='green', alpha=0.8)
        plt.plot(indices, pred_rescaled[:, 1], label='Predicted RY Moment', color='orange', linestyle='--')
        plt.title(f'Trial {trial_to_leave_out} - Instance {idx+1} (Col {instance_id}): RY Moment')
        plt.xlabel('Time Step')
        plt.ylabel('Moment (Nm/kg)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'instance_{idx+1}_col_{instance_id}_pred.png'), dpi=150)
        plt.close()

    # No metrics to summarize; proceed to save predictions if any.

    if all_instance_predictions:
        all_predictions_df = pd.concat(all_instance_predictions)
        all_predictions_df.to_csv(os.path.join(pred_dir, f'trial_{trial_to_leave_out}_all_predictions.csv'), index=False)

    print(f"\nCompleted trial {trial_to_leave_out}. Results are in {pred_dir} and {plot_dir}")
    
    del model, feature_scaler, target_scaler, test_instances
    torch.cuda.empty_cache()
    gc.collect()
    return None

def main():
    parser = argparse.ArgumentParser(description='LSTM-based Knee Moment Prediction Testing (LOCO)')
    parser.add_argument('--start_trial', type=int, default=1, help='Starting trial ID')
    parser.add_argument('--end_trial', type=int, default=33, help='Ending trial ID')
    parser.add_argument('--single_trial', type=int, help='Test only one specific trial')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing LOCO data folders')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for LSTM input')
    parser.add_argument('--step', type=int, default=1, help='Step size for sequence creation')
    parser.add_argument('--debug_single_instance', action='store_true', help='Process only one instance per trial')
    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"{ '='*20} LOCO TESTING SCRIPT {'='*20}")
    print("="*70)
    print(f"Data Dir : {args.data_dir}")
    print(f"Seq Len  : {args.seq_length}")
    print(f"Step     : {args.step}")
    if args.single_trial:
        print(f"Mode     : Single Trial ({args.single_trial})")
    else:
        print(f"Mode     : Batch ({args.start_trial} to {args.end_trial})")
    print("="*70 + "\n")

    trials = [args.single_trial] if args.single_trial else range(args.start_trial, args.end_trial + 1)
    for trial in trials:
        test_for_trial(trial, args.data_dir, args.seq_length, args.step, args.debug_single_instance)

if __name__ == "__main__":
    main()
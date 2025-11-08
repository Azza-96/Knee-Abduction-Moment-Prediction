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
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def fit_scalers_from_train(subject_to_leave_out, data_dir):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    all_train_features = []
    all_train_targets = []
    all_subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('subj')]

    for subj_folder in tqdm(all_subjects, desc='Fitting Scalers'):
        subj_id = int(subj_folder.replace('subj', ''))
        if subj_id == subject_to_leave_out:
            continue
        subj_path = os.path.join(data_dir, subj_folder)
        try:
            l_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis1.csv'))
            l_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis2.csv'))
            l_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_LLML_Acc_axis3.csv'))
            r_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis1.csv'))
            r_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis2.csv'))
            r_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_RLML_Acc_axis3.csv'))
            l_moment = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_left_knee_moment_axis2.csv'))
            r_moment = pd.read_csv(os.path.join(subj_path, f'subj{subj_id}_right_knee_moment_axis2.csv'))
        except FileNotFoundError as e:
            print(f'Skipping subject {subj_id}: {e}')
            continue

        n_trials = l_knee_x.shape[1]
        for trial in range(n_trials):
            acc_data = np.stack([
                l_knee_x.iloc[:, trial], l_knee_y.iloc[:, trial], l_knee_z.iloc[:, trial],
                r_knee_x.iloc[:, trial], r_knee_y.iloc[:, trial], r_knee_z.iloc[:, trial]
            ], axis=1)
            moment_data = np.stack([l_moment.iloc[:, trial], r_moment.iloc[:, trial]], axis=1)
            all_train_features.append(acc_data)
            all_train_targets.append(moment_data)

    if not all_train_features:
        raise ValueError("No training data found.")
    all_train_features = np.concatenate(all_train_features, axis=0)
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    feature_scaler.fit(all_train_features)
    target_scaler.fit(all_train_targets)
    print("Scalers fitted.")
    return feature_scaler, target_scaler

def load_test_data_frames(subject_to_leave_out, data_dir):
    subj_folder = f'subj{subject_to_leave_out}'
    subj_path = os.path.join(data_dir, subj_folder)
    try:
        l_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis1.csv'))
        l_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis2.csv'))
        l_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_LLML_Acc_axis3.csv'))
        r_knee_x = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis1.csv'))
        r_knee_y = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis2.csv'))
        r_knee_z = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_RLML_Acc_axis3.csv'))
        l_moment = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_left_knee_moment_axis2.csv'))
        r_moment = pd.read_csv(os.path.join(subj_path, f'subj{subject_to_leave_out}_right_knee_moment_axis2.csv'))
        print(f"Test subject {subject_to_leave_out} data loaded.")
        return l_knee_x, l_knee_y, l_knee_z, r_knee_x, r_knee_y, r_knee_z, l_moment, r_moment
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def test_for_subject(subject_to_leave_out, data_dir, seq_length=100, step=5, single_trial=False):
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB = 6, 64, 2, 2, 0.3
    model_path = f'model_without_subj_{subject_to_leave_out}.pth'
    model_path = "models/"+model_path
    # Create output folders
    pred_dir = f'predictions/subj_{subject_to_leave_out}'
    plot_dir = f'plots/subj_{subject_to_leave_out}'
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Create and display folder structure
    print(f"\nOutput Directories:")
    print(f"├── Predictions: {pred_dir}")
    print(f"└── Plots     : {plot_dir}\n")

    # Scalers and data loading
    print("Step 1: Preparing Data")
    print("─" * 50)
    feature_scaler, target_scaler = fit_scalers_from_train(subject_to_leave_out, data_dir)
    data_frames = load_test_data_frames(subject_to_leave_out, data_dir)
    if data_frames is None:
        return None

    l_knee_x, l_knee_y, l_knee_z, r_knee_x, r_knee_y, r_knee_z, l_moment, r_moment = data_frames
    n_trials = 1 if single_trial else l_knee_x.shape[1]
    print(f"Found {n_trials} trials for subject {subject_to_leave_out:02d}")
    print("Data preparation completed successfully\n")

    # Load and prepare model
    print("Step 2: Model Initialization")
    print("─" * 50)
    print(f"Model Configuration:")
    print(f"├── Input Size    : {INPUT_SIZE}")
    print(f"├── Hidden Size   : {HIDDEN_SIZE}")
    print(f"├── Num Layers    : {NUM_LAYERS}")
    print(f"├── Output Size   : {OUTPUT_SIZE}")
    print(f"└── Dropout Rate  : {DROPOUT_PROB}")
    
    model = AdvancedLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Strip 'module.' prefix if present (for DataParallel compatibility)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"\nSuccessfully loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"\nError: Model {model_path} not found.")
        return None
    except RuntimeError as e:
        print(f"\nError loading model state dict: {e}")
        return None

    # GPU Warm-up
    print("\nWarming up GPU for optimal performance...")
    dummy = torch.randn(32, seq_length, INPUT_SIZE, device=device)
    with torch.inference_mode():
        _ = model(dummy)
    print("GPU warm-up completed successfully")
    model.eval()
    print("Model initialization completed\n")

    # Test all trials
    print("Step 3: Processing Trials")
    print("─" * 50)
    trial_indices = range(n_trials)
    print(f"Starting prediction for {n_trials} trials...")
    print(f"Each trial will generate:")
    print(f"├── Predictions CSV")
    print(f"├── Performance Metrics")
    print(f"└── Visualization Plot\n")
    
    # Metrics collection removed per request. Predictions and plots will still be saved.

    # Process each trial efficiently
    with torch.inference_mode():
        for trial_idx in trial_indices:
            print(f"\nProcessing Trial {trial_idx + 1}")
            
            # Efficient data preprocessing
            acc_data = np.stack([
                l_knee_x.iloc[:, trial_idx], l_knee_y.iloc[:, trial_idx], l_knee_z.iloc[:, trial_idx],
                r_knee_x.iloc[:, trial_idx], r_knee_y.iloc[:, trial_idx], r_knee_z.iloc[:, trial_idx]
            ], axis=1)
            moment_data = np.stack([l_moment.iloc[:, trial_idx], r_moment.iloc[:, trial_idx]], axis=1)
            
            # Scale data
            acc_scaled = feature_scaler.transform(acc_data)
            moment_scaled = target_scaler.transform(moment_data)

            # Create and prepare sequences
            X_seq, y_seq, _ = create_sequences(acc_scaled, moment_scaled, seq_length, step)
            
            # Process in larger batches for efficiency
            batch_size = 64  # Adjust based on your GPU memory
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_seq, dtype=torch.float32)

            # Efficient batched prediction with larger batch size for speed
            predictions = []
            batch_size = min(128, len(X_tensor))  # Use larger batches for faster processing
            for i in tqdm(range(0, len(X_tensor), batch_size), 
                         desc=f"Predicting Trial {trial_idx + 1}/{n_trials}",
                         ncols=100):
                batch = X_tensor[i:i + batch_size]
                pred = model(batch)
                predictions.append(pred.cpu().numpy())
            
            # Combine and rescale predictions
            predictions = np.concatenate(predictions, axis=0)
            predictions_rescaled = target_scaler.inverse_transform(predictions)
            ground_truth_rescaled = target_scaler.inverse_transform(y_tensor.numpy())

            # Save predictions and ground truth for this trial
            predictions_df = pd.DataFrame({
                'time_step': np.arange(len(predictions_rescaled)),
                'LY_predicted': predictions_rescaled[:, 0],
                'RY_predicted': predictions_rescaled[:, 1],
                'LY_ground_truth': ground_truth_rescaled[:, 0],
                'RY_ground_truth': ground_truth_rescaled[:, 1]
            })
            predictions_df.to_csv(os.path.join(pred_dir, f'trial_{trial_idx+1}_predictions.csv'), index=False)
            print(f"├── Saved predictions for trial {trial_idx + 1}")

            # Metrics calculations removed. If you want per-trial metrics later,
            # reintroduce computation here and append to a metrics list.

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
            plt.savefig(os.path.join(plot_dir, f'trial_{trial_idx+1}_pred.png'), dpi=150, bbox_inches='tight')
            plt.close('all')
            gc.collect()

    # Metrics saving removed per request. No metrics files will be created.
    means = None

    # Step 4: Finalizing Results
    print("\nStep 4: Finalizing Results")
    print("─" * 50)
    
    print("Combining all trial predictions...")
    all_trial_predictions = []
    for trial_idx in trial_indices:
        trial_df = pd.read_csv(os.path.join(pred_dir, f'trial_{trial_idx+1}_predictions.csv'))
        trial_df['trial'] = trial_idx + 1
        all_trial_predictions.append(trial_df)
    
    # Save combined predictions for the subject
    all_predictions_df = pd.concat(all_trial_predictions, axis=0)
    combined_file = os.path.join(pred_dir, f'subject_{subject_to_leave_out}_all_predictions.csv')
    all_predictions_df.to_csv(combined_file, index=False)
    
    print("\nSummary of Generated Files:")
    print(f"├── Individual Trial Predictions: {n_trials} files")
    print(f"└── Combined Predictions File  : {os.path.basename(combined_file)}")
    
    print(f"\n{'='*70}")
    print(f"{'='*15} SUBJECT {subject_to_leave_out:02d} COMPLETED {'='*15}")
    print(f"{'='*70}\n")
    
    # Cleanup
    del model, feature_scaler, target_scaler, data_frames
    torch.cuda.empty_cache()
    gc.collect()
    return means

def main():
    parser = argparse.ArgumentParser(description='LSTM-based Knee Moment Prediction Testing Script')
    parser.add_argument('--start_subject', type=int, default=1, help='Starting subject ID for batch processing')
    parser.add_argument('--end_subject', type=int, default=10, help='Ending subject ID for batch processing')
    parser.add_argument('--single_subject', type=int, help='Test only one specific subject')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing subject data')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for LSTM input')
    parser.add_argument('--step', type=int, default=1, help='Step size for sequence creation (5 = faster, less dense)')
    parser.add_argument('--debug_single_trial', action='store_true', help='Process only one trial per subject')
    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*70)
    print(f"{'='*15} LSTM KNEE MOMENT PREDICTION {'='*15}")
    print("="*70)
    print(f"Device          : {device}")
    print(f"Data Directory  : {args.data_dir}")
    print(f"Sequence Length : {args.seq_length}")
    print(f"Step Size      : {args.step}")
    if args.single_subject:
        print(f"Mode           : Single Subject (Subject {args.single_subject})")
    else:
        print(f"Mode           : Batch Processing (Subjects {args.start_subject} to {args.end_subject})")
    if args.debug_single_trial:
        print("Debug Mode     : Single Trial per Subject")
    print("="*70 + "\n")

    subjects = [args.single_subject] if args.single_subject else range(args.start_subject, args.end_subject + 1)
    all_means = []

    # Print processing information
    if args.single_subject:
        print(f"\n{'='*15} Processing Subject {args.single_subject:02d} {'='*15}")
    else:
        print(f"\n{'='*15} Processing Subjects {args.start_subject:02d}-{args.end_subject:02d} {'='*15}")

    for subj in subjects:
        means = test_for_subject(subj, args.data_dir, args.seq_length, args.step, args.debug_single_trial)
        if means is not None:
            all_means.append(means)

if __name__ == "__main__":
    main()
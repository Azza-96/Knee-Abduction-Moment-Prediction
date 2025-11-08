import argparse
import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import numpy as np
import os
import warnings

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

def convert_pth_to_onnx(pth_path, onnx_path, seq_length=100, input_size=6):
    """Convert a .pth model file to ONNX with dynamic batch size support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model hyperparameters (matching the training script)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2
    DROPOUT_PROB = 0.3
    
    # Initialize model
    model = AdvancedLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_PROB).to(device)
    
    # Load the state dict
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Dummy input for export (batch=1, but dynamic)
    dummy_input = torch.randn(1, seq_length, input_size).to(device)
    
    # Export to ONNX with dynamic batch axis
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Dynamic batch size
            'output': {0: 'batch_size'}
        }
    )
    print(f'Model converted: {pth_path} -> {onnx_path}')
    return model, device  # Return for verification

def verify_onnx_vs_pth(model, onnx_path, seq_length=100, input_size=6, atol=1e-5):
    """Verify ONNX model equivalence to PyTorch model with batch=1 and batch=32."""
    device = next(model.parameters()).device
    
    # Load ONNX session
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Test cases: batch=1 and batch=32
    batch_sizes = [1, 32]
    all_passed = True
    
    for batch_size in batch_sizes:
        # Generate random input
        test_input_torch = torch.randn(batch_size, seq_length, input_size).to(device)
        test_input_onnx = test_input_torch.detach().cpu().numpy().astype(np.float32)
        
        # PyTorch forward
        with torch.no_grad():
            torch_out = model(test_input_torch).detach().cpu().numpy().astype(np.float32)
        
        # ONNX forward
        ort_out = session.run(None, {input_name: test_input_onnx})[0]
        
        # Compare
        close = np.allclose(ort_out, torch_out, atol=atol)
        print(f'  Batch={batch_size}: Verification {"PASSED" if close else "FAILED"} (atol={atol})')
        if not close:
            all_passed = False
            max_diff = np.max(np.abs(ort_out - torch_out))
            print(f'    Max absolute difference: {max_diff}')
    
    if all_passed:
        print('  Overall verification: PASSED')
    else:
        print('  Overall verification: FAILED')
    print()

def main():
    parser = argparse.ArgumentParser(description='Convert .pth models to ONNX and verify with dynamic batch sizes')
    parser.add_argument('--pth_dir', type=str, default='.', help='Directory containing .pth files (default: current dir)')
    parser.add_argument('--start_subject', type=int, default=2, help='Starting subject (default: 2)')
    parser.add_argument('--end_subject', type=int, default=10, help='Ending subject (default: 10)')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for dummy input (default: 100)')
    parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for verification (default: 1e-5)')
    args = parser.parse_args()
    
    pth_dir = args.pth_dir
    start_subject = args.start_subject
    end_subject = args.end_subject
    seq_length = args.seq_length
    atol = args.atol
    
    input_size = 6  # Fixed
    
    for subject in range(start_subject, end_subject + 1):
        pth_file = os.path.join(pth_dir, f'model_without_subj_{subject}.pth')
        onnx_file = os.path.join(pth_dir, f'model_without_subj_{subject}.onnx')
        
        if os.path.exists(pth_file):
            print(f'Processing subject {subject}:')
            model, device = convert_pth_to_onnx(pth_file, onnx_file, seq_length, input_size)
            verify_onnx_vs_pth(model, onnx_file, seq_length, input_size, atol)
            del model  # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f'Warning: {pth_file} not found, skipping subject {subject}')

if __name__ == '__main__':
    main()
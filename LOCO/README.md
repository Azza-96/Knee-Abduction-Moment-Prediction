# Knee Moment Prediction with Bidirectional LSTM  
### **Leave-One-Trial-Out (LOCO) Cross-Validation Pipeline**  


This repository implements a **deep learning pipeline** to predict **bilateral knee adduction moments (KAM)** from **6-axis accelerometer data** collected from the lower limbs using a **Bidirectional LSTM** model. The system uses **Leave-One-Trial-Out (LOCO)** cross-validation to ensure **generalization across different movements**, a critical requirement in biomechanical and wearable sensor applications.

---

## 🎯 **Key Features**

| Feature | Description |
|-------|-----------|
| **LOCO Training** | Train on all trials except one, test on the held-out trial |
| **Bidirectional LSTM** | Captures temporal dependencies in both directions |
| **Real-time Ready** | ONNX export with dynamic batch size |
 
| **Visualization** | Prediction vs Ground Truth plots for every trial |
| **Scalable Scripts** | Train/test individual or batch trials |
| **Memory Efficient** | Garbage collection, CUDA cleanup, progress bars |

---

## 📁 **Repository Structure**

```
.
├── data/                          # (Your data here) Trial folders: LOCO_trial1/, LOCO_trial3/, ...
├── models/                        # Saved .pth and .onnx models
├── predictions/                   # CSV predictions per trial
├── plots/                         # Prediction vs GT plots
├── train.py                       # Single-trial training
├── train_loco.py                  # Batch LOCO training
├── test.py                        # Quick test on random trials (debug)
├── test_loco.py                   # Full evaluation with plots
├── convert.py                     # Convert .pth → .onnx + verify equivalence
├── code.ipynb                     # Jupyter notebook version (full pipeline)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore
```

---

## 🛠 **Setup & Installation**

### 1. Clone the Repository
```bash
git clone https://github.com/Azza-96/Knee-Abduction-Moment-Prediction.git
cd LOCO
```

### 2. (Recommended) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **requirements.txt**
```txt
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
onnxruntime
scipy
```

---

## 📊 **Data Format**

Place your data in the `data/` folder with the following structure:

```
data/
├── LOCO_trial1/
│   ├── trial1_LLML_Acc_axis1.csv
│   ├── trial1_LLML_Acc_axis2.csv
│   ├── trial1_LLML_Acc_axis3.csv
│   ├── trial1_RLML_Acc_axis1.csv
│   ├── trial1_RLML_Acc_axis2.csv
│   ├── trial1_RLML_Acc_axis3.csv
│   ├── trial1_left_knee_moment_axis2.csv
│   └── trial1_right_knee_moment_axis2.csv
├── LOCO_trial3/
│   └── ...
└── ...
```

- **Inputs**: 6 acceleration channels (3 per knee: X, Y, Z)
- **Outputs**: Left & Right knee **flexion/extension moment (axis 2)** in Nm/kg
- Each CSV has **one trial per column**, rows = time steps

---

## 🚀 **Usage**

### 1. **Train a Single Trial (LOCO)**

```bash
python train.py --trial 10 --data_dir data
```

- Leaves out `trial10`, trains on all others
- Saves:  
  - `model_without_trial_10.pth`  
  - `model_without_trial_10.onnx`

---

### 2. **Train All Trials (Batch LOCO)**

```bash
python train_loco.py --data_dir data
```

> Trains models for all trials in the data directory.

---

### 3. **Test & Evaluate a Trial**

```bash
python test_loco.py --single_trial 10 --data_dir data --step 5
```

#### Arguments:
| Flag | Description |
|------|-----------|
| `--single_trial` | Test only this trial |
| `--start_trial` / `--end_trial` | Batch test range |
| `--step` | Sequence step (1 = dense, 5 = faster) |
| `--debug_single_trial` | Test only first trial |

#### Output:
```
predictions/trial_10/
├── instance_1_col_0.csv
├── ...
└── trial_10_all_predictions.csv

plots/trial_10/
├── instance_1_col_0_pred.png
└── ...
```

---

### 4. **Quick Debug Test (Random Trials)**

```bash
python test.py
```

- Tests 5 random trials from trial 10
- Shows inline plots

---

### 5. **Convert `.pth` → `.onnx` (Verify Equivalence)**

```bash
python convert.py --pth_dir . --start_trial 3 --end_trial 33
```

- Converts all `.pth` models
- Verifies output equality (batch=1 and batch=32)

---

## 🔬 **Model Architecture**

```python
Bidirectional LSTM
├── Input:  (batch, 100, 6)   → 100 timesteps, 6 accel channels
├── LSTM:   6 → 64 (bidirectional, 2 layers, dropout=0.3)
├── FC:     128 → 2          → [Left KAM, Right KAM]
└── Output: (batch, 2)
```

- **Sequence Length**: 100 timesteps (~1 second at 100 Hz)
- **Step Size**: 1 (training), 5 (inference for speed)
- **Optimizer**: Adam + L2 regularization
- **Early Stopping**: Patience = 10

---

## 🛠 **ONNX Export**

- **Dynamic batch size** supported
- Compatible with:
  - ONNX Runtime (Python, C++, JavaScript)
  - TensorRT, OpenVINO, etc.
  - Edge devices (with ONNX Runtime Mobile)

```python
ort_session = ort.InferenceSession("model_without_trial_10.onnx")
output = ort_session.run(None, {"input": input_array})
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](../../LICENSE.md) file for details.
# Knee Moment Prediction with Bidirectional LSTM  
### **Leave-One-Subject-Out (LOSO) Cross-Validation Pipeline**  


This repository implements a **deep learning pipeline** to predict **bilateral knee adduction moments (KAM)** from **6-axis accelerometer data** collected from the lower limbs using a **Bidirectional LSTM** model. The system uses **Leave-One-Subject-Out (LOSO)** cross-validation to ensure **generalization across individuals**, a critical requirement in biomechanical and wearable sensor applications.

---

## 🎯 **Key Features**

| Feature | Description |
|-------|-----------|
| **LOSO Training** | Train on all subjects except one, test on the held-out subject |
| **Bidirectional LSTM** | Captures temporal dependencies in both directions |
| **Real-time Ready** | ONNX export with dynamic batch size |
 
| **Visualization** | Prediction vs Ground Truth plots for every trial |
| **Scalable Scripts** | Train/test individual or batch subjects |
| **Memory Efficient** | Garbage collection, CUDA cleanup, progress bars |

---

## 📁 **Repository Structure**

```
.
├── data/                          # (Your data here) Subject folders: subj1/, subj2/, ...
├── models/                        # Saved .pth and .onnx models
├── predictions/                   # CSV predictions per trial/subject
├── plots/                         # Prediction vs GT plots
├── train.py                       # Single-subject training
├── train_loso.py                  # Batch LOSO training (subjects 2–10)
├── test.py                        # Quick test on random trials (debug)
├── test_loso.py                   # Full evaluation with metrics + plots
├── convert.py                     # Convert .pth → .onnx + verify equivalence
├── code.ipynb                     # Jupyter notebook version (full pipeline)
├── progress_bar_iter.py           # Custom progress bar (optional)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore
```

---

## 🛠 **Setup & Installation**

### 1. Clone the Repository
```bash
git clone https://github.com/Azza-96/Knee-Abduction-Moment-Prediction.git
cd LOSO
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
├── subj1/
│   ├── subj1_LLML_Acc_axis1.csv
│   ├── subj1_LLML_Acc_axis2.csv
│   ├── subj1_LLML_Acc_axis3.csv
│   ├── subj1_RLML_Acc_axis1.csv
│   ├── subj1_RLML_Acc_axis2.csv
│   ├── subj1_RLML_Acc_axis3.csv
│   ├── subj1_left_knee_moment_axis2.csv
│   └── subj1_right_knee_moment_axis2.csv
├── subj2/
│   └── ...
└── ...
```

- **Inputs**: 6 acceleration channels (3 per knee: X, Y, Z)
- **Outputs**: Left & Right knee **flexion/extension moment (axis 2)** in Nm/kg
- Each CSV has **one trial per column**, rows = time steps

---

## 🚀 **Usage**

### 1. **Train a Single Subject (LOSO)**

```bash
python train.py --subject 9 --data_dir data
```

- Leaves out `subj9`, trains on all others
- Saves:  
  - `model_without_subj_9.pth`  
  - `model_without_subj_9.onnx`

---

### 2. **Train All Subjects (Batch LOSO)**

```bash
python train_loso.py --start_subject 2 --end_subject 10 --data_dir data
```

> Trains 9 models: `model_without_subj_2.onnx`, ..., `model_without_subj_10.onnx`

---

### 3. **Test & Evaluate a Subject**

```bash
python test_loso.py --single_subject 9 --data_dir data --step 5
```

#### Arguments:
| Flag | Description |
|------|-----------|
| `--single_subject` | Test only this subject |
| `--start_subject` / `--end_subject` | Batch test range |
| `--step` | Sequence step (1 = dense, 5 = faster) |
| `--debug_single_trial` | Test only first trial |

#### Output:
```
predictions/subj_9/
├── trial_1_predictions.csv
└── ...

plots/subj_9/
├── trial_1_pred.png
├── ...
```

---

### 4. **Quick Debug Test (Random Trials)**

```bash
python test.py
```

- Tests 5 random trials from subject 9
- Shows inline plots

---

### 5. **Convert `.pth` → `.onnx` (Verify Equivalence)**

```bash
python convert.py --pth_dir . --start_subject 2 --end_subject 10
```

- Converts all `.pth` models
- Verifies output equality (batch=1 and batch=32)

---

## 📊 **Sample Output**

### Prediction vs Ground Truth (Subject 9, Trial 3)
![Sample Plot](/plots/subj_9/trial_3_pred.png)

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
ort_session = ort.InferenceSession("model_without_subj_9.onnx")
output = ort_session.run(None, {"input": input_array})
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](../../LICENSE.md) file for details.
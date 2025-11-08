# Prediction of Knee Abduction Moment from Wearable Sensor Data

**Author**: Azza Tayari  

**Email**: Tayari.Azza@eniso.u-sousse.tn

---

## 📝 **Project Overview**

This repository contains the complete codebase for a PhD research project focused on predicting knee abduction moments (KAM) using data from wearable accelerometers. The primary objective of this research is to develop and validate a robust deep learning model capable of accurately estimating KAM in real-time. This work has significant applications in clinical biomechanics, sports performance analysis, and remote patient monitoring.

The project is structured into several key components:

1.  **Data Extraction & Preprocessing (`ExtractDataForML_KAM_Prediction/`)**: A collection of MATLAB scripts designed to extract, filter, and synchronize raw data from V3D-exported files. This stage is crucial for preparing the dataset for machine learning by aligning time-series data and segmenting gait cycles.

2.  **Leave-One-Subject-Out (LOSO) Cross-Validation (`LOSO/`)**: A deep learning pipeline featuring a Bidirectional LSTM to predict KAM. This validation scheme is implemented to ensure the model generalizes to new, unseen individuals.

3.  **Leave-One-Condition-Out (LOCO) Cross-Validation (`LOCO/`)**: A pipeline similar to LOSO, but designed to validate the model's ability to generalize to new movement conditions or trials.

4.  **Cycle Processing & Analysis (`ProcessedCycles/`, `Segement_cycles_Save_indice/`)**: MATLAB and Python scripts for segmenting gait cycles, calculating peak moments, and performing statistical analysis on both ground truth and predicted data.

---

## 📂 **Repository Structure**

```
├── ExtractDataForML_KAM_Prediction/ # MATLAB scripts for data extraction and filtering
├── LOSO/                            # Leave-One-Subject-Out pipeline (Bi-LSTM)
├── LOCO/                            # Leave-One-Condition-Out pipeline (Bi-LSTM)
├── ProcessedCycles/                 # MATLAB scripts for processing gait cycles
├── ProcessedCyclesPrediction/       # MATLAB scripts for processing predicted cycles
├── Segement_cycles_Save_indice/     # MATLAB scripts for cycle segmentation and analysis
├── statPeak/                        # Statistical analysis results
├── all_trials_predicted_left_columns.csv
├── all_trials_predicted_right_columns.csv
└── README.md                        # This file
```

---

## 🚀 **Core Methodologies**

### **Deep Learning Models**

Both the `LOSO` and `LOCO` pipelines employ a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network. This architecture is exceptionally well-suited for time-series data, as it can effectively capture temporal dependencies from both past and future contexts.

-   **Input**: 6-axis accelerometer data (3 axes from each lower limb).
-   **Output**: Bilateral knee abduction moments (Left and Right).
-   **Framework**: PyTorch.
-   **Deployment**: Models are exported to the ONNX (Open Neural Network Exchange) format to facilitate real-time inference.

### **Cross-Validation Strategies**

To ensure the robustness and generalizability of the models, two stringent cross-validation methods are utilized:

1.  **Leave-One-Subject-Out (LOSO)**: The model is trained on data from all subjects except for one, which is held out for testing. This process is repeated for each subject, ensuring that the model is not biased towards a specific individual’s gait pattern.

2.  **Leave-One-Condition-Out (LOCO)**: The model is trained on data from all recorded trials/conditions except for one, which is used for testing. This validates the model's performance on new or different types of movements.

---

## 🛠 **Getting Started**

Each of the sub-directories (`LOSO`, `LOCO`) contains its own detailed `README.md` file with specific instructions for setup, training, and testing.

In general, the workflow is as follows:

1.  **Prepare Data**: Use the MATLAB scripts in `ExtractDataForML_KAM_Prediction/` to process your raw V3D data.
2.  **Train Models**: Navigate to either the `LOSO/` or `LOCO/` directory and follow the instructions in the respective `README.md` to train the Bi-LSTM models.
3.  **Evaluate**: Use the provided testing scripts to evaluate model performance and generate predictions.
4.  **Analyze**: Further process the predictions using the scripts in `Segement_cycles_Save_indice/` and `ProcessedCyclesPrediction/`.

### **Dependencies**

-   **Python**: See `requirements.txt` in the `LOSO` and `LOCO` folders. Key libraries include PyTorch, NumPy, Pandas, Scikit-learn, and ONNXRuntime.
-   **MATLAB**: Requires a standard MATLAB installation with the Signal Processing Toolbox.

---

## © **Copyright and Citation**

© 2025 Azza Tayari. All rights reserved.

This repository and its contents (including code, data, models, and documentation) are the intellectual property of the authors and are associated with the manuscript:

Tayari, A., Rezgui, T., Bennour, S., & Safra, I. (2025). 
"Prediction of Knee Adduction Moment Using Foot Acceleration: Toward Personalized Gait Retraining for Knee Osteoarthritis Management."
Under review in *Machine Learning with Applications.*

Author Affiliations:
1. Mechanical Laboratory of Sousse (LMS), National School of Engineers of Sousse, University of Sousse, Tunisia – tayari.azza@eniso.u-sousse.tn  
2. Applied Mechanics and Systems Research Laboratory (LASMAP), Tunisia Polytechnic School, University of Carthage, Tunisia – taysir.rezgui@ept.ucar.tn  
3. National School of Engineers of Monastir, University of Monastir, Tunisia – sami.bennour@enim.u-monastir.tn  
4. Computer Engineering, Production and Maintenance Laboratory (LGIPM), University of Lorraine, Metz, France  
5. Department of Industrial and Systems Engineering, College of Engineering, Princess Nourah bint Abdulrahman University, Riyadh, Saudi Arabia – imsafra@pnu.edu.sa  

This material is provided exclusively for academic review and non-commercial research reference during the peer-review process.
No part of this repository may be reproduced, distributed, or used for derivative works without the explicit written permission of the author.

If you use or refer to this work, please cite:
Tayari, A. (2025). Prediction of Knee Adduction Moment Using Foot Acceleration: Toward Personalized Gait Retraining for Knee Osteoarthritis Management.
GitHub repository:  https://github.com/Azza-96/Knee-Abduction-Moment-Prediction/tree/main

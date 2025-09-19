# FFP Visualization Tools

This folder contains Python scripts to generate performance charts and analysis for the Future Frame Prediction (FFP) anomaly detection model.

## Quick Setup

### 1. Create Virtual Environment
```bash
python3 -m venv ffp_env
source ffp_env/bin/activate  # On Windows: ffp_env\Scripts\activate

### 2. Install required libraries
pip install pandas matplotlib numpy scikit-learn

### 3. Dataset setup

# Place your dataset in the Data folder with this structure:
Data/
├── ped1/
│   ├── training/frames/01-36/
│   ├── testing/frames/01-14/
│   └── ped1.mat              # Ground truth file (REQUIRED)
├── ped2/  
│   ├── training/frames/01-16/
│   ├── testing/frames/01-12/
│   └── ped2.mat              # Ground truth file (REQUIRED)
└── StreetScene/ (optional)
    ├── training/
    └── testing/


### 4. Generate Performance Chart
python ResultsIteration_UCSDPED.py




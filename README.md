# Future Frame Prediction for Video Anomaly Detection

## Project Overview

This repository contains a modernized implementation of Future Frame Prediction (FFP) for video anomaly detection, based on the seminal work by Liu et al. (2018). The implementation has been upgraded from TensorFlow 1.4 to TensorFlow 2.x and enhanced with additional evaluation capabilities as part of dissertation research comparing temporal reasoning approaches in surveillance video analysis.

### Original Work Citation
```
Liu, W., Luo, W., Lian, D., & Gao, S. (2018). 
Future frame prediction for anomaly detection--a new baseline. 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 6536-6545.
```

**Original Repository:** https://github.com/StevenLiuWen/ano_pred_cvpr2018  
**License:** MIT (maintained from original work)

## Key Modifications & Enhancements

### Technical Upgrades:
- **TensorFlow 1.4 → 2.x Migration**: Complete framework modernization with detailed upgrade report

## Repository Structure

```
dissertation_future_frame_prediction/
├── Codes_tf2/              # Core FFP implementation
│   ├── train.py           # Training script
│   ├── test.py            # Testing/evaluation script
│   ├── evaluate.py        # AUC computation and metrics
│   ├── model.py           # U-Net generator + discriminator architecture
│   ├── utils.py           # Utility functions and data loading
│   └── psnrs/             # Generated PSNR results per checkpoint
├── TF2_Upgrade_Log/        # TensorFlow migration documentation
│   └── tf2_upgrade_report.txt  # Detailed upgrade log and changes
├── Visualisation/         # Performance analysis and visualization tools
│   ├── ResultsIteration_UCSDPED.py  # Performance vs iteration charts
│   ├── Comparative_LineChart.py     # Cross-model comparison
│   └── README.md          # Visualization setup and usage guide
├── assets/               # Model configurations and checkpoints
├── Data/                 # Dataset placement directory (see setup below)
└── README.md
```

## Dataset Setup Instructions

### Required Datasets
- **UCSD PED1**: 36 training + 14 testing sequences
- **UCSD PED2**: 16 training + 12 testing sequences

### Expected Directory Structure
```
Data/
├── ped1/
│   ├── training/frames/01-36/    # Individual video folders
│   ├── testing/frames/01-14/     # Testing video folders
│   └── ped1.mat                  # Ground truth annotations (REQUIRED)
├── ped2/  
│   ├── training/frames/01-16/    # Individual video folders
│   ├── testing/frames/01-12/     # Testing video folders (exactly 12)
│   └── ped2.mat                  # Ground truth annotations (REQUIRED)
```

### Dataset Download Sources
- **UCSD Datasets:** http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- **StreetScene:** Available through research collaboration

**Important Notes:**
- Include `.mat` files containing ground truth pixel-level annotations
- Ensure proper folder naming with leading zeros (01, 02, ..., 12)
- Remove system-generated hidden files (.DS_Store, Thumbs.db)

## Installation & Usage

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/dineshgowri/dissertation_future_frame_prediction.git
cd dissertation_future_frame_prediction

# Create virtual environment
python3 -m venv ffp_env
source ffp_env/bin/activate  # On Windows: ffp_env\Scripts\activate

# Install dependencies
pip uninstall numpy -y 
pip install "numpy<2.0"
pip install tensorflow==2.13.0 
pip install tf-slim
pip install opencv-python matplotlib scipy scikit-learn pillow h5py
pip install pypng
```

### 2. Dataset Preparation
```bash
# Create Data directory and place datasets following structure above
mkdir -p Data/ped1 Data/ped2
# Copy your downloaded datasets into appropriate folders
# Ensure .mat files are included for ground truth annotations
```

### 3. Training
```bash
cd Codes_tf2
# Train on PED2 dataset (recommended for validation)
!python train.py \
    --dataset ped2 \
    --train_folder /Data/ped2/training/frames \
    --test_folder /Data/ped2/testing/frames \
    --batch 4 \
    --iters 80000 \
    --gpu 0
```

### 4. Testing & Evaluation
```bash
# Test trained model
!python inference.py  --dataset  ped2 --test_folder  ../Data/ped2/testing/frames/ --gpu  1 --snapshot_dir ./checkpoints/pretrains/ped2/model.ckpt-10000
```

## Citation & Academic Use

If you use this implementation in your research, please cite both the original paper and acknowledge this repository:

### Original Paper Citation:
```bibtex
@inproceedings{liu2018future,
  title={Future frame prediction for anomaly detection--a new baseline},
  author={Liu, Wen and Luo, Weixin and Lian, Dongze and Gao, Shenghua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6536--6545},
  year={2018}
}
```

### Repository Acknowledgment:
```
This implementation builds upon the original FFP codebase by Liu et al. (2018), 
with modernization and enhancements for TensorFlow 2.x compatibility and 
extended evaluation capabilities for dissertation research.

Original repository: https://github.com/StevenLiuWen/ano_pred_cvpr2018
Enhanced implementation: https://github.com/dineshgowri/dissertation_future_frame_prediction
```

## Research Context

This work is part of Master's dissertation research.

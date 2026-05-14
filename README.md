# Satellite Cloud Diffusion (CloudDiffusion)

A deep learning project for short-term cloud motion forecasting using geostationary satellite data. Currently implementing baselines and moving towards Diffusion-based models.

## Current Project Status
- **Baseline**: ConvLSTM (Convolutional Long Short-Term Memory) implemented.
- **Pipeline**: Fully automated from raw `.nc` (NetCDF) files to normalized `.npy` frames.
- **Tracking**: Integrated with Weights & Biases (wandb) for experiment management.
- **Metrics**: Evaluating using MSE (loss), SSIM (structural similarity), and CSI (Critical Success Index).

## Project Structure

```text
├── src/
│   ├── models/           # Model architectures
│   │   ├── convlstm.py   # Baseline ConvLSTM implementation
│   │   └── unet.py       # (Planned) Diffusion U-Net
│   ├── engine.py         # Training engine (Trainer class)
│   ├── dataset.py        # PyTorch Dataset for .npy frames
│   ├── manifest.py       # Utility to build manifest.json
│   └── utils.py          # Metrics (SSIM, CSI) and helper functions
├── train.py              # Main training entry point
├── config.yaml           # Centralized hyperparameters and settings
├── data/
│   └── manifest.json     # Generated file listing all sequence samples
└── notebooks/
    ├── inference.ipynb   # Qualitative model evaluation
    └── vis.ipynb         # Data exploration and visualization
```

## Getting Started

1. **Preprocess Data**:
   ```bash
   python src/preprocess.py --raw-dir data/raw --out-dir data/processed
   ```
2. **Build Manifest**:
   ```bash
   python src/manifest.py
   ```
3. **Train Model**:
   ```bash
   python train.py
   ```

## Metrics
- **MSE**: Standard pixel-wise error.
- **SSIM**: Measures structural/textural preservation of clouds.
- **CSI**: Critical Success Index (Threat Score) for cloud detection at a 0.5 threshold.

---
**Note**: The current dataset consists of **NASA GOES-16** geostationary data. Integration with **MOSDAC** data is planned for the next phase.

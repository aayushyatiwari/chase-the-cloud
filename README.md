# Satellite Cloud Diffusion (CloudDiffusion)

A deep learning project for short-term cloud motion forecasting. Originally developed for NASA GOES-16 data, now fully adapted for **INSAT-3DR** (MOSDAC) geostationary satellite data.

## Project Overview
The goal is to predict future cloud positions (Nowcasting) using a sequence of past satellite observations.
- **Data Source**: INSAT-3DR Thermal Infrared (TIR1) channel.
- **Sequence Length**: 6 input frames to predict the next 1 frame (T=6).
- **Model Architecture**: ConvLSTM baseline (moving towards Diffusion-based models).

## Setup & Environment
The project uses a dedicated Conda environment with GPU support.

### 1. Environment Installation
```bash
# Install Miniconda (if not already present)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate environment
conda create -n sat-cloud python=3.11 -y
conda activate sat-cloud

# Install core dependencies
conda install nomkl -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install h5py netCDF4 numpy matplotlib scipy scikit-image wandb opencv -y
```

### 2. Weights & Biases
Log in to track your experiments:
```bash
wandb login
```

## Data Pipeline

### 1. Preprocessing
Convert raw INSAT-3DR HDF5 files to normalized `.npy` frames:
```bash
python src/preprocess.py --raw-dir data/data --out-dir data/processed
```
- **Conversion**: Raw counts are mapped to Brightness Temperature (K) using the internal LUT (`IMG_TIR1_TEMP`).
- **Normalization**: Clipped between 180K and 300K and scaled to [0, 1].

### 2. Manifest Building
Generate the dataset manifest for training:
```bash
python src/manifest.py
```

## Training
Launch the training loop:
```bash
python train.py
```

## Visualization
Explore the data and model outputs using:
- `notebooks/INSAT_Check.ipynb`: Quick qualitative check of preprocessed frames.
- `notebooks/vis.ipynb`: Advanced visualization and analysis.

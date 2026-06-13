# Technical Notes & Documentation: INSAT-3DR Transition

## 1. Project Evolution
Originally built for NASA GOES-16 data (NetCDF format), the project has been refactored to support **INSAT-3DR** data from MOSDAC. This transition involved significant changes to the data ingestion pipeline and the underlying computing environment.

## 2. Data Source: INSAT-3DR
The current dataset consists of L1C imagery in HDF5 (`.h5`) format.
- **Channel**: Thermal Infrared 1 (TIR1).
- **Format**: HDF5 datasets containing raw counts and lookup tables.
- **Dimensions**: Full disk imagery (~1616x1737), currently cropped to a 256x256 region for efficient training.

### Brightness Temperature (BT) Conversion
Unlike GOES data which often provides BT directly, INSAT-3DR raw data requires a LUT mapping:
1. Load `IMG_TIR1` (Raw Counts, range ~480-950).
2. Load `IMG_TIR1_TEMP` (Lookup Table, 1024 elements).
3. Map: `BT = LUT[RawCounts]`.
This results in Kelvin values typically ranging from 180K (high cold clouds) to 310K (warm surface).

## 3. Preprocessing & Normalization
- **Normalization Range**: [180K, 300K]. 
  - Values < 180K are clipped to 0.0 (Cloud tops).
  - Values > 300K are clipped to 1.0 (Surface).
- **Output**: Clean `.npy` files stored in `data/processed/`.
- **Reasoning**: This range focuses on cloud-top temperature gradients, which are essential for motion tracking.

## 4. Environment & GPU Optimization
A custom Miniconda environment `sat-cloud` was established to resolve library conflicts (specifically MKL and OMP issues).
- **Python**: 3.11
- **PyTorch**: 2.6.0+cu124 (installed via pip for robust GPU support).
- **CUDA**: 12.4 compatibility for RTX 4050.
- **MKL Fix**: `nomkl` package was installed via Conda to prevent `undefined symbol: omp_get_num_procs` errors during visualization.

## 5. Dataset Structure (The Manifest)
The `src/manifest.py` script builds a sliding window dataset:
- **T = 6**: The model looks at 6 consecutive frames (30-min intervals if data is continuous).
- **Target**: The 7th frame in the sequence.
- **Validation**: 10% of sequences are reserved for validation, split sequentially to prevent data leakage from temporal correlation.

## 6. Training Engine
The training logic remains decoupled in `src/engine.py`.
- **Loss**: Mean Squared Error (MSE).
- **Metrics**: 
  - **SSIM**: Structural Similarity Index (higher is better).
  - **CSI**: Critical Success Index at 0.5 threshold (cloud detection accuracy).
- **Logging**: Integrated with wandb for real-time tracking of both training and system metrics.

## 7. Known Caveats
- **Missing Frames**: The manifest builder automatically skips sequences where any of the 7 required frames are missing or corrupted.
- **Static Cropping**: Currently uses a fixed center-ish crop (`row: 680-936, col: 740-996`). Future iterations should support random cropping or specific geographical regions of interest.
- **Time Step**: Assumes consistent time intervals between HDF5 files.

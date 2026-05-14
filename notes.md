# Technical Notes & Observations

## Training Infrastructure (The Engine)
We use a decoupled "Engine" architecture (`src/engine.py`) to keep `train.py` clean.
- **Trainer Class**: Encapsulates the training loop, validation, and checkpointing logic.
- **Checkpoints**: Stored in `checkpoints/`. Each checkpoint contains the model state, optimizer state, and the epoch/loss info.
- **Hardware**: Automatically detects and uses CUDA if available.

## Baseline: ConvLSTM
The ConvLSTM is our first temporal baseline.
- **Logic**: It replaces the matrix multiplication in a standard LSTM with convolutions. This allows the model to "remember" spatial patterns (like cloud shapes) and their movement over time.
- **Architecture**: Stacked layers of ConvLSTMCells. The final output is passed through a 1x1 convolution to generate a single-channel prediction.

## Experiment Tracking (WandB)
Integrated via `config.yaml`. We track:
- `train_loss` (MSE)
- `val_loss` (MSE)
- `val_ssim` (Structural similarity)
- `val_csi` (Cloud detection accuracy)

## TODO & Future Roadmap
- [x] **ConvLSTM Baseline**: Successfully implemented and trained.
- [x] **Metrics**: SSIM and CSI added for better evaluation.
- [ ] **U-Net Diffusion**: Next major architecture shift.
- [ ] **MOSDAC Data**: Adapt pipeline for `.hdf5` formats and different spatial resolutions.

## Data Source Note
All current experiments and preprocessing logic are tuned for **GOES-16 (Channel 13)** data. MOSDAC data will require updates to the cropping and normalization logic.

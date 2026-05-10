# CloudDiffusion

- .nc data se .npy karna h (preprocessing)
- .npy se dataclass
- dataclass feeded to dataloader

> key question : do we batchprocess or do we go one sample {(input:output)} at a time?

- abhi sab kuch is for GOES data only. will have to change when the MOSDAC data arrives.

# todo when the mosdac data arrives:
1. .hdf5 data hoga, converting that  to .npy or tensor. although we prefer the former.
2. cropping area change karna hoga. ab that will be hardcoded.
3. do we normalize?
4. t=6 out=1 remains the same, meaning everthing after preprocessing will be same.


## Project Structure

```text
├── data/
│   ├── download.py       # Script to download raw NetCDF files from S3 (noaa-goes16)
│   ├── preprocess.py     # Converts Radiance to Brightness Temperature, crops, and normalizes
│   └── manifest.json     # Generated JSON listing training samples
├── src/
│   ├── dataset.py        # PyTorch Dataset implementation
│   ├── explore.py        # Exploration scripts for raw data
│   └── manifest.py       # Utility to build the manifest.json
├── notebooks/
│   └── vis.ipynb         # Visualization of results and data
└── README.md
```

## TODO
priority wise

- [x] **Data Correctness Verification**: # need a method to check the correct of 1146 files i have on here. working on this.
    - Figure out a robust method to check for missing frames, temporal gaps, and data artifacts.
    - Ensure spatial consistency across the dataset.
    - only after doing this will we proceed.
- [x] **Refine Preprocessing**: Update `data/preprocess.py` based on findings from the correctness checks.
- [x] **Finalize Dataset Class**: Complete and test the `Clouds` dataset class in `src/dataset.py`.
- [ ] **Model Definition**: Implement the Diffusion Model architecture (U-Net based) for cloud forecasting.
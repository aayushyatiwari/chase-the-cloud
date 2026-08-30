# Technical Notes: Cloud Motion Nowcasting on INSAT-3DR

## 1. Project Scope

Short-term cloud motion forecasting (nowcasting) from geostationary satellite
imagery: given `T` consecutive thermal-infrared frames, predict the **single
next frame**. The project began on NASA GOES-16 data (NetCDF) and now runs on
**INSAT-3DR** L1C imagery from MOSDAC (HDF5).

- **Channel**: Thermal Infrared 1 (TIR1), single band.
- **Task**: `T` frames in → 1 frame out (not a multi-frame rollout).
- **Baseline architecture**: ConvLSTM (`src/models/convlstm.py`).
- **Legacy**: `src/explore.py` is GOES/NetCDF-era code and is not part of the
  current pipeline.

## 2. Data Source and Brightness Temperature

INSAT-3DR L1C files contain raw sensor counts plus a lookup table, unlike GOES
which often supplies brightness temperature (BT) directly. Conversion:

1. Read `IMG_TIR1` (raw counts, range ~480–950), taking index 0 of the `(1,H,W)` array.
2. Read `IMG_TIR1_TEMP` (1024-element LUT).
3. Map: `BT = LUT[raw_counts]`.

Resulting BT spans roughly 180K (high, cold cloud tops) to 310K (warm surface).
Full-disk imagery is ~1616×1737.

## 3. Preprocessing (`src/preprocess.py`)

Pipeline per file: HDF5 → BT via LUT → crop → normalize → NaN fill → `.npy`.

- **Crop**: fixed window, rows `680:936`, cols `740:996` → **256×256**.
- **Normalization**: `(BT - 180) / (300 - 180)`, clipped to `[0, 1]`.
- **Output**: `float32` `.npy` in `data/processed/`.

### Orientation convention (important)

The normalization is **not** cloud-bright. It maps:

| BT | normalized | physical meaning |
|---|---|---|
| ≤ 180K | **0.0** | coldest / highest cloud tops |
| ≥ 300K | **1.0** | warm surface, clear sky |

**Cold cloud is the LOW end of the range.** Any thresholding of cloud must
therefore test `value < threshold`, not `>`. This convention is the single
easiest thing to get backwards in this codebase — it silently inverted the CSI
metric once already (see §8). Note also that `explore.py` plots with `cmap='gray_r'`
so cold cloud renders bright, while the notebooks plot normalized data with
`cmap='gray'` where cloud renders dark.

The [180K, 300K] window is chosen to concentrate resolution on cloud-top
temperature gradients, which carry the motion signal.

## 4. Dataset Construction (`src/manifest.py`)

Builds a sliding-window manifest over the processed frames. For each window of
`T+1` frames, the first `T` are inputs and the last is the target.

Two guarantees the builder enforces:

**Sorted by parsed timestamp, not filename.** `_timestamp()` parses the
`DDMONYYYY_HHMM` field out of the filename. Lexicographic filename sorting is
only coincidentally correct within a single month — across months it orders
day-major then month-alphabetically (`01AUG` < `01JUL` < `01JUN`), which would
scramble the time series entirely. Any multi-month ingest depends on this.

**Temporal continuity.** `_is_continuous()` rejects any window whose
consecutive frames are not one time step apart (`step_minutes=30`,
`tolerance_minutes=5`). Without this check, a missing frame produces a window
that spans more time than it claims — the model is trained to predict 30
minutes ahead while the actual target is 60 minutes ahead, or an input sequence
silently skips an hour of cloud motion. The tolerance keeps benign scan-start
jitter (±3 min) while dropping genuine frame dropouts.

Skips are reported separately (`skipped_gap` vs `skipped_invalid`) so data loss
is attributable rather than a single opaque number.

### Loader (`src/dataset.py`)

`Clouds.__getitem__` loads the frames and **raises on failure**. It deliberately
does not substitute a fallback sample: drawing a random replacement index would
sample the *whole* dataset, bypassing the train/val index ranges and leaking
training frames into validation, while also hiding corrupt files and making
validation non-reproducible.

## 5. Train / Validation Split (`train.py`)

The split is **sequential** (no shuffling) because the data is a time series —
random splitting would place near-identical adjacent frames on both sides.

Sequential splitting alone is not sufficient. Because the manifest uses a
**stride-1** sliding window, manifest sample `i` spans raw frames `[i, i+T]`, so
adjacent samples overlap in `T` of their `T+1` frames. A naive contiguous index
cut therefore puts samples on either side of the boundary that share up to `T`
raw frames — and the last training sample's *target* frame appears among the
first validation sample's *inputs*.

The fix is a **buffer of `T` samples dropped at the boundary**, assigned to
neither split:

```python
train_dataset = Subset(full_dataset, range(0, train_size))
val_dataset   = Subset(full_dataset, range(train_size + T, len(full_dataset)))
```

This guarantees no raw frame is shared between any training and any validation
sample. Cost is `T` discarded samples.

## 6. Metrics (`src/utils.py`, `src/engine.py`)

**MSE** — the training objective (`nn.MSELoss`), pixelwise on normalized values.
The only metric that produces gradients; the others are diagnostic and run
under `torch.no_grad()`.

**SSIM** — windowed structural similarity, standard Wang et al. formulation
computed via convolution with an 11×11 Gaussian (σ=1.5). Local means/variances/
covariance come from `conv2d`, using `Var(X) = E[X²] − E[X]²`. Stabilizers
`C1=0.01²`, `C2=0.03²` assume **data range 1.0**. Higher is better.

**CSI (Critical Success Index)** — meteorological skill score for the
**cold-cloud class**:

```
CSI = hits / (hits + misses + false_alarms)
```

True negatives are excluded by design, so the large clear-sky majority cannot
inflate the score. Three properties of this implementation matter:

- **Cold class**: thresholds `< threshold`, matching §3. Default `0.5`
  corresponds to **240K**, close to conventional cold cloud-top cutoffs
  (~235–241K) used to flag deep convection in IR imagery.
- **Pooled, not averaged**: `csi_counts()` returns raw hit/miss/false-alarm
  counts which are accumulated across the whole epoch before
  `csi_from_counts()` forms the ratio. CSI is a ratio of sums; averaging
  per-batch ratios is a different (and biased) quantity.
- **Undefined, not zero**: with no cloud in either prediction or target the
  denominator is 0 and the result is `NaN`. Returning `0.0` would score a
  correct "no cloud anywhere" forecast as a total miss.

`calculate_csi()` remains as a single-batch convenience wrapper returning a
tensor, used by `notebooks/inference_check.ipynb`.

### Persistence baseline

`Trainer.validate()` scores a **persistence forecast** (repeat the last input
frame) on the same batches and returns `persistence_loss`, `persistence_ssim`,
`persistence_csi` next to the model metrics.

This is not optional bookkeeping. Over a 30-minute step, clouds move little, so
persistence is a strong forecast and a model that does not clearly beat it has
learned no motion. Absolute metric values are misleading without it — on this
data a constant all-warm image scores CSI ≈ 0.57 under the pre-fix warm-class
definition. Every reported number should be quoted as a delta against
persistence.

## 7. Training Flow

End to end, from raw download to logged metrics:

**Stage 1 — Preprocess** (`python src/preprocess.py --raw-dir data/data --out-dir data/processed`)
Each `.h5` → LUT-mapped BT → 256×256 crop → normalized to `[0,1]` → NaN-filled
→ `.npy`. Idempotent; existing outputs are skipped unless `--overwrite`.

**Stage 2 — Build manifest** (`python src/manifest.py`)
Sort `data/processed/*.npy` by parsed timestamp → validate each frame → slide a
`T+1` window → reject gap-spanning and invalid windows → write
`data/manifest.json` as a list of `{input_frames: [...T paths], target_frame: path}`.

**Stage 3 — Launch** (`python train.py`)
Load `config.yaml` → init wandb → select device → construct `Clouds` →
sequential split with the `T`-sample buffer (§5) → `DataLoader`s
(train `shuffle=True`, val `shuffle=False`) → build `ConvLSTM` → Adam + MSE →
`EarlyStopping` → `Trainer`.

**Stage 4 — Per-epoch train** (`Trainer.train_one_epoch`)
Per batch: dataset yields `(B, T, H, W)` and `(B, H, W)`; the engine inserts the
channel axis via `inputs.unsqueeze(2)` → `(B, T, 1, H, W)` and
`targets.unsqueeze(1)` → `(B, 1, H, W)`. Then zero grad → forward → MSE →
backward → step. Returns mean training loss.

**Stage 5 — Per-epoch validate** (`Trainer.validate`)
Under `no_grad`, for each batch compute model output and the persistence
prediction `inputs[:, -1]`; accumulate MSE and SSIM sums and pooled CSI counts
for both; return the six metrics.

**Stage 6 — Checkpoint, early stop, log**
Save on any validation-loss improvement to
`checkpoints/model_epoch_{epoch}.pt` (state dict + optimizer state + loss).
`EarlyStopping(patience=5, min_delta=0.001)` monitors validation loss. All
metrics are logged to wandb under a `val_` prefix.

### Tensor shapes

| stage | shape |
|---|---|
| Dataset item (inputs, target) | `(T, 256, 256)`, `(256, 256)` |
| Model input | `(B, T, 1, 256, 256)` |
| ConvLSTM hidden / cell state per layer | `(B, hidden_dim, 256, 256)` |
| Model output | `(B, 1, 256, 256)` |

The ConvLSTM keeps full spatial resolution at every layer (padding preserves
H×W, no downsampling), so memory scales with `hidden_dim × H × W × num_layers`.
The final `conv_last` is a 1×1 convolution from `hidden_dim` to 1 channel,
applied to the last layer's final hidden state.

## 8. Correctness Fixes Applied

Recorded with rationale, since several of these are easy to reintroduce.

| # | Issue | Resolution |
|---|---|---|
| 1 | Manifest built windows across missing frames, so 8.7% of samples had a mislabeled lead time | `_is_continuous()` gap rejection (§4) |
| 2 | Frames sorted lexicographically — correct for one month, scrambles across months | sort by parsed timestamp (§4) |
| 3 | CSI thresholded `> 0.5`, scoring the warm 76% majority instead of cold cloud | threshold `< 0.5` for the cold class (§6) |
| 4 | CSI averaged per-batch ratios instead of pooling counts | `csi_counts()` + `csi_from_counts()` (§6) |
| 5 | CSI returned `0.0` on an empty denominator, penalizing correct no-cloud forecasts | returns `NaN` (§6) |
| 6 | No baseline, so absolute metric values looked strong without evidence of skill | persistence baseline in `validate()` (§6) |
| 7 | Loader substituted a random sample on load failure, leaking train frames into val and hiding corruption | raises instead (§4) |
| 8 | Contiguous index split shared up to `T` raw frames across the train/val boundary | `T`-sample buffer dropped (§5) |

## 9. Reference Data Characteristics

Measured on the current corpus; useful for sanity checks.

- **Coverage**: July 2023 only, single fixed 256×256 crop, 30-minute nominal cadence.
- **Volume**: 1469 raw `.h5` ≈ **36 GB**; the same month processed ≈ **374 MB**
  (~100× reduction). A full year of processed frames is only ~4.5 GB, so the
  storage constraint is raw-archive retention, not training data.
- **Continuity**: 1488 slots expected for the month, 1469 present → **19 missing
  frames** across 23 irregular intervals, including a near-daily 06:45→07:45
  hole (instrument housekeeping) on 17 of 31 days. These 19 gaps poison up to 6
  windows each, which is why 114 samples are rejected.
- **Manifest**: 1349 samples after gap rejection (1463 before).
- **Class balance**: ~76% of pixels are warm (`> 0.5`), ~24% cold. The cold class
  is the minority — which is exactly why CSI must score it.
- **Clipping**: ~0.6% of pixels sit exactly at 0.0, ~0.06% at 1.0.

### Indicative results

ConvLSTM (2 layers, `hidden_dim=64`), validation split, versus persistence:

| metric | model | persistence | delta |
|---|---|---|---|
| MSE | 0.0100 | 0.0124 | −0.0024 |
| SSIM | 0.536 | 0.468 | +0.068 |
| CSI (cold, pooled) | 0.669 | 0.657 | +0.011 |

The margin over persistence is thin. Treat "beats persistence convincingly" as
the bar for any architecture change, not MSE in isolation.

## 10. Known Caveats and Open Issues

**Data coverage is the dominant limitation.** One month, one crop, one season
means validation measures a few days' extrapolation within the monsoon, not
generalization. Metrics are a sanity check, not a reliable model-selection
signal. Roughly a year of data — or at minimum 3–4 months spread across
seasons — is needed before results generalize; duration across regimes matters
more than raw sample count.

**Validation samples are highly autocorrelated.** Even with the boundary buffer,
adjacent samples share `T-1` input frames, so the effective number of
independent weather states is far below the nominal sample count.

**No test set.** Validation drives both early stopping and best-checkpoint
selection, so reported validation metrics are selection-biased upward. A third
held-out split is needed before quoting final numbers.

**Model output is unbounded.** `conv_last` has no sigmoid or clamp; predictions
reach roughly `[-0.04, 1.03]`. This violates SSIM's data-range-1 assumption
(which fails silently) and leaves CSI thresholding undefined outside `[0,1]`.

**Missing data is encoded as maximum cloud.** `np.nan_to_num(nan=0.0)` runs
*after* normalization, and 0.0 means 180K — the coldest cloud top. Sensor
dropouts and off-disk pixels therefore become synthetic deep convection. Impact
is small on the current inland crop but would be severe on a crop touching the
disk edge. A mask channel or a neutral fill value is the real fix.

**`_is_valid()`'s NaN check cannot fire.** It runs on `.npy` files that have
already been through `nan_to_num`, so no NaN survives to be caught. A frame
that was 90% NaN becomes 90% zeros — a plausible-looking giant cloud — and
passes validation. Partial-corruption detection needs to happen in
`preprocess.py`, before the fill.

**`EarlyStopping(min_delta=0.001)` is coarse** relative to a validation MSE of
~0.010, effectively demanding ~10% relative improvement per epoch and stopping
prematurely.

**No gradient clipping and no LR scheduler**, which matters more as `num_layers`
grows and the recurrent stack deepens over `T` steps.

**Checkpoints are named `model_epoch_{N}.pt` with no run or architecture
identifier**, so successive runs with different `num_layers` interleave in
`checkpoints/` and a stale file can be loaded against a mismatched config.
Notebooks that hardcode a checkpoint filename are especially exposed.

**Static crop.** Fixed geographic region; random or region-of-interest cropping
would add spatial diversity.

## 11. Environment

Conda environment with GPU support (`torchgpu` / `sat-cloud`):

- Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4.
- `h5py`, `netCDF4`, `numpy`, `matplotlib`, `scipy`, `scikit-image`, `wandb`, `opencv`.
- `nomkl` installed via conda to avoid `undefined symbol: omp_get_num_procs`
  (MKL/OMP conflict) during visualization.

See `README.md` for setup and run commands.

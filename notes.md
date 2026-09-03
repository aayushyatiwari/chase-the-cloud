# Technical Notes: Cloud Motion Nowcasting on INSAT-3DR

Last substantive update: 2026-09-03.

## 1. Task

Given `T=6` consecutive TIR1 frames at 30-minute spacing, predict the next
single frame (30 minutes ahead). Not a multi-frame rollout.

Experiment matrix, four cells:

|  | frame prediction | residual (predict the change) |
|---|---|---|
| ConvLSTM | `src/models/convlstm.py` | `+ ResidualWrapper` |
| SimVP | `src/models/simvp.py` | `+ ResidualWrapper` |

`src/explore.py` is GOES/NetCDF-era legacy and not part of the pipeline.

## 2. Data

INSAT-3DR L1C from MOSDAC, `ASIA_MER` sector, HDF5.

| | |
|---|---|
| frame shape | 1616 × 1737 (rows × cols) |
| resolution | 4.0 km/pixel (verified from the file's X/Y axes: 6,946 km over 1737 cols) |
| extent | 44.5–110°E, 10°S–45.5°N |
| area | 41.2 M km² |
| cadence | 30 min |
| raw frames | 2,964 across 62 days — July 2023 (1,469) + July 2024 (1,495) |
| raw size | 72 GB; processed float32 full-sector 32 GB |

Files carry counts plus a 1024-entry LUT: `BT = IMG_TIR1_TEMP[IMG_TIR1[0]]`.
LUT span is 179.86–340.06 K for TIR1/TIR2, 179.69–325.34 K for WV,
179.69–339.79 K for MIR.

## 3. Preprocessing (`src/preprocess.py`)

HDF5 → BT via LUT → normalize → NaN fill → `.npy` (C, H, W) in
`data/processed_full/`. Full sector kept; no crop at this stage.

`NORM_RANGES['TIR1'] = (180.0, 340.0)` — the sensor's own LUT span.

**This was 180–300 K until 2026-09-03 and that was a bug.** Daytime land over
India in July reaches 333 K, so the old ceiling pinned **8.46%** of all target
pixels to exactly 1.0 — 15–21% of the sector on midday frames, 0% at night.
Saturated targets are invisible to the loss: it cannot distinguish a correct
prediction from an overshoot. Measured after the fix, over all 2,683 targets:

| | before | after |
|---|---|---|
| pixels == 1.0 | 8.4600% | **0.0000%** |
| pixels == 0.0 | 0.2336% | 0.2033% |

The remaining 0.20% at zero is the LUT floor at 179.86 K, irreducible.

`MIR: (230.0, 315.0)` has the same disease — LUT reaches 339.8 K and MIR picks
up solar reflection. Fix before enabling that channel.

### Orientation (easy to get backwards)

Cold cloud is the **LOW** end. 0.0 = 180 K = highest/coldest tops; 1.0 = 340 K =
warm surface. Cloud thresholds must test `value < threshold`. This silently
inverted CSI once.

Target distribution, measured: mean 0.596, median 0.646, std 0.157. Only 1.52%
of pixels below 0.15 and 1.05% above 0.85.

### Known gap

`np.nan_to_num(nan=0.0)` runs *after* normalization, so missing data becomes
180 K — synthetic deep convection. No NaNs are present in the current corpus
(checked), but this bites the moment a frame with dropouts arrives. `_is_valid()`
in manifest.py cannot catch it either, since it inspects post-fill `.npy`.

## 4. Manifest (`src/manifest.py`)

Sorts by **parsed timestamp**, not filename — lexicographic ordering scrambles
across months (`01AUG` < `01JUL` < `01JUN`). Rejects any window whose
consecutive frames are not one 30-min step apart (±5 min tolerance), so no
window has a mislabeled lead time. The July 2023 → July 2024 seam is dropped
automatically by this check.

**Built at stride 7 (non-overlapping): 399 windows**, 169 rejected on gaps, 0
invalid. Verified disjoint — 2,793 frame references, 2,793 distinct.

At stride 1 the same data gave 2,683 windows, but adjacent windows share 6 of 7
frames, so that was ~396 independent sequences wearing a 6.8× inflated number.
`build(..., stride=T+1)` partitions instead. A rejected window still advances by
1, so a gap costs only the windows spanning it rather than knocking the whole
partition out of phase.

## 5. Splits (`config.yaml` → `manifest.split_indices`)

Date ranges, **not fractions**. A fraction silently moves when data is added:
`train_split: 0.8` meant "through 25 Jul 2023" before the 2024 merge and would
have meant "through 19 Jul 2024" after, making no metric comparable across the
change.

| split | dates | days | windows | × 49 tiles |
|---|---|---|---|---|
| train | 2023-07-01 → 2024-07-15 | 46 | 291 | 14,259 |
| val | 2024-07-17 → 2024-07-23 | 7 | 46 | 2,254 |
| test | 2024-07-25 → 2024-07-31 | 7 | 47 | 2,303 |

16 and 24 July 2024 belong to no split — buffer days. The 11–12 July 2024
acquisition gap (13 of 48 windows on the 11th) sits inside train.

A window joins a split only if **every one of its 7 frames** is inside the
range, so nothing reaches across a boundary for its inputs.

**Verified**, not assumed: zero shared windows and **zero shared frames**
between every pair of splits, and all 24 hours present in each split so the
diurnal cycle is represented identically.

`evaluate_test: false`. Val drives early stopping, the LR schedule and
checkpoint selection, so it is not an unbiased estimate. Run test once, at the
end.

## 6. Output convention: nothing is bounded

No sigmoid, no clamp, anywhere — not in training, not in evaluation, not on the
residual path. Both architectures end in a plain Conv2d with no activation;
`ResidualWrapper` adds the last frame to it.

Reasoning: targets are already inside [0,1] because the **inputs** were
normalized, so MSE penalizes drift on its own. A clamp in training has zero
gradient outside the range, so any pixel that wanders out can never be pulled
back. And a squashing head on one path but not the other would mean the four
cells train against different objectives — any measured difference could be the
parameterization or could be the nonlinearity, with no way to separate them.

Removed 2026-09-03: `clamp(0,1)` in `ResidualWrapper.forward`, `BoundedOutput`
(sigmoid) on the frame path, `clamp(0,1)` in `Trainer.validate`.

**Open:** the frame path now starts near 0 while the target mean is 0.596; the
residual path starts at persistence. Initializing the final conv bias to the
train-split mean would equalize the starting points so the comparison isolates
the parameterization. Not implemented — decide before running the matrix.

## 7. Metrics (`src/utils.py`, `src/engine.py`)

`Trainer.validate` returns loss / SSIM / PSNR for the model and for a
**persistence** forecast (repeat the last input frame) on the same batches.

- **MSE** — the objective.
- **SSIM** — Wang et al., 11×11 Gaussian σ=1.5, stabilizers assume data range 1.0.
- **PSNR** — pooled as squared error and elements, converted to dB once at the
  end. Averaging per-batch PSNR averages logarithms.
- **CSI** — implemented in `utils.py` (`csi_counts` / `csi_from_counts`, pooled,
  returns NaN on an empty denominator) but **not currently wired into
  `validate()`**.

Persistence is not optional bookkeeping. Over 30 minutes clouds move little, so
persistence is strong and absolute numbers are meaningless without it. Quote
every result as a delta against persistence.

Note when comparing to published work: **we forecast 30 minutes, the FY-2G paper
forecasts 1 hour.** Ours is the easier lead time.

## 8. Sampling: why tiling, not random crops

A 256×256 crop is 1/43 of the sector, so training has to crop. Random crops are
badly non-uniform:

| pixel | times trained on in 100 epochs |
|---|---|
| interior | 3.25× |
| edge midpoint | 0.013× |
| corner | 0.00005× |

Interior pixels are sampled **65,536×** more often than corners — a row-0 pixel
is reachable from one crop offset, an interior pixel from 256. The undersampled
band is 255 px deep on every side, **51.7% of the sector**, and after 100 epochs
14% of the frame has under a 50% chance of ever being seen.

That band is real weather, not off-limb space: 6.3% cloud fraction vs 10.2%
interior, mean |Δ30 min| 0.032 vs 0.037.

And val/test use `tile_grid`, which covers uniformly — so the current setup
trains center-heavy and evaluates uniformly. That is a train/eval mismatch that
biases every cell equally and for reasons unrelated to the thing being measured.

**Fix: tile the sector.** `tile_grid(1616, 1737, 256, stride=256)` gives 49
tiles (6×6 plus a flush row and column against the right/bottom edges, which
overlap slightly). Every pixel used, deterministic, identical to val/test.

Do **not** copy the paper's 128→64 center-output design. It exists because
clouds advect into a patch from outside the model's view. At 14.35 m/s and a
30-minute step, cloud moves 6.5 px at 4 km, so 95% of a 256 patch is retained,
versus 88% for their 64 patch at hourly steps. They spend 75% of their pixels as
non-targets to fix a 12% problem; for us it is a 5% problem.

## 9. Training budget

Stride 7 removes 6.7× redundancy; tiling adds 49× real coverage. Stacking both
is what made this look unaffordable.

| | samples/epoch |
|---|---|
| today (1 random crop) | 1,960 |
| stride 1 × 49 tiles | 96,040 |
| **stride 7 × 49 tiles** | **14,259** (291 sequences × 49) |
| FY-2G paper | 12,800 (800 × 16) |

50 epochs = 712,950 samples, against the paper's 640,000. Comparable, and at
14,259/epoch an ordinary training loop works — 50 validation checkpoints, no
chunked-epoch machinery needed.

Measured memory, batch 1 including backward, linear in pixel count:

| | per pixel | at 256² | max frame, 22 GB, batch 2 |
|---|---|---|---|
| ConvLSTM | 38.2 MB | 2.51 GB | 536 × 536 |
| SimVP | 14.4 MB | 0.94 GB | 873 × 873 |

ConvLSTM is 2.7× heavier — it keeps every hidden state across 6 timesteps and 3
layers for BPTT. It is the binding constraint; size for it and run SimVP the
same, or the comparison is not clean. On 6 GB (4050 laptop) batch 2 is the
ceiling at 256². On 24 GB, batch 8; gradient accumulation reaches the paper's 32.

Wall clock at 712,950 samples: 73 h on the 4050 (measured 371 ms/sample),
~20 h on one 24 GB card, ~10 h on two (both estimates).

Four cells × 3 seeds = 12 runs. Cut seeds before cutting cells — with one seed
you cannot separate a real effect from initialization luck, and a fresh ConvLSTM
started entirely negative on 5 of 8 seeds tested.

## 10. Benchmark: FY-2G / Multi-GRU-RCN (Atmosphere 2020, 11, 1151)

| | theirs | ours |
|---|---|---|
| frame | 512 × 512 | 1616 × 1737 |
| resolution | 13.3 km N-S, 10–24 km E-W (~15.8 km effective) | 4.0 km |
| area | 65.2 M km² | 41.2 M km² |
| cadence / lead time | 1 h / 1 h | 30 min / 30 min |
| days | 200 train, 20 val, 20 test (2018, all seasons) | 46 / 7 / 7 (July only) |
| sequences | non-overlapping, n=6 (5 in → 1 out) | non-overlapping, 6 in → 1 out |
| patches | 16 per frame, 128 in → 64 center out | 49 per frame, 256 in → 256 out |
| cases | 12,800 train | 14,259 train |
| batch / LR | 32 / 1e-3 | 2–8 / 1e-4 |
| hardware | one Tesla T4, 12.3 h for ConvLSTM | — |

Their stated "13 km in both directions" is inconsistent with their own stated
extent: 61° of latitude over 512 px is 13.3 km, but 110° of longitude over 512 px
is 24 km at the equator and 10 km at 65°N. It is a plate carrée grid.

They **do** crop into patches — this is not a whole-frame method.

## 11. Open items

- [x] `src/manifest.py`: stride parameter, built at stride 7 (399 windows).
- [ ] `train.py` / `Clouds`: tiled training crops (49) instead of `random_crop`.
- [ ] **New wandb project** — old runs used 180–300 K normalization, a sigmoid
      output head, clamped metrics and a fractional split. Nothing before
      2026-09-03 is comparable.
- [ ] Decide the frame-path bias init (§6).
- [ ] Wire CSI into `validate()`, or drop it from `utils.py`.
- [ ] `val_crop_stride: 512` gives 16 tiles → 736 val samples. At 256 (49 tiles)
      it is 2,254 and matches train coverage. Tile count now drives val sample
      count, since stride-7 leaves only 46 val windows.
- [ ] Restore `epochs` to ~50 and pick `batch_size` for the target GPU.
- [ ] Rotate the MOSDAC password — it is in plaintext in
      `~/code/chase-the-cloudv2/data/get_data.sh` (not in any git repo).

## 12. Correctness fixes applied

| # | Issue | Resolution |
|---|---|---|
| 1 | Windows built across missing frames → mislabeled lead time | `_is_continuous()` gap rejection |
| 2 | Lexicographic frame sort scrambles across months | sort by parsed timestamp |
| 3 | CSI thresholded `> 0.5`, scoring the warm majority | threshold `< 0.5` |
| 4 | CSI averaged per-batch ratios | pooled counts |
| 5 | CSI returned 0.0 on empty denominator | returns NaN |
| 6 | No baseline | persistence in `validate()` |
| 7 | Loader substituted a random sample on failure, leaking splits | raises |
| 8 | Contiguous index split shared frames across the boundary | date splits, whole-window containment, buffer days |
| 9 | 180–300 K normalization clipped 8.46% of targets to 1.0 | 180–340 K, the LUT span |
| 10 | Clamp in `ResidualWrapper` killed gradients outside range | removed |
| 11 | Sigmoid on the frame path only → different objective per cell | removed |
| 12 | Metrics clamped before scoring | unclipped |
| 13 | Fractional split moved silently when data was added | date-based splits |
| 14 | No test set | third split, gated behind `evaluate_test` |
| 15 | Random crops sampled interior 65,536× more than corners | tiled crops |

## 13. Environment

`torchgpu` conda env — Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, plus
`h5py`, `numpy`, `matplotlib`, `scipy`, `scikit-image`, `wandb`, `opencv`,
`pyyaml`. `nomkl` avoids an MKL/OMP symbol clash during visualization.

The `chase-the-cloud` env is **incomplete** (no `yaml`). Use `torchgpu`.

See `README.md` for setup and run commands.

# Chase the Cloud

Short-term cloud forecasting from Indian weather-satellite images. Give the
model the last 3 hours of sky and it predicts what the sky looks like 30
minutes from now.

```
 06:00   06:30   07:00   07:30   08:00   08:30        09:00
 [img]   [img]   [img]   [img]   [img]   [img]   ->   [predicted]
 <------------- 6 frames in ----------------->        1 frame out
```

---

## The dataset

Images from **INSAT-3D / 3DR**, the Indian weather satellites, over the Asia
sector (roughly 10°S–45°N, 44°E–110°E).

| | |
|---|---|
| Source files | 2964 HDF5 granules, July 2023 – July 2024 |
| Usable frames | 2961 (3 were blank) |
| One frame every | 30 minutes |
| Image size | 1616 × 1737 pixels, one pixel ≈ 4 km |
| Channels used | TIR1, TIR2, WV, MIR |
| Size on disk | 66.5 GB |

**What the four channels are.** Each is the same scene photographed at a
different wavelength, and each shows something different:

- **TIR1** (10.8 µm) — the main cloud picture. Cold = high cloud tops = storms.
  This is the one we predict.
- **TIR2** (12.0 µm) — nearly the same view; helps identify thin cirrus.
- **WV** (6.9 µm) — water vapour high in the atmosphere. Shows the airflow
  steering the clouds.
- **MIR** (3.9 µm) — useful for low cloud and fog.

The satellite also records visible and shortwave channels. We skipped those:
they go black at night, and we need a forecaster that works around the clock.

---

## Preprocessing decisions

### 1. We store the satellite's raw numbers, not temperatures

A satellite pixel is not a temperature. It is a **count** — a whole number from
0 to 1023 that the detector produced. Each file ships a small conversion table
that says what its counts mean:

```
count  555  ->  292.31 K
count  557  ->  292.01 K
count  562  ->  291.27 K
```

The obvious approach is to convert everything to temperature once, and save
that. We don't. We save the counts and keep the tables in a separate 27 MB
file, then convert while the model is being fed.

Why: a count needs only 10 bits; a temperature needs 32. Storing counts is
**half the size and loses nothing** — the conversion is a lookup, so it can be
redone any time. It also means we can change our minds about anything
downstream without touching the 66 GB again.

### 2. Every file gets its own conversion table

The tables are not identical across files. The satellite is recalibrated as its
detectors drift, by up to 13 K on the MIR channel. So count 555 means one
temperature in a July file and a slightly different one in a January file.

Using each file's own table is what makes the frames **comparable**. Reusing one
table everywhere would inject fake variation the model would try to learn as
weather.

### 3. But the 0-to-1 scaling is global

After converting to temperature, every frame is squeezed into the range 0–1
using **one fixed range per channel**, the same for all 2961 frames:

```json
{ "TIR1": [179.86, 335.84], "TIR2": [179.93, 340.07],
  "WV":   [179.69, 308.57], "MIR": [179.69, 339.79] }
```

So 180 K is always 0.0 and 335.84 K is always 1.0. Checked against all 8.3
billion pixels: the real data lands exactly inside these bounds, so nothing is
being cut off.

### 4. Three dead frames removed

Three granules came back blank or near-blank and were dropped:

```
3RIMG_31JUL2023_0420   100% empty
3RIMG_11JUL2024_2015    98% empty
3RIMG_22JUL2024_2057    11% empty
```

### 5. Whole images saved, small squares taken later

We keep the full 1616 × 1737 image on disk and cut 256 × 256 squares only when
training. Each image yields a 7 × 7 grid of 49 squares. Cutting at training
time means the square size can change without redoing anything.

**Files produced:**

```
data/processed_counts/*.npy     2961 frames, raw counts        66.5 GB
data/luts.npz                   the conversion tables          27 MB
data/norm_ranges.json           the 0-to-1 ranges              112 B
data/manifest_counts.json       which 7 frames form a sequence 328 KB
```

---

## Models and their configs

Two architectures, one per GPU, same data and same settings otherwise — so the
comparison is fair.

| | **ConvLSTM** | **SimVP** |
|---|---|---|
| Idea | Watches frames in order, carrying a memory forward | Squashes all 6 frames at once and reconstructs |
| Parameters | 747 K | 6.8 M |
| Config | `config.yaml` | `config_simvp.yaml` |
| Size | 3 layers, 64 hidden channels | hid_S 64, hid_T 256, N_S 4, N_T 4 |
| Batch size | 8 | 24 |
| Est. per epoch | ~70 min | ~25 min |

Both take **all 4 channels in** and predict **TIR1 only** out. Extra channels
are allowed to help without having to be predicted themselves — like glancing at
the wind to guess where a cloud goes, without forecasting the wind.

---

## Training methodology

### Splitting by date, not at random

```
train   2023-07-01 .. 2024-07-15     290 sequences
  (16 July: buffer, unused)
val     2024-07-17 .. 2024-07-23      46 sequences
  (24 July: buffer, unused)
test    2024-07-25 .. 2024-07-31      47 sequences
```

Random splitting would be cheating: frames 30 minutes apart look almost
identical, so the model would be tested on weather it had already seen. Splitting
by date, with a buffer day between, keeps the test genuinely unseen.

With 49 squares per frame, 290 train sequences become **14,210 training
examples**.

### Settings

| | |
|---|---|
| Loss | Mean squared error |
| Optimiser | Adam |
| Learning rate | 0.0001 |
| LR schedule | halve it after 3 epochs with no improvement, floor 1e-6 |
| Max epochs | 50 |
| Early stopping | give up after 10 epochs with no improvement |
| Seed | 42 |

**What the learning rate does.** It is the size of the step the model takes when
correcting itself. Too big and it overshoots the answer; too small and it takes
forever. Starting at 0.0001 and halving it when progress stalls is the usual
compromise: big strides early, careful shuffling later.

### The score to beat

Every validation pass also scores **persistence** — the forecast "in 30 minutes
it will look exactly like it does now." Over half an hour that is a
surprisingly good guess, and it is the honest baseline. A model that doesn't
clearly beat persistence has learned nothing about how clouds move.

Reported each epoch: loss, SSIM (how similar the images look), PSNR (error in
decibels) — model and persistence side by side.

---

## Running it

```bash
# one-off preprocessing, from raw granules
./run_preprocess.sh

# check shapes, memory and epoch time without committing to a run
python train.py --dry-run

# the two experiments, one per GPU
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=1 python train.py --config config_simvp.yaml
```

Checkpoints go to `checkpoints/<timestamp>_<model>_<size>/`. Metrics go to
Weights & Biases under project `cloud-diffusion-v2`, group `4ch-counts`.

---

## Where things stand

- Data preprocessed, transferred to the HPC container, verified end to end.
- Both experiments launched, one per Quadro RTX 6000.
- `evaluate_test: false` — the test week stays untouched until a winner is
  picked. Using it to make a decision is what stops it being an honest estimate.

### Notes for the HPC container

- `/dev/shm` is only 70 MB there, so `num_workers: 0`. With workers on, the
  data loader crashes with a confusing "bus error".
- Driver 470 caps us at CUDA 11.4, so PyTorch stays pinned to the CUDA 11.8
  build. Don't upgrade it.
- The file listing which frames form a sequence stores absolute paths, so it
  must be rebuilt on the machine that uses it: `python -m src.manifest`.

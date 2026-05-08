# Satellite Data Pipeline — Technical Notes

## What We're Working With

GOES-16 is a geostationary satellite sitting above the Americas. It continuously takes infrared photos of Earth. Each photo is stored as a `.nc` (NetCDF) file.

We're using **Channel 13** — the clean longwave infrared channel (10.3μm). This channel measures how much heat each spot on Earth is radiating outward. It doesn't need sunlight, so it works day and night.

**Why Channel 13 for clouds?**
- Cold pixels = high altitude clouds (ice tops)
- Warm pixels = clear sky or low clouds
- Gives a clean signal for cloud motion

---

## The `.nc` File Format

NetCDF is the standard format for scientific array data. Think of it as a dictionary of arrays, each with metadata attached.

A GOES `.nc` file has three kinds of things inside:

**Dimensions** — define the axes. For a 2D image: `y` (rows) and `x` (columns).

**Variables** — the actual arrays. The important ones:
- `Rad` — raw radiance values, one per pixel. Shape: `(1500, 2500)` for CONUS scans.
- `DQF` — Data Quality Flag. `0` means good pixel, anything else means bad/missing.
- `planck_fk1`, `planck_fk2`, `planck_bc1`, `planck_bc2` — four scalar constants used for conversion.

**Attributes** — metadata like units, sensor info, valid range.

### Reading a `.nc` file

```python
from netCDF4 import Dataset

ds = Dataset("file.nc")
rad = ds.variables['Rad'][:]   # load the Rad array
ds.close()
```

The `[:]` loads the full array into memory. Without it, you just get a reference.

### Masked Arrays

NetCDF variables return **masked arrays** — a numpy array with a boolean mask on top. Masked pixels are bad data (off-earth, sensor error, etc.).

```python
rad = ds.variables['Rad'][:]   # MaskedArray — safe
rad = ds.variables['Rad'][:].data  # raw buffer — DANGEROUS
```

Using `.data` strips the mask and gives you the raw underlying buffer. If a pixel was masked, its value is undefined garbage. **Never use `.data`.**

The safe way to fill masked values:
```python
rad = np.ma.filled(ds.variables['Rad'][:], fill_value=0.0)
```

---

## Radiance to Brightness Temperature

Raw radiance values are in physical units (`mW m-2 sr-1 (cm-1)-1`). They're not directly interpretable. We convert them to **Brightness Temperature (BT)** in Kelvin using the Planck function.

### The Formula

```python
BT = (fk2 / np.log(fk1 / rad + 1) - bc1) / bc2
```

**What each variable is:**
- `rad` — `(1500, 2500)` array. One radiance value per pixel.
- `fk1`, `fk2`, `bc1`, `bc2` — four scalar constants. Same for every pixel. Loaded once from the file. Encode the physics of Channel 13's wavelength and sensor calibration.

**How it runs:**
NumPy broadcasts the four scalars across the entire 2D array. The formula runs per pixel with no explicit loop. For 3.75 million pixels (1500×2500), this takes milliseconds on CPU.

**The compute cost breakdown:**
- Math (Planck formula): essentially free — vectorized, no GPU needed
- IO (reading `.nc`, writing `.npy`): the actually expensive part

---

## The Preprocessing Pipeline

Raw `.nc` files are converted to `.npy` files for fast loading during training.

### Steps per file

1. Load `Rad` from `.nc`
2. Apply Planck formula → get BT in Kelvin
3. Crop to region of interest
4. Normalize to `[0, 1]`
5. Replace `nan` with `0`
6. Save as `.npy`

### Safe File Writing

A common bug: if preprocessing crashes mid-write, the output file exists on disk but is corrupted (header written, data missing). Next run skips it because the file "exists."

Fix — write to a temp file, rename atomically on success:

```python
tmp_path = out_path + ".tmp"
np.save(tmp_path, bt)
os.rename(tmp_path, out_path)   # atomic on Linux
```

`os.rename` is atomic — either the complete file appears at `out_path`, or nothing. A half-written `.tmp` never gets picked up by the existence check.

### Normalization

Hardcoded ranges like `bt_min=200, bt_max=300` break when actual data falls outside that range. Percentile-based normalization adapts to real data:

```python
def normalize(bt):
    bt_min = np.nanpercentile(bt, 1)
    bt_max = np.nanpercentile(bt, 99)
    return np.clip((bt - bt_min) / (bt_max - bt_min), 0, 1)
```

`nanpercentile` ignores `nan` values. Using 1st and 99th percentile instead of min/max avoids outliers distorting the range.

**Important:** Before calling this, convert the masked array to a plain float array:

```python
bt = np.array(bt, dtype=np.float32)
```

Otherwise numpy raises a `read-only` error because it can't write into a masked array's buffer during the percentile computation.

---

## Parsing Timestamps from Filenames

GOES filenames encode the timestamp directly:

```
OR_ABI-L1b-RadC-M6C13_G16_s20231820001179_e20231820003563_c20231820004012
                                ↑
                         s = start time
```

The `s` field format: `sYYYYDDDHHMMSSt`
- `YYYY` — year
- `DDD` — day of year (1–365)
- `HH` — hour
- `MM` — minute
- `SS` — second
- `t` — tenths of second (ignore)

```python
from datetime import datetime, timedelta

def parse_goes_time(filename):
    part = [p for p in filename.split('_') if p.startswith('s')][0]
    s = part[1:]
    year   = int(s[0:4])
    doy    = int(s[4:7])
    hour   = int(s[7:9])
    minute = int(s[9:11])
    second = int(s[11:13])
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
    return dt
```

**Logic:** `datetime(year, 1, 1)` creates January 1st of that year. Adding `timedelta(days=doy-1)` shifts to the correct day. The rest adds hours, minutes, seconds.

This timestamp is used to detect gaps between frames — consecutive frames in a training window must be exactly 10 minutes apart.

---

## The Manifest

A manifest is a JSON file that lists every training sample. Each sample is a sliding window of T consecutive frames (input) + the next frame (target).

### Why a manifest?

Without it, your DataLoader has to figure out which files belong together every time training starts. With it:

- Shuffling is cheap — shuffle the manifest, not the files on disk
- DataLoader is simple — just reads what's listed
- Scaling is easy — run `build_manifest` on a new data folder, nothing else changes
- Gap detection is centralized — one place to enforce temporal consistency

### Building the manifest

```python
def build_manifest(data_dir, T=4, output_path="manifest.json"):
    files = sorted(Path(data_dir).glob("*.npy"))  # sorted = chronological order

    samples = []
    for i in range(len(files) - T):
        samples.append({
            "input_frames": [str(files[i + j]) for j in range(T)],
            "target_frame": str(files[i + T])
        })

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
```

**The logic:** For each position `i`, take T files starting at `i` as input, and the file at `i+T` as target. This is a sliding window — each step moves forward by one frame.

**What the manifest looks like:**
```json
[
  {
    "input_frames": ["frame_00.npy", "frame_01.npy", "frame_02.npy", "frame_03.npy"],
    "target_frame": "frame_04.npy"
  },
  {
    "input_frames": ["frame_01.npy", "frame_02.npy", "frame_03.npy", "frame_04.npy"],
    "target_frame": "frame_05.npy"
  }
]
```

Notice frames overlap between consecutive samples. This is intentional — maximizes use of available data.

---

## Dataset and DataLoader

### The separation of concerns

| Component | Who writes it | What it does |
|-----------|--------------|--------------|
| `Dataset` | You | Knows how to read one sample |
| `DataLoader` | PyTorch | Calls Dataset repeatedly, batches results, prefetches |

### Dataset

```python
class CloudDataset(Dataset):
    def __init__(self, manifest_path, T=4):
        with open(manifest_path) as f:
            self.samples = json.load(f)   # load manifest into memory (just paths, not arrays)
        self.T = T

    def __len__(self):
        return len(self.samples)          # total number of training samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = np.stack([np.load(p) for p in sample["input_frames"]])  # (T, H, W)
        target = np.load(sample["target_frame"])                          # (H, W)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
```

**What's boilerplate:** `__init__`, `__len__`, `return torch.tensor(...)` — standard PyTorch Dataset pattern, always looks like this.

**What's logic:** The `__getitem__` body — how you load and stack frames. This is specific to your data format and changes per project.

`np.stack` turns a list of `(H, W)` arrays into a single `(T, H, W)` array. The stacking runs inside the worker process, so it's parallelized automatically.

### DataLoader

```python
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
```

**What each argument does:**
- `batch_size=8` — stack 8 samples into one batch. Output shape: `(8, T, H, W)`
- `shuffle=True` — randomize sample order each epoch. Shuffles manifest indices, not files on disk.
- `num_workers=2` — 2 background processes prefetch the next batch while the GPU trains on the current one. Hides IO latency.

**The mental model:**
```
Dataset      →    knows how to read ONE sample
DataLoader   →    calls Dataset repeatedly, batches, prefetches ahead
```

---

## Finding the Best Crop Region

### The problem

A CONUS scan is `1500×2500` pixels. We need a `256×256` crop for training. Where we crop matters — open ocean with no clouds is useless training data.

### The algorithm

Find the region with the highest **variance** across multiple frames. High variance means the pixel values are changing a lot over time — that's where cloud motion is happening.

**Why variance?** A region of flat ocean has near-zero variance across frames (always the same temperature). A region with active convection has high variance (clouds appearing, moving, dissipating).

### Step 1 — Sliding window variance per frame

```python
def variance_map(bt, window=256, stride=64):
    H, W = bt.shape
    bt_clean = bt.copy()
    bt_clean[bt_clean > 320] = np.nan   # mask out fill values (too hot = bad data)
    bt_clean[bt_clean < 150] = np.nan   # mask out fill values (too cold = bad data)

    rows = range(200, H - window - 200, stride)   # skip 200px edge on each side
    cols = range(200, W - window - 200, stride)

    vmap = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            patch = bt_clean[r:r+window, c:c+window]
            valid = np.sum(~np.isnan(patch))
            if valid > 0.8 * window * window:     # at least 80% real pixels
                vmap[i, j] = np.nanvar(patch)
    return vmap
```

**Why skip edges (`range(200, ...)`):** GOES edge pixels are geometrically distorted and often masked. They produce artificially high variance that isn't real cloud activity.

**Why the 80% valid pixel check:** A patch that's mostly masked values will have misleadingly low or high variance. Only score patches with mostly real data.

**`stride=64`:** Check a window every 64 pixels instead of every pixel. Coarse enough to be fast, fine enough to find a good region.

### Step 2 — Average variance across frames

```python
avg_vmap = None
for f in files[:30]:               # use 30 frames, not all — enough to be representative
    try:
        bt = get_bt(f)
        vmap = variance_map(bt)
        avg_vmap = vmap if avg_vmap is None else avg_vmap + vmap
    except:
        print(f"skipping {f.name}")

avg_vmap /= len(files)
```

One frame might have a storm that's gone in the next. Averaging across 30 frames finds regions that are consistently active, not just lucky in one frame.

### Step 3 — Pick the best window

```python
best_idx = np.unravel_index(np.argmax(avg_vmap), avg_vmap.shape)

row_offset = 200   # must match the range() start used in variance_map
col_offset = 200

row_start = row_offset + best_idx[0] * stride
col_start = col_offset + best_idx[1] * stride
```

**The offset correction is critical.** `best_idx` is an index into `vmap`, which starts at row 200, not row 0. Without adding `row_offset`, the coordinates point to the wrong place on the full image.

`np.argmax` returns the flat index of the maximum. `np.unravel_index` converts it to `(row_idx, col_idx)` in the vmap grid.

---

## Summary — What Each File Does

| File | Purpose |
|------|---------|
| `preprocess.py` | Converts raw `.nc` → normalized `.npy` |
| `build_manifest.py` | Scans processed folder, builds `manifest.json` |
| `src/dataset.py` | `CloudDataset` — reads manifest, loads frames |
| `src/explore.py` | `check()` — visualize a single raw file |
| `rough.py` | Scratch exploration — variance map, crop finding |
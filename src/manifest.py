import os
import json
from pathlib import Path
import numpy as np

def _is_valid(path):
    try:
        arr = np.load(path)
        return arr.ndim == 2 and not np.all(np.isnan(arr))
    except Exception:
        return False

def build(data_dir, T=6, output_path='data/manifest.json'):
    files = sorted(Path(data_dir).glob('*.npy'))
    valid = {f for f in files if _is_valid(f)}
    
    samples = []
    skipped = 0
    for i in range(len(files) - T):
        window = files[i : i + T + 1]
        if all(f in valid for f in window):
            samples.append({
                "input_frames": [str(f.resolve()) for f in window[:T]],
                "target_frame": str(window[T].resolve())
            })
        else:
            skipped += 1
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(output_path), 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Built {len(samples)} samples, skipped {skipped} sequences")

if __name__ == '__main__':
    build('data/processed/')
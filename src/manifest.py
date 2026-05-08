import os 
import json 
from pathlib import Path

def build(data_dir, T=6, output_path = 'data/manifest.json'):
    files = sorted(Path(data_dir).glob('*.npy'))

    samples = []
    for i in range(len(files) - T):
        samples.append({
            "input_frames": [str(files[i + j]) for j in range(T)],
            "target_frame": str(files[i + T])
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Built {len(samples)} samples successfully")

if __name__ == '__main__':
    build('data/processed/')
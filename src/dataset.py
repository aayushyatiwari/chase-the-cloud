import json
import numpy as np
import torch
from torch.utils.data import Dataset

class Clouds(Dataset):
    def __init__(self, manifest_path='data/manifest.json', T=6):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.T = T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load failures are raised, not substituted: silently swapping in a
        # random sample would draw from the whole dataset, leaking training
        # frames into validation and hiding corrupt data.
        sample = self.samples[idx]
        inputs = np.stack([np.load(p) for p in sample['input_frames']])
        target = np.load(sample['target_frame'])
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
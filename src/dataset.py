import json 
import os 
import torch 
import numpy as np
from torch.utils.data import Dataset

class Clouds(Dataset):
    def __init__(self, manifest_path='data/manifest.json', T=6):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.T = T 
    
    def __len__ (self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = np.stack([np.load(p) for p in sample['input_frames']])
        outputs = np.load(sample['target_frame'])
        return torch.tensor(inputs, dtype=torch.float32) , torch.tensor(outputs, dtype=torch.float32)

'''
NOTE:
One — np.stack in __getitem__ runs inside the worker process, so the stacking cost is parallelized across workers automatically.
Two — shuffle=True shuffles the manifest indices, not the files on disk. This is why the manifest pattern is powerful — cheap shuffling.
'''
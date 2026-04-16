import torch
from torch.utils.data import Dataset
import numpy as np


class TorchBEVDataset(Dataset):
    def __init__(self, nusc, bev_dataset):
        self.nusc = nusc
        self.bev_dataset = bev_dataset
        self.samples = nusc.sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        bev, target = self.bev_dataset.get_sample(sample)

        # Convert to torch tensors
        bev = torch.from_numpy(bev).float()
        target = torch.from_numpy(target).float()

        # Change BEV to (C, H, W)
        bev = bev.permute(2, 0, 1)

        return bev, target
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBEVDetector(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super().__init__()

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Head
        self.head = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        # Output: (B, 6, H, W)
        return x
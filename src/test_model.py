import torch
from model import SimpleBEVDetector

model = SimpleBEVDetector()

x = torch.randn(2, 3, 500, 500)

y = model(x)

print("Output shape:", y.shape)

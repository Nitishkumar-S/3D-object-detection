import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from bev_grid_generator import BEVGridGenerator
from bev_dataset import BEVDataset
from torch_dataset import TorchBEVDataset
from model import SimpleBEVDetector
from loss import detection_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load nuScenes
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot='/work/dlclarge1/solpuren-object_detection/3D-object-detection/data/sets/nuscenes',
        verbose=False
    )

    # BEV + Dataset
    bev_generator = BEVGridGenerator(
        x_range=(-50, 50),
        y_range=(-50, 50),
        resolution=0.2
    )

    bev_dataset = BEVDataset(nusc, bev_generator)
    dataset = TorchBEVDataset(nusc, bev_dataset)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    # Model
    model = SimpleBEVDetector().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0

        for i, (bev, target) in enumerate(loader):
            bev = bev.to(device)
            target = target.to(device)

            # Forward
            pred = model(bev)

            # Loss
            loss, obj_loss, box_loss = detection_loss(pred, target)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i}")
                print(f"  Total Loss: {loss.item():.4f}")
                print(f"  Obj Loss:   {obj_loss.item():.4f}")
                print(f"  Box Loss:   {box_loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
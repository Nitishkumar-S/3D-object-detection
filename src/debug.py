import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from bev_grid_generator import BEVGridGenerator
from bev_dataset import BEVDataset


def main():
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot='/work/dlclarge1/solpuren-object_detection/3D-object-detection/data/sets/nuscenes',
        verbose=False
    )

    bev_generator = BEVGridGenerator(
        x_range=(-50, 50),
        y_range=(-50, 50),
        resolution=0.2
    )

    bev_dataset = BEVDataset(nusc, bev_generator)

    # pick a sample
    sample = nusc.sample[0]

    bev, target = bev_dataset.get_sample(sample)

    print("BEV shape:", bev.shape)
    print("Target shape:", target.shape)
    print("Num objects:", target[..., 0].sum())

    # Visualization
    density = bev[:, :, 0]
    objectness = target[:, :, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(density, cmap='gray', origin='lower')

    ys, xs = np.where(objectness == 1)
    plt.scatter(xs, ys, c='red', s=10)

    plt.title("BEV + object centers")
    plt.savefig("visualization/debug_bev.png")
    print("Saved visualization/debug_bev.png")


if __name__ == "__main__":
    main()
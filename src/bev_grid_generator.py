import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class BEVGridGenerator:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), resolution=0.2):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.resolution = resolution

        self.H = int((self.x_max - self.x_min) / self.resolution)
        self.W = int((self.y_max - self.y_min) / self.resolution)

    def generate_bev(self, pc):
        points = pc.points  # (4, N)

        x = points[0]
        y = points[1]
        z = points[2]

        # Clip height (robustness)
        z = np.clip(z, -3, 3)
        # Filter points within range
        mask = (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max)
        )

        x = x[mask]
        y = y[mask]
        z = z[mask]

        # Convert to grid indices
        x_idx = ((x - self.x_min) / self.resolution).astype(np.int32)
        y_idx = ((y - self.y_min) / self.resolution).astype(np.int32)

        # Initialize grids
        density = np.zeros((self.H, self.W), dtype=np.float32)
        max_height = np.full((self.H, self.W), -np.inf, dtype=np.float32)
        sum_height = np.zeros((self.H, self.W), dtype=np.float32)

        # Density (count)
        np.add.at(density, (y_idx, x_idx), 1)

        # Sum height
        np.add.at(sum_height, (y_idx, x_idx), z)

        # Max height
        np.maximum.at(max_height, (y_idx, x_idx), z)

        # Mean height
        mean_height = np.zeros_like(density)
        valid_mask = density > 0
        mean_height[valid_mask] = sum_height[valid_mask] / density[valid_mask]

        max_height[max_height == -np.inf] = 0

        # Normalization
        # Density normalization (log scale)
        density = np.log1p(density) / np.log(64)

        # Height normalization
        z_min, z_max = -2, 2
        max_height = np.clip((max_height - z_min) / (z_max - z_min), 0, 1)
        mean_height = np.clip((mean_height - z_min) / (z_max - z_min), 0, 1)

        bev = np.stack([density, max_height, mean_height], axis=-1)

        return bev

    def visualize(self, bev, save_prefix="bev_grid"):
        density = bev[:, :, 0] 
        max_height = bev[:, :, 1] 
        mean_height = bev[:, :, 2] 
        
        os.makedirs("visualization", exist_ok=True) 
        
        plt.figure(figsize=(6, 6)) 
        plt.imshow(density, cmap='hot', origin='lower') 
        plt.title("Density (log scale)") 
        plt.colorbar() 
        plt.savefig(f"visualization/{save_prefix}_density.png") 
        plt.close() 
        
        plt.figure(figsize=(6, 6)) 
        plt.imshow(max_height, cmap='viridis', origin='lower') 
        plt.title("Max Height") 
        plt.colorbar() 
        plt.savefig(f"visualization/{save_prefix}_max_height.png") 
        plt.close() 

        plt.figure(figsize=(6, 6)) 
        plt.imshow(mean_height, cmap='viridis', origin='lower') 
        plt.title("Mean Height") 
        plt.colorbar() 
        plt.savefig(f"visualization/{save_prefix}_mean_height.png") 
        plt.close()

def main():
    # Update this path
    dataroot = "data/sets/nuscenes"

    nusc = NuScenes(
        version='v1.0-mini',
        dataroot='/work/dlclarge1/solpuren-object_detection/3D-object-detection/data/sets/nuscenes',
        verbose=False
    )

    # Take first sample
    sample = nusc.sample[0]

    # Get LiDAR data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    lidar_path = os.path.join(dataroot, lidar_data['filename'])

    pc = LidarPointCloud.from_file(lidar_path)

    # Generate BEV
    bev_generator = BEVGridGenerator(
        x_range=(-50, 50),
        y_range=(-50, 50),
        resolution=0.2
    )

    bev = bev_generator.generate_bev(pc)

    # Save visualization
    os.makedirs("visualization", exist_ok=True)
    bev_generator.visualize(bev, "bev_grid")

    print("BEV grid saved to visualization/")

if __name__ == "__main__":
    main()

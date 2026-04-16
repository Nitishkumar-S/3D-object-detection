import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import os

# from bev_grid_generator import BEVGridGenerator # --- IGNORE ---


class BEVDataset:
    def __init__(self, nusc, bev_generator):
        self.nusc = nusc
        self.bev_generator = bev_generator

    def get_sample(self, sample):
        
        # Load LiDAR
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(lidar_path)

        bev = self.bev_generator.generate_bev(pc)

        # Create target map
        H, W, _ = bev.shape
        target = np.zeros((H, W, 6), dtype=np.float32)

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Filter only cars
            if "vehicle" not in ann['category_name']:
                continue

            # Get box in global frame
            box = self.nusc.get_box(ann_token)

            # Get ego pose and calibrated sensor
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

            # GLOBAL → EGO
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # EGO → LIDAR
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            # Now box is in LiDAR frame
            x, y, z = box.center
            w, l, h = box.wlh
            yaw = box.orientation.yaw_pitch_roll[0]

            # Check if inside BEV range
            if not (
                self.bev_generator.x_min <= x < self.bev_generator.x_max and
                self.bev_generator.y_min <= y < self.bev_generator.y_max
            ):
                continue

            # Convert to grid
            i = int((x - self.bev_generator.x_min) / self.bev_generator.resolution)
            j = int((y - self.bev_generator.y_min) / self.bev_generator.resolution)

            # Normalize box dimensions to grid scale
            w_grid = w / self.bev_generator.resolution
            l_grid = l / self.bev_generator.resolution

            # Offsets (center inside cell)
            x_cell = (x - self.bev_generator.x_min) / self.bev_generator.resolution
            y_cell = (y - self.bev_generator.y_min) / self.bev_generator.resolution

            offset_x = x_cell - i
            offset_y = y_cell - j

            # Fill target
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx = i + dx
                    ny = j + dy

                    if 0 <= nx < W and 0 <= ny < H:
                        target[ny, nx, 0] = 1
            target[j, i, 1] = offset_x
            target[j, i, 2] = offset_y
            target[j, i, 3] = np.log(w_grid + 1e-6)
            target[j, i, 4] = np.log(l_grid + 1e-6)
            target[j, i, 5] = yaw

        return bev, target
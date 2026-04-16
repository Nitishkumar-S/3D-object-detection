from nuscenes.nuscenes import NuScenes, LidarPointCloud
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import os

nusc = NuScenes(
    version='v1.0-mini',
    dataroot='/work/dlclarge1/solpuren-object_detection/3D-object-detection/data/sets/nuscenes',
    verbose=True
)

sample = nusc.sample[0]

# print(sample.keys())
# print(sample)

ann_token = sample['anns'][0]
ann = nusc.get('sample_annotation', ann_token)

# print(ann)

# Get LiDAR token
lidar_token = sample['data']['LIDAR_TOP']

# Get metadata
lidar_data = nusc.get('sample_data', lidar_token)

# File path
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

# print(lidar_path)

# Load point cloud
pc = LidarPointCloud.from_file(lidar_path)

# print(pc.points.shape)

nusc.render_sample(sample['token'])

plt.savefig("visualization/sample_vis.png")
# plt.close()

# BEV pointcloud
x = pc.points[0, :]
y = pc.points[1, :]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=0.5)

# Add boxes
ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
ego_translation = np.array(ego_pose['translation'])
ego_rotation = Quaternion(ego_pose['rotation'])

for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)

    # global → ego
    ann_translation = np.array(ann['translation']) - ego_translation
    ann_translation = ego_rotation.inverse.rotate(ann_translation)

    x_ann, y_ann = ann_translation[0], ann_translation[1]
    w, l, _ = ann['size']

    rect = plt.Rectangle(
        (x_ann - w/2, y_ann - l/2),
        w, l,
        fill=False,
        linewidth=1
    )
    plt.gca().add_patch(rect)
plt.xlabel("x (forward)")
plt.ylabel("y (left-right)")
plt.title("LiDAR BEV (Top View)")
plt.axis('equal')

plt.savefig("visualization/bev_pointcloud.png")
plt.close()


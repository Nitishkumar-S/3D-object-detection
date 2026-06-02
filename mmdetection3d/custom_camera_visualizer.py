import os
import cv2
import argparse
import mmengine
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Project 3D predictions to Camera Image')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('pkl_path', help='Path to predictions .pkl')
    parser.add_argument('--out-dir', default='video_frames_camera', help='Output folder')
    parser.add_argument('--score-thr', type=float, default=0.25, help='Confidence threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.test_mode = True
    
    # We don't need point cloud sweeps for image projection
    pipeline = cfg.test_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.pipeline = [s for s in pipeline if s['type'] != 'LoadPointsFromMultiSweeps']
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    preds = mmengine.load(args.pkl_path)
    
    print("Starting Camera Projection Visualizer...")
    
    # Define the 12 lines that make up a 3D bounding box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4], # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical pillars
    ]
    
    for i in range(len(dataset)):
        data_info = dataset.get_data_info(i)
        
        # 1. Load the Front Camera Image (FIXED PATH)
        raw_path = data_info['images']['CAM_FRONT']['img_path']
        filename = os.path.basename(raw_path)
        
        # Manually reconstruct the correct nuScenes path
        img_path = os.path.join('data', 'nuscenes', 'samples', 'CAM_FRONT', filename)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Get Projection Matrices from the dataset info
        # lidar2cam transforms 3D LiDAR space to 3D Camera space
        lidar2cam = np.array(data_info['images']['CAM_FRONT']['lidar2cam'])
        # cam2img transforms 3D Camera space to 2D Image pixels
        cam2img = np.array(data_info['images']['CAM_FRONT']['cam2img'])[:3, :3]
        
        pred_dict = preds[i]
        if 'pred_instances_3d' in pred_dict:
            bboxes = pred_dict['pred_instances_3d']['bboxes_3d']
            scores = pred_dict['pred_instances_3d']['scores_3d'].numpy()
            
            # Filter by confidence score
            mask = scores > args.score_thr
            bboxes = bboxes[mask]
            
            if len(bboxes) > 0:
                # Get the 8 corners of every bounding box (in LiDAR coordinates)
                corners_3d = bboxes.corners.numpy() # Shape: (N, 8, 3)
                N = corners_3d.shape[0]
                
                # Convert to homogeneous coordinates (N, 8, 4) to do matrix multiplication
                corners_3d_hom = np.concatenate([corners_3d, np.ones((N, 8, 1))], axis=-1)
                
                # Transform from LiDAR to Camera coordinates
                corners_cam = corners_3d_hom @ lidar2cam.T
                corners_cam = corners_cam[..., :3]
                
                # Project from Camera coordinates to 2D Image Pixels
                corners_img = corners_cam @ cam2img.T
                corners_2d = corners_img[..., :2] / corners_img[..., 2:3]
                
                # 3. Draw the boxes on the image
                for b in range(N):
                    # Skip drawing boxes that are physically behind the camera lens (Z < 0)
                    if np.any(corners_cam[b, :, 2] < 0.1):
                        continue
                        
                    # Draw the 12 edges using OpenCV
                    for edge in edges:
                        pt1 = (int(corners_2d[b, edge[0], 0]), int(corners_2d[b, edge[0], 1]))
                        pt2 = (int(corners_2d[b, edge[1], 0]), int(corners_2d[b, edge[1], 1]))
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2) # Neon Green
        
        # 4. Save the finished image
        plt.figure(figsize=(16, 9))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        name = f"frame_{i:04d}.png"
        plt.savefig(os.path.join(args.out_dir, name), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        if i % 10 == 0:
            print(f"Rendered {i} / {len(dataset)} camera frames...")

    print(f"Success! Check the {args.out_dir} folder.")

if __name__ == '__main__':
    main()
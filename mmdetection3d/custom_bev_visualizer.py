import os
import argparse
import mmengine
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Render BEV Video Frames')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('pkl_path', help='Path to predictions .pkl')
    parser.add_argument('--out-dir', default='video_frames', help='Output folder')
    parser.add_argument('--score-thr', type=float, default=0.25, help='Confidence threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.test_mode = True
    
    # Safely remove Multi-Sweeps to prevent path bugs and speed up rendering
    pipeline = cfg.test_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.pipeline = [s for s in pipeline if s['type'] != 'LoadPointsFromMultiSweeps']
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    preds = mmengine.load(args.pkl_path)
    
    print("Starting Bulletproof BEV Matplotlib Renderer...")
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # 1. Extract Points Safely
        points = data['inputs']['points']
        if hasattr(points, 'numpy'): pts = points.numpy()
        elif hasattr(points, 'tensor'): pts = points.tensor.cpu().numpy()
        else: pts = points
            
        pred_dict = preds[i]
        if 'pred_instances_3d' not in pred_dict: continue
        
        # 2. Extract Bounding Boxes Safely
        bboxes = pred_dict['pred_instances_3d']['bboxes_3d']
        scores = pred_dict['pred_instances_3d']['scores_3d']
        
        if hasattr(bboxes, 'cpu'): bboxes = bboxes.cpu().numpy()
        elif hasattr(bboxes, 'tensor'): bboxes = bboxes.tensor.cpu().numpy()
        if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
            
        # Filter by confidence
        valid = scores > args.score_thr
        bboxes = bboxes[valid]
        
        # --- MATPLOTLIB DRAWING ---
        plt.figure(figsize=(8, 8))
        plt.style.use('dark_background') 
        
        # Draw Point Cloud (Subsampled for cleaner look and speed)
        plt.scatter(pts[::3, 0], pts[::3, 1], s=0.2, c='gray', alpha=0.5)
        
        # Draw Bounding Boxes
        for box in bboxes:
            x, y, z, w, l, h, yaw = box[:7]
            
            # Calculate rotated 2D corners for the box
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            cx = np.array([l/2, l/2, -l/2, -l/2])
            cy = np.array([w/2, -w/2, -w/2, w/2])
            
            rot_x = cx * cos_y - cy * sin_y + x
            rot_y = cx * sin_y + cy * cos_y + y
            
            # Plot the green bounding box
            plt.plot(np.append(rot_x, rot_x[0]), np.append(rot_y, rot_y[0]), c='lime', linewidth=1.5)
            # Plot a red line indicating the front of the vehicle
            plt.plot([x, (rot_x[0]+rot_x[1])/2], [y, (rot_y[0]+rot_y[1])/2], c='red', linewidth=1.5)
            
        # Lock the camera view to ~50 meters around the ego-vehicle
        plt.xlim(-51.2, 51.2)
        plt.ylim(-51.2, 51.2)
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save and forcefully close to prevent memory leaks
        name = f"frame_{i:04d}.png"
        plt.savefig(os.path.join(args.out_dir, name), dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        if i % 10 == 0:
            print(f"Rendered {i} / {len(dataset)} frames...")

    print(f"Success! Check the {args.out_dir} folder.")

if __name__ == '__main__':
    main()
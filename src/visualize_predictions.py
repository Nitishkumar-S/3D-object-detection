import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from bev_grid_generator import BEVGridGenerator
from bev_dataset import BEVDataset
from model import SimpleBEVDetector


def get_corners(cx, cy, w, l, yaw):
    """
    cx, cy → center (in grid coords)
    w, l   → width & length (grid units)
    yaw    → rotation (radians)
    """

    # rectangle corners (centered at origin)
    corners = np.array([
        [-l/2, -w/2],
        [-l/2,  w/2],
        [ l/2,  w/2],
        [ l/2, -w/2]
    ])

    # rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # rotate + translate
    rotated = corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    return rotated

def decode_predictions(pred, threshold=0.5):
    """
    pred: (6, H, W)
    """
    
    offset_x = pred[1]
    offset_y = pred[2]
    width = torch.exp(pred[3])
    length = torch.exp(pred[4])
    yaw = pred[5]

    obj = torch.sigmoid(pred[0])  # objectness


    print("Max objectness:", obj.max().item())
    print("Mean objectness:", obj.mean().item())

    # Apply max pooling to find local maxima
    obj_unsqueezed = obj.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    pooled = F.max_pool2d(obj_unsqueezed, kernel_size=5, stride=1, padding=2)

    # Keep only local peaks
    keep = (obj_unsqueezed == pooled) & (obj_unsqueezed > threshold)

    keep = keep.squeeze()
    ys, xs = torch.where(keep)
    # top-k filtering
    max_boxes = 50

    if len(xs) > max_boxes:
        scores = obj[ys, xs]
        topk = torch.topk(scores, max_boxes)

        ys = ys[topk.indices]
        xs = xs[topk.indices]

    print("Candidates after threshold:", len(xs))

    boxes = []

    for y, x in zip(ys, xs):
        score = obj[y, x].item()

        dx = offset_x[y, x].item()
        dy = offset_y[y, x].item()

        w = width[y, x].item()
        l = length[y, x].item()
        angle = yaw[y, x].item()
        
        print("Sample w,l:", w, l)
        # if w < 3 or w > 20:
        #     continue
        # if l < 5 or l > 30:
        #     continue

        boxes.append((x.item(), y.item(), dx, dy, w, l, angle, score))

    return boxes


def plot_boxes(bev, boxes):
    density = bev[:, :, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(density, cmap='gray', origin='lower')

    for box in boxes:
        x, y, dx, dy, w, l, yaw, score = box

        # center with offset
        cx = x + dx
        cy = y + dy

        # skip very weird predictions (optional safety)
        if w <= 0 or l <= 0:
            continue

        corners = get_corners(cx, cy, w, l, yaw)

        # close the box
        corners = np.vstack([corners, corners[0]])

        plt.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=1)

    plt.title(f"Predictions: {len(boxes)} boxes")
    plt.savefig("visualization/boxes.png")
    print("Saved visualization/boxes.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    sample = nusc.sample[0]

    bev, target = bev_dataset.get_sample(sample)

    bev_tensor = torch.from_numpy(bev).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Load trained model
    model = SimpleBEVDetector().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(bev_tensor)[0]  # (6, H, W)

    boxes = decode_predictions(pred, threshold=0.2)

    plot_boxes(bev, boxes)


if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F


def detection_loss(pred, target):
    """
    pred:   (B, 6, H, W)
    target: (B, H, W, 6)
    """

    # Rearrange target
    target = target.permute(0, 3, 1, 2)

    obj_pred = pred[:, 0]
    obj_target = target[:, 0]

    # Objectness loss
    obj_loss = F.binary_cross_entropy_with_logits(
        obj_pred,
        obj_target,
        pos_weight=torch.tensor(5.0, device=pred.device)
    )

    # Box loss (only where object exists)
    mask = obj_target > 0  # (B, H, W)

    box_pred = pred[:, 1:6]     # (B, 5, H, W)
    box_target = target[:, 1:6] # (B, 5, H, W)

    # Expand mask to match channels
    mask = mask.unsqueeze(1)  # (B, 1, H, W)

    # Apply mask
    box_pred = box_pred[mask.expand_as(box_pred)]
    box_target = box_target[mask.expand_as(box_target)]

    if box_pred.numel() > 0:
        box_loss = F.l1_loss(box_pred, box_target, reduction='mean')
    else:
        box_loss = torch.tensor(0.0, device=pred.device)

    return obj_loss + 5.0 * box_loss, obj_loss, box_loss
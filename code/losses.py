# --- IMPORTS ---
import torch
import torch.nn as nn
import torchvision.ops as ops

# --- CUSTOM LOSSES ---
class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        b1_x1, b1_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
        b1_x2, b1_y2 = b1_x1 + pred_boxes[:, 2], b1_y1 + pred_boxes[:, 3]
        
        b2_x1, b2_y1 = target_boxes[:, 0], target_boxes[:, 1]
        b2_x2, b2_y2 = b2_x1 + target_boxes[:, 2], b2_y1 + target_boxes[:, 3]
        
        preds = torch.stack([b1_x1, b1_y1, b1_x2, b1_y2], dim=1)
        targets = torch.stack([b2_x1, b2_y1, b2_x2, b2_y2], dim=1)
        
        loss = ops.complete_box_iou_loss(preds, targets, reduction='mean')
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
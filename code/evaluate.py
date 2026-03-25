# --- IMPORTS ---
import torch
import numpy as np
import torchvision.ops as ops
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
from settings import DEVICE, BEST_MODEL_PATH, CSV_PATH, TRAIN_IMG_PATH, IMG_SIZE
from dataset import get_dataloaders
from models import HighResPneumoniaDetector

# --- HELPERS ---
def invert_box_flip(box):
    x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    return torch.stack([1.0 - (x + w), y, w, h], dim=1)

def xywh2xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x, y, x + w, y + h], dim=1)

if __name__ == "__main__":
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    model = HighResPneumoniaDetector().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\n[INFO] Starting TTA Inference...")
    all_probs, all_targets, all_reg_preds, all_true_boxes = [], [], [], []

    with torch.no_grad():
        for imgs, targets, boxes in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            
            # TTA Standard Pass
            cls_1, reg_1 = model(imgs)
            prob_1 = torch.sigmoid(cls_1)
            
            # TTA Flipped Pass
            imgs_flip = torch.flip(imgs, [3]) 
            cls_2, reg_2 = model(imgs_flip)
            prob_2 = torch.sigmoid(cls_2)
            
            # Ensemble
            final_prob = (prob_1 + prob_2) / 2.0
            final_reg = (reg_1 + invert_box_flip(reg_2)) / 2.0
            
            all_probs.append(final_prob.cpu().view(-1))
            all_targets.append(targets.cpu().view(-1))
            all_reg_preds.append(final_reg.cpu())
            all_true_boxes.append(boxes.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_reg_preds = torch.cat(all_reg_preds)
    all_true_boxes = torch.cat(all_true_boxes)

    # Threshold Optimization
    best_f1, best_thresh = 0, 0.5
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (all_probs > th).astype(int)
        _, r, f1, _ = precision_recall_fscore_support(all_targets, preds, average=None, zero_division=0)
        if len(f1) > 1 and f1[1] > best_f1:
            best_f1, best_thresh = f1[1], th

    print(f"\n[SUCCESS] Optimal Threshold: {best_thresh}")
    final_preds = (all_probs > best_thresh).astype(int)
    
    print("\n[INFO] Classification Report:")
    print(classification_report(all_targets, final_preds, target_names=['Healthy', 'Pneumonia']))

    mask_tp = (all_targets == 1) & (final_preds == 1)
    if mask_tp.sum() > 0:
        ious_diag = torch.diag(ops.box_iou(xywh2xyxy(all_reg_preds[mask_tp]), xywh2xyxy(all_true_boxes[mask_tp]))).numpy()
        mean_px_err = np.mean(torch.abs(all_reg_preds[mask_tp] - all_true_boxes[mask_tp]).numpy()) * IMG_SIZE
        print(f"\n[INFO] Bounding Box Precision (over {mask_tp.sum()} TP):")
        print(f"   -> Mean IoU: {np.mean(ious_diag):.4f}")
        print(f"   -> Mean Pixel Error: {mean_px_err:.1f} px")
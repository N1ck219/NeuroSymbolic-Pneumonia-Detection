# --- IMPORTS ---
import torch
import numpy as np
from tqdm import tqdm
from settings import DEVICE, IMG_SIZE, BEST_MODEL_PATH

def extract_neurosymbolic_features(model, dataloader):
    """
    Runs inference on the validation set, extracts bounding boxes,
    and computes ROI physics (intensity and texture).
    Returns the metadata required for ASP facts generation.
    """
    print("\n[INFO] Starting GPU inference & feature extraction...")
    model.eval()
    
    results_meta = [] 
    patient_targets = {} 
    global_idx = 0 

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting ROI Features"):
            imgs = batch_data[0].to(DEVICE)
            targets = batch_data[1]
            
            # Extract patient IDs or generate sequential ones
            if len(batch_data) == 4:
                p_ids = batch_data[3]
            else:
                p_ids = [f"patient_{global_idx + j}" for j in range(len(imgs))]
                
            cls_out, reg_out = model(imgs)
            probs = torch.sigmoid(cls_out).view(-1)
            
            for j in range(len(probs)):
                pid = p_ids[j]
                patient_targets[pid] = int(targets[j].item())
                score_val = probs[j].item()
                
                # Filter out extremely low-confidence predictions to save ASP processing time
                if score_val > 0.05:
                    norm_box = reg_out[j].cpu().numpy()
                    x, y, w, h = (norm_box * IMG_SIZE).astype(int)
                    
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(IMG_SIZE, x + w), min(IMG_SIZE, y + h)
                    
                    mean_intensity, std_texture = 0.0, 0.0
                    
                    # Compute physical metrics if the bounding box is valid
                    if x2 > x1 and y2 > y1:
                        roi = imgs[j, 0, y1:y2, x1:x2] 
                        mean_intensity = roi.mean().item()
                        std_texture = roi.std().item()

                    results_meta.append({
                        'patientId': pid,
                        'score': score_val,
                        'pred_box': [x, y, w, h],
                        'intensity': mean_intensity,
                        'texture': std_texture
                    })
                
                global_idx += 1

    print(f"[SUCCESS] Extraction complete. Found {len(results_meta)} candidate boxes.")
    return results_meta, patient_targets
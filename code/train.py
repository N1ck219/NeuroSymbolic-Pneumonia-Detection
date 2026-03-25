# --- IMPORTS ---
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from settings import DEVICE, BATCH_SIZE, EPOCHS, PATIENCE, CSV_PATH, TRAIN_IMG_PATH, BEST_MODEL_PATH
from dataset import get_dataloaders
from losses import CIoULoss
from models import HighResPneumoniaDetector

# --- TRAINING ENGINE ---
def run_training(model, train_loader, val_loader, optimizer, epochs, phase_name="Phase"):
    accumulation_steps = 8 
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE), reduction='mean')
    criterion_ciou = CIoULoss()
    criterion_size = nn.L1Loss() 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())
    
    print(f"\n=== START {phase_name} (Virtual Batch: {BATCH_SIZE * accumulation_steps}) ===")
    
    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        valid_batches = 0 
        optimizer.zero_grad() 
        
        loop = tqdm(train_loader, desc=f"Train Ep {epoch+1}", leave=False)
        for i, (imgs, targets, boxes) in enumerate(loop):
            imgs, targets, boxes = imgs.to(DEVICE), targets.to(DEVICE).unsqueeze(1), boxes.to(DEVICE)
            
            cls_pred, reg_pred = model(imgs)
            reg_pred = torch.clamp(reg_pred, min=1e-6, max=1.0 - 1e-6)
            
            loss_cls = criterion_cls(cls_pred, targets)
            mask = targets.view(-1) == 1
            if mask.sum() > 0:
                loss_reg = criterion_ciou(reg_pred[mask], boxes[mask]) + 15.0 * criterion_size(reg_pred[mask][:, 2:], boxes[mask][:, 2:])
            else:
                loss_reg = torch.tensor(0.0, device=DEVICE)
            
            loss = (1.0 * loss_cls + 2.0 * loss_reg) / accumulation_steps
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue 
            
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad() 
            
            run_loss += loss.item() * accumulation_steps
            valid_batches += 1
            loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train = run_loss / (valid_batches + 1e-8)
        
        # Validation
        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for imgs, targets, boxes in tqdm(val_loader, desc=f"Val Ep {epoch+1}", leave=False):
                imgs, targets, boxes = imgs.to(DEVICE), targets.to(DEVICE).unsqueeze(1), boxes.to(DEVICE)
                cls_pred, reg_pred = model(imgs)
                reg_pred = torch.clamp(reg_pred, min=1e-6, max=1.0 - 1e-6)
                loss_cls = criterion_cls(cls_pred, targets)
                
                mask = targets.view(-1) == 1
                if mask.sum() > 0:
                    loss_reg = criterion_ciou(reg_pred[mask], boxes[mask]) + 15.0 * criterion_size(reg_pred[mask][:, 2:], boxes[mask][:, 2:])
                else:
                    loss_reg = torch.tensor(0.0, device=DEVICE)
                
                batch_loss = (1.0 * loss_cls + 2.0 * loss_reg).item()
                if not math.isnan(batch_loss): run_val += batch_loss

        avg_val = run_val / len(val_loader)
        scheduler.step(avg_val)
        
        print(f"Ep {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}", end=" ")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("--> [NEW BEST MODEL SAVED]")
        else:
            patience_counter += 1
            print(f"| Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("[INFO] Early Stopping Triggered.")
            break
            
    model.load_state_dict(best_weights)
    return model

# --- EXECUTION SCRIPT ---
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    model = HighResPneumoniaDetector().to(DEVICE)

    print("\n[INFO] PHASE 1: Warmup Heads & CBAM (Backbone Frozen)")
    for param in model.backbone.parameters(): param.requires_grad = False
    
    opt_phase1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    model = run_training(model, train_loader, val_loader, opt_phase1, epochs=8, phase_name="WARMUP-DECOUPLED")

    print("\n[INFO] PHASE 2: Unfreeze All - Full Fine Tuning")
    for param in model.parameters(): param.requires_grad = True
    
    opt_phase2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 
    model = run_training(model, train_loader, val_loader, opt_phase2, epochs=EPOCHS, phase_name="FULL-TRAINING")
    print("\n[SUCCESS] Training Completed.")
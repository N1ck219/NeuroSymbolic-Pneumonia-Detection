# --- IMPORTS ---
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from tqdm import tqdm
import torchvision.models as models
import timm

# Import
from settings import DEVICE, CSV_PATH, TRAIN_IMG_PATH, BEST_MODEL_PATH, ABLATION_MODEL_PATH, RESNET_WEIGHTS, EFFNET_WEIGHTS, SERESNET_WEIGHTS
from dataset import get_dataloaders
from models import HighResPneumoniaDetector, NoCeNNPneumoniaDetector

# --- 1. SETUP ---
NOISE_LEVELS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.90]
SAVE_DIR = os.path.join("..", "results", "confusion_matrix_noise")
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n========================================================")
print("   STARTING ROBUSTNESS TEST TO NOISE (ALL MODELS)  ")
print(f"   Saving in: {SAVE_DIR}")
print("========================================================\n")

def add_noise(img_tensor, noise_factor):
    if noise_factor <= 0.0: return img_tensor
    noise = torch.randn_like(img_tensor) * noise_factor
    return torch.clamp(img_tensor + noise, 0.0, 1.0)

if __name__ == "__main__":
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    # --- 2. MODELS ---
    models_to_test = []

    resnet = models.resnet50(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    models_to_test.append({"name": "ResNet50", "model": resnet, "path": RESNET_WEIGHTS, "thresh": 0.50})
    
    effnet = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "EfficientNet-B0", "model": effnet, "path": EFFNET_WEIGHTS, "thresh": 0.50})

    seresnet = timm.create_model('seresnet50', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "SE-ResNet50", "model": seresnet, "path": SERESNET_WEIGHTS, "thresh": 0.50})

    no_cenn = NoCeNNPneumoniaDetector()
    models_to_test.append({"name": "Ablation Model (No CeNN)", "model": no_cenn, "path": ABLATION_MODEL_PATH, "thresh": 0.40})

    hybrid = HighResPneumoniaDetector()
    models_to_test.append({"name": "Proposed Hybrid CeNN", "model": hybrid, "path": BEST_MODEL_PATH, "thresh": 0.40})

    all_results = {}

    # --- 3. LOOP MODELS ---
    for item in models_to_test:
        model_name = item["name"]
        model = item["model"].to(DEVICE)
        weights_path = item["path"]
        chosen_thresh = item["thresh"]
        
        print(f"\n---> Starting test for: {model_name}")
        
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        except FileNotFoundError:
            print(f"[ERROR] Weights not found in '{weights_path}'. Skipping this model.")
            continue
            
        model.eval()
        results = {'noise': [], 'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'specificity': [], 'auc': [], 'mcc': []}
        
        fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
        fig_cm.suptitle(model_name, fontsize=22, fontweight='bold', y=1.02)
        axes_cm = axes_cm.flatten()
        
        with torch.no_grad():
            for idx, noise_lvl in enumerate(NOISE_LEVELS):
                all_probs, all_targets = [], []
                loop = tqdm(val_loader, desc=f"Noise {int(noise_lvl*100):02d}%", leave=False, dynamic_ncols=True)
                
                for batch_data in loop:
                    clean_imgs_batch = batch_data[0]
                    targets = batch_data[1].to(DEVICE)
                    
                    noisy_input = add_noise(clean_imgs_batch, noise_lvl).to(DEVICE)
                    
                    out_1 = model(noisy_input)
                    cls_1 = out_1[0] if isinstance(out_1, tuple) else out_1
                    prob_1 = torch.sigmoid(cls_1)
                    
                    flipped_input = torch.flip(noisy_input, [3])
                    out_2 = model(flipped_input)
                    cls_2 = out_2[0] if isinstance(out_2, tuple) else out_2
                    prob_2 = torch.sigmoid(cls_2)
                    
                    final_prob = (prob_1 + prob_2) / 2.0
                    
                    all_probs.append(final_prob.cpu().view(-1))
                    all_targets.append(targets.cpu().view(-1))
                    
                all_probs = torch.cat(all_probs).numpy()
                all_targets = torch.cat(all_targets).numpy()
                preds = (all_probs > chosen_thresh).astype(int)
                
                acc = accuracy_score(all_targets, preds)
                p, r, f1, _ = precision_recall_fscore_support(all_targets, preds, average=None, zero_division=0)
                auc = roc_auc_score(all_targets, all_probs)
                mcc = matthews_corrcoef(all_targets, preds)
                
                cm = confusion_matrix(all_targets, preds)
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                results['noise'].append(noise_lvl * 100)
                results['accuracy'].append(acc)
                results['recall'].append(r[1])
                results['precision'].append(p[1])
                results['f1'].append(f1[1])
                results['specificity'].append(specificity)
                results['auc'].append(auc)
                results['mcc'].append(mcc)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False, annot_kws={"size": 16},
                            xticklabels=['Healthy', 'Pneumonia'], yticklabels=['True: Healthy', 'True: Pneumonia'])
                
                axes_cm[idx].set_title(f"Noise: {int(noise_lvl*100)}%\nAcc: {acc:.1%} | F1: {f1[1]:.2f}", fontsize=14, fontweight='bold')
                
                if idx % 3 != 0:
                    axes_cm[idx].set_ylabel('')
                    axes_cm[idx].set_yticks([])

        plt.tight_layout()
        save_path_cm = os.path.join(SAVE_DIR, f"{model_name.replace(' ', '_')}_CM.png")
        plt.savefig(save_path_cm, bbox_inches='tight', dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(results['noise'], results['accuracy'], marker='o', color='blue', linewidth=2, label='Accuracy')
        plt.plot(results['noise'], results['recall'], marker='s', color='red', linestyle='--', linewidth=2, label='Recall')
        plt.plot(results['noise'], results['f1'], marker='^', color='green', linestyle=':', linewidth=2, label='F1-Score')
        plt.plot(results['noise'], results['specificity'], marker='d', color='purple', linestyle='-.', linewidth=2, label='Specificity')
        plt.title(f"Degradation Analysis - {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Gaussian Noise (%)")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path_line = os.path.join(SAVE_DIR, f"{model_name.replace(' ', '_')}_Degradation.png")
        plt.savefig(save_path_line, bbox_inches='tight', dpi=150)
        plt.close()
        
        all_results[model_name] = results
        
        del model
        torch.cuda.empty_cache()

    # --- 4. EXPORT NUMERICAL RESULTS TO CSV ---
    flat_results = []
    for model_name, res in all_results.items():
        for i in range(len(res['noise'])):
            flat_results.append({
                'Model': model_name,
                'Noise (%)': res['noise'][i],
                'Accuracy': res['accuracy'][i],
                'Precision': res['precision'][i],
                'Recall': res['recall'][i],
                'F1-Score': res['f1'][i],
                'Specificity': res['specificity'][i],
                'AUC-ROC': res['auc'][i],
                'MCC': res['mcc'][i]
            })
            
    df_noise = pd.DataFrame(flat_results)
    df_noise = df_noise.round(4) # Arrotonda a 4 decimali
    
    csv_filename = os.path.join(SAVE_DIR, "noise_robustness_metrics.csv")
    df_noise.to_csv(csv_filename, index=False, sep=';', decimal=',')
    
    print("\n[SUCCESS] All images saved.")
    print(f"[SUCCESS] Numerical tabular data exported to: {csv_filename}")
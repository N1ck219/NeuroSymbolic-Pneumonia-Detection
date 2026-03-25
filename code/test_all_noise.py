# --- IMPORTS ---
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm
import torchvision.models as models
import timm
from settings import DEVICE, CSV_PATH, TRAIN_IMG_PATH, BEST_MODEL_PATH, ABLATION_MODEL_PATH, RESNET_WEIGHTS, EFFNET_WEIGHTS, SERESNET_WEIGHTS

# Import dai moduli locali
from settings import DEVICE, CSV_PATH, TRAIN_IMG_PATH, BEST_MODEL_PATH, ABLATION_MODEL_PATH
from dataset import get_dataloaders
from models import HighResPneumoniaDetector, NoCeNNPneumoniaDetector

# --- 1. CONFIGURAZIONE E SETUP CARTELLE ---
NOISE_LEVELS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.90]
SAVE_DIR = os.path.join("..", "results", "confusion_matrix_noise")
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n========================================================")
print("   AVVIO TEST DI ROBUSTEZZA AL RUMORE (TUTTI I MODELLI)  ")
print(f"   Salvataggio in: {SAVE_DIR}")
print("========================================================\n")

def add_noise(img_tensor, noise_factor):
    if noise_factor <= 0.0: return img_tensor
    noise = torch.randn_like(img_tensor) * noise_factor
    return torch.clamp(img_tensor + noise, 0.0, 1.0)

if __name__ == "__main__":
    # Caricamento Dataloader
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    # --- 2. INIZIALIZZAZIONE MODELLI ---
    # Definiamo i modelli, il loro percorso pesi e la soglia clinica ottimale
    models_to_test = []

    # A. ResNet50
    resnet = models.resnet50(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    models_to_test.append({"name": "ResNet50", "model": resnet, "path": RESNET_WEIGHTS, "thresh": 0.50})
    
    # B. EfficientNet-B0
    effnet = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "EfficientNet-B0", "model": effnet, "path": EFFNET_WEIGHTS, "thresh": 0.50})

    # C. SE-ResNet50
    seresnet = timm.create_model('seresnet50', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "SE-ResNet50", "model": seresnet, "path": SERESNET_WEIGHTS, "thresh": 0.50})

    # D. Ablation Study (Senza CeNN)
    no_cenn = NoCeNNPneumoniaDetector()
    models_to_test.append({"name": "Ablation Model (No CeNN)", "model": no_cenn, "path": ABLATION_MODEL_PATH, "thresh": 0.40})

    # E. Proposed Model (Con CeNN)
    hybrid = HighResPneumoniaDetector()
    models_to_test.append({"name": "Proposed Hybrid CeNN", "model": hybrid, "path": BEST_MODEL_PATH, "thresh": 0.40})

    # Struttura per salvare i risultati aggregati (se ti servono per il grafico a linee finale)
    all_results = {}

    # --- 3. LOOP SUI MODELLI ---
    for item in models_to_test:
        model_name = item["name"]
        model = item["model"].to(DEVICE)
        weights_path = item["path"]
        chosen_thresh = item["thresh"]
        
        print(f"\n---> Avvio test per: {model_name}")
        
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        except FileNotFoundError:
            print(f"[ERROR] Pesi non trovati in '{weights_path}'. Salto questo modello.")
            continue
            
        model.eval()
        
        results = {'noise': [], 'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        
        # Setup Grafico Matrici
        fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
        fig_cm.suptitle(model_name, fontsize=22, fontweight='bold', y=1.02)
        axes_cm = axes_cm.flatten()
        
        with torch.no_grad():
            for idx, noise_lvl in enumerate(NOISE_LEVELS):
                all_probs, all_targets = [], []
                loop = tqdm(val_loader, desc=f"Noise {int(noise_lvl*100):02d}%", leave=False, dynamic_ncols=True)
                
                # Attenzione: il dataloader del codice base restituisce (imgs, targets, boxes)
                for batch_data in loop:
                    clean_imgs_batch = batch_data[0]
                    targets = batch_data[1].to(DEVICE)
                    
                    noisy_input = add_noise(clean_imgs_batch, noise_lvl).to(DEVICE)
                    
                    # --- TTA Pass 1 (Normale) ---
                    out_1 = model(noisy_input)
                    cls_1 = out_1[0] if isinstance(out_1, tuple) else out_1
                    prob_1 = torch.sigmoid(cls_1)
                    
                    # --- TTA Pass 2 (Flippato) ---
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
                
                p_mal, r_mal, f1_mal = p[1], r[1], f1[1]
                
                results['noise'].append(noise_lvl * 100)
                results['accuracy'].append(acc)
                results['recall'].append(r_mal)
                results['precision'].append(p_mal)
                results['f1'].append(f1_mal)
                
                # Disegno Confusion Matrix (sempre in colore 'Blues')
                cm = confusion_matrix(all_targets, preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False, annot_kws={"size": 16},
                            xticklabels=['Healthy', 'Pneumonia'], yticklabels=['True: Healthy', 'True: Pneumonia'])
                
                # Titoletto pulito come richiesto
                axes_cm[idx].set_title(f"Noise: {int(noise_lvl*100)}%\nAcc: {acc:.1%}", fontsize=16, fontweight='bold')
                
                if idx % 3 != 0:
                    axes_cm[idx].set_ylabel('')
                    axes_cm[idx].set_yticks([])

        # Salvataggio Matrici di Confusione
        plt.tight_layout()
        save_path_cm = os.path.join(SAVE_DIR, f"{model_name.replace(' ', '_')}_CM.png")
        plt.savefig(save_path_cm, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[OK] Salvato {save_path_cm}")

        # Salvataggio Grafico Lineare (Accuracy e Recall) per il singolo modello
        plt.figure(figsize=(8, 5))
        plt.plot(results['noise'], results['accuracy'], marker='o', color='blue', linewidth=2, label='Accuracy')
        plt.plot(results['noise'], results['recall'], marker='s', color='red', linestyle='--', linewidth=2, label='Recall')
        plt.title(f"Degradation - {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Gaussian Noise (%)")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path_line = os.path.join(SAVE_DIR, f"{model_name.replace(' ', '_')}_Degradation.png")
        plt.savefig(save_path_line, bbox_inches='tight', dpi=150)
        plt.close()
        
        all_results[model_name] = results
        
        # Svuota memoria GPU
        del model
        torch.cuda.empty_cache()

    print("\n[SUCCESS] Tutte le immagini sono state salvate in:", SAVE_DIR)
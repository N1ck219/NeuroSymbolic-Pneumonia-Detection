# --- IMPORTS ---
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from thop import profile
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef

from settings import DEVICE, BATCH_SIZE, CSV_PATH, TRAIN_IMG_PATH, IMG_SIZE, BEST_MODEL_PATH, ABLATION_MODEL_PATH, RESNET_WEIGHTS, EFFNET_WEIGHTS, SERESNET_WEIGHTS
from dataset import get_dataloaders
from models import HighResPneumoniaDetector, NoCeNNPneumoniaDetector 

print("\n========================================================")
print("   STARTING HARDWARE PROFILING AND MODEL COMPLEXITY ANALYSIS    ")
print("========================================================\n")

# --- 1. SUPPORT FUNCTIONS ---
def get_model_complexity_info(model, input_size=(1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 ** 2)
    dummy_input = torch.randn(*input_size).to(device)
    macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 
    return total_params, model_size_mb, flops

def measure_inference_speed(model, dataloader, device=DEVICE, num_batches=20):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        
    model.eval()
    total_time = 0.0
    batches_processed = 0
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_batches: break
            imgs = batch_data[0].to(device)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            _ = model(imgs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()
            
            if i > 0:
                total_time += (end_time - start_time)
                batches_processed += 1
                
    avg_time_ms = (total_time / batches_processed) * 1000 if batches_processed > 0 else 0
    throughput = (batches_processed * BATCH_SIZE) / total_time if total_time > 0 else 0
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    return avg_time_ms, throughput, peak_mem_mb

def evaluate_predictive_metrics(model, dataloader, thresh, device=DEVICE):
    model.eval()
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for batch_data in dataloader:
            imgs = batch_data[0].to(device)
            targets = batch_data[1].to(device)
            
            out = model(imgs)
            cls_out = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(cls_out)
            
            all_probs.append(probs.cpu().view(-1))
            all_targets.append(targets.cpu().view(-1))
            
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    preds = (all_probs > thresh).astype(int)
    
    acc = accuracy_score(all_targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, preds, average=None, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    mcc = matthews_corrcoef(all_targets, preds)
    
    cm = confusion_matrix(all_targets, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return acc, r[1], p[1], f1[1], spec, auc, mcc

if __name__ == "__main__":
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    # --- 2. MODEL INITIALIZATION ---
    models_to_test = []

    resnet = models.resnet50(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    models_to_test.append({"name": "ResNet50 (Baseline)", "model": resnet, "path": RESNET_WEIGHTS, "thresh": 0.50})

    effnet = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "EfficientNet-B0 (Baseline)", "model": effnet, "path": EFFNET_WEIGHTS, "thresh": 0.50})

    seresnet = timm.create_model('seresnet50', pretrained=False, in_chans=1, num_classes=1)
    models_to_test.append({"name": "SE-ResNet50 (Baseline)", "model": seresnet, "path": SERESNET_WEIGHTS, "thresh": 0.50})

    no_cenn_model = NoCeNNPneumoniaDetector()
    models_to_test.append({"name": "Ablation: No CeNN", "model": no_cenn_model, "path": ABLATION_MODEL_PATH, "thresh": 0.40})

    hybrid_model = HighResPneumoniaDetector()
    models_to_test.append({"name": "Proposed Hybrid CeNN", "model": hybrid_model, "path": BEST_MODEL_PATH, "thresh": 0.40})

    # --- 3. PROFILING EXECUTION ---
    hw_results = []

    for item in models_to_test:
        name = item["name"]
        net = item["model"].to(DEVICE)
        weights_path = item["path"]
        thresh = item["thresh"]
        
        print(f"\n[INFO] Evaluating & Profiling: {name}...")
        
        try:
            net.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            acc, rec, prec, f1, spec, auc, mcc = evaluate_predictive_metrics(net, val_loader, thresh)
        except FileNotFoundError:
            print(f"[WARN] Pesi non trovati in '{weights_path}'. Le metriche predittive saranno impostate a 0.")
            acc = rec = prec = f1 = spec = auc = mcc = 0.0

        tot_p, size_mb, flops = get_model_complexity_info(net, device=DEVICE)
        avg_time, throughput, peak_mem = measure_inference_speed(net, val_loader, device=DEVICE)
        
        hw_results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "AUC-ROC": auc,
            "MCC": mcc,
            "Parameters (M)": tot_p / 1e6,
            "Size (MB)": size_mb,
            "FLOPs (G)": flops / 1e9,
            "Inf. Time/Batch (ms)": avg_time,
            "Throughput (img/s)": throughput,
            "Peak VRAM (MB)": peak_mem
        })
        
        del net
        torch.cuda.empty_cache()

    # --- 4. EXPORT TO CSV ---
    df_hw = pd.DataFrame(hw_results)
    df_hw = df_hw.round(4)

    print("\n" + "="*110)
    print("                      FINAL COMPLEXITY AND RESOURCE REPORT                      ")
    print("="*110)
    print(df_hw.to_markdown(index=False))

    save_dir = os.path.join("..", "results", "hardware_profiling")
    os.makedirs(save_dir, exist_ok=True)
    
    csv_filename = os.path.join(save_dir, "hardware_comparison_metrics_final.csv")
    df_hw.to_csv(csv_filename, index=False, sep=';', decimal=',')
    
    print(f"\n[SUCCESS] Numerical tabular data exported to: {csv_filename}")
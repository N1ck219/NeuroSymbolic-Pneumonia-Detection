# --- IMPORTS ---
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from thop import profile
import pandas as pd

# Import dai moduli locali
from settings import DEVICE, BATCH_SIZE, CSV_PATH, TRAIN_IMG_PATH, IMG_SIZE
from dataset import get_dataloaders
from models import HighResPneumoniaDetector, NoCeNNPneumoniaDetector 

print("\n========================================================")
print("   AVVIO PROFILAZIONE HARDWARE E COMPLESSITÀ MODELLI    ")
print("========================================================\n")

# --- 1. FUNZIONI DI SUPPORTO ---
def get_model_complexity_info(model, input_size=(1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 ** 2)
    dummy_input = torch.randn(*input_size).to(device)
    macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 
    return total_params, model_size_mb, flops

def measure_inference_speed(model, dataloader, device=DEVICE, num_batches=20):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
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

if __name__ == "__main__":
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    # --- 2. INIZIALIZZAZIONE MODELLI ---
    models_dict = {}

    # A. ResNet50 
    resnet = models.resnet50(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    models_dict["ResNet50 (Baseline)"] = resnet

    # B. EfficientNet-B0
    effnet = models.efficientnet_b0(weights=None)
    effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 1)
    models_dict["EfficientNet-B0 (Baseline)"] = effnet

    # C. SE-ResNet50
    seresnet = timm.create_model('seresnet50', pretrained=False, in_chans=1, num_classes=1)
    models_dict["SE-ResNet50 (Baseline)"] = seresnet

    # D. ABLATION STUDY: Tuo Modello SENZA CeNN
    no_cenn_model = NoCeNNPneumoniaDetector()
    models_dict["Ablation: No CeNN"] = no_cenn_model

    # E. PROPOSED MODEL: Tuo Modello CON CeNN originale
    hybrid_model = HighResPneumoniaDetector()
    models_dict["Proposed Hybrid CeNN"] = hybrid_model

    # --- 3. ESECUZIONE PROFILAZIONE ---
    hw_results = []

    for name, net in models_dict.items():
        print(f"\n[INFO] Profilazione in corso per: {name}...")
        net = net.to(DEVICE)
        
        tot_p, size_mb, flops = get_model_complexity_info(net, device=DEVICE)
        avg_time, throughput, peak_mem = measure_inference_speed(net, val_loader, device=DEVICE)
        
        hw_results.append({
            "Model": name,
            "Parameters (M)": tot_p / 1e6,
            "Size on Disk (MB)": size_mb,
            "FLOPs (G)": flops / 1e9,
            "Inference Time / Batch (ms)": avg_time,
            "Throughput (img/sec)": throughput,
            "Peak VRAM (MB)": peak_mem
        })
        
        del net
        torch.cuda.empty_cache()

    # --- 4. STAMPA E SALVATAGGIO ---
    df_hw = pd.DataFrame(hw_results)
    df_hw = df_hw.round(2)

    print("\n" + "="*80)
    print("                      REPORT COMPLESSITÀ E RISORSE DEFINITIVO                      ")
    print("="*80)
    print(df_hw.to_markdown(index=False))

    # Setup Directory di salvataggio
    save_dir = os.path.join("..", "results", "hardware_profiling")
    os.makedirs(save_dir, exist_ok=True)
    
    csv_filename = os.path.join(save_dir, "hardware_comparison_metrics_final.csv")
    df_hw.to_csv(csv_filename, index=False, sep=';', decimal=',')
    print(f"\n[SUCCESS] Dati esportati in '{csv_filename}'! Pronti per la tesi.")
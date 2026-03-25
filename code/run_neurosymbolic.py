# --- IMPORTS ---
import torch
from settings import DEVICE, BEST_MODEL_PATH, CSV_PATH, TRAIN_IMG_PATH
from dataset import get_dataloaders
from models import HighResPneumoniaDetector
from extractor import extract_neurosymbolic_features
from asp_solver import evaluate_asp_rules

if __name__ == "__main__":
    # 1. Load Data and Model
    _, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMG_PATH)
    
    model = HighResPneumoniaDetector().to(DEVICE)
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"[INFO] Successfully loaded weights from {BEST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[ERROR] Weights file '{BEST_MODEL_PATH}' not found. Please train the model first.")
        exit()

    # 2. Extract DL Metadata (Run only once)
    results_meta, patient_targets = extract_neurosymbolic_features(model, val_loader)

    # 3. Test ASP Rule Versions
    # You can easily test multiple versions back-to-back without re-running the DL inference!
    evaluate_asp_rules(results_meta, patient_targets, rules_path="rules/rules_v20.lp")
    
    # evaluate_asp_rules(results_meta, patient_targets, rules_path="rules/rules_v29.lp")
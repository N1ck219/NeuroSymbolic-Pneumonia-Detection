# --- IMPORTS ---
import os
import torch

# --- DEVICE CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
# CSV_PATH = r"../RSNA/stage_2_train_labels.csv"
# TRAIN_IMG_PATH = r"../RSNA/stage_2_train_images"
CSV_PATH = "C:/Users/Nicol/OneDrive - Università degli Studi di Udine/UniversitaOneDrive/Università 5 OneDrive/OneDriveSmallProject/CeNN-RSNA/RSNA/stage_2_train_labels.csv"
TRAIN_IMG_PATH = 'C:/Users/Nicol/OneDrive - Università degli Studi di Udine/UniversitaOneDrive/Università 5 OneDrive/OneDriveSmallProject/CeNN-RSNA/RSNA/stage_2_train_images'

# --- MODELS DIRECTORY SETUP ---
MODELS_DIR = os.path.join("..", "trained_models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_final.pth")
ABLATION_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_no_cenn.pth")

RESNET_WEIGHTS = os.path.join(MODELS_DIR, "best_model_resnet.pth")
SERESNET_WEIGHTS = os.path.join(MODELS_DIR, "best_model_seresnet.pth")
EFFNET_WEIGHTS = os.path.join(MODELS_DIR, "best_model_efficientnet.pth")

# --- HYPERPARAMETERS ---
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 5
ACCUMULATION_STEPS = 8
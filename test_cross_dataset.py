import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
FERPLUS_WEIGHTS = "/content/drive/MyDrive/FERPlus_Results/best_ferplus_aggressive.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
RAFDB_VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
RAFDB_VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# ==========================================
# STRICT ALPHABETICAL LABEL TRANSLATION
# ==========================================
# FERPlus (Based on your image_a25a83.png): 
# 0:angry, 1:contempt, 2:disgust, 3:fear, 4:happy, 5:neutral, 6:sad, 7:suprise
#
# RAF-DB (Based on your image_a25a0d.png, subtracted by 1): 
# 0:Surprise, 1:Fear, 2:Disgust, 3:Happy, 4:Sad, 5:Angry, 6:Neutral

FER_TO_RAF_MAP = {
    0: 5,  # FER angry   -> RAF Angry
    1: -1, # FER contempt-> RAF (N/A)
    2: 2,  # FER disgust -> RAF Disgust
    3: 1,  # FER fear    -> RAF Fear
    4: 3,  # FER happy   -> RAF Happy
    5: 6,  # FER neutral -> RAF Neutral
    6: 4,  # FER sad     -> RAF Sad
    7: 0   # FER suprise -> RAF Surprise
}

def translate_predictions(predictions, device):
    mapped_preds = torch.zeros_like(predictions)
    for fer_idx, raf_idx in FER_TO_RAF_MAP.items():
        mapped_preds[predictions == fer_idx] = raf_idx
    return mapped_preds.to(device)

def run_cross_dataset_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting Zero-Shot Cross-Dataset Inference (FERPlus -> RAF-DB)\n{'='*65}")

    model = FRITNet(num_classes=8, transformer_depth=2).to(device)
    
    print(f"--> Loading base FERPlus weights from: {FERPLUS_WEIGHTS}")
    model.load_state_dict(torch.load(FERPLUS_WEIGHTS))
    model.eval()

    print(f"--> Loading RAF-DB Test dataset from: {RAFDB_VAL_CSV}")
    val_dataset = RAFDBDataset(csv_file=RAFDB_VAL_CSV, root_dir=RAFDB_VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    val_correct, val_total = 0, 0
    
    print("--> Running inference...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Inference")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Convert RAF-DB 1-7 labels to 0-6 index
            targets = labels - 1 
            
            logits, features, _, _ = model(images)
            
            # Get the FERPlus prediction (0-7)
            _, fer_predicted = torch.max(logits.data, 1)
            
            # Translate to RAF-DB format (0-6) using the corrected dictionary
            raf_predicted = translate_predictions(fer_predicted, device)
            
            # Only count predictions that actually exist in RAF-DB (ignore contempt)
            valid_mask = (raf_predicted != -1)
            
            val_total += valid_mask.sum().item()
            val_correct += (raf_predicted[valid_mask] == targets[valid_mask]).sum().item()

    accuracy = val_correct / val_total
    
    print("\n===================================")
    print(f"Zero-Shot Cross-Dataset Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("===================================")

if __name__ == "__main__":
    run_cross_dataset_inference()
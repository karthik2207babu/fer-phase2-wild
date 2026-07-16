import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import your dataset and model
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
FERPLUS_WEIGHTS = "/content/drive/MyDrive/FERPlus_Results/best_ferplus_aggressive.pth"

# ==========================================
# VERIFIED LOCAL RAF-DB PATHS
# ==========================================
BASE_PATH = "/content/data/Datasets/RAF-DB"
RAFDB_VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
RAFDB_VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# ==========================================
# LABEL TRANSLATION DICTIONARY
# ==========================================
# FERPlus Standard (0-7): 0:Neutral, 1:Happy, 2:Surprise, 3:Sad, 4:Anger, 5:Disgust, 6:Fear, 7:Contempt
# RAF-DB Standard (1-7, converted to 0-6): 0:Surprise, 1:Fear, 2:Disgust, 3:Happy, 4:Sad, 5:Anger, 6:Neutral

FER_TO_RAF_MAP = {
    0: 6,  # FER Neutral -> RAF Neutral
    1: 3,  # FER Happy   -> RAF Happy
    2: 0,  # FER Surprise-> RAF Surprise
    3: 4,  # FER Sad     -> RAF Sad
    4: 5,  # FER Anger   -> RAF Anger
    5: 2,  # FER Disgust -> RAF Disgust
    6: 1,  # FER Fear    -> RAF Fear
    7: -1  # FER Contempt-> RAF doesn't have Contempt (always counted as wrong)
}

def translate_predictions(predictions, device):
    mapped_preds = torch.zeros_like(predictions)
    for fer_idx, raf_idx in FER_TO_RAF_MAP.items():
        mapped_preds[predictions == fer_idx] = raf_idx
    return mapped_preds.to(device)

def run_cross_dataset_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting Zero-Shot Cross-Dataset Inference (FERPlus -> RAF-DB)\n{'='*65}")

    # 1. Initialize the 8-class model to match the loaded weights
    model = FRITNet(num_classes=8, transformer_depth=2).to(device)
    
    # 2. Load the optimized FERPlus weights
    print(f"--> Loading base FERPlus weights from: {FERPLUS_WEIGHTS}")
    model.load_state_dict(torch.load(FERPLUS_WEIGHTS))
    model.eval()

    # 3. Initialize your custom RAF-DB dataset using the test paths
    print(f"--> Loading RAF-DB Test dataset from: {RAFDB_VAL_CSV}")
    val_dataset = RAFDBDataset(csv_file=RAFDB_VAL_CSV, root_dir=RAFDB_VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    val_correct, val_total = 0, 0
    
    print("--> Running inference...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Inference")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Convert RAF-DB 1-7 labels to 0-6 index for mathematical comparison
            targets = labels - 1 
            
            logits, features, _, _ = model(images)
            
            # Get the FERPlus prediction (0-7)
            _, fer_predicted = torch.max(logits.data, 1)
            
            # Translate to RAF-DB format (0-6)
            raf_predicted = translate_predictions(fer_predicted, device)
            
            val_total += targets.size(0)
            val_correct += (raf_predicted == targets).sum().item()

    accuracy = val_correct / val_total
    
    print("\n===================================")
    print(f"Zero-Shot Cross-Dataset Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("===================================")

if __name__ == "__main__":
    run_cross_dataset_inference()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd

# Import your dataset and model
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
RESULTS_DIR = "/content/drive/MyDrive/FERPlus_Results"

# Updated Paths for RAF-DB
BASE_PATH = "/content/data/Datasets/RAF-DB"
RAFDB_VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
RAFDB_VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# ==========================================
# STRICT ALPHABETICAL LABEL TRANSLATION
# ==========================================
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

def run_batch_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Batch Zero-Shot Cross-Dataset Inference\n{'='*70}")

    # Ensure the directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} not found.")
        return

    # Grab all .pth files in the directory
    weight_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.pth')]
    
    if not weight_files:
        print(f"No .pth files found in {RESULTS_DIR}.")
        return
        
    print(f"--> Found {len(weight_files)} model weights to test.\n")

    # Initialize your custom RAF-DB validation dataset
    print(f"--> Loading RAF-DB Test dataset from: {RAFDB_VAL_CSV}")
    val_dataset = RAFDBDataset(csv_file=RAFDB_VAL_CSV, root_dir=RAFDB_VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Dictionary to store results
    results_board = {}

    for weight_name in weight_files:
        weight_path = os.path.join(RESULTS_DIR, weight_name)
        print(f"\nEvaluating: {weight_name} ...")
        
        # Initialize fresh model
        model = FRITNet(num_classes=8, transformer_depth=2).to(device)
        
        try:
            # Load weights (strict=False to prevent crashes on slight architecture mismatches)
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            model.eval()

            val_correct, val_total = 0, 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Testing {weight_name[:20]}", leave=False)
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    
                    targets = labels - 1 
                    logits, features, _, _ = model(images)
                    
                    _, fer_predicted = torch.max(logits.data, 1)
                    raf_predicted = translate_predictions(fer_predicted, device)
                    
                    valid_mask = (raf_predicted != -1)
                    
                    val_total += valid_mask.sum().item()
                    val_correct += (raf_predicted[valid_mask] == targets[valid_mask]).sum().item()

            accuracy = val_correct / val_total if val_total > 0 else 0
            results_board[weight_name] = accuracy
            print(f"--> Accuracy: {accuracy*100:.2f}%")

        except Exception as e:
            print(f"--> Failed to evaluate {weight_name}. Error: {str(e)}")
            results_board[weight_name] = -1.0 

    # --- PRINT FINAL LEADERBOARD ---
    print(f"\n{'='*70}\nFINAL CROSS-DATASET LEADERBOARD (FERPlus -> RAF-DB)\n{'='*70}")
    
    # Sort results by accuracy (highest first)
    sorted_results = sorted(results_board.items(), key=lambda item: item[1], reverse=True)
    
    for i, (name, acc) in enumerate(sorted_results, 1):
        if acc == -1.0:
            print(f"{i}. {name:<35} | ERROR/FAILED")
        else:
            print(f"{i}. {name:<35} | {acc*100:.2f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_batch_inference()
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
RESULTS_DIR = "/content/drive/MyDrive/RAFDB_Results"
BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def run_batch_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Batch Test-Time Augmentation (TTA) Evaluation\n{'='*70}")

    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} not found.")
        return

    # Grab all .pth files in the directory
    weight_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.pth')]
    
    if not weight_files:
        print(f"No .pth files found in {RESULTS_DIR}.")
        return
        
    print(f"--> Found {len(weight_files)} model weights to test.\n")

    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    
    # Batch size is halved because TTA expands tensors 5x in memory during evaluation
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers=2)

    results_board = {}

    for weight_name in weight_files:
        weight_path = os.path.join(RESULTS_DIR, weight_name)
        print(f"\nEvaluating with TTA: {weight_name} ...")
        
        model = FRITNet(num_classes=7, transformer_depth=2).to(device)
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"TTA Testing {weight_name[:20]}", leave=False)
                for images_raw, labels in pbar:
                    labels = labels.to(device)
                    targets = labels - 1 
                    
                    batch_logits = torch.zeros(labels.size(0), 7).to(device)
                    images = images_raw.to(device)
                    
                    # Pass 1: Original
                    logits_1, _, _, _ = model(images)
                    batch_logits += F.softmax(logits_1, dim=1)
                    
                    # Pass 2: Flipped
                    images_flipped = torch.flip(images, dims=[3])
                    logits_2, _, _, _ = model(images_flipped)
                    batch_logits += F.softmax(logits_2, dim=1)
                    
                    # Pass 3: Zoomed (Interpolate & Crop)
                    images_zoomed = F.interpolate(images, scale_factor=1.05, mode='bilinear', align_corners=False)
                    images_zoomed = transforms.CenterCrop(224)(images_zoomed)
                    logits_3, _, _, _ = model(images_zoomed)
                    batch_logits += F.softmax(logits_3, dim=1)
                    
                    # Pass 4 & 5: Brightness adjustments via tensor clamping
                    images_bright = torch.clamp(images * 1.2, 0, 1)
                    logits_4, _, _, _ = model(images_bright)
                    batch_logits += F.softmax(logits_4, dim=1)
                    
                    images_dark = torch.clamp(images * 0.8, 0, 1)
                    logits_5, _, _, _ = model(images_dark)
                    batch_logits += F.softmax(logits_5, dim=1)
                    
                    # Average the softmax probabilities across all 5 variations
                    final_probs = batch_logits / 5.0
                    _, predicted = torch.max(final_probs, 1)
                    
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            accuracy = val_correct / val_total if val_total > 0 else 0
            results_board[weight_name] = accuracy
            print(f"--> TTA Accuracy: {accuracy*100:.2f}%")

        except Exception as e:
            print(f"--> Failed to evaluate {weight_name}. Error: {str(e)}")
            results_board[weight_name] = -1.0 

    # --- PRINT FINAL LEADERBOARD ---
    print(f"\n{'='*70}\nFINAL TTA LEADERBOARD (RAF-DB)\n{'='*70}")
    
    sorted_results = sorted(results_board.items(), key=lambda item: item[1], reverse=True)
    
    for i, (name, acc) in enumerate(sorted_results, 1):
        if acc == -1.0:
            print(f"{i}. {name:<40} | ERROR/FAILED")
        else:
            print(f"{i}. {name:<40} | {acc*100:.2f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_batch_tta()
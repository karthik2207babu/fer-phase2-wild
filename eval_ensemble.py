import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
WEIGHTS_SAMPLER = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_regularized_sampler.pth"
WEIGHTS_MIXUP = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_curriculum_mixup.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def run_logit_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Logit Ensemble Evaluation\n{'='*70}")

    # 1. Initialize and load Model 1 (Sampler - Sharp Boundaries)
    print(f"--> Loading Model 1: {os.path.basename(WEIGHTS_SAMPLER)}")
    model_1 = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model_1.load_state_dict(torch.load(WEIGHTS_SAMPLER, map_location=device))
    model_1.eval()

    # 2. Initialize and load Model 2 (MixUp - Structural Geometry)
    print(f"--> Loading Model 2: {os.path.basename(WEIGHTS_MIXUP)}")
    model_2 = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model_2.load_state_dict(torch.load(WEIGHTS_MIXUP, map_location=device))
    model_2.eval()

    # 3. Load Dataset
    print("--> Initializing Validation Dataloader...")
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    val_correct = 0
    val_total = 0

    print("\n--> Executing Simultaneous Inference...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Ensemble Evaluation")
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            targets = labels - 1 
            
            # Forward pass through Model 1
            logits_1, _, _, _ = model_1(images)
            
            # Forward pass through Model 2
            logits_2, _, _, _ = model_2(images)
            
            # Element-wise summation of unnormalized logits
            summed_logits = logits_1 + logits_2
            
            # Determine max value index from the summed tensor
            _, predicted = torch.max(summed_logits, 1)
            
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    accuracy = val_correct / val_total
    
    print("\n======================================================")
    print(f"Final Ensemble Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("======================================================")

if __name__ == "__main__":
    run_logit_ensemble()
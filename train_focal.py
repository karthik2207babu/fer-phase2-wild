import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration for Iterative Fine-Tuning ---
BATCH_SIZE = 64
EPOCHS = 15           # Short run to prevent overfitting
LEARNING_RATE = 1e-5  # Micro-LR to prevent breaking local minimum
WEIGHT_DECAY = 1e-2  

# Load your peak single weight file directly
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_curriculum_mixup.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_focal_tuned.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def train_focal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting Focal Loss Iterative Fine-Tuning | LR: {LEARNING_RATE}\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Model and Load Peak Single Weights
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    print(f"--> Loading peak single weights from: {BASE_WEIGHTS}")
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))

    # 2. Load Datasets and Sampler
    print("--> Loading RAF-DB datasets...")
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Initialize Updated CombinedFERLoss (featuring corrected Focal Loss)
    criterion = CombinedFERLoss(feat_dim=128, num_classes=7, alpha=0.1).to(device)

    # Uniform low learning rate across all layers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Focal Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            optimizer.zero_grad()

            # Clean forward pass (No MixUp - Focal Loss requires pure targets)
            logits, features, aux_global, aux_local = model(images)
            loss = criterion(logits, features, targets + 1, aux_global, aux_local) 
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += targets.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Validation Phase
        model.eval()
        val_correct, val_total = 0.0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1
                
                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME)
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best Focal weights: {v_acc:.4f} at {UNIQUE_WEIGHT_NAME}")

if __name__ == "__main__":
    train_focal()
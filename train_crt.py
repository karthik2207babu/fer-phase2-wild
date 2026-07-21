import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np

# Import custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

# Load your 88.07% averaged soup weights
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/averaged_models_init.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_crt.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def train_crt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Classifier-only Re-Training (cRT)\n{'='*70}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Model and Load 88.07% Weights
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
    
    # 2. FREEZE EVERYTHING EXCEPT CLASSIFICATION HEADS
    print("--> Freezing backbone and transformers...")
    for param in model.parameters():
        param.requires_grad = False
        
    print("--> Unfreezing classification heads...")
    # Adjust these string matches based on exactly what you named your final FC layers in model.py
    # Common names: 'fc', 'classifier', 'main_logits', 'aux_global', 'aux_local'
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if 'fc' in name.lower() or 'logit' in name.lower() or 'classifier' in name.lower():
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    print(f"--> Total trainable parameters for cRT: {unfrozen_params}")

    # 3. Load Datasets and Class-Balanced Sampler
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    # This sampler is the core of cRT: it forces the network to see Fear and Happiness equally
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Standard Cross Entropy is fine here because the sampler is perfectly balanced
    criterion = nn.CrossEntropyLoss().to(device)

    # Only pass the unfrozen parameters to the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"cRT Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            optimizer.zero_grad()
            logits, _, _, _ = model(images)
            loss = criterion(logits, targets) 
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved new best cRT weights: {v_acc:.4f}")

if __name__ == "__main__":
    train_crt()
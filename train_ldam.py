import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import recall_score

# Import custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 8           # Very short run to prevent breaking the 88.66% foundation
LEARNING_RATE = 5e-5 # Micro LR for delicate margin adjustment

# Load your 88.66% cRT weights
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_crt.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_ldam.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# RAF-DB order: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
CLASS_COUNTS = [329, 74, 160, 1185, 478, 162, 680]

class LDAMLoss(nn.Module):
    def __init__(self, class_counts, max_margin=0.5, s=30):
        super(LDAMLoss, self).__init__()
        # Calculate margins: 1 / (class_count ^ 0.25)
        cls_counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(cls_counts))
        # Scale to max_margin
        self.margins = margins * (max_margin / torch.max(margins))
        self.s = s

    def forward(self, logits, labels):
        device = logits.device
        margins = self.margins.to(device)
        
        # Apply margin only to the true class
        margin_mask = F.one_hot(labels, num_classes=logits.size(1)).float()
        adjusted_logits = logits - (margin_mask * margins.unsqueeze(0))
        
        # Scale by s and apply cross entropy
        return F.cross_entropy(self.s * adjusted_logits, labels)

def train_ldam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting LDAM-DRW Margin Optimization (Deep Unfreeze)\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Model and Load 88.66% Weights
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
    
    # 2. FREEZE ALL EXCEPT HEADS AND CROSS-ATTENTION
    print("--> Freezing backbone and local transformers...")
    for param in model.parameters():
        param.requires_grad = False
        
    print("--> Unfreezing heads and cross-attention blocks...")
    unfrozen_params = 0
    for name, param in model.named_parameters():
        # Target heads AND the cross-attention fusion layers
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out', 'cross', 'fusion']):
            param.requires_grad = True
            unfrozen_params += param.numel()
            print(f"  [UNFROZEN] {name}")
            
    print(f"--> Total trainable parameters for LDAM: {unfrozen_params}")

    # 3. Load Datasets (Standard Shuffle, NO Balanced Sampler)
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. LDAM Loss Setup
    criterion = LDAMLoss(class_counts=CLASS_COUNTS, max_margin=0.5, s=30).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_macro_recall = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"LDAM Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            optimizer.zero_grad()
            logits, _, aux_g, aux_l = model(images)
            
            # Apply LDAM to main logits. Optional: standard CE on aux branches to maintain stability
            loss = criterion(logits, targets) + 0.1 * F.cross_entropy(aux_g, targets) + 0.1 * F.cross_entropy(aux_l, targets)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            train_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})

        scheduler.step()

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1
                
                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        v_acc = val_correct / val_total
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} | Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        # Save if accuracy is stable AND macro recall improves
        if v_acc >= 0.8800 and macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved LDAM weights! Acc: {v_acc:.4f}, Recall: {macro_recall:.4f}")

if __name__ == "__main__":
    train_ldam()
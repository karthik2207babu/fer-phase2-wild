import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import recall_score

# Custom modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import NAWLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 15           
LEARNING_RATE = 1e-3  # FIXED: Increased to allow the fresh L2NormLinear head to learn
SAVE_DIR = "/content/drive/MyDrive/FER_NLA_Results"

# Using Model 1
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_regularized_sampler.pth"
UNIQUE_WEIGHT_CRT = "best_model1_standard_crt.pth"
UNIQUE_WEIGHT_NAW = "best_model1_naw_crt.pth"

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def setup_frozen_model(device):
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    if os.path.exists(BASE_WEIGHTS):
        print(f"--> Loading Model 1 weights from: {BASE_WEIGHTS}")
        model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device), strict=False)
    else:
        raise FileNotFoundError(f"Cannot find {BASE_WEIGHTS}.")
    
    print("--> Freezing backbone and transformers. Unfreezing classification heads...")
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out']):
            if 'attn' not in name.lower() and 'attention' not in name.lower() and 'norm' not in name.lower():
                param.requires_grad = True
    return model

def run_standard_crt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nPHASE 1: Standard cRT on Model 1 | LR: {LEARNING_RATE}\n{'='*75}")
    
    model = setup_frozen_model(device)
    
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    # Balanced Sampler for Standard cRT
    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    sample_weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Standard cRT Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets_idx = labels - 1 
            
            optimizer.zero_grad()
            logits, _, _, _ = model(images)
            loss = criterion(logits, targets_idx) 
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets_idx).sum().item()
            train_total += targets_idx.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets_idx = labels - 1
                
                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                
                val_total += targets_idx.size(0)
                val_correct += (predicted == targets_idx).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets_idx.cpu().numpy())

        v_acc = val_correct / val_total
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}: Val Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_CRT))
            print(f"--> Saved best Standard cRT weights: Acc = {v_acc:.4f}")

def run_naw_crt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nPHASE 2: NAW-cRT on Model 1 | LR: {LEARNING_RATE}\n{'='*75}")
    
    model = setup_frozen_model(device)
    
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    # Standard Dataloader (NAW handles the imbalance directly)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # NAW Loss with Lambda=1.0
    criterion = NAWLoss(num_classes=7, total_epochs=EPOCHS, lambda_param=1.0).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_macro_recall = 0.0

    for epoch in range(EPOCHS):
        criterion.set_epoch(epoch)
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"NAW-cRT Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets_idx = labels - 1 
            
            optimizer.zero_grad()
            logits, _, aux_g, aux_l = model(images)
            loss = criterion(logits, targets_idx, aux_g, aux_l)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets_idx).sum().item()
            train_total += targets_idx.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets_idx = labels - 1
                
                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                
                val_total += targets_idx.size(0)
                val_correct += (predicted == targets_idx).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets_idx.cpu().numpy())

        v_acc = val_correct / val_total
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}: Val Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAW))
            print(f"--> Saved best NAW-cRT weights: Recall = {macro_recall:.4f}")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_standard_crt()
    run_naw_crt()
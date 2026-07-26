import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import recall_score

from dataset import RAFDBDataset
from model import FRITNet
from loss import NAWLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 15           
LEARNING_RATE = 1e-3  
SAVE_DIR = "/content/drive/MyDrive/FER_NLA_Results"

# Base: Model 1
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_regularized_sampler.pth"
UNIQUE_WEIGHT_NAME = "best_model1_naw_tuned.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def load_weights_safely(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device)
    model_keys = set(model.state_dict().keys())
    
    first_ckpt_key = next(iter(state_dict.keys())) if len(state_dict) > 0 else ""
    if first_ckpt_key.startswith("module.") and not any(k.startswith("module.") for k in model_keys):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif not first_ckpt_key.startswith("module.") and any(k.startswith("module.") for k in model_keys):
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    return model

def train_naw_tuned():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting NAW-cRT Tuning | L2Norm Calibrated\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model = load_weights_safely(model, BASE_WEIGHTS, device)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out']):
            if 'attn' not in name.lower() and 'attention' not in name.lower() and 'norm' not in name.lower():
                param.requires_grad = True

    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ==========================================
    # TUNED NAW PARAMETERS FOR L2NORM
    # ==========================================
    # We must explicitly pass these if your NAWLoss class accepts them. 
    # If your NAWLoss hardcodes mu/sigma internally, you must update the loss.py file first!
    criterion = NAWLoss(num_classes=7, total_epochs=EPOCHS, lambda_param=1.0).to(device)
    
    # *CRITICAL NOTE*: If your current loss.py doesn't let you pass mu_gt, mu_nn, sigma_gt, sigma_nn,
    # you need to open loss.py and manually change the Gaussian centers to:
    # mu_gt = 0.8
    # mu_nn = 0.15
    # sigma = 0.15

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_macro_recall = 0.0

    for epoch in range(EPOCHS):
        criterion.set_epoch(epoch)
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved best Tuned NAW weights: Recall = {macro_recall:.4f}")

if __name__ == "__main__":
    train_naw_tuned()
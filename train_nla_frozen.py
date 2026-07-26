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
LEARNING_RATE = 1e-4  
SAVE_DIR = "/content/drive/MyDrive/FER_NLA_Results"

# Baseline Model Soup Weights
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/averaged_models_init.pth"
UNIQUE_WEIGHT_NAME = "best_rafdb_naw_crt_verified.pth"

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")


def load_weights_safely(model, weights_path, device):
    print(f"--> Reading checkpoint file: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    first_ckpt_key = next(iter(state_dict.keys())) if len(state_dict) > 0 else ""
    if first_ckpt_key.startswith("module.") and not any(k.startswith("module.") for k in model_keys):
        print("--> Detected 'module.' prefix in checkpoint. Stripping prefix...")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif not first_ckpt_key.startswith("module.") and any(k.startswith("module.") for k in model_keys):
        print("--> Adding 'module.' prefix to match model architecture...")
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"--> Diagnostic Check: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    
    # Filter out class head size mismatches (e.g. 8-class vs 7-class heads)
    critical_missing = [k for k in missing if not any(h in k for h in ['classifier', 'head', 'fc', 'logit'])]
    
    if len(critical_missing) > 0:
        print("\n================ ERROR: CRITICAL KEY MISMATCH ================")
        print(f"Found {len(critical_missing)} backbone/transformer keys that failed to load!")
        print("First 5 missing critical keys:", critical_missing[:5])
        print("==============================================================")
        raise RuntimeError("Checkpoint loading failed. Backbone is sitting at random initialization.")
        
    print("--> SUCCESS: Pre-trained feature extractor verified and fully loaded.")
    return model


def train_naw_crt_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting Diagnostic NAW-cRT | Verified Load | LR: {LEARNING_RATE}\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Model and Load Weights Safely
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model = load_weights_safely(model, BASE_WEIGHTS, device)
    
    # 2. Freeze Feature Extractor
    print("--> Freezing backbone, LFA, SAFM, and Transformers...")
    for param in model.parameters():
        param.requires_grad = False
        
    print("--> Unfreezing classification heads...")
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out']):
            if 'attn' not in name.lower() and 'attention' not in name.lower() and 'norm' not in name.lower():
                param.requires_grad = True
                unfrozen_params += param.numel()
                print(f"  [UNFROZEN] {name}")
    
    print(f"--> Total trainable parameters for cRT: {unfrozen_params:,}")

    # 3. Datasets and Weighted Sampler
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    sample_weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. NAW Loss (Lambda=1.0 isolates NAW-CE without JSD)
    criterion = NAWLoss(num_classes=7, total_epochs=EPOCHS, lambda_param=1.0).to(device)

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
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        t_acc = train_correct / train_total

        # Validation
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
        
        print(f"Epoch {epoch+1}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        # Checkpoint strictly on Macro Recall
        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved new best weights based on Macro Recall: {macro_recall:.4f} (Val Acc: {v_acc:.4f})")

if __name__ == "__main__":
    train_naw_crt_diagnostic()
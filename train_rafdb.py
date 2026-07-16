import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 50           
LEARNING_RATE = 3e-4  
WEIGHT_DECAY = 1e-2  # High weight decay to combat sampler duplication

FERPLUS_WEIGHTS = "/content/drive/MyDrive/FERPlus_Results/best_ferplus_aggressive.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_mixup_regularized.pth"

# --- Verified Local Paths ---
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def load_pretrained_weights(model, weights_path):
    print(f"--> Loading base FERPlus weights from: {weights_path}")
    state_dict = torch.load(weights_path)
    
    targets = ['.head.', 'aux_global_head', 'aux_local_head']
    keys_to_delete = [k for k in state_dict.keys() if any(t in k for t in targets)]
    
    print(f"--> Stripping {len(keys_to_delete)} mismatching 8-class head tensors...")
    for k in keys_to_delete:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    print("--> Successfully injected pre-trained facial geometry.")
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting RAF-DB Fine-Tuning | Dynamic MixUp | WD: {WEIGHT_DECAY}\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model = load_pretrained_weights(model, FERPLUS_WEIGHTS)

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

    criterion = CombinedFERLoss(feat_dim=128, num_classes=7, alpha=0.2).to(device)

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        
        # ---------------------------------------------------------
        # TWO-STAGE GRADIENT FREEZING
        # ---------------------------------------------------------
        if epoch == 0:
            print("\n--> STAGE 1: Freezing feature extractors. Training classification heads only.")
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        elif epoch == 5:
            print("\n--> STAGE 2: Unfreezing full network for precise alignment.")
            for param in model.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------
        # DYNAMIC MIXUP SCHEDULE
        # ---------------------------------------------------------
        mixup_active = False
        mixup_alpha = 0.0
        
        if epoch >= 7:
            mixup_active = True
            # Starts at 0.1, increases by 0.1 every 5 epochs
            increments = (epoch - 7) // 5
            # Cap at 0.5 to prevent complete structural degradation
            mixup_alpha = min(0.1 + (increments * 0.1), 0.5)
            
            print(f"--> Dynamic MixUp Active: Alpha = {mixup_alpha:.1f}")

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            optimizer.zero_grad()

            # --- MIXUP ROUTINE ---
            if mixup_active and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam) # Ensure primary image is always > 50%
                
                index = torch.randperm(images.size(0)).to(device)
                
                mixed_images = lam * images + (1 - lam) * images[index]
                
                logits, features, aux_global, aux_local = model(mixed_images)
                
                # Interpolate the loss between the two mixed targets
                loss_a = criterion(logits, features, targets + 1, aux_global, aux_local)
                loss_b = criterion(logits, features, targets[index] + 1, aux_global, aux_local)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                # Accuracy tracking is mapped to the primary/dominant image
                _, predicted = torch.max(logits.data, 1)
                train_correct += (predicted == targets).sum().item()

            # --- STANDARD ROUTINE ---
            else:
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

        # Validation Phase (Always clean, unmixed data)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1
                
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, targets + 1, aux_global, aux_local)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME)
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best RAF-DB weights: {v_acc:.4f} at {UNIQUE_WEIGHT_NAME}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('RAF-DB Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('RAF-DB Loss')
    plt.legend()
    
    plot_path = os.path.join(SAVE_DIR, "training_results_mixup_regularized.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
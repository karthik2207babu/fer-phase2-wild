import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import copy

# Import custom modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss
from sam import SAM

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10           
LEARNING_RATE = 5e-6  # Extremely low for SAM fine-tuning
WEIGHT_DECAY = 1e-2  

# Load your 87.68% peak weights
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_focal_tuned.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_sam_ema.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def update_ema(ema_model, model, alpha=0.999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_sam_ema():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting SAM + EMA Optimization | Gentle TTA\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize Base Model
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
    
    # Initialize EMA Shadow Model
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # Load Datasets
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sampler = WeightedRandomSampler(weights=class_weights[train_labels], num_samples=len(train_labels), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = CombinedFERLoss(feat_dim=128, num_classes=7, alpha=0.1).to(device)

    # Initialize SAM wrapper around AdamW
    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"SAM Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            # SAM Step 1: Forward pass and calculate gradients
            logits, features, aux_global, aux_local = model(images)
            loss = criterion(logits, features, targets + 1, aux_global, aux_local) 
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM Step 2: Forward pass at the perturbed local maximum
            criterion(model(images)[0], model(images)[1], targets + 1, model(images)[2], model(images)[3]).backward()
            optimizer.second_step(zero_grad=True)
            
            # Update EMA weights
            update_ema(ema_model, model)
            
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        # Validation Phase using the EMA model + Gentle Flip TTA
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1
                
                # Standard Forward
                logits_orig, _, _, _ = ema_model(images)
                probs_orig = torch.softmax(logits_orig, dim=1)
                
                # Flipped Forward
                images_flipped = torch.flip(images, dims=[3])
                logits_flip, _, _, _ = ema_model(images_flipped)
                probs_flip = torch.softmax(logits_flip, dim=1)
                
                # 80/20 Blend
                final_probs = (0.8 * probs_orig) + (0.2 * probs_flip)
                _, predicted = torch.max(final_probs, 1)
                
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc (EMA+TTA): {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(ema_model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved new best SAM+EMA weights: {v_acc:.4f}")

if __name__ == "__main__":
    train_sam_ema()
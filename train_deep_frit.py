import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from PIL import Image

from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 60           # Increased to 60 to accommodate warmup and deeper network
MAX_LR = 1e-4         # Slightly higher max LR, controlled by warmup
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 15

# Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

ZIP_PATH = "/content/drive/MyDrive/affectnet.zip"
EXTRACT_PATH = "/content/data"
AFFECTNET_DIR = os.path.join(EXTRACT_PATH, "affectnet/affectnet/Train") 
PSEUDO_CSV = "/content/drive/MyDrive/pseudo_labeled_affectnet.csv"

# Dedicated save directory for the Deep architecture
SAVE_DIR = "/content/drive/MyDrive/FER_Phase5_Deep_Transformer"

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class PseudoLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        label = int(self.data.iloc[idx]['label']) 
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Deep Transformer Training on: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(AFFECTNET_DIR):
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        os.system(f'unzip -q -n "{ZIP_PATH}" -d "{EXTRACT_PATH}"')

    raf_train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    pseudo_dataset = PseudoLabelDataset(csv_file=PSEUDO_CSV, transform=raf_train_dataset.transform)
    combined_train_dataset = ConcatDataset([raf_train_dataset, pseudo_dataset])
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize deeper model
    model = FRITNet(num_classes=7).to(device)

    # Replaced CombinedFERLoss with standard CrossEntropy + Label Smoothing
    # This prevents the deeper network from becoming overconfident and overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True

    # Increased weight decay to 1e-2 for stronger regularization on the deeper network
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': MAX_LR * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': MAX_LR},
        {'params': model.safm.parameters(), 'lr': MAX_LR},
        {'params': model.transformer.parameters(), 'lr': MAX_LR}
    ], weight_decay=1e-2)

    # ---------------------------------------------------------
    # WARMUP SCHEDULER IMPLEMENTATION
    # ---------------------------------------------------------
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    log_file_path = os.path.join(SAVE_DIR, "training_log_deep.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch,MixUp_Prob,LR,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = optimizer.param_groups[1]['lr']
        
        # MixUp Decay: Starts dropping after the warmup phase
        mixup_prob = max(0.0, 1.0 - (max(0, epoch - WARMUP_EPOCHS) // 5) * 0.1)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [MixUp: {mixup_prob:.1f}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Shift labels to 0-index for nn.CrossEntropyLoss
            labels = labels - 1 
            optimizer.zero_grad()
            
            if np.random.rand() < mixup_prob:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
                logits, _, _, _ = model(mixed_images)
                
                loss_a = criterion(logits, targets_a)
                loss_b = criterion(logits, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                dominant_labels = targets_a if lam > 0.5 else targets_b
            else:
                logits, _, _, _ = model(images)
                loss = criterion(logits, labels)
                dominant_labels = labels
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == dominant_labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) - 1
                logits, _, _, _ = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}: LR: {current_lr:.6f} | T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch+1},{mixup_prob:.1f},{current_lr:.6f},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_frit_weights_deep.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break
        
        scheduler.step()

if __name__ == "__main__":
    train()
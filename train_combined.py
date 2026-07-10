import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from PIL import Image

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 80           
LEARNING_RATE = 5e-5 
EARLY_STOPPING_PATIENCE = 15

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Pseudo-Label & Zip Paths
ZIP_PATH = "/content/drive/MyDrive/affectnet.zip"
EXTRACT_PATH = "/content/data"
AFFECTNET_DIR = os.path.join(EXTRACT_PATH, "affectnet/affectnet/Train") 
PSEUDO_CSV = "/content/drive/MyDrive/pseudo_labeled_affectnet.csv"

# Save Path
SAVE_DIR = "/content/drive/MyDrive/raf_trained_scratch_weights"

# --- MixUp Function ---
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

# --- Dedicated Loader for the Pseudo-Labeled Data ---
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
    print(f"Starting 10-Token Architecture Training from Scratch on: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================
    # ZIP EXTRACTION
    # =========================================================
    if not os.path.exists(AFFECTNET_DIR):
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        print(f"\nExtracting AffectNet ZIP from {ZIP_PATH}...")
        os.system(f'unzip -q -n "{ZIP_PATH}" -d "{EXTRACT_PATH}"')
        print("Extraction Complete\n")
    else:
        print("\nAffectNet Dataset already extracted.\n")

    raf_train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    pseudo_dataset = PseudoLabelDataset(csv_file=PSEUDO_CSV, transform=raf_train_dataset.transform)
    combined_train_dataset = ConcatDataset([raf_train_dataset, pseudo_dataset])
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Ready! Training on {len(combined_train_dataset)} Total Images.")

    # =========================================================
    # --- TRUE MRAN ARCHITECTURE: TRAINING FROM SCRATCH ---
    # =========================================================
    
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    print("--> Initialized fresh 10-token FRITNet architecture for RAF-DB (7 classes).")

    # Initialize Loss with alpha 0.2 (will be dynamically toggled)
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.2).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-3)
    # =========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    log_file_path = os.path.join(SAVE_DIR, "training_log_scratch.txt")
    log_file = open(log_file_path, "w")
    log_file.write("Epoch,MixUp_Prob,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        if epoch < 15:
            mixup_prob = 0.0
        else:
            mixup_prob = max(0.0, 0.5 - ((epoch - 15) // 5) * 0.1)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [MixUp: {mixup_prob:.1f}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if np.random.rand() < mixup_prob:
                # DYNAMIC ROUTING: Disable SupCon for MixUp batches
                criterion.alpha = 0.0
                
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
                logits, features, aux_global, aux_local = model(mixed_images)
                
                loss_a = criterion(logits, features, targets_a, aux_global, aux_local)
                loss_b = criterion(logits, features, targets_b, aux_global, aux_local)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                dominant_labels = targets_a if lam > 0.5 else targets_b
            else:
                # DYNAMIC ROUTING: Enable SupCon (0.2) for clean batches to stabilize FaceNet
                criterion.alpha = 0.2
                
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, labels, aux_global, aux_local)
                dominant_labels = labels
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            
            train_correct += (predicted == (dominant_labels - 1)).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        # Validation is always clean, so SupCon is computed
        criterion.alpha = 0.2 
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, labels, aux_global, aux_local)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == (labels - 1)).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} (MixUp: {mixup_prob:.1f}): T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        log_file.write(f"{epoch+1},{mixup_prob:.1f},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_frit_weights_scratch.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\n===================================")
            print("Early stopping triggered.")
            print("===================================")
            break
        scheduler.step()

    log_file.close()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    
    plot_path = os.path.join(SAVE_DIR, "training_results_plot_scratch.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
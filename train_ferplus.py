import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from prepare_data import prepare_ferplus_data, get_ferplus_dataloaders
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 80           
LEARNING_RATE = 5e-5 
EARLY_STOPPING_PATIENCE = 15

# Colab Paths
ZIP_PATH = "/content/drive/MyDrive/FERPLUS.zip"
EXTRACT_PATH = "/content/ferplus_extracted"

# Save Path for the optimized FERPlus base weights
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Optimized_Base"

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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optimized FERPlus Base Training on: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================
    # EXTRACT AND LOAD DATA VIA PREPARE_DATA.PY
    # =========================================================
    data_root = prepare_ferplus_data(zip_path=ZIP_PATH, extract_path=EXTRACT_PATH)
    train_loader, val_loader = get_ferplus_dataloaders(root_dir=data_root, batch_size=BATCH_SIZE)

    # =========================================================
    # --- MODEL & LOSS INITIALIZATION ---
    # =========================================================
    model = FRITNet(num_classes=8, transformer_depth=2).to(device)
    print("--> Initialized 10-token FRITNet architecture for FERPlus (8 classes).")

    criterion = CombinedFERLoss(feat_dim=128, alpha=0.2).to(device)
    
    # OVERRIDE: Balanced 8-class weights for FERPlus
    criterion.class_weights = torch.ones(8).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    log_file_path = os.path.join(SAVE_DIR, "training_log_ferplus.txt")
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
            
            # Shift ImageFolder's 0-indexed labels to 1-indexed for CombinedFERLoss
            loss_labels = labels + 1 
            
            optimizer.zero_grad()
            
            if np.random.rand() < mixup_prob:
                criterion.alpha = 0.0
                
                mixed_images, targets_a, targets_b, lam = mixup_data(images, loss_labels, alpha=0.2, device=device)
                logits, features, aux_global, aux_local = model(mixed_images)
                
                loss_a = criterion(logits, features, targets_a, aux_global, aux_local)
                loss_b = criterion(logits, features, targets_b, aux_global, aux_local)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                dominant_labels = targets_a if lam > 0.5 else targets_b
                acc_labels = dominant_labels - 1 
            else:
                criterion.alpha = 0.2
                
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, loss_labels, aux_global, aux_local)
                acc_labels = labels
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            
            train_correct += (predicted == acc_labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        criterion.alpha = 0.2 
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                loss_labels = labels + 1
                
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, loss_labels, aux_global, aux_local)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} (MixUp: {mixup_prob:.1f}): T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        log_file.write(f"{epoch+1},{mixup_prob:.1f},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_ferplus_optimized.pth")
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
    
    plot_path = os.path.join(SAVE_DIR, "training_results_plot_ferplus.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
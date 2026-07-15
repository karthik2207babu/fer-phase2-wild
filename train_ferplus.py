import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset_ferplus import prepare_ferplus_data, get_ferplus_dataloaders
from model import FRITNet
from loss import FERPlusMRANLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 60           
LEARNING_RATE = 0.01 
WEIGHT_DECAY_CNN = 1e-4
WEIGHT_DECAY_TRANSFORMER = 1e-2 

ZIP_PATH = "/content/drive/MyDrive/FERPLUS.zip"
EXTRACT_PATH = "/content/ferplus_extracted"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Results"

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting Run: Aggressive Regularization | LR: {LEARNING_RATE} | Epochs: {EPOCHS}\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    data_root = prepare_ferplus_data(zip_path=ZIP_PATH, extract_path=EXTRACT_PATH)
    train_loader, val_loader = get_ferplus_dataloaders(root_dir=data_root, batch_size=BATCH_SIZE)

    model = FRITNet(num_classes=8, transformer_depth=2).to(device)
    
    # Increased smoothing to heavily penalize overconfidence
    criterion = FERPlusMRANLoss(smoothing=0.25).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True
        
    print("--> CNN Backbone fully unfrozen for fine-tuning.")

    optimizer = optim.SGD([
        {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': LEARNING_RATE * 0.1, 'weight_decay': WEIGHT_DECAY_CNN},
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY_CNN},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY_CNN},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY_TRANSFORMER} 
    ], momentum=0.9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LEARNING_RATE*0.1, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.3
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Apply strict 0.2 MixUp after a 5-epoch warmup
        current_mixup = 0.0 if epoch < 5 else 0.2
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [MixUp: {current_mixup:.1f}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if current_mixup > 0.0:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=current_mixup, device=device)
                logits, features, aux_global, aux_local = model(mixed_images)
                
                loss_a = criterion(logits, features, targets_a, aux_global, aux_local)
                loss_b = criterion(logits, features, targets_b, aux_global, aux_local)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                dominant_labels = targets_a if lam > 0.5 else targets_b
            else:
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, labels, aux_global, aux_local)
                dominant_labels = labels
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            
            train_correct += (predicted == dominant_labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, features, aux_global, aux_local = model(images)
                loss = criterion(logits, features, labels, aux_global, aux_local)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} (MixUp: {current_mixup:.1f}): T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_ferplus_aggressive.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 20:
            print("\n===================================")
            print("Early stopping triggered.")
            print("===================================")
            break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    
    plot_path = os.path.join(SAVE_DIR, "training_results_ferplus_aggressive.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
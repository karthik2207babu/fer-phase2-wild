import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset_ferplus import prepare_ferplus_data, get_ferplus_dataloaders
from model import FRITNet  
from loss_ferplus import FERPlusMRANLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 60                # Increased to 60 for MixUp convergence
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15 # Increased to handle MixUp noise
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Results"

# MixUp function integrated
def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_ferplus():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting FERPlus MixUp Training on: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    data_root = prepare_ferplus_data()
    train_loader, val_loader = get_ferplus_dataloaders(data_root, batch_size=BATCH_SIZE)

    model = FRITNet(num_classes=8).to(device)
    criterion = FERPlusMRANLoss().to(device)

    # UNFREEZING THE BACKBONE
    print("--> Unfreezing CNN Backbone for Grayscale Domain Adaptation...")
    for param in model.backbone.parameters():
        param.requires_grad = True 

    # Applying the 0.1x multiplier to the backbone
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    log_file_path = os.path.join(SAVE_DIR, "ferplus_training_log_mixup.txt")
    log_file = open(log_file_path, "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Apply MixUp to the batch
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
            
            logits, _, aux_global, aux_local = model(mixed_images)
            
            # Calculate loss as a linear combination of both targets
            loss_a = criterion(logits, aux_global, aux_local, targets_a)
            loss_b = criterion(logits, aux_global, aux_local, targets_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            
            dominant_labels = targets_a if lam > 0.5 else targets_b
            train_correct += (predicted == dominant_labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Validation uses standard clean images, NO MixUp
                logits, _, aux_global, aux_local = model(images)
                
                loss = criterion(logits, aux_global, aux_local, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        t_acc = train_correct / train_total
        v_acc = val_correct / val_total
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        log_file.write(f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_frit_weights_ferplus_mixup.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break
            
        scheduler.step()

    log_file.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "ferplus_training_plot_mixup.png"))

if __name__ == "__main__":
    train_ferplus()
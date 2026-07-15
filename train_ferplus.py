import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Directly utilizing the verified data loader
from dataset_ferplus import prepare_ferplus_data, get_ferplus_dataloaders
from model import FRITNet
from loss import FERPlusMRANLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 60           
LEARNING_RATE = 0.01 
WEIGHT_DECAY = 1e-4

ZIP_PATH = "/content/drive/MyDrive/FERPLUS.zip"
EXTRACT_PATH = "/content/ferplus_extracted"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Results"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting Run: ferplus_sgd_0.01 (No MixUp) | Optimizer: SGD | LR: {LEARNING_RATE} | Epochs: {EPOCHS}\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    data_root = prepare_ferplus_data(zip_path=ZIP_PATH, extract_path=EXTRACT_PATH)
    train_loader, val_loader = get_ferplus_dataloaders(root_dir=data_root, batch_size=BATCH_SIZE)

    model = FRITNet(num_classes=8, transformer_depth=2).to(device)
    criterion = FERPlusMRANLoss(smoothing=0.15).to(device)

    # Freeze only the first 12 layers (basic edge detectors)
    for param in model.backbone.features[:12].parameters():
        param.requires_grad = False
        
    print("--> Early CNN layers frozen. Fine-tuning remaining blocks at 0.1x LR.")

    # Standard SGD with Momentum. Note the backbone getting lr * 0.1
    optimizer = optim.SGD([
        {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': LEARNING_RATE * 0.1},
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], momentum=0.9, weight_decay=WEIGHT_DECAY)

    # Extended OneCycleLR schedule to 60 epochs for maximum convergence
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, features, aux_global, aux_local = model(images)
            loss = criterion(logits, features, labels, aux_global, aux_local)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
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
        
        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path = os.path.join(SAVE_DIR, "best_ferplus_sgd_clean.pth")
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
    
    plot_path = os.path.join(SAVE_DIR, "training_results_ferplus_clean.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
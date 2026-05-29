import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss
import config  # Importing your new portability config

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 12

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # Initialize Datasets using paths from config
    train_dataset = RAFDBDataset(csv_file=config.TRAIN_CSV, root_dir=config.TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=config.VAL_CSV, root_dir=config.VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Ready! Training on {len(train_dataset)} images.")

    # Initialize Model
    model = FRITNet(num_classes=7).to(device)
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.2).to(device)

    # Ensure save directory exists
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Unfreeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.multiscale.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4) # Increased decay for 10-token stability

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Open log file
    log_file = open(config.LOG_FILE_PATH, "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass with Auxiliary outputs
            logits, features, aux_global, aux_local = model(images)
            loss = criterion(logits, features, labels, aux_global, aux_local)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == (labels - 1)).sum().item()
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
                val_correct += (predicted == (labels - 1)).sum().item()

        t_acc, v_acc = train_correct/train_total, val_correct/val_total
        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        log_file.write(f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), config.BEST_WEIGHTS_PATH)
            print(f"--> Saved new best weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break
        scheduler.step()

    log_file.close()
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc']); plt.plot(history['val_acc']); plt.title('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(history['train_loss']); plt.plot(history['val_loss']); plt.title('Loss')
    plt.savefig(config.PLOT_PATH)
    print(f"Graphs saved to {config.PLOT_PATH}")

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-4  # Significantly lowered for fine-tuning
WEIGHT_DECAY = 1e-4

FERPLUS_WEIGHTS = "/content/drive/MyDrive/FERPlus_Results/best_ferplus_aggressive.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"

# --- Verified Local Paths ---
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def load_pretrained_weights(model, weights_path):
    print(f"--> Loading base FERPlus weights from: {weights_path}")
    state_dict = torch.load(weights_path)
    
    # Isolate and delete the classification heads from the saved weights
    # FERPlus had 8 classes, RAF-DB has 7. We must drop these matrices.
    keys_to_delete = [
        k for k in state_dict.keys() 
        if k.startswith('head.') or k.startswith('aux_global_head.') or k.startswith('aux_local_head.')
    ]
    
    for k in keys_to_delete:
        del state_dict[k]
        
    # strict=False allows the model to load everything EXCEPT the missing heads.
    # The new 7-class heads will initialize with random weights.
    model.load_state_dict(state_dict, strict=False)
    print("--> Successfully injected pre-trained facial geometry.")
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nStarting RAF-DB Fine-Tuning | LR: {LEARNING_RATE} | Epochs: {EPOCHS}\n{'='*65}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize model specifically for 7 classes
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model = load_pretrained_weights(model, FERPLUS_WEIGHTS)

    # 2. Datasets & Loaders
    print("--> Loading RAF-DB datasets...")
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Loss & Optimizer (AdamW is preferred for fine-tuning)
    criterion = CombinedFERLoss(feat_dim=128, num_classes=7, alpha=0.2).to(device)

    # The backbone receives an even smaller learning rate (LR * 0.1) 
    # to protect the foundational edge detectors.
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)

    # Cosine Annealing smoothly drops the LR to 0 by the final epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Align RAF-DB 1-7 labels to 0-6 index for PyTorch
            targets = labels - 1 
            
            optimizer.zero_grad()
            logits, features, aux_global, aux_local = model(images)
            
            loss = criterion(logits, features, targets + 1, aux_global, aux_local) 
            # Note: CombinedFERLoss internally expects raw labels (1-7) to subtract 1 for the focal loss,
            # but relies on raw labels for SupCon matrix matching. Passing targets+1 retains compatibility.

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Validation Phase
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
            weights_path = os.path.join(SAVE_DIR, "best_rafdb_finetuned.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best RAF-DB weights: {v_acc:.4f}")

    # Plot and save training graphs
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
    
    plot_path = os.path.join(SAVE_DIR, "training_results_rafdb.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
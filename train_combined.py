import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 5e-5 # Lower LR since we are fine-tuning from a strong baseline
EARLY_STOPPING_PATIENCE = 12

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Pseudo-Label Paths
PSEUDO_CSV = "/content/drive/MyDrive/pseudo_labeled_affectnet.csv"
PRETRAINED_WEIGHTS = "/content/drive/MyDrive/FER_Phase3_Results/best_frit_weights_mixup.pth"
SAVE_DIR = "/content/drive/MyDrive/FER_Phase4_Pseudo"

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
    print(f"Starting Pseudo-Label Combined Training on: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Original RAF-DB Train Dataset
    raf_train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    
    # 2. Load Pseudo-Labeled Dataset (re-using the exact same augmentations)
    pseudo_dataset = PseudoLabelDataset(csv_file=PSEUDO_CSV, transform=raf_train_dataset.transform)
    
    # 3. Combine them
    combined_train_dataset = ConcatDataset([raf_train_dataset, pseudo_dataset])
    
    # Validation remains strictly RAF-DB to track our true target metric
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Ready! Training on {len(combined_train_dataset)} Total Images ({len(raf_train_dataset)} RAF + {len(pseudo_dataset)} Pseudo).")

    model = FRITNet(num_classes=7).to(device)

    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading Base Weights: {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
    else:
        print("WARNING: Base weights not found. Training from scratch.")

    # Restoring standard SupCon configuration
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.1).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    log_file_path = os.path.join(SAVE_DIR, "training_log_pseudo.txt")
    log_file = open(log_file_path, "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

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
            weights_path = os.path.join(SAVE_DIR, "best_frit_weights_pseudo.pth")
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
    
    plot_path = os.path.join(SAVE_DIR, "training_results_plot_pseudo.png")
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")

if __name__ == "__main__":
    train()
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Configuration ---
BATCH_SIZE = 16 
EPOCHS = 10
LEARNING_RATE = 1e-4

# Path setup for your specific nested folders
BASE_PATH = r"C:\Users\chinn\OneDrive\Desktop\Datasets\RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")

# These point to the folders containing the 1, 2, 3... subfolders
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # 1. Load Data
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Ready! {len(train_dataset)} training images found.")

    # 2. Model & Loss
    model = FRITNet(num_classes=7).to(device)
    criterion = CombinedFERLoss(feat_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # 3. Training
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, features = model(images)
            loss = criterion(logits, features, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == (labels - 1)).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Simple Val Check
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == (labels - 1)).sum().item()

        v_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "best_frit_weights.pth")

    print("Success. Run completed.")

if __name__ == "__main__":
    train()
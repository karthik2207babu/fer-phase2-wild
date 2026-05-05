import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Colab Configuration ---
BATCH_SIZE = 64  
EPOCHS = 25  
LEARNING_RATE = 1e-4

# Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # 1. Load Data
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    
    # Balanced Sampler (UNCHANGED)
    labels = train_dataset.annotations.iloc[:, 1].values
    class_counts = torch.bincount(torch.tensor(labels))[1:].numpy()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels - 1]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Ready! Training on {len(train_dataset)} images.")

    # 2. Model, Loss, Optimizer
    model = FRITNet(num_classes=7).to(device)
    criterion = CombinedFERLoss(feat_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    log_file = open("training_log.txt", "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, features = model(images)

            # ✅ ORIGINAL LOSS (NO MIXUP)
            loss = criterion(logits, features, labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == (labels - 1)).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation (UNCHANGED)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, features = model(images)
                loss = criterion(logits, features, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == (labels - 1)).sum().item()

        # Metrics
        t_loss = train_loss / len(train_loader)
        t_acc = train_correct / train_total
        v_loss = val_loss / len(val_loader)
        v_acc = val_correct / val_total
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")
        log_file.write(f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "best_frit_weights.pth")
            print(f"--> Saved new best weights: {v_acc:.4f}")

    log_file.close()

    # Graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_results_plot.png")
    print("Graphs saved as training_results_plot.png")


if __name__ == "__main__":
    train()
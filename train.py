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
BATCH_SIZE = 64
EPOCHS = 50                      # 👇 CHANGED: Increased from 50
LEARNING_RATE = 1e-4

# =========================================
# Early stopping patience
# =========================================
EARLY_STOPPING_PATIENCE = 12     # 👇 CHANGED: Increased from 8

# Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    train_dataset = RAFDBDataset(
        csv_file=TRAIN_CSV,
        root_dir=TRAIN_ROOT,
        phase='train'
    )

    val_dataset = RAFDBDataset(
        csv_file=VAL_CSV,
        root_dir=VAL_ROOT,
        phase='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    print(f"Ready! Training on {len(train_dataset)} images.")

    model = FRITNet(num_classes=7).to(device)

    # 👇 CHANGED: Added alpha=0.2 to balance clustering with classification
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.2).to(device)

    # =========================================
    # UNFREEZE BACKBONE FOR MAXIMUM CAPACITY
    # =========================================
    for param in model.backbone.parameters():
        param.requires_grad = True

    # =========================================
    # DIFFERENTIAL OPTIMIZER
    # =========================================
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.multiscale.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=5e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # =========================================
    # Early stopping counter
    # =========================================
    epochs_without_improvement = 0

    # 👇 CHANGED: Saving log directly to Drive
    log_file = open("/content/drive/MyDrive/FER_Phase3_Results/training_log_5token.txt", "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:

            images = images.to(device)
            labels = labels.to(device)

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

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits, features = model(images)

                loss = criterion(logits, features, labels)

                val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)

                val_total += labels.size(0)

                val_correct += (predicted == (labels - 1)).sum().item()

        t_loss = train_loss / len(train_loader)
        t_acc = train_correct / train_total

        v_loss = val_loss / len(val_loader)
        v_acc = val_correct / val_total

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch+1}: T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}")

        log_file.write(
            f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n"
        )

        log_file.flush()

        # =========================================
        # Save best model
        # =========================================
        if v_acc > best_val_acc:

            best_val_acc = v_acc

            # 👇 CHANGED: Saving weights directly to Drive
            torch.save(model.state_dict(), "/content/drive/MyDrive/FER_Phase3_Results/best_frit_weights_5token.pth")

            print(f"--> Saved new best weights: {v_acc:.4f} directly to Drive")

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # =========================================
        # Early stopping
        # =========================================
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print("\n===================================")
            print("Early stopping triggered.")
            print("===================================")

            break

        scheduler.step()

    log_file.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 👇 CHANGED: Saving plot directly to Drive
    plt.savefig("/content/drive/MyDrive/FER_Phase3_Results/training_results_plot_5token.png")

    print("Graphs saved to Drive as training_results_plot_5token.png")


if __name__ == "__main__":
    train()
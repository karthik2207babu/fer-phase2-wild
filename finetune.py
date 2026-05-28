import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# =========================================
# COOLDOWN CONFIGURATION (SGD + Freeze)
# =========================================
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3  # SGD needs a slightly higher LR than Adam
EARLY_STOPPING_PATIENCE = 8

# Dataset Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Points to your local 86.08% file
LOAD_WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase3_Results/best_frit_weights.pth"
SAVE_WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase3_Results/best_frit_finetuned_sgd.pth"

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SGD cooldown fine-tuning on: {device}")

    # 1. Load Datasets & Apply Clean Transforms
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clean_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Clean data, only flipping
        transforms.ToTensor(),
        normalize
    ])
    train_dataset.transform = clean_train_transform
    print("Heavy augmentations disabled. Using clean data.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Load Model & Weights
    model = FRITNet(num_classes=7).to(device)
    
    if os.path.exists(LOAD_WEIGHTS_PATH):
        model.load_state_dict(torch.load(LOAD_WEIGHTS_PATH, map_location=device))
        print(f"✅ Loaded 86.08% weights from: {LOAD_WEIGHTS_PATH}")
    else:
        raise FileNotFoundError(f"Could not find weights at {LOAD_WEIGHTS_PATH}")

    # Lower alpha to prioritize pure classification accuracy
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.2).to(device)

    # =====================================================
    # 3. HARD FREEZE BACKBONE (Kills Identity Memorization)
    # =====================================================
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen. Network must rely on emotional features.")

    # =====================================================
    # 4. SGD OPTIMIZER (Glides into better generalization)
    # =====================================================
    optimizer = optim.SGD([
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.multiscale.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.8608
    epochs_without_improvement = 0

    log_file = open("/content/drive/MyDrive/FER_Phase3_Results/finetune_sgd_log.txt", "w")
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

        # Validation
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
        log_file.write(f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_WEIGHTS_PATH)
            print(f"--> 🔥 New Peak Reached! Saved weights: {v_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\n===================================")
            print("Fine-tuning plateaued. Early stopping.")
            print("===================================")
            break

        scheduler.step()

    log_file.close()

    # Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('SGD Fine-Tuning Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('SGD Fine-Tuning Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = "/content/drive/MyDrive/FER_Phase3_Results/finetune_sgd_plot.png"
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")
    print(f"Best Fine-Tuned Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    finetune()
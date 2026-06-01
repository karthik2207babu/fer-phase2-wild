import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import RAFDBDataset
from model import FRITNet
from loss import CombinedFERLoss

# --- Fine-Tuning Configuration ---
BATCH_SIZE = 64
EPOCHS = 15             # 15-epoch target precision cooldown
LEARNING_RATE = 1e-5   # Dropped 10x lower to squeeze out performance without breaking weights

# Google Drive Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")
SAVE_DIR = "/content/drive/MyDrive/FER_Phase3_Results"

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting fine-tuning phase on: {device}")

    # Use basic augmentations or validation transforms if you want a clean data pass
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Ready! Fine-tuning on {len(train_dataset)} images.")

    # 1. Initialize Architecture
    model = FRITNet(num_classes=7).to(device)

    # 2. Look up and load the exact weights from your previous run
    weights_path = os.path.join(SAVE_DIR, "best_frit_weights_hybrid.pth")
    if os.path.exists(weights_path):
        print(f"==> Found existing weights at: {weights_path}")
        print("--> Loading weights securely into FRITNet pipeline...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("--> Load successful! Weights are ready for precision fine-tuning.")
    else:
        print(f"ERROR: Could not find weights file at {weights_path}")
        print("Please check your Drive path or file name before running.")
        return

    # Use alpha=0.5 consistency regularization as before
    criterion = CombinedFERLoss(feat_dim=128, alpha=0.5).to(device)

    # Ensure backbone layers remain unfrozen for micro-adjustments
    for param in model.backbone.parameters():
        param.requires_grad = True

    # 3. Setup Optimizer with 10x lower base LR
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, 
        {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
        {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)

    # Cosine annealing adapted for the short 15-epoch stretch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.8680  # Anchored to your peak performance from the last run

    log_file_path = os.path.join(SAVE_DIR, "fine_tuning_log_hybrid.txt")
    log_file = open(log_file_path, "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{EPOCHS}")

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

        # Save only if it beats the previous all-time peak of 86.80%
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            weights_path_finetune = os.path.join(SAVE_DIR, "best_frit_weights_hybrid_finetuned.pth")
            torch.save(model.state_dict(), weights_path_finetune)
            print(f"--> [NEW SOTA BREAKTHROUGH] Saved fine-tuned weights: {v_acc:.4f}")
            
        scheduler.step()

    log_file.close()
    
    # Save visualization tracking performance changes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Fine-Tuning Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Fine-Tuning Loss'); plt.legend()
    
    plot_path = os.path.join(SAVE_DIR, "finetuning_results_plot.png")
    plt.savefig(plot_path)
    print(f"Fine-tuning metrics graph exported cleanly to: {plot_path}")

if __name__ == "__main__":
    finetune()
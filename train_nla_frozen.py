import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import recall_score

# Custom modules
from dataset import RAFDBDataset, RandomMasking
from model import FRITNet
from loss import NAWLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 15           # cRT converges very fast
LEARNING_RATE = 1e-4
SAVE_DIR = "/content/drive/MyDrive/FER_NLA_Results"

# Use the Model Soup weights that produced the 88.66% cRT baseline
BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/averaged_models_init.pth"
UNIQUE_WEIGHT_NAME = "best_rafdb_naw_crt.pth"

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")


# --- Dual-View Dataset Wrapper (RAF-DB Only) ---
class DualViewDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        self.clean_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            RandomMasking(min_area=0.04, max_area=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_name = str(self.base_dataset.annotations.iloc[idx, 0])
        label = int(self.base_dataset.annotations.iloc[idx, 1])
        img_path = os.path.join(self.base_dataset.root_dir, str(label), img_name)

        image_pil = Image.open(img_path).convert('RGB')
        return self.clean_transform(image_pil), self.aug_transform(image_pil), label


def train_naw_crt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting NAW-cRT | Frozen Representation + Ambiguity Classifier\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Model and Load Soup Weights
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    
    if os.path.exists(BASE_WEIGHTS):
        print(f"--> Loading Soup weights from: {BASE_WEIGHTS}")
        model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device), strict=False)
    else:
        raise FileNotFoundError(f"Cannot find {BASE_WEIGHTS}. Run failed.")
    
    # 2. FREEZE EVERYTHING EXCEPT CLASSIFICATION HEADS
    print("--> Freezing backbone, LFA, SAFM, and Transformers...")
    for param in model.parameters():
        param.requires_grad = False
        
    print("--> Unfreezing classification heads...")
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out']):
            if 'attn' not in name.lower() and 'attention' not in name.lower() and 'norm' not in name.lower():
                param.requires_grad = True
                unfrozen_params += param.numel()
                print(f"  [UNFROZEN] {name}")
    
    print(f"--> Total trainable parameters for NAW-cRT: {unfrozen_params:,}")

    # 3. Load Pure RAF-DB Dataset (Standard Shuffle, NAW handles imbalance)
    raf_train = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    train_dataset = DualViewDataset(raf_train)
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. NAW Loss and Optimizer
    criterion = NAWLoss(num_classes=7, total_epochs=EPOCHS, lambda_param=0.5).to(device)

    # Only pass the unfrozen parameters to the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_macro_recall = 0.0

    for epoch in range(EPOCHS):
        criterion.set_epoch(epoch)
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"NAW-cRT Epoch {epoch+1}/{EPOCHS}")
        for imgs_orig, imgs_aug, labels in pbar:
            imgs_orig, imgs_aug, labels = imgs_orig.to(device), imgs_aug.to(device), labels.to(device)
            targets_idx = labels - 1 
            
            optimizer.zero_grad()
            logits_orig, _, aux_g, aux_l = model(imgs_orig)
            logits_aug, _, _, _ = model(imgs_aug)

            loss = criterion(logits_orig, targets_idx, aux_g, aux_l, logits_aug=logits_aug)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits_orig.data, 1)
            train_correct += (predicted == targets_idx).sum().item()
            train_total += targets_idx.size(0)
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        t_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets_idx = labels - 1
                
                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                
                val_total += targets_idx.size(0)
                val_correct += (predicted == targets_idx).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets_idx.cpu().numpy())

        v_acc = val_correct / val_total
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        # Save on Best Accuracy 
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_macro_recall = macro_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved new best NAW-cRT weights: Acc = {v_acc:.4f}, Recall = {macro_recall:.4f}")

if __name__ == "__main__":
    train_naw_crt()
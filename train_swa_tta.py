import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

# Import custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
SWA_LR = 5e-5

BASE_WEIGHTS = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_crt.pth"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_rafdb_swa.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def train_swa_and_test_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting SWA Polish + Gentle TTA Evaluation\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load the 88.66% Baseline
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
    
    # 2. Maintain cRT Freezing Logic (Protect the Backbone)
    print("--> Freezing backbone, unfreezing heads...")
    unfrozen_params = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(keyword in name.lower() for keyword in ['fc', 'logit', 'classifier', 'head', 'main', 'aux', 'out']):
            if 'attn' not in name.lower() and 'attention' not in name.lower() and 'norm' not in name.lower() and 'cross' not in name.lower():
                param.requires_grad = True
                unfrozen_params += param.numel()
                
    # 3. Setup SWA
    swa_model = AveragedModel(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    # 4. Load Datasets (Using cRT Balanced Sampler)
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_labels = train_dataset.annotations.iloc[:, 1].values - 1
    class_counts = np.bincount(train_labels)
    sample_weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss().to(device)

    # 5. SWA Training Loop
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"SWA Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1 
            
            optimizer.zero_grad()
            logits, _, _, _ = model(images)
            loss = criterion(logits, targets) 
            
            loss.backward()
            optimizer.step()

        # Update SWA parameters at the end of each epoch
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Save the smoothed SWA weights
    torch.save(swa_model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
    print(f"\n--> Saved SWA polished weights to {UNIQUE_WEIGHT_NAME}")

    # 6. Evaluation with Gentle TTA (80/20 Blend)
    print("\n--> Executing Gentle TTA Evaluation (0.8 Orig + 0.2 Flip)...")
    swa_model.eval()
    val_correct, val_total = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1
            
            # Forward Original
            logits_orig, _, _, _ = swa_model(images)
            probs_orig = F.softmax(logits_orig, dim=1)
            
            # Forward Flipped
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _, _, _ = swa_model(images_flip)
            probs_flip = F.softmax(logits_flip, dim=1)
            
            # Blend Probs
            final_probs = (0.8 * probs_orig) + (0.2 * probs_flip)
            _, predicted = torch.max(final_probs, 1)
            
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    accuracy = val_correct / val_total
    print("\n" + "="*50)
    print(f"FINAL SWA + TTA ACCURACY: {accuracy*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_swa_and_test_tta()
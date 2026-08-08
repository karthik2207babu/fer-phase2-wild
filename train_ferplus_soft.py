import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image

from model import FRITNet

# ------------------ Config ------------------
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 5
NUM_CLASSES = 7

# Paths (update these)
PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Improved_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# RAF-DB class order (drop contempt)
RAF_DB_ORDER = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
ALL_VOTES = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

class FERPlusSoftDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels).convert('RGB')
        if self.transform:
            image = self.transform(image)
        soft_label = torch.tensor(row['soft_label'], dtype=torch.float32)
        # Also get the majority class for sampler weighting
        hard_label = torch.argmax(soft_label).item()
        return image, soft_label, hard_label

def prepare_dataframes(pixels_path, labels_path):
    print("Loading and merging FERPlus CSVs...")
    pixels_df = pd.read_csv(pixels_path)
    labels_df = pd.read_csv(labels_path)
    df = pd.concat([pixels_df[['pixels']], labels_df], axis=1)

    valid_rows = []
    for idx, row in df.iterrows():
        total_votes = sum([row[c] for c in ALL_VOTES])
        if total_votes == 0 or (row['unknown'] + row['NF']) > 0.5 * total_votes:
            continue
        votes_7 = np.array([row[c] for c in RAF_DB_ORDER], dtype=np.float32)
        sum_7 = votes_7.sum()
        if sum_7 == 0:
            continue
        soft_label = votes_7 / sum_7
        valid_rows.append({
            'pixels': row['pixels'],
            'Usage': row['Usage'],
            'soft_label': soft_label
        })
    final_df = pd.DataFrame(valid_rows)
    print(f"Kept {len(final_df)} images out of {len(df)}")
    return final_df

# ----- Loss: soft-target cross-entropy -----
def soft_target_ce(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()

# ----- Data preparation -----
df = prepare_dataframes(PIXELS_CSV, LABELS_CSV)
train_df = df[df['Usage'] == 'Training']
val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

# Stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FERPlusSoftDataset(train_df, transform=train_transform)
val_dataset = FERPlusSoftDataset(val_df, transform=val_transform)

# Weighted sampler (based on majority class counts)
# Get hard labels from the dataframe (soft_label is a numpy array)
train_hard_labels = [np.argmax(row) for row in train_df['soft_label'].values]
class_counts = np.bincount(train_hard_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[train_hard_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Class counts in train: {class_counts}")

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FRITNet(num_classes=NUM_CLASSES, transformer_depth=2, use_srt=False).to(device)

# The backbone is already pre-trained (TruncatedFaceNet loads VGGFace2 weights by default)

# Optimizer with lower LR for backbone
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
    {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
    {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
    {'params': model.transformer.parameters(), 'lr': LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

# Warmup + Cosine scheduler
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----- Training -----
best_val_acc = 0.0
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Two-stage freezing
    if epoch == 0:
        print("Stage 1: Freezing backbone, training heads only")
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    elif epoch == 5:
        print("Stage 2: Unfreezing backbone")
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, soft_labels, hard_labels in pbar:
        images, soft_labels = images.to(device), soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        optimizer.zero_grad()

        logits, _, aux_g, aux_l = model(images, training=True)
        # Main loss
        loss = soft_target_ce(logits, soft_labels)
        # Auxiliary losses (with smaller weight)
        loss += 0.1 * soft_target_ce(aux_g, soft_labels)
        loss += 0.1 * soft_target_ce(aux_l, soft_labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(logits, 1)
        train_preds.extend(pred.cpu().numpy())
        train_targets.extend(hard_labels.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    train_acc = (np.array(train_preds) == np.array(train_targets)).mean()

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, soft_labels, hard_labels in val_loader:
            images, soft_labels = images.to(device), soft_labels.to(device)
            hard_labels = hard_labels.to(device)
            logits, _, _, _ = model(images, training=False)
            loss = soft_target_ce(logits, soft_labels)
            val_loss += loss.item()
            _, pred = torch.max(logits, 1)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(hard_labels.cpu().numpy())

    val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val Loss={avg_val_loss:.4f}")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_ferplus_improved.pth"))
        print(f"--> Saved best model (val acc {val_acc:.4f})")

print(f"Training complete. Best FERPlus validation accuracy: {best_val_acc:.4f}")
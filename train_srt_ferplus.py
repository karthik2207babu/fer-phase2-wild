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
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUM_CLASSES = 7
TRANSFORMER_DEPTH = 4
USE_SRT = True

# Paths
FERPLUS_PIXELS = "/content/drive/MyDrive/fer2013.csv"
FERPLUS_LABELS = "/content/drive/MyDrive/fer2013new.csv"
BASE_WEIGHTS = "/content/drive/MyDrive/FERPlus_Depth4_AdamW/best_depth4_adamw.pth"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_SRT"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ Dataset (same as before) ------------------
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
        hard_label = torch.argmax(soft_label).item()
        return image, soft_label, hard_label

def prepare_ferplus_data(pixels_path, labels_path):
    print("Loading FERPlus CSVs...")
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
    print(f"FERPlus: Kept {len(final_df)} images out of {len(df)}")
    return final_df

def soft_target_ce(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()

# ----- Data -----
df = prepare_ferplus_data(FERPLUS_PIXELS, FERPLUS_LABELS)
train_df = df[df['Usage'] == 'Training']
val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FERPlusSoftDataset(train_df, transform=train_transform)
val_dataset = FERPlusSoftDataset(val_df, transform=val_transform)

train_hard_labels = [np.argmax(row) for row in train_df['soft_label'].values]
class_counts = np.bincount(train_hard_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[train_hard_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Model with SRT -----
model = FRITNet(num_classes=NUM_CLASSES, transformer_depth=TRANSFORMER_DEPTH, use_srt=USE_SRT).to(device)

# Load the best depth-4 checkpoint
if os.path.exists(BASE_WEIGHTS):
    print(f"Loading base weights from {BASE_WEIGHTS}")
    state_dict = torch.load(BASE_WEIGHTS, map_location=device)
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("Loaded base weights. SRT layers are randomly initialised.")
else:
    print("Base weights not found – starting from scratch.")

# Freeze backbone, fine‑tune the rest with SRT
for name, param in model.named_parameters():
    if 'backbone' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
    {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
    {'params': model.transformer.parameters(), 'lr': LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_preds, train_targets = [], []
    pbar = tqdm(train_loader, desc=f"SRT Epoch {epoch+1}/{EPOCHS}")
    for images, soft_labels, hard_labels in pbar:
        images, soft_labels = images.to(device), soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        optimizer.zero_grad()
        logits, _, aux_g, aux_l = model(images, training=True)
        loss = soft_target_ce(logits, soft_labels)
        loss += 0.1 * soft_target_ce(aux_g, soft_labels)
        loss += 0.1 * soft_target_ce(aux_l, soft_labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(logits, 1)
        train_preds.extend(pred.cpu().numpy())
        train_targets.extend(hard_labels.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    train_acc = (np.array(train_preds) == np.array(train_targets)).mean()

    # Validation (SRT disabled)
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, _, hard_labels in val_loader:
            images, hard_labels = images.to(device), hard_labels.to(device)
            logits, _, _, _ = model(images, training=False)
            _, pred = torch.max(logits, 1)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(hard_labels.cpu().numpy())

    val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    scheduler.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_srt_model.pth"))
        print(f"--> Saved best SRT model (val acc {val_acc:.4f})")

print(f"SRT fine‑tuning complete. Best val acc: {best_val_acc:.4f}")
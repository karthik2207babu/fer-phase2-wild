import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from PIL import Image

from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 80                # increased for better convergence
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4

# Paths – update these to your actual CSV files
PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"       # official FER2013 pixels
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"    # official vote counts
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Soft_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# RAF‑DB order (7 classes, dropping contempt)
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
        return image, soft_label

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
        # Extract 7 classes in RAF‑DB order
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

def soft_target_ce(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training FERPlus with soft labels on {device}")

    df = prepare_dataframes(PIXELS_CSV, LABELS_CSV)
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = FRITNet(num_classes=7, transformer_depth=2).to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine scheduler
    def lambda_lr(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, soft_labels in pbar:
            images, soft_labels = images.to(device), soft_labels.to(device)

            optimizer.zero_grad()
            logits, _, aux_g, aux_l = model(images)

            loss = soft_target_ce(logits, soft_labels)
            loss_g = soft_target_ce(aux_g, soft_labels)
            loss_l = soft_target_ce(aux_l, soft_labels)
            loss = loss + 0.1 * loss_g + 0.1 * loss_l

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            _, target = torch.max(soft_labels, 1)
            train_total += target.size(0)
            train_correct += (pred == target).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, soft_labels in val_loader:
                images, soft_labels = images.to(device), soft_labels.to(device)
                logits, _, _, _ = model(images)
                loss = soft_target_ce(logits, soft_labels)
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                _, target = torch.max(soft_labels, 1)
                val_total += target.size(0)
                val_correct += (pred == target).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val Loss={avg_val_loss:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_ferplus_soft_ds.pth"))
            print(f"--> Saved best FERPlus soft model with val acc {val_acc:.4f}")

    print(f"Training complete. Best FERPlus soft validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
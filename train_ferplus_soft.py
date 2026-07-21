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

# Import custom modules
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

# File paths - Update these if your Drive paths differ
PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"
SAVE_DIR = "/content/drive/MyDrive/RAFDB_Results"
UNIQUE_WEIGHT_NAME = "best_ferplus_soft.pth"

# 1. Class Count Decision:
# We strictly drop 'contempt' and enforce the exact 7-way RAF-DB label order.
# This ensures the classification head remains 100% compatible with your RAF-DB transfer.
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
        
        # Parse 48x48 pixel string to 2D numpy array
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        
        # Convert to RGB image to match FRITNet's expected 3-channel input
        image = Image.fromarray(pixels).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        soft_label = torch.tensor(row['soft_label'], dtype=torch.float32)
        
        return image, soft_label

def prepare_dataframes(pixels_path, labels_path):
    print("--> Loading and merging FERPlus CSVs by row index...")
    pixels_df = pd.read_csv(pixels_path)
    labels_df = pd.read_csv(labels_path)
    
    # Merge safely (using labels_df's Usage to avoid column duplication)
    df = pd.concat([pixels_df[['pixels']], labels_df], axis=1)
    
    initial_count = len(df)
    valid_rows = []
    
    for idx, row in df.iterrows():
        total_votes = sum([row[c] for c in ALL_VOTES])
        
        # Filter 1: Drop if no votes or non-face/unknown votes > 50%
        if total_votes == 0 or (row['unknown'] + row['NF']) > 0.5 * total_votes:
            continue
            
        # Filter 2: Extract 7 valid classes in RAF-DB order (dropping contempt)
        votes_7 = np.array([row[c] for c in RAF_DB_ORDER], dtype=np.float32)
        sum_7 = votes_7.sum()
        
        # If dropping contempt leaves us with 0 valid votes, skip the image
        if sum_7 == 0:
            continue
            
        # Normalize to probability distribution
        soft_label = votes_7 / sum_7
        
        valid_rows.append({
            'pixels': row['pixels'],
            'Usage': row['Usage'], 
            'soft_label': soft_label
        })
        
    final_df = pd.DataFrame(valid_rows)
    dropped = initial_count - len(final_df)
    print(f"--> Data preparation complete. Dropped {dropped} rows due to unknown/NF dominance or missing votes.")
    return final_df

def soft_target_ce(logits, soft_labels):
    # Cross Entropy for soft targets: -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()

def train_ferplus_soft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\nStarting FERPlus Pretraining (Soft-Label Distributions)\n{'='*75}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Process Data
    df = prepare_dataframes(PIXELS_CSV, LABELS_CSV)
    
    # Split based on official Usage flags
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])] # Combining tests for validation
    
    print(f"--> Training samples: {len(train_df)}")
    print(f"--> Validation samples: {len(val_df)}")

    # 224x224 RGB Alignment Emulation 
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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

    # Initialize Model
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, soft_labels in pbar:
            images, soft_labels = images.to(device), soft_labels.to(device)
            
            optimizer.zero_grad()
            logits, _, aux_g, aux_l = model(images)
            
            # Joint Optimization with Soft-Target CE
            loss_main = soft_target_ce(logits, soft_labels)
            loss_g = soft_target_ce(aux_g, soft_labels)
            loss_l = soft_target_ce(aux_l, soft_labels)
            
            loss = loss_main + 0.1 * loss_g + 0.1 * loss_l
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, soft_labels in val_loader:
                images, soft_labels = images.to(device), soft_labels.to(device)
                
                logits, _, _, _ = model(images)
                
                # To calculate accuracy, we treat the class with the highest probability 
                # in the soft label as the "ground truth" for evaluation purposes.
                _, predicted = torch.max(logits.data, 1)
                _, targets = torch.max(soft_labels, 1)
                
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        v_acc = val_correct / val_total
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, UNIQUE_WEIGHT_NAME))
            print(f"--> Saved new best FERPlus soft-label weights: {v_acc:.4f}")

if __name__ == "__main__":
    train_ferplus_soft()
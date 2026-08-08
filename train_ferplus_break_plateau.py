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
from transformer import FRITTransformer

# ------------------ Config ------------------
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE_BACKBONE = 5e-6          # Very low, but unfrozen from start
LEARNING_RATE_TRANSFORMER = 3e-4        # Higher for deeper transformer
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 7
TRANSFORMER_DEPTH = 4
DROPOUT = 0.6
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.8
PATIENCE = 15                           # Early stopping

# Paths (update these)
PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_Break_Plateau"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# ----- MixUp function -----
def mixup_data_soft(x, y_soft, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y_soft, y_soft[index]
    return mixed_x, y_a, y_b, lam

# ----- Data preparation -----
df = prepare_dataframes(PIXELS_CSV, LABELS_CSV)
train_df = df[df['Usage'] == 'Training']
val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FERPlusSoftDataset(train_df, transform=train_transform)
val_dataset = FERPlusSoftDataset(val_df, transform=val_transform)

# Weighted sampler
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

model = FRITNet(num_classes=NUM_CLASSES, transformer_depth=TRANSFORMER_DEPTH).to(device)

# Override transformer with dropout=0.6
model.transformer = FRITTransformer(
    embed_dim=128,
    num_heads=8,
    num_local_layers=TRANSFORMER_DEPTH,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)

# ----- Optimizer: SGD with Nesterov -----
# Separate LR groups
optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
    {'params': model.lfa.parameters(), 'lr': LEARNING_RATE_TRANSFORMER * 0.5},
    {'params': model.safm.parameters(), 'lr': LEARNING_RATE_TRANSFORMER * 0.5},
    {'params': model.transformer.parameters(), 'lr': LEARNING_RATE_TRANSFORMER},
], weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, nesterov=True)

# Cosine annealing with warmup
def lr_lambda(epoch):
    if epoch < 3:
        return (epoch + 1) / 3
    else:
        progress = (epoch - 3) / (EPOCHS - 3)
        return 0.5 * (1 + np.cos(np.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----- Training -----
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, soft_labels, hard_labels in pbar:
        images, soft_labels = images.to(device), soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        # MixUp
        if np.random.rand() < MIXUP_PROB:
            mixed_x, y_a, y_b, lam = mixup_data_soft(images, soft_labels, alpha=MIXUP_ALPHA)
            logits, _, aux_g, aux_l = model(mixed_x)
            loss_a = soft_target_ce(logits, y_a) + 0.1 * soft_target_ce(aux_g, y_a) + 0.1 * soft_target_ce(aux_l, y_a)
            loss_b = soft_target_ce(logits, y_b) + 0.1 * soft_target_ce(aux_g, y_b) + 0.1 * soft_target_ce(aux_l, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            # For accuracy, use dominant hard label
            index = torch.randperm(images.size(0)).to(device)
            dominant = hard_labels if lam > 0.5 else hard_labels[index]
        else:
            logits, _, aux_g, aux_l = model(images)
            loss = soft_target_ce(logits, soft_labels)
            loss += 0.1 * soft_target_ce(aux_g, soft_labels)
            loss += 0.1 * soft_target_ce(aux_l, soft_labels)
            dominant = hard_labels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(logits, 1)
        train_preds.extend(pred.cpu().numpy())
        train_targets.extend(dominant.cpu().numpy())
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
            logits, _, _, _ = model(images)
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
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_ferplus_break.pth"))
        print(f"--> Saved best model (val acc {val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print(f"Training complete. Best FERPlus validation accuracy: {best_val_acc:.4f}")
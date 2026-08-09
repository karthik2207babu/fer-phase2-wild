import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset, Dataset
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from copy import deepcopy

from model import FRITNet

# ------------------ Config ------------------
BATCH_SIZE = 128
EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 5
NUM_CLASSES = 7
TRANSFORMER_DEPTH = 6
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.0                     # DISABLED (set to 0.5 to enable)
MIXUP_START_EPOCH = 10
EMA_DECAY = 0.999
USE_SWA = True
SWA_START_EPOCH = 80
PATIENCE = 15                        # Early stopping

# Paths
FERPLUS_PIXELS = "/content/drive/MyDrive/fer2013.csv"
FERPLUS_LABELS = "/content/drive/MyDrive/fer2013new.csv"
RAFDB_TRAIN_CSV = "/content/data/Datasets/RAF-DB/train_labels.csv"
RAFDB_TRAIN_ROOT = "/content/data/Datasets/RAF-DB/DATASET/train"

SAVE_DIR = "/content/drive/MyDrive/FERPlus_Ultimate_Fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ FERPlus Dataset (soft labels) ------------------
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

# ------------------ RAF‑DB Dataset (convert to soft one‑hot) ------------------
class RAFDBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = str(self.annotations.iloc[idx, 0])
        label = int(self.annotations.iloc[idx, 1])  # 1..7
        img_path = os.path.join(self.root_dir, str(label), img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        soft_label = torch.zeros(NUM_CLASSES)
        soft_label[label - 1] = 1.0
        hard_label = label - 1
        return image, soft_label, hard_label

# ------------------ Transforms (heavy) ------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- Load FERPlus -----
ferplus_df = prepare_ferplus_data(FERPLUS_PIXELS, FERPLUS_LABELS)
ferplus_train_df = ferplus_df[ferplus_df['Usage'] == 'Training']
ferplus_val_df = ferplus_df[ferplus_df['Usage'].isin(['PublicTest', 'PrivateTest'])]

ferplus_train_dataset = FERPlusSoftDataset(ferplus_train_df, transform=train_transform)
ferplus_val_dataset = FERPlusSoftDataset(ferplus_val_df, transform=val_transform)

# ----- Load RAF‑DB -----
raf_train_dataset = RAFDBDataset(RAFDB_TRAIN_CSV, RAFDB_TRAIN_ROOT, transform=train_transform)

# ----- Combine training sets -----
combined_train = ConcatDataset([ferplus_train_dataset, raf_train_dataset])

# ----- Weighted sampler -----
ferplus_hard = [np.argmax(row) for row in ferplus_train_df['soft_label'].values]
raf_hard = []
for idx in range(len(raf_train_dataset)):
    _, _, h = raf_train_dataset[idx]
    raf_hard.append(h)
all_hard = ferplus_hard + raf_hard
class_counts = np.bincount(all_hard, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[all_hard]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(ferplus_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Combined train samples: {len(combined_train)}, FERPlus val samples: {len(ferplus_val_dataset)}")
print(f"Class counts in combined train: {class_counts}")

# ----- Model (depth=6) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FRITNet(num_classes=NUM_CLASSES, transformer_depth=TRANSFORMER_DEPTH).to(device)

# Optional: load depth-4 best weights (recommended)
checkpoint_path = "/content/drive/MyDrive/FERPlus_Depth4_AdamW/best_depth4_adamw.pth"
if os.path.exists(checkpoint_path):
    print("Loading depth-4 weights as initialization...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded.")

# ----- Optimizer -----
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
    {'params': model.lfa.parameters(), 'lr': LEARNING_RATE},
    {'params': model.safm.parameters(), 'lr': LEARNING_RATE},
    {'params': model.transformer.parameters(), 'lr': LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

# ----- Scheduler -----
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----- EMA and SWA (after optimizer) -----
ema_model = deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False

if USE_SWA:
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=LEARNING_RATE * 0.1)   # NOW optimizer is defined

# ----- Loss & MixUp functions -----
def soft_target_ce(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()

def mixup_data(x, y_soft, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y_soft, y_soft[index]
    return mixed_x, y_a, y_b, lam

def update_ema(ema_model, model, decay=EMA_DECAY):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

# ----- Training -----
best_val_acc = 0.0
best_ema_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    if epoch == 0:
        print("Stage 1: Freezing backbone")
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    elif epoch == 5:
        print("Stage 2: Unfreezing backbone")
        for param in model.parameters():
            param.requires_grad = True

    use_mixup = (MIXUP_PROB > 0 and epoch >= MIXUP_START_EPOCH)

    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, soft_labels, hard_labels in pbar:
        images, soft_labels = images.to(device), soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        if use_mixup and np.random.rand() < MIXUP_PROB:
            mixed_x, y_a, y_b, lam = mixup_data(images, soft_labels, alpha=MIXUP_ALPHA)
            logits, _, aux_g, aux_l = model(mixed_x)
            loss_a = soft_target_ce(logits, y_a) + 0.1 * soft_target_ce(aux_g, y_a) + 0.1 * soft_target_ce(aux_l, y_a)
            loss_b = soft_target_ce(logits, y_b) + 0.1 * soft_target_ce(aux_g, y_b) + 0.1 * soft_target_ce(aux_l, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            dominant = hard_labels if lam > 0.5 else hard_labels[torch.randperm(images.size(0)).to(device)]
        else:
            logits, _, aux_g, aux_l = model(images)
            loss = soft_target_ce(logits, soft_labels)
            loss += 0.1 * soft_target_ce(aux_g, soft_labels)
            loss += 0.1 * soft_target_ce(aux_l, soft_labels)
            dominant = hard_labels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema(ema_model, model)
        _, pred = torch.max(logits, 1)
        train_preds.extend(pred.cpu().numpy())
        train_targets.extend(dominant.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    train_acc = (np.array(train_preds) == np.array(train_targets)).mean()

    # Validation
    model.eval()
    ema_model.eval()
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

    # EMA validation
    ema_preds, ema_targets = [], []
    with torch.no_grad():
        for images, _, hard_labels in val_loader:
            images, hard_labels = images.to(device), hard_labels.to(device)
            logits_ema, _, _, _ = ema_model(images)
            _, pred_ema = torch.max(logits_ema, 1)
            ema_preds.extend(pred_ema.cpu().numpy())
            ema_targets.extend(hard_labels.cpu().numpy())
    ema_acc = (np.array(ema_preds) == np.array(ema_targets)).mean()

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, EMA Val Acc={ema_acc:.4f}, Val Loss={avg_val_loss:.4f}")

    scheduler.step()
    if USE_SWA and epoch >= SWA_START_EPOCH:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"--> Saved best model (val acc {val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if ema_acc > best_ema_acc:
        best_ema_acc = ema_acc
        torch.save(ema_model.state_dict(), os.path.join(SAVE_DIR, "best_ema_model.pth"))
        print(f"--> Saved best EMA model (val acc {ema_acc:.4f})")

    if epochs_no_improve >= PATIENCE and epoch > 20:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# SWA final eval
if USE_SWA:
    swa_model.eval()
    swa_preds, swa_targets = [], []
    with torch.no_grad():
        for images, _, hard_labels in val_loader:
            images, hard_labels = images.to(device), hard_labels.to(device)
            logits_swa, _, _, _ = swa_model(images)
            _, pred_swa = torch.max(logits_swa, 1)
            swa_preds.extend(pred_swa.cpu().numpy())
            swa_targets.extend(hard_labels.cpu().numpy())
    swa_acc = (np.array(swa_preds) == np.array(swa_targets)).mean()
    print(f"SWA Validation Accuracy: {swa_acc:.4f}")
    torch.save(swa_model.state_dict(), os.path.join(SAVE_DIR, "swa_model.pth"))

print(f"Training complete. Best val acc: {best_val_acc:.4f}, Best EMA: {best_ema_acc:.4f}")
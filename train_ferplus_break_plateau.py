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
from loss import NAWLoss   # we'll adapt NAW to soft labels

# ------------------ Config ------------------
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE_BACKBONE = 1e-5
LEARNING_RATE_OTHER = 5e-5
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 7

# Paths to your best model from previous run
CHECKPOINT_PATH = "/content/drive/MyDrive/FERPlus_Final_Results/best_ferplus_final.pth"
SAVE_DIR = "/content/drive/MyDrive/FERPlus_FineTune_NLA"
os.makedirs(SAVE_DIR, exist_ok=True)

# (Same dataset and dataloader code as before – reuse)
# I'll include it for completeness

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

# ----- Data preparation -----
PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"
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

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FRITNet(num_classes=NUM_CLASSES, transformer_depth=2).to(device)

# Load your best 84.24% checkpoint
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
print("Loaded best previous model.")

# ----- Loss: NAW adapted to soft labels -----
# We'll use NAWLoss, but we need to pass hard labels (majority class) for NAW weighting.
# We'll compute w_NAW from the majority class probability (p_gt) and nearest negative (p_nn).
# Then we weight the soft-target CE loss.
criterion_soft = nn.KLDivLoss(reduction='batchmean')  # actually we'll use our soft_target_ce
# We'll create a wrapper that uses NAW weights

class NLA_SoftLoss(nn.Module):
    def __init__(self, num_classes, total_epochs):
        super().__init__()
        self.naw = NAWLoss(num_classes=num_classes, total_epochs=total_epochs, lambda_param=0.5)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.naw.set_epoch(epoch)

    def forward(self, logits, soft_labels, hard_labels):
        # Compute NAW weights on hard labels
        # NAWLoss expects (logits, targets) and returns a loss, but we need weights.
        # We'll compute p_gt and p_nn manually, then use the NAW formula.
        # For simplicity, we'll use the NAWLoss's internal method to compute weights.
        # But NAWLoss doesn't expose weights directly. We can call _compute_naw_weights.
        # However, that requires p_gt, p_nn. We can compute them from logits and hard_labels.
        probs = F.softmax(logits, dim=1)
        p_gt = probs.gather(1, hard_labels.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, hard_labels.unsqueeze(1), False)
        p_nn = probs[mask].view(probs.size(0), -1).max(dim=1)[0]
        # Now compute NAW weights using the same formula as in NAWLoss
        w_star = self.naw._compute_naw_weights(p_gt, p_nn, logits.device)
        # Compute soft-target CE
        log_probs = F.log_softmax(logits, dim=1)
        soft_ce = -(soft_labels * log_probs).sum(dim=1)
        # Weighted loss
        loss = ((1.0 + w_star) * soft_ce).mean()
        return loss

loss_fn = NLA_SoftLoss(num_classes=NUM_CLASSES, total_epochs=EPOCHS).to(device)

# Optimizer
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
    {'params': model.lfa.parameters(), 'lr': LEARNING_RATE_OTHER},
    {'params': model.safm.parameters(), 'lr': LEARNING_RATE_OTHER},
    {'params': model.transformer.parameters(), 'lr': LEARNING_RATE_OTHER},
], weight_decay=WEIGHT_DECAY)

# Cosine scheduler (no warmup needed for fine-tuning)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----- Fine-tuning loop -----
best_val_acc = 0.0
for epoch in range(EPOCHS):
    loss_fn.set_epoch(epoch)
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, soft_labels, hard_labels in pbar:
        images, soft_labels = images.to(device), soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        optimizer.zero_grad()
        logits, _, aux_g, aux_l = model(images)
        # Main loss (with NLA weighting)
        loss = loss_fn(logits, soft_labels, hard_labels)
        # Aux losses (unweighted soft CE)
        loss += 0.1 * F.kl_div(F.log_softmax(aux_g, dim=1), soft_labels, reduction='batchmean')
        loss += 0.1 * F.kl_div(F.log_softmax(aux_l, dim=1), soft_labels, reduction='batchmean')
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
            logits, _, _, _ = model(images)
            # For validation, we use soft CE to monitor loss
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean()
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
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_finetuned_nla.pth"))
        print(f"--> Saved best model (val acc {val_acc:.4f})")

print(f"Fine-tuning complete. Best validation accuracy: {best_val_acc:.4f}")
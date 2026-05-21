import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import FRITNet

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running on: {DEVICE}")

# =========================================================
# ZIP + EXTRACTION
# =========================================================

ZIP_PATH = "/content/drive/MyDrive/affectnet.zip"

EXTRACT_PATH = "/content/data"

os.makedirs(EXTRACT_PATH, exist_ok=True)

print("\nExtracting AffectNet ZIP...")

os.system(
    f'unzip -q -n "{ZIP_PATH}" -d "{EXTRACT_PATH}"'
)

print("Extraction Complete")

# =========================================================
# DATASET PATHS
# =========================================================

BASE_PATH = "/content/data/affectnet/affectnet"

TRAIN_DIR = os.path.join(BASE_PATH, "Train")
TEST_DIR = os.path.join(BASE_PATH, "Test")

# =========================================================
# RAF-DB PRETRAINED WEIGHTS
# =========================================================

RAF_WEIGHTS = "/content/drive/MyDrive/FER_Phase2_Results/best_frit_weights.pth"

# =========================================================
# SAVE DIRECTORY
# =========================================================

SAVE_DIR = "/content/drive/MyDrive/AffectNet_Results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LABEL MAPPING
# =========================================================

emotion_map = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprise": 7
}

idx_to_emotion = {
    v: k for k, v in emotion_map.items()
}

NUM_CLASSES = 8

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================================================
# DATASET
# =========================================================

class AffectNetDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []

        self.transform = transform

        for emotion_name in os.listdir(root_dir):

            emotion_path = os.path.join(
                root_dir,
                emotion_name
            )

            if not os.path.isdir(emotion_path):
                continue

            emotion_name_lower = emotion_name.lower()

            if emotion_name_lower not in emotion_map:
                continue

            label = emotion_map[
                emotion_name_lower
            ]

            for img_name in os.listdir(emotion_path):

                if img_name.lower().endswith(
                    ('.jpg', '.png', '.jpeg')
                ):

                    img_path = os.path.join(
                        emotion_path,
                        img_name
                    )

                    self.samples.append(
                        (img_path, label)
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(
            img_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label

# =========================================================
# DATA LOADERS
# =========================================================

train_dataset = AffectNetDataset(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = AffectNetDataset(
    TEST_DIR,
    transform=val_transform
)

print(f"\nTrain Images: {len(train_dataset)}")
print(f"Test Images : {len(test_dataset)}")

# =========================================================
# SAFETY CHECKS
# =========================================================

if len(train_dataset) == 0:

    raise ValueError(
        "No training images found. Check AffectNet path."
    )

if len(test_dataset) == 0:

    raise ValueError(
        "No test images found. Check AffectNet path."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =========================================================
# MODEL
# =========================================================

model = FRITNet(
    num_classes=NUM_CLASSES
).to(DEVICE)

print("\nLoading RAF-DB pretrained weights...")

checkpoint = torch.load(
    RAF_WEIGHTS,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint,
    strict=False
)

print("Weights loaded successfully")

# =========================================================
# FREEZE BACKBONE INITIALLY
# =========================================================

for param in model.backbone.parameters():

    param.requires_grad = False

print("Backbone frozen")

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=5e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================================================
# TRAINING
# =========================================================

best_acc = 0.0

train_acc_history = []
val_acc_history = []

start_time = time.time()

# =========================================================
# EPOCH LOOP
# =========================================================

for epoch in range(EPOCHS):

    print("\n")
    print("===================================================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("===================================================")

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    pbar = tqdm(train_loader)

    for images, labels in pbar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, features = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            logits.data,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    train_acc = train_correct / train_total

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, features = model(images)

            _, predicted = torch.max(
                logits.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = val_correct / val_total

    scheduler.step()

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"\nTrain Accuracy      : {train_acc:.4f}")
    print(f"Validation Accuracy : {val_acc:.4f}")

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_acc > best_acc:

        best_acc = val_acc

        weight_path = os.path.join(
            SAVE_DIR,
            "best_affectnet_frit.pth"
        )

        torch.save(
            model.state_dict(),
            weight_path
        )

        print(f"Saved Best Model: {weight_path}")

# =========================================================
# TRAINING COMPLETE
# =========================================================

total_time = (
    time.time() - start_time
) / 60

print("\n")
print("===================================================")
print("TRAINING COMPLETE")
print("===================================================")

print(f"Best Validation Accuracy : {best_acc:.4f}")
print(f"Training Time (min)      : {total_time:.2f}")

# =========================================================
# SAVE TRAINING PLOT
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_acc_history,
    label="Train Accuracy"
)

plt.plot(
    val_acc_history,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("AffectNet Fine-Tuning")

plt.legend()
plt.grid(True)

plot_path = os.path.join(
    SAVE_DIR,
    "affectnet_training_plot.png"
)

plt.savefig(plot_path)

plt.close()

print(f"Saved Plot: {plot_path}")

# =========================================================
# SAVE TRAINING LOG
# =========================================================

log_path = os.path.join(
    SAVE_DIR,
    "affectnet_training_log.txt"
)

with open(log_path, "w") as f:

    f.write(
        f"Best Validation Accuracy: {best_acc:.4f}\n"
    )

    f.write(
        f"Training Time(min): {total_time:.2f}\n"
    )

print(f"Saved Logs: {log_path}")
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

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running on: {DEVICE}")

# =========================================================
# ZIP EXTRACTION
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
# SAVE DIRECTORY
# =========================================================

SAVE_DIR = "/content/drive/MyDrive/AffectNet_Scratch_Results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LABEL MAP
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

if len(train_dataset) == 0:
    raise ValueError("Training images not found")

if len(test_dataset) == 0:
    raise ValueError("Test images not found")

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

print("\nModel initialized from scratch")

# =========================================================
# LOSS
# =========================================================

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_counts = [0] * NUM_CLASSES

for _, label in train_dataset.samples:
    class_counts[label] += 1

total_samples = sum(class_counts)

class_weights = []

for count in class_counts:

    weight = total_samples / (NUM_CLASSES * count)

    class_weights.append(weight)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(DEVICE)

print("\nClass Weights:")
print(class_weights)

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================================================
# HISTORY
# =========================================================

best_acc = 0.0

train_acc_history = []
val_acc_history = []

train_loss_history = []
val_loss_history = []

log_file = os.path.join(
    SAVE_DIR,
    "affectnet_training_log.txt"
)

start_time = time.time()

# =========================================================
# TRAIN LOOP
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
    train_loss /= len(train_loader)

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, features = model(images)

            loss = criterion(
                logits,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                logits.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(test_loader)

    scheduler.step()

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    # =====================================================
    # PRINT LOGS
    # =====================================================

    log_text = (
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    print(log_text)

    with open(log_file, "a") as f:
        f.write(log_text + "\n")

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_acc > best_acc:

        best_acc = val_acc

        weight_path = os.path.join(
            SAVE_DIR,
            "best_affectnet_scratch_weights.pth"
        )

        torch.save(
            model.state_dict(),
            weight_path
        )

        print(f"Saved Best Model: {weight_path}")

# =========================================================
# SAVE ACCURACY PLOT
# =========================================================

plt.figure(figsize=(8,5))

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

plt.title("AffectNet Accuracy")

plt.legend()
plt.grid(True)

acc_plot = os.path.join(
    SAVE_DIR,
    "affectnet_accuracy_plot.png"
)

plt.savefig(acc_plot)

plt.close()

# =========================================================
# SAVE LOSS PLOT
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(
    train_loss_history,
    label="Train Loss"
)

plt.plot(
    val_loss_history,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("AffectNet Loss")

plt.legend()
plt.grid(True)

loss_plot = os.path.join(
    SAVE_DIR,
    "affectnet_loss_plot.png"
)

plt.savefig(loss_plot)

plt.close()

# =========================================================
# FINAL SUMMARY
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

print(f"\nSaved Results To: {SAVE_DIR}")
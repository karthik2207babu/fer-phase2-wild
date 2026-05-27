import os
import zipfile
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from model import FRITNet

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# =========================================================
# ZIP EXTRACTION
# =========================================================

ZIP_PATH = "/content/drive/MyDrive/masked FER-2013.zip"

EXTRACT_ROOT = "/content/data"

EXTRACTED_PATH = os.path.join(
    EXTRACT_ROOT,
    "Masked-fer2013"
)

if not os.path.exists(EXTRACTED_PATH):

    print("\nExtracting Masked FER Dataset...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_ROOT)

    print("Extraction complete.")

else:
    print("\nDataset already extracted.")

# =========================================================
# DATASET PATHS
# =========================================================

TRAIN_DIR = os.path.join(EXTRACTED_PATH, "train")

VAL_DIR = os.path.join(EXTRACTED_PATH, "validation")

SAVE_DIR = "/content/drive/MyDrive/MaskedFER_Results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# EMOTION MAP
# =========================================================

emotion_map = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3,
    "surprise": 4
}

NUM_CLASSES = 5

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASET
# =========================================================

class MaskedFERDataset(Dataset):

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

            label = emotion_map[emotion_name_lower]

            for img_name in os.listdir(emotion_path):

                if img_name.lower().endswith(
                    ('.jpg', '.jpeg', '.png')
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

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================================================
# WEIGHTED FOCAL LOSS
# =========================================================

class WeightedFocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction='none',
            weight=self.alpha
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train():

    train_dataset = MaskedFERDataset(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = MaskedFERDataset(
        VAL_DIR,
        transform=val_transform
    )

    print(f"\nTrain Images: {len(train_dataset)}")

    print(f"Validation Images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    class_counts = [0] * NUM_CLASSES

    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)

    class_weights = [
        total_samples / (NUM_CLASSES * count)
        for count in class_counts
    ]

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(DEVICE)

    print("\nClass Weights:")
    print(class_weights)

    # =====================================================
    # MODEL
    # =====================================================

    model = FRITNet(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = WeightedFocalLoss(
        alpha=class_weights,
        gamma=2.0
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_acc = 0.0

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    start_time = time.time()

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(EPOCHS):

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        train_loss = 0.0

        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader)

        for images, labels in pbar:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits, features = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()

            pbar.set_postfix(
                {'loss': f"{loss.item():.4f}"}
            )

        t_acc = train_correct / train_total

        t_loss = train_loss / len(train_loader)

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss = 0.0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                logits, features = model(images)

                loss = criterion(logits, labels)

                val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()

        v_acc = val_correct / val_total

        v_loss = val_loss / len(val_loader)

        scheduler.step()

        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        print(
            f"Epoch {epoch+1}: "
            f"T-Acc: {t_acc:.4f}, "
            f"V-Acc: {v_acc:.4f}"
        )

        # =================================================
        # SAVE BEST
        # =================================================

        if v_acc > best_val_acc:

            best_val_acc = v_acc

            save_path = os.path.join(
                SAVE_DIR,
                "best_masked_fer_weights.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"--> Saved Best Weights: "
                f"{v_acc:.4f}"
            )

    total_time = (
        time.time() - start_time
    ) / 60

    print("\n===================================")
    print("TRAINING COMPLETE")
    print("===================================")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    print(f"Training Time (min): {total_time:.2f}")

    # =====================================================
    # PLOTS
    # =====================================================

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plot_path = os.path.join(
        SAVE_DIR,
        "masked_fer_training_plot.png"
    )

    plt.savefig(plot_path)

    print(f"Saved plots to: {plot_path}")

if __name__ == "__main__":
    train()
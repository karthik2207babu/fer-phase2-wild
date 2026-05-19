import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import FRITNet

# =========================================================
# CONFIG
# =========================================================

BASE_PATH = r"C:\Users\chinn\OneDrive\Desktop\Datasets\SFEW"

MODEL_PATH = "best_frit_weights.pth"

BATCH_SIZE = 64

# =========================================================
# CLASS MAPPING
# =========================================================

CLASS_NAMES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]

folder_to_label = {
    "Surprise": 0,
    "Fear": 1,
    "Disgust": 2,
    "Happy": 3,
    "Sad": 4,
    "Angry": 5,
    "Neutral": 6
}

# =========================================================
# DATASET
# =========================================================

class FullSFEWDataset(Dataset):

    def __init__(self, base_path, transform=None):

        self.samples = []
        self.transform = transform

        splits = ["train", "val", "test"]

        for split in splits:

            split_path = os.path.join(base_path, split)

            if not os.path.exists(split_path):
                continue

            print(f"Scanning: {split_path}")

            for class_name in folder_to_label:

                class_dir = os.path.join(split_path, class_name)

                if not os.path.exists(class_dir):
                    continue

                label = folder_to_label[class_name]

                for file_name in os.listdir(class_dir):

                    if file_name.lower().endswith((
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp"
                    )):

                        img_path = os.path.join(class_dir, file_name)

                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running on: {device}")

# =========================================================
# LOAD DATASET
# =========================================================

dataset = FullSFEWDataset(
    base_path=BASE_PATH,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\nLoaded {len(dataset)} total SFEW images")

# =========================================================
# LOAD MODEL
# =========================================================

model = FRITNet(num_classes=7).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully")

# =========================================================
# EVALUATION
# =========================================================

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        logits, _ = model(images)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# =========================================================
# OVERALL ACCURACY
# =========================================================

accuracy = accuracy_score(all_labels, all_preds)

print("\n===================================")
print(f"SFEW Overall Accuracy: {accuracy:.4f}")
print("===================================\n")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES
    )
)

# =========================================================
# CLASS-WISE ACCURACY
# =========================================================

print("\nClass-wise Accuracy:\n")

cm = confusion_matrix(all_labels, all_preds)

class_acc = cm.diagonal() / cm.sum(axis=1)

for idx, acc in enumerate(class_acc):

    print(f"{CLASS_NAMES[idx]:10s}: {acc:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("SFEW Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("sfew_confusion_matrix.png")

print("\nConfusion matrix saved as sfew_confusion_matrix.png")
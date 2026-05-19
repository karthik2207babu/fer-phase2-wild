import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model import FRITNet

# =====================================================
# CONFIG
# =====================================================

TEST_DIR = r"C:\Users\chinn\OneDrive\Desktop\Datasets\FER2013\test"

MODEL_PATH = "best_frit_weights.pth"

BATCH_SIZE = 64

# =====================================================
# FER2013 LABEL MAP
# =====================================================

# Folder name -> label index
FER2013_LABELS = {
    "surprise": 0,
    "fear": 1,
    "disgust": 2,
    "happy": 3,
    "sad": 4,
    "angry": 5,
    "neutral": 6
}

EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

# =====================================================
# CUSTOM DATASET
# =====================================================

class FER2013FolderDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []
        self.transform = transform

        for folder_name in os.listdir(root_dir):

            folder_path = os.path.join(root_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            if folder_name not in FER2013_LABELS:
                continue

            label = FER2013_LABELS[folder_name]

            for file_name in os.listdir(folder_path):

                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):

                    img_path = os.path.join(folder_path, file_name)

                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# TRANSFORMS
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")

# =====================================================
# DATA
# =====================================================

test_dataset = FER2013FolderDataset(
    root_dir=TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Loaded {len(test_dataset)} FER2013 test images")

# =====================================================
# MODEL
# =====================================================

model = FRITNet(num_classes=7).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully")

# =====================================================
# INFERENCE
# =====================================================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        logits, _ = model(images)

        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

print("\n===================================")
print(f"FER2013 Accuracy: {accuracy:.4f}")
print("===================================\n")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print(classification_report(
    y_true,
    y_pred,
    target_names=EMOTIONS
))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))

plt.imshow(cm, cmap="Blues")

plt.title("FER2013 Confusion Matrix")

plt.colorbar()

plt.xticks(
    np.arange(len(EMOTIONS)),
    EMOTIONS,
    rotation=45
)

plt.yticks(
    np.arange(len(EMOTIONS)),
    EMOTIONS
)

# annotate cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("fer2013_confusion_matrix.png")

print("Confusion matrix saved as fer2013_confusion_matrix.png")

plt.show()
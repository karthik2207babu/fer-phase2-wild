import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import RAFDBDataset
from model import FRITNet

# =========================================
# CONFIG
# =========================================

BATCH_SIZE = 64
MODEL_PATH = "best_frit_weights.pth"

# ✅ LOCAL RAF-DB PATH
BASE_PATH = r"C:\Users\chinn\OneDrive\Desktop\Datasets\RAF-DB"

VAL_CSV = BASE_PATH + r"\test_labels.csv"
VAL_ROOT = BASE_PATH + r"\DATASET\test"

EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

# =========================================
# DEVICE
# =========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# =========================================
# DATA
# =========================================

val_dataset = RAFDBDataset(
    csv_file=VAL_CSV,
    root_dir=VAL_ROOT,
    phase='val'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0   # safer on Windows
)

print(f"Loaded {len(val_dataset)} validation images")

# =========================================
# MODEL
# =========================================

model = FRITNet(num_classes=7).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully")

# =========================================
# INFERENCE
# =========================================

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)

        preds = torch.argmax(logits, dim=1)

        # RAF labels start from 1
        y_true.extend((labels - 1).cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================================
# OVERALL ACCURACY
# =========================================

overall_acc = np.mean(y_true == y_pred)

print("\n===================================")
print(f"Overall Accuracy: {overall_acc:.4f}")
print("===================================\n")

# =========================================
# CLASS-WISE ACCURACY
# =========================================

print("Class-wise Accuracy:\n")

for i, emotion in enumerate(EMOTIONS):

    total = np.sum(y_true == i)

    correct = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    acc = correct / total if total > 0 else 0

    print(f"{emotion:10s}: {acc:.4f}")

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

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

# Annotate cells
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

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as confusion_matrix.png")

plt.show()

# =========================================
# FULL CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS
    )
)
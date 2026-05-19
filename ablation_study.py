import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

from dataset import RAFDBDataset
from loss import CombinedFERLoss

from backbone import TruncatedFaceNet
from lfa import LFAModule
from multiscale import MultiScaleGlobalConv
from safm import SAFM
from transformer import FRITTransformer

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-4

BASE_PATH = "/content/data/Datasets/RAF-DB"

TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")

TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

SAVE_DIR = "/content/drive/MyDrive/FER_Ablation_Results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running on: {device}")

# =========================================================
# DATA
# =========================================================

train_dataset = RAFDBDataset(
    csv_file=TRAIN_CSV,
    root_dir=TRAIN_ROOT,
    phase='train'
)

val_dataset = RAFDBDataset(
    csv_file=VAL_CSV,
    root_dir=VAL_ROOT,
    phase='val'
)

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

print(f"Train Images: {len(train_dataset)}")
print(f"Validation Images: {len(val_dataset)}")

# =========================================================
# MODEL VARIANTS
# =========================================================

class BaselineNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.bridge = nn.Sequential(
            nn.Conv2d(1792, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.bridge(x)

        x = F.interpolate(
            x,
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        )

        logits, features = self.transformer(x)

        return logits, features


class LFANet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.lfa = LFAModule()

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.lfa(x)

        logits, features = self.transformer(x)

        return logits, features


class LFAMultiScaleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.lfa = LFAModule()

        self.multiscale = MultiScaleGlobalConv()

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.lfa(x)

        x = self.multiscale(x)

        logits, features = self.transformer(x)

        return logits, features


class FullFRITNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.lfa = LFAModule()

        self.multiscale = MultiScaleGlobalConv()

        self.safm = SAFM()

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.lfa(x)

        x = self.multiscale(x)

        x = self.safm(x)

        logits, features = self.transformer(x)

        return logits, features

# =========================================================
# VARIANTS
# =========================================================

variants = {
    "Baseline": BaselineNet,
    "LFA": LFANet,
    "LFA_MultiScale": LFAMultiScaleNet,
    "Full_FRIT": FullFRITNet
}

# =========================================================
# RESULTS
# =========================================================

results = []

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_variant(name, model_class):

    print("\n")
    print("===================================================")
    print(f"TRAINING VARIANT: {name}")
    print("===================================================")

    model = model_class().to(device)

    criterion = CombinedFERLoss(
        feat_dim=128
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # =====================================================
    # GFLOPS
    # =====================================================

    macs, params = get_model_complexity_info(
        model,
        (3, 224, 224),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"MACs/GFLOPs : {macs}")
    print(f"Parameters  : {params}")

    best_acc = 0.0

    train_history = []
    val_history = []

    start_time = time.time()

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"{name} Epoch {epoch+1}/{EPOCHS}"
        )

        for images, labels in pbar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, features = model(images)

            loss = criterion(
                logits,
                features,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            train_total += labels.size(0)

            train_correct += (
                predicted == (labels - 1)
            ).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits, features = model(images)

                _, predicted = torch.max(logits.data, 1)

                val_total += labels.size(0)

                val_correct += (
                    predicted == (labels - 1)
                ).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_history.append(train_acc)
        val_history.append(val_acc)

        scheduler.step()

        print(
            f"Epoch {epoch+1}: "
            f"T-Acc={train_acc:.4f}, "
            f"V-Acc={val_acc:.4f}"
        )

        # =================================================
        # SAVE BEST
        # =================================================

        if val_acc > best_acc:

            best_acc = val_acc

            weight_path = os.path.join(
                SAVE_DIR,
                f"{name}_best.pth"
            )

            torch.save(
                model.state_dict(),
                weight_path
            )

            print(f"Saved Best Weights: {weight_path}")

    # =====================================================
    # TIME
    # =====================================================

    total_time = (time.time() - start_time) / 60

    # =====================================================
    # SAVE PLOT
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(train_history, label='Train Acc')
    plt.plot(val_history, label='Val Acc')

    plt.title(f"{name} Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(
        SAVE_DIR,
        f"{name}_plot.png"
    )

    plt.savefig(plot_path)

    plt.close()

    print(f"Saved Plot: {plot_path}")

    # =====================================================
    # STORE RESULTS
    # =====================================================

    results.append({
        "Variant": name,
        "Accuracy": round(best_acc, 4),
        "MACs": macs,
        "Params": params,
        "TrainTime": round(total_time, 2)
    })

# =========================================================
# RUN ALL VARIANTS
# =========================================================

for variant_name, variant_class in variants.items():

    train_variant(
        variant_name,
        variant_class
    )

# =========================================================
# FINAL TABLE
# =========================================================

print("\n")
print("==============================================================")
print("FINAL ABLATION RESULTS")
print("==============================================================")

print(
    f"{'Variant':<20}"
    f"{'Accuracy':<15}"
    f"{'MACs/GFLOPs':<20}"
    f"{'Parameters':<20}"
    f"{'TrainTime(min)':<20}"
)

for r in results:

    print(
        f"{r['Variant']:<20}"
        f"{r['Accuracy']:<15}"
        f"{r['MACs']:<20}"
        f"{r['Params']:<20}"
        f"{r['TrainTime']:<20}"
    )

# =========================================================
# SAVE RESULTS FILE
# =========================================================

results_path = os.path.join(
    SAVE_DIR,
    "ablation_results.txt"
)

with open(results_path, "w") as f:

    f.write("FINAL ABLATION RESULTS\n")
    f.write("====================================================\n\n")

    for r in results:

        f.write(f"Variant: {r['Variant']}\n")
        f.write(f"Accuracy: {r['Accuracy']}\n")
        f.write(f"MACs/GFLOPs: {r['MACs']}\n")
        f.write(f"Parameters: {r['Params']}\n")
        f.write(f"Training Time(min): {r['TrainTime']}\n")
        f.write("--------------------------------------------------\n")

print("\n==============================================================")
print("ABLATION STUDY COMPLETE")
print("==============================================================")

print(f"\nSaved Results To: {results_path}")
print(f"Saved Everything To: {SAVE_DIR}")
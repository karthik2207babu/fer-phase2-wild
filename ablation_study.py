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

from model import (
    TruncatedFaceNet,
    LFAModule,
    MultiScaleModule,
    SAFM,
    FRITTransformer
)

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

# =========================================================
# SAVE DIRECTORY
# =========================================================

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
# DATASETS
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
print(f"Val Images: {len(val_dataset)}")

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


class LFAMGTCNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.lfa = LFAModule()

        self.mgtc = MultiScaleModule()

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.lfa(x)

        x = self.mgtc(x)

        logits, features = self.transformer(x)

        return logits, features


class FullFRITNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = TruncatedFaceNet()

        self.lfa = LFAModule()

        self.mgtc = MultiScaleModule()

        self.safm = SAFM()

        self.transformer = FRITTransformer()

    def forward(self, x):

        x = self.backbone(x)

        x = self.lfa(x)

        x = self.mgtc(x)

        x = self.safm(x)

        logits, features = self.transformer(x)

        return logits, features

# =========================================================
# VARIANTS
# =========================================================

variants = {
    "Baseline": BaselineNet,
    "LFA": LFANet,
    "LFA_MGTC": LFAMGTCNet,
    "Full_FRIT": FullFRITNet
}

# =========================================================
# RESULTS STORAGE
# =========================================================

final_results = []

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_variant(name, model_class):

    print("\n")
    print("===================================================")
    print(f"TRAINING VARIANT: {name}")
    print("===================================================")

    model = model_class().to(device)

    # =====================================================
    # FREEZE BACKBONE
    # =====================================================

    for param in model.backbone.parameters():
        param.requires_grad = False

    # =====================================================
    # LOSS
    # =====================================================

    criterion = CombinedFERLoss(
        feat_dim=128
    ).to(device)

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # =====================================================
    # GFLOPS + PARAMS
    # =====================================================

    macs, params = get_model_complexity_info(
        model,
        (3, 224, 224),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"GFLOPs/MACs : {macs}")
    print(f"Parameters  : {params}")

    # =====================================================
    # HISTORY
    # =====================================================

    history_train = []
    history_val = []

    best_acc = 0.0

    start_time = time.time()

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(EPOCHS):

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0.0

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

            preds = torch.argmax(logits, dim=1)

            train_total += labels.size(0)

            train_correct += (
                preds == (labels - 1)
            ).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
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

                preds = torch.argmax(logits, dim=1)

                val_total += labels.size(0)

                val_correct += (
                    preds == (labels - 1)
                ).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history_train.append(train_acc)
        history_val.append(val_acc)

        scheduler.step()

        print(
            f"\nEpoch {epoch+1}: "
            f"T-Acc={train_acc:.4f}, "
            f"V-Acc={val_acc:.4f}"
        )

        # =================================================
        # SAVE BEST MODEL
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
    # TRAINING TIME
    # =====================================================

    total_time = time.time() - start_time

    # =====================================================
    # SAVE PLOT
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(history_train, label='Train Acc')
    plt.plot(history_val, label='Val Acc')

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

    final_results.append({
        "Variant": name,
        "Best Accuracy": round(best_acc, 4),
        "GFLOPs/MACs": macs,
        "Parameters": params,
        "Training Time (min)": round(total_time / 60, 2)
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
# FINAL RESULTS TABLE
# =========================================================

print("\n")
print("==============================================================")
print("FINAL ABLATION RESULTS")
print("==============================================================")

print(
    f"{'Variant':<15}"
    f"{'Accuracy':<15}"
    f"{'GFLOPs/MACs':<20}"
    f"{'Parameters':<20}"
    f"{'Train Time(min)':<20}"
)

for result in final_results:

    print(
        f"{result['Variant']:<15}"
        f"{result['Best Accuracy']:<15}"
        f"{result['GFLOPs/MACs']:<20}"
        f"{result['Parameters']:<20}"
        f"{result['Training Time (min)']:<20}"
    )

# =========================================================
# SAVE RESULTS FILE
# =========================================================

results_file = os.path.join(
    SAVE_DIR,
    "ablation_results.txt"
)

with open(results_file, "w") as f:

    f.write("FINAL ABLATION RESULTS\n")
    f.write("====================================================\n\n")

    for result in final_results:

        line = (
            f"Variant: {result['Variant']}\n"
            f"Accuracy: {result['Best Accuracy']}\n"
            f"GFLOPs/MACs: {result['GFLOPs/MACs']}\n"
            f"Parameters: {result['Parameters']}\n"
            f"Training Time(min): {result['Training Time (min)']}\n"
            f"--------------------------------------------------\n"
        )

        f.write(line)

print("\n==============================================================")
print("ABLATION STUDY COMPLETE")
print("==============================================================")

print(f"\nSaved Results File: {results_file}")
print(f"Saved Everything To: {SAVE_DIR}")
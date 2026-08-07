"""
=================================================================
FRITNet Ablation Study — Existing Components Only
Variants: Bridge, LFA, SAFM, Transformer Depth, L2Norm Head, 
          Backbone Freezing, Auxiliary Loss
=================================================================
"""
import argparse
import os
import random
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from backbone import TruncatedFaceNet
from lfa import LFAModule
from safm import SAFM
from transformer import FRITTransformer
from dataset import RAFDBDataset

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
NUM_CLASSES = 7

BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

SAVE_DIR = "/content/drive/MyDrive/FER_Ablation_Results"
os.makedirs(SAVE_DIR, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# EXISTING COMPONENTS FROM YOUR CODEBASE
# ------------------------------------------------------------------

class L2NormLinear(nn.Module):
    """Already exists in your model.py — just copied here for convenience."""
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


# ------------------------------------------------------------------
# ABLATION VARIANTS (ALL EXISTING COMPONENTS)
# ------------------------------------------------------------------

class BaselineVariant(nn.Module):
    """Bridge + Transformer. No LFA, no SAFM."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.bridge = nn.Sequential(
            nn.Conv2d(1792, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


class LFAVariant(nn.Module):
    """LFA replaces Bridge."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


class SAFMVariant(nn.Module):
    """Bridge + SAFM."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.bridge = nn.Sequential(
            nn.Conv2d(1792, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        x = self.safm(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


class FullVariant(nn.Module):
    """Your current FRITNet: LFA + SAFM + Transformer."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


class DeepTransformerVariant(nn.Module):
    """Full + Deeper Transformer (3 layers instead of 2)."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=3,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


class L2NormHeadVariant(nn.Module):
    """Full + L2Norm Classification Head (already in your model.py)."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = L2NormLinear(128, NUM_CLASSES, scale=30.0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        logits = self.classifier(features)
        return logits, features, aux_g, aux_l


class UnfrozenBackboneVariant(nn.Module):
    """Full but backbone is NOT frozen."""
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained="vggface2", freeze_early_layers=False)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5,
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, features, aux_g, aux_l = self.transformer(x)
        return logits, features, aux_g, aux_l


# ==================================================================
# TRAINING LOOP
# ==================================================================

def build_loaders(batch_size, num_workers=2):
    train_ds = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase="train")
    val_ds = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase="val")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_variant(name, model_class, train_loader, val_loader, epochs, use_aux=False, max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")

    model = model_class().to(device)

    # Freeze backbone only if the model was initialized with freeze_early_layers=True
    # For UnfrozenBackboneVariant, we skip this
    if not isinstance(model, UnfrozenBackboneVariant):
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Layer-wise LR
    param_groups = []
    if hasattr(model, 'lfa'):
        param_groups.append({'params': model.lfa.parameters(), 'lr': LEARNING_RATE})
    if hasattr(model, 'safm'):
        param_groups.append({'params': model.safm.parameters(), 'lr': LEARNING_RATE})
    param_groups.append({'params': model.transformer.parameters(), 'lr': LEARNING_RATE})
    param_groups.append({'params': model.classifier.parameters(), 'lr': LEARNING_RATE})

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"{name} | Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = labels.to(device) - 1

            optimizer.zero_grad()
            logits, features, aux_g, aux_l = model(images)

            loss = criterion(logits, targets)
            if use_aux and aux_g is not None and aux_l is not None:
                loss = loss + 0.2 * criterion(aux_g, targets) + 0.2 * criterion(aux_l, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            train_total += targets.size(0)
            train_correct += (pred == targets).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break

                images = images.to(device)
                targets = labels.to(device) - 1

                logits, _, _, _ = model(images)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                val_total += targets.size(0)
                val_correct += (pred == targets).sum().item()

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = val_correct / max(val_total, 1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Per-class breakdown for minority classes
        cls_accs = {}
        for c in range(NUM_CLASSES):
            mask = np.array(all_targets) == c
            if mask.sum() > 0:
                cls_accs[c] = float((np.array(all_preds)[mask] == c).sum() / mask.sum())
            else:
                cls_accs[c] = 0.0

        print(
            f"Epoch {epoch+1}: Train={train_acc:.4f} | Val={val_acc:.4f} | "
            f"Fear={cls_accs.get(1,0):.3f} | Disgust={cls_accs.get(2,0):.3f} | "
            f"Happy={cls_accs.get(3,0):.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_best.pth"))

        scheduler.step()

    elapsed = (time.time() - start) / 60.0
    return {
        'variant': name,
        'best_val_acc': round(best_val_acc, 4),
        'train_history': history['train_acc'],
        'val_history': history['val_acc'],
        'elapsed_min': round(elapsed, 2),
    }


# ==================================================================
# MAIN
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="FRITNet Ablation — Existing Components Only")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick", action="store_true", help="Smoke test: 2 epochs, 3 batches")
    parser.add_argument("--aux", action="store_true", help="Use auxiliary global/local losses")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(args.batch_size)

    epochs = 2 if args.quick else args.epochs
    max_batches = 3 if args.quick else None

    variants = {
        '01_Baseline': BaselineVariant,
        '02_LFA': LFAVariant,
        '03_SAFM': SAFMVariant,
        '04_LFA_SAFM': FullVariant,
        '05_LFA_SAFM_Deep': DeepTransformerVariant,
        '06_LFA_SAFM_L2Norm': L2NormHeadVariant,
        '07_LFA_SAFM_Unfrozen': UnfrozenBackboneVariant,
    }

    results = []
    for name, model_cls in variants.items():
        result = train_variant(
            name=name,
            model_class=model_cls,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            use_aux=args.aux,
            max_batches=max_batches,
        )
        results.append(result)

    # Leaderboard
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<5} {'Variant':<30} {'Val Acc':<10} {'Time(min)':<10}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['variant']:<30} {r['best_val_acc']:<10.4f} {r['elapsed_min']:<10.2f}")

    # Save
    out_path = os.path.join(SAVE_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
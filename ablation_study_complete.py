# ablation_study_complete.py
import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

# Import your custom modules
from backbone import TruncatedFaceNet
from config import BASE_PATH, TRAIN_CSV, TRAIN_ROOT, VAL_CSV, VAL_ROOT
from dataset import RAFDBDataset
from lfa import LFAModule
from safm import SAFM
from transformer import FRITTransformer
from loss import NAWLoss   # only NAWLoss is available

SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
NUM_CLASSES = 7
USE_WEIGHTED_SAMPLER = True   # set to False for standard shuffle

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_save_dir():
    if os.path.exists("/content/drive"):
        save_dir = "/content/drive/MyDrive/FER_Ablation_Results"
    else:
        save_dir = os.path.join(os.getcwd(), "ablation_results")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# ----------------------------------------------------------------------
#  Define model variants
# ----------------------------------------------------------------------

# 1. Baseline: No LFA, no SAFM, no transformer (just global average pool)
class BaselineNoAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.bridge = nn.Sequential(
            nn.Conv2d(1792, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        return self.classifier(x)

# 2. LFA only (no SAFM, no transformer)
class LFAOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)               # (B, 128, 28, 28)
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        return self.classifier(x)

# 3. LFA + SAFM (no transformer)
class LFA_SAFM_NoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        return self.classifier(x)

# 4. Full FRITNet (LFA + SAFM + Transformer) – your original
class FRITNetFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=2,
            num_classes=NUM_CLASSES, dropout=0.5
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, _, _, _ = self.transformer(x)
        return logits

# 5. Deeper Transformer (4 layers) – to test overfitting capacity
class FRITNetDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        self.safm = SAFM(kernel_size=7)
        self.transformer = FRITTransformer(
            embed_dim=128, num_heads=8, num_local_layers=4,
            num_classes=NUM_CLASSES, dropout=0.5
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)
        logits, _, _, _ = self.transformer(x)
        return logits

# ----------------------------------------------------------------------
#  Training function (with optional NLA loss)
# ----------------------------------------------------------------------
def train_variant(name, model_class, train_loader, val_loader, save_dir, epochs,
                  use_nla=False, use_weighted_sampler=USE_WEIGHTED_SAMPLER):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nVariant: {name}  |  NLA: {use_nla}  |  Weighted: {use_weighted_sampler}\n{'='*70}")

    model = model_class().to(device)

    # Loss
    if use_nla:
        criterion = NAWLoss(num_classes=NUM_CLASSES, total_epochs=epochs, lambda_param=0.5).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_recall = 0.0
    best_epoch = 0
    best_per_class = {}
    train_hist, val_hist = [], []
    start_time = time.time()

    for epoch in range(epochs):
        if use_nla:
            criterion.set_epoch(epoch)   # needed for NAW scheduling

        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        pbar = tqdm(train_loader, desc=f"{name} E{epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1   # RAF-DB labels are 1..7

            optimizer.zero_grad()
            logits = model(images)

            if use_nla:
                # NAWLoss expects (logits, targets) – no aux needed for ablation
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(train_targets, train_preds)
        train_recall = recall_score(train_targets, train_preds, average='macro', zero_division=0)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1
                logits = model(images)
                _, pred = torch.max(logits, 1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)
        per_class = {}
        for c in range(NUM_CLASSES):
            mask = (np.array(val_targets) == c)
            if mask.sum() > 0:
                per_class[c] = (np.array(val_preds)[mask] == c).sum() / mask.sum()
            else:
                per_class[c] = 0.0

        scheduler.step()
        train_hist.append(train_acc)
        val_hist.append(val_acc)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Macro Recall={val_recall:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_recall = val_recall
            best_epoch = epoch + 1
            best_per_class = per_class
            torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_best.pth"))

    elapsed = time.time() - start_time

    # Return summary
    return {
        "variant": name,
        "use_nla": use_nla,
        "use_weighted": use_weighted_sampler,
        "best_val_acc": best_val_acc,
        "best_val_recall": best_val_recall,
        "best_epoch": best_epoch,
        "per_class": best_per_class,
        "elapsed_min": elapsed / 60.0,
        "train_hist": train_hist,
        "val_hist": val_hist,
    }

# ----------------------------------------------------------------------
#  Data loader builder (with optional WeightedRandomSampler)
# ----------------------------------------------------------------------
def build_data_loaders(batch_size, use_weighted_sampler=True, num_workers=2):
    train_dataset = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    if use_weighted_sampler:
        # Compute class weights
        labels = train_dataset.annotations.iloc[:, 1].values - 1
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick", action="store_true", help="run 1 epoch for sanity")
    parser.add_argument("--no-weighted", action="store_true", help="disable weighted sampler")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    save_dir = get_save_dir()
    epochs = 1 if args.quick else args.epochs
    use_weighted = not args.no_weighted

    # Build data loaders once (weighted or not)
    train_loader, val_loader = build_data_loaders(args.batch_size, use_weighted_sampler=use_weighted)

    # Define all experiments: (name, model_class, use_nla)
    experiments = [
        ("Baseline_NoAttn", BaselineNoAttn, False),
        ("LFA_Only", LFAOnly, False),
        ("LFA_SAFM_NoTrans", LFA_SAFM_NoTransformer, False),
        ("FRITNet_Full", FRITNetFull, False),
        ("FRITNet_Deep", FRITNetDeep, False),
        ("FRITNet_Full_NLA", FRITNetFull, True),
        ("FRITNet_Deep_NLA", FRITNetDeep, True),
    ]

    results = []
    for name, model_class, use_nla in experiments:
        res = train_variant(
            name=name,
            model_class=model_class,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            epochs=epochs,
            use_nla=use_nla,
            use_weighted_sampler=use_weighted
        )
        results.append(res)

    # Save summary to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)

    # Print leaderboard
    print("\n" + "="*80)
    print("ABLATION LEADERBOARD (RAF-DB)")
    print("="*80)
    sorted_df = df.sort_values("best_val_acc", ascending=False)
    print(sorted_df[["variant", "best_val_acc", "best_val_recall", "elapsed_min"]].to_string(index=False))

    # Also save per-class breakdown
    with open(os.path.join(save_dir, "per_class_accuracy.txt"), "w") as f:
        for res in results:
            f.write(f"{res['variant']} (NLA={res['use_nla']}, weighted={res['use_weighted']})\n")
            for c, acc in res['per_class'].items():
                f.write(f"  Class {c}: {acc:.4f}\n")
            f.write("\n")

    print(f"\nAll results saved to {save_dir}")

if __name__ == "__main__":
    main()
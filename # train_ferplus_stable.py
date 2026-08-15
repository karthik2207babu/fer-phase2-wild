# train_ferplus_stable.py

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import FRITNet


# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 64
EPOCHS = 60

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 12

NUM_CLASSES = 7
TRANSFORMER_DEPTH = 2

SEED = 42

PIXELS_CSV = "/content/drive/MyDrive/fer2013.csv"
LABELS_CSV = "/content/drive/MyDrive/fer2013new.csv"

SAVE_DIR = "/content/drive/MyDrive/FERPlus_Stable_20260815_v1"
BEST_WEIGHTS = os.path.join(
    SAVE_DIR,
    "best_ferplus_stable_7cls_20260815.pth"
)
LOG_PATH = os.path.join(
    SAVE_DIR,
    "training_log_ferplus_stable_7cls_20260815.csv"
)
PLOT_PATH = os.path.join(
    SAVE_DIR,
    "training_curve_ferplus_stable_7cls_20260815.png"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# =========================================================
# FERPLUS LABEL DEFINITIONS
# =========================================================

# Current project formulation: 7 classes.
RAF_DB_ORDER = [
    "surprise",
    "fear",
    "disgust",
    "happiness",
    "sadness",
    "anger",
    "neutral"
]

ALL_VOTES = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
    "unknown",
    "NF"
]


# =========================================================
# DATASET
# =========================================================

class FERPlusSoftDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pixels = np.fromstring(
            row["pixels"],
            sep=" ",
            dtype=np.uint8
        ).reshape(48, 48)

        image = Image.fromarray(pixels).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        soft_label = torch.tensor(
            row["soft_label"],
            dtype=torch.float32
        )

        hard_label = int(torch.argmax(soft_label).item())

        return image, soft_label, hard_label


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_dataframes(pixels_path, labels_path):
    print("\nLoading FERPlus CSVs...")

    pixels_df = pd.read_csv(pixels_path)
    labels_df = pd.read_csv(labels_path)

    if len(pixels_df) != len(labels_df):
        raise ValueError(
            f"CSV length mismatch: "
            f"pixels={len(pixels_df)}, labels={len(labels_df)}"
        )

    df = pd.concat(
        [
            pixels_df[["pixels"]],
            labels_df
        ],
        axis=1
    )

    valid_rows = []

    for _, row in df.iterrows():

        total_votes = sum(float(row[c]) for c in ALL_VOTES)

        if total_votes <= 0:
            continue

        unknown_fraction = (
            float(row["unknown"]) + float(row["NF"])
        ) / total_votes

        if unknown_fraction > 0.5:
            continue

        votes = np.array(
            [float(row[c]) for c in RAF_DB_ORDER],
            dtype=np.float32
        )

        vote_sum = votes.sum()

        if vote_sum <= 0:
            continue

        soft_label = votes / vote_sum

        valid_rows.append(
            {
                "pixels": row["pixels"],
                "Usage": row["Usage"],
                "soft_label": soft_label
            }
        )

    final_df = pd.DataFrame(valid_rows)

    print(f"Valid samples: {len(final_df)} / {len(df)}")

    return final_df


# =========================================================
# TRANSFORMS
# =========================================================

# Keep augmentation much closer to the clean FER/MRAN-style setup.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224), padding=8),
    transforms.RandomHorizontalFlip(),
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
# SOFT-TARGET LOSS
# =========================================================

def soft_target_ce(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)

    return -(
        soft_labels * log_probs
    ).sum(dim=1).mean()


def total_loss(
    logits,
    aux_global,
    aux_local,
    soft_labels
):
    main_loss = soft_target_ce(
        logits,
        soft_labels
    )

    global_loss = soft_target_ce(
        aux_global,
        soft_labels
    )

    local_loss = soft_target_ce(
        aux_local,
        soft_labels
    )

    return (
        main_loss
        + 0.1 * global_loss
        + 0.1 * local_loss
    )


# =========================================================
# LR SCHEDULER
# =========================================================

def build_scheduler(optimizer):
    def lr_lambda(epoch):

        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)

        progress = (
            epoch - WARMUP_EPOCHS
        ) / max(
            1,
            EPOCHS - WARMUP_EPOCHS
        )

        return 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


# =========================================================
# MAIN TRAINING
# =========================================================

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("FERPLUS STABLE SOFT-LABEL TRAINING")
    print("=" * 70)
    print(f"Device      : {device}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Backbone LR : {BACKBONE_LR}")
    print(f"Head LR     : {HEAD_LR}")
    print(f"Save dir    : {SAVE_DIR}")
    print("=" * 70)

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    df = prepare_dataframes(
        PIXELS_CSV,
        LABELS_CSV
    )

    train_df = df[
        df["Usage"] == "Training"
    ].reset_index(drop=True)

    val_df = df[
        df["Usage"].isin(
            ["PublicTest", "PrivateTest"]
        )
    ].reset_index(drop=True)

    train_dataset = FERPlusSoftDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = FERPlusSoftDataset(
        val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nTrain samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = FRITNet(
        num_classes=NUM_CLASSES,
        transformer_depth=TRANSFORMER_DEPTH
    ).to(device)

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.lfa.parameters(),
                "lr": HEAD_LR
            },
            {
                "params": model.safm.parameters(),
                "lr": HEAD_LR
            },
            {
                "params": model.transformer.parameters(),
                "lr": HEAD_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = build_scheduler(
        optimizer
    )

    # -----------------------------------------------------
    # TWO-STAGE FINE-TUNING
    # -----------------------------------------------------

    for param in model.backbone.parameters():
        param.requires_grad = False

    print("\nStage 1:")
    print("Backbone frozen.")
    print("LFA + SAFM + Transformer trainable.")

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    with open(LOG_PATH, "w") as log_file:
        log_file.write(
            "epoch,"
            "lr_backbone,"
            "lr_head,"
            "train_loss,"
            "train_acc,"
            "val_loss,"
            "val_acc\n"
        )

        # -------------------------------------------------
        # EPOCH LOOP
        # -------------------------------------------------

        for epoch in range(EPOCHS):

            # ---------------------------------------------
            # UNFREEZE BACKBONE
            # ---------------------------------------------

            if epoch == WARMUP_EPOCHS:

                print("\nStage 2:")
                print("Unfreezing backbone.")

                for param in model.backbone.parameters():
                    param.requires_grad = True

            # ---------------------------------------------
            # TRAIN
            # ---------------------------------------------

            model.train()

            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{EPOCHS}"
            )

            for images, soft_labels, hard_labels in pbar:

                images = images.to(
                    device,
                    non_blocking=True
                )

                soft_labels = soft_labels.to(
                    device,
                    non_blocking=True
                )

                hard_labels = hard_labels.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits, _, aux_global, aux_local = model(
                    images
                )

                loss = total_loss(
                    logits,
                    aux_global,
                    aux_local,
                    soft_labels
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

                train_loss_sum += loss.item()

                predicted = torch.argmax(
                    logits,
                    dim=1
                )

                train_correct += (
                    predicted == hard_labels
                ).sum().item()

                train_total += (
                    hard_labels.size(0)
                )

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            # ---------------------------------------------
            # VALIDATION
            # ---------------------------------------------

            model.eval()

            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for (
                    images,
                    soft_labels,
                    hard_labels
                ) in val_loader:

                    images = images.to(
                        device,
                        non_blocking=True
                    )

                    soft_labels = soft_labels.to(
                        device,
                        non_blocking=True
                    )

                    hard_labels = hard_labels.to(
                        device,
                        non_blocking=True
                    )

                    logits, _, aux_global, aux_local = model(
                        images
                    )

                    loss = total_loss(
                        logits,
                        aux_global,
                        aux_local,
                        soft_labels
                    )

                    val_loss_sum += loss.item()

                    predicted = torch.argmax(
                        logits,
                        dim=1
                    )

                    val_correct += (
                        predicted == hard_labels
                    ).sum().item()

                    val_total += (
                        hard_labels.size(0)
                    )

            # ---------------------------------------------
            # METRICS
            # ---------------------------------------------

            train_loss = (
                train_loss_sum /
                max(1, len(train_loader))
            )

            val_loss = (
                val_loss_sum /
                max(1, len(val_loader))
            )

            train_acc = (
                train_correct /
                max(1, train_total)
            )

            val_acc = (
                val_correct /
                max(1, val_total)
            )

            current_backbone_lr = optimizer.param_groups[0]["lr"]
            current_head_lr = optimizer.param_groups[1]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"\nEpoch {epoch + 1:03d} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            log_file.write(
                f"{epoch + 1},"
                f"{current_backbone_lr:.8e},"
                f"{current_head_lr:.8e},"
                f"{train_loss:.6f},"
                f"{train_acc:.6f},"
                f"{val_loss:.6f},"
                f"{val_acc:.6f}\n"
            )

            log_file.flush()

            # ---------------------------------------------
            # SAVE BEST
            # ---------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    BEST_WEIGHTS
                )

                print(
                    f"--> NEW BEST: "
                    f"{best_val_acc * 100:.2f}%"
                )

                print(
                    f"--> Saved: {BEST_WEIGHTS}"
                )

            else:
                epochs_without_improvement += 1

            # ---------------------------------------------
            # EARLY STOPPING
            # ---------------------------------------------

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

            scheduler.step()

    # =====================================================
    # FINAL
    # =====================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Best FERPlus validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )
    print(
        f"Best weights: {BEST_WEIGHTS}"
    )
    print(
        f"Log: {LOG_PATH}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FERPlus Stable Soft-Label Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_PATH,
        dpi=150
    )
    plt.close()

    print(
        f"Training curve saved to: {PLOT_PATH}"
    )


if __name__ == "__main__":
    train()
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

# Import your custom architecture components
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

ZIP_PATH = "/content/drive/MyDrive/affectnet.zip"
EXTRACT_PATH = "/content/data"

os.makedirs(EXTRACT_PATH, exist_ok=True)

print("\nExtracting AffectNet ZIP...")
# -q: quiet, -n: never overwrite existing files
os.system(f'unzip -q -n "{ZIP_PATH}" -d "{EXTRACT_PATH}"')
print("Extraction Complete")

# =========================================================
# DATASET PATHS & SAVING
# =========================================================

BASE_PATH = "/content/data/affectnet/affectnet"
TRAIN_DIR = os.path.join(BASE_PATH, "Train")
TEST_DIR = os.path.join(BASE_PATH, "Test")

# Save directly to Drive
SAVE_DIR = "/content/drive/MyDrive/AffectNet_FRIT_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

emotion_map = {
    "anger": 0, "contempt": 1, "disgust": 2, "fear": 3,
    "happy": 4, "neutral": 5, "sad": 6, "surprise": 7
}
NUM_CLASSES = 8

# =========================================================
# RANDOM MASKING
# =========================================================

class RandomMasking:
    def __init__(self, p=0.5, min_area=0.04, max_area=0.3):
        self.p = p
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        C, H, W = img.shape
        area = H * W

        mask_area = torch.empty(1).uniform_(self.min_area, self.max_area).item() * area
        aspect_ratio = torch.empty(1).uniform_(0.3, 3.3).item()

        h = int((mask_area * aspect_ratio) ** 0.5)
        w = int((mask_area / aspect_ratio) ** 0.5)

        if h >= H or w >= W:
            return img

        top = torch.randint(0, H // 2, (1,)).item()
        left = torch.randint(0, W - w, (1,)).item()

        img[:, top:top+h, left:left+w] = 0
        return img

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    RandomMasking(p=0.5), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================================================
# DATASET
# =========================================================

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for emotion_name in os.listdir(root_dir):
            emotion_path = os.path.join(root_dir, emotion_name)
            if not os.path.isdir(emotion_path):
                continue
            
            emotion_name_lower = emotion_name.lower()
            if emotion_name_lower not in emotion_map:
                continue

            label = emotion_map[emotion_name_lower]
            for img_name in os.listdir(emotion_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_name)
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
# MAIN TRAINING LOGIC
# =========================================================

def train():
    train_dataset = AffectNetDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = AffectNetDataset(TEST_DIR, transform=val_transform)

    print(f"\nTrain Images: {len(train_dataset)}")
    print(f"Test Images : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    class_weights = [total_samples / (NUM_CLASSES * count) for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    model = FRITNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Create the master log file
    log_file_path = os.path.join(SAVE_DIR, "affectnet_frit_master_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")
    
    start_time = time.time()

    # =========================================================
    # EPOCH LOOP
    # =========================================================
    for epoch in range(EPOCHS):
        current_epoch = epoch + 1
        print(f"\n================ EPOCH {current_epoch}/{EPOCHS} ================")
        
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="Training")

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, features = model(images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        t_acc = train_correct / train_total
        t_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                logits, features = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        v_acc = val_correct / val_total
        v_loss = val_loss / len(test_loader)

        scheduler.step()

        # --- RECORD METRICS ---
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        log_text = f"Epoch {current_epoch} | T-Loss: {t_loss:.4f} | T-Acc: {t_acc:.4f} | V-Loss: {v_loss:.4f} | V-Acc: {v_acc:.4f}"
        print(log_text)
        
        # Append to log file and flush immediately
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{current_epoch},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")

        # =========================================================
        # SAVE WEIGHTS FOR THIS SPECIFIC EPOCH
        # =========================================================
        weight_filename = f"affectnet_weights_epoch_{current_epoch}.pth"
        weight_path = os.path.join(SAVE_DIR, weight_filename)
        torch.save(model.state_dict(), weight_path)
        print(f"--> Saved Weights: {weight_filename}")

        # =========================================================
        # PLOT AND SAVE GRAPHS FOR THIS SPECIFIC EPOCH
        # =========================================================
        plt.figure(figsize=(12, 5))
        
        # Accuracy Subplot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Acc', marker='o')
        plt.plot(history['val_acc'], label='Val Acc', marker='o')
        plt.title(f'Accuracy (Up to Epoch {current_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss Subplot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss', marker='o')
        plt.plot(history['val_loss'], label='Val Loss', marker='o')
        plt.title(f'Loss (Up to Epoch {current_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_filename = f"affectnet_plot_epoch_{current_epoch}.png"
        plot_path = os.path.join(SAVE_DIR, plot_filename)
        plt.savefig(plot_path)
        
        # Close the plot to free up memory
        plt.close()
        print(f"--> Saved Plot: {plot_filename}")

    total_time = (time.time() - start_time) / 60
    print("\n===================================================")
    print("TRAINING COMPLETE")
    print("===================================================")
    print(f"Training Time (min)      : {total_time:.2f}")
    print(f"All files saved to       : {SAVE_DIR}")

if __name__ == "__main__":
    train()
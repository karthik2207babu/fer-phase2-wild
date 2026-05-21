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

BATCH_SIZE = 64      # Matched to your successful run
EPOCHS = 30          # Increased to 30
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# =========================================================
# DATASET PATHS & SAVING
# =========================================================

BASE_PATH = "/content/data/affectnet/affectnet"
TRAIN_DIR = os.path.join(BASE_PATH, "Train")
TEST_DIR = os.path.join(BASE_PATH, "Test")

SAVE_DIR = "/content/drive/MyDrive/AffectNet_FRIT_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

emotion_map = {
    "anger": 0, "contempt": 1, "disgust": 2, "fear": 3,
    "happy": 4, "neutral": 5, "sad": 6, "surprise": 7
}
NUM_CLASSES = 8

# =========================================================
# RANDOM MASKING (Ported from dataset.py)
# =========================================================

class RandomMasking:
    def __init__(self, p=0.5, min_area=0.04, max_area=0.3):
        self.p = p
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        # img shape: (C, H, W)
        if torch.rand(1).item() > self.p:
            return img

        C, H, W = img.shape
        area = H * W

        # Random mask size
        mask_area = torch.empty(1).uniform_(self.min_area, self.max_area).item() * area
        aspect_ratio = torch.empty(1).uniform_(0.3, 3.3).item()

        h = int((mask_area * aspect_ratio) ** 0.5)
        w = int((mask_area / aspect_ratio) ** 0.5)

        if h >= H or w >= W:
            return img

        # Restrict mask to upper face region
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
    RandomMasking(p=0.5), # Applied AFTER ToTensor()
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
    # 1. Load Data
    train_dataset = AffectNetDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = AffectNetDataset(TEST_DIR, transform=val_transform)

    print(f"\nTrain Images: {len(train_dataset)}")
    print(f"Test Images : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Dynamic Class Weights for AffectNet
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    class_weights = [total_samples / (NUM_CLASSES * count) for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"\nDynamic Class Weights: {class_weights}")

    # 3. Model, Loss, Optimizer
    print("\nInitializing FRITNet with VGGFace2 pre-trained backbone...")
    model = FRITNet(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. History Tracking
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    log_file_path = os.path.join(SAVE_DIR, "affectnet_frit_log.txt")
    log_file = open(log_file_path, "w")
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")
    
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(EPOCHS):
        print(f"\n================ EPOCH {epoch+1}/{EPOCHS} ================")
        
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="Training")

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Unpack both logits and features from FRITNet
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

        # --- METRICS & SAVING ---
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        log_text = f"Epoch {epoch+1} | T-Loss: {t_loss:.4f} | T-Acc: {t_acc:.4f} | V-Loss: {v_loss:.4f} | V-Acc: {v_acc:.4f}"
        print(log_text)
        
        log_file.write(f"{epoch+1},{t_loss:.4f},{t_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        log_file.flush()

        if v_acc > best_acc:
            best_acc = v_acc
            weight_path = os.path.join(SAVE_DIR, "best_affectnet_frit_weights.pth")
            torch.save(model.state_dict(), weight_path)
            print(f"--> Saved New Best Model (Acc: {v_acc:.4f}) to {weight_path}")

    log_file.close()

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('AffectNet FRITNet Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('AffectNet FRITNet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "affectnet_frit_results_plot.png")
    plt.savefig(plot_path)
    print(f"\nGraphs saved as {plot_path}")

    total_time = (time.time() - start_time) / 60
    print("\n===================================================")
    print("TRAINING COMPLETE")
    print("===================================================")
    print(f"Best Validation Accuracy : {best_acc:.4f}")
    print(f"Training Time (min)      : {total_time:.2f}")

if __name__ == "__main__":
    # Assuming you run zip extraction prior to executing this script
    train()
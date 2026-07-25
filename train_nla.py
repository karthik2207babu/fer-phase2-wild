import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image
from sklearn.metrics import recall_score

# Your existing modules
from dataset import RAFDBDataset
from model import FRITNet
from loss import NAWLoss

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 50           
BASE_LR = 1e-4
SAVE_DIR = "/content/drive/MyDrive/FER_NLA_Results"

# Colab Paths
BASE_PATH = "/content/data/Datasets/RAF-DB"
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# ADDED: FERPlus weights path
FERPLUS_WEIGHTS = "/content/drive/MyDrive/FERPlus_Results/best_ferplus_aggressive.pth"

# --- Dual-View Dataset Wrapper ---
class DualViewDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        self.clean_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_name = str(self.base_dataset.annotations.iloc[idx, 0])
        label = int(self.base_dataset.annotations.iloc[idx, 1])
        img_path = os.path.join(self.base_dataset.root_dir, str(label), img_name)

        image_pil = Image.open(img_path).convert('RGB')
        return self.clean_transform(image_pil), self.aug_transform(image_pil), label

# ADDED: Weight loading function
def load_pretrained_weights(model, weights_path):
    print(f"--> Loading base FERPlus weights from: {weights_path}")
    state_dict = torch.load(weights_path)
    
    targets = ['.head.', 'aux_global_head', 'aux_local_head']
    keys_to_delete = [k for k in state_dict.keys() if any(t in k for t in targets)]
    
    print(f"--> Stripping {len(keys_to_delete)} mismatching 8-class head tensors...")
    for k in keys_to_delete:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    print("--> Successfully injected pre-trained facial geometry.")
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Pure RAF-DB Training with NLA Framework\n{'='*70}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    raf_train = RAFDBDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_ROOT, phase='train')
    train_dataset = DualViewDataset(raf_train)
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"--> Training strictly on RAF-DB dataset: {len(raf_train)} images.")

    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    
    # ADDED: Load FERPlus weights before starting
    if os.path.exists(FERPLUS_WEIGHTS):
        model = load_pretrained_weights(model, FERPLUS_WEIGHTS)
    else:
        print("WARNING: FERPlus weights not found! Training from scratch.")

    criterion = NAWLoss(num_classes=7, total_epochs=EPOCHS, lambda_param=0.5).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': BASE_LR * 0.1},
        {'params': model.lfa.parameters(), 'lr': BASE_LR},
        {'params': model.safm.parameters(), 'lr': BASE_LR},
        {'params': model.transformer.parameters(), 'lr': BASE_LR},
        {'params': model.classifier.parameters(), 'lr': BASE_LR}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_macro_recall = 0.0

    for epoch in range(EPOCHS):
        criterion.set_epoch(epoch)
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [NLA Active]")
        for imgs_orig, imgs_aug, labels in pbar:
            imgs_orig, imgs_aug, labels = imgs_orig.to(device), imgs_aug.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits_orig, _, aux_g, aux_l = model(imgs_orig)
            logits_aug, _, _, _ = model(imgs_aug)

            loss = criterion(logits_orig, labels, aux_g, aux_l, logits_aug=logits_aug)
            loss.backward()
            optimizer.step()
            
            # ADDED: Track training accuracy
            targets_idx = labels - 1 if labels.min() >= 1 else labels
            _, predicted = torch.max(logits_orig.data, 1)
            train_total += targets_idx.size(0)
            train_correct += (predicted == targets_idx).sum().item()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        t_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets = labels - 1 if labels.min() >= 1 else labels

                logits, _, _, _ = model(images)
                _, predicted = torch.max(logits.data, 1)

                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        v_acc = val_correct / val_total
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # UPDATED: Print both Train and Val accuracy
        print(f"Epoch {epoch+1}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Macro Recall: {macro_recall:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_macro_recall = macro_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_frit_nla_weights.pth"))
            print(f"--> Saved new best NLA weights: Acc = {v_acc:.4f}, Recall = {macro_recall:.4f}")

if __name__ == "__main__":
    train()
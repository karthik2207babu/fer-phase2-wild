import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Pointing to the new deep transformer weights
WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase5_Deep_Transformer/best_frit_weights_deep.pth"

def evaluate_deep_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # CRITICAL: We must specify transformer_depth=6 to load these specific weights
    model = FRITNet(num_classes=7, transformer_depth=6).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    val_correct = 0
    val_total = 0
    
    print(f"\nEvaluating Deep Architecture: {os.path.basename(WEIGHTS_PATH)}")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running Flip-TTA"):
            images, labels = images.to(device), labels.to(device)
            
            # Original
            logits_orig, _, _, _ = model(images)
            probs_orig = F.softmax(logits_orig, dim=1)
            
            # Flipped
            flipped_images = torch.flip(images, dims=[3])
            logits_flip, _, _, _ = model(flipped_images)
            probs_flip = F.softmax(logits_flip, dim=1)
            
            # Ensemble
            final_probs = (probs_orig + probs_flip) / 2.0
            
            _, predicted = torch.max(final_probs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == (labels - 1)).sum().item()
    
    final_acc = val_correct / val_total
    print("\n" + "="*50)
    print(f"Base Accuracy was: ~86.93%")
    print(f"Deep TTA Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_deep_tta()
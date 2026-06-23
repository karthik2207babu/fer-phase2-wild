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

# Test both weight files to see which one scales higher with TTA
# WEIGHTS_TO_CHECK = [
#     "/content/drive/MyDrive/FER_Phase4_Pseudo/best_frit_weights_pseudo.pth",      # The 87.03% run
#     "/content/drive/MyDrive/FER_Phase4_Pseudo_MixUpDecay/best_frit_weights_decay.pth" # The 86.96% run
# ]
WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase5_Deep_Transformer/best_frit_weights_deep.pth"

def evaluate_with_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = FRITNet(num_classes=7).to(device)
    
    for weights_path in WEIGHTS_TO_CHECK:
        if not os.path.exists(weights_path):
            print(f"Skipping: {weights_path} (File not found)")
            continue
            
        print(f"\nEvaluating: {os.path.basename(weights_path)}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Running Flip-TTA"):
                images, labels = images.to(device), labels.to(device)
                
                # 1. Forward pass on original images
                logits_orig, _, _, _ = model(images)
                probs_orig = F.softmax(logits_orig, dim=1)
                
                # 2. Forward pass on horizontally flipped images
                flipped_images = torch.flip(images, dims=[3]) # Flip along the width axis
                logits_flip, _, _, _ = model(flipped_images)
                probs_flip = F.softmax(logits_flip, dim=1)
                
                # 3. Ensemble the probabilities (Averaging)
                final_probs = (probs_orig + probs_flip) / 2.0
                
                _, predicted = torch.max(final_probs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == (labels - 1)).sum().item()
        
        final_acc = val_correct / val_total
        print(f"==> Original Base Accuracy was around: ~87.0%")
        print(f"==> New TTA Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
        print("="*50)

if __name__ == "__main__":
    evaluate_with_tta()
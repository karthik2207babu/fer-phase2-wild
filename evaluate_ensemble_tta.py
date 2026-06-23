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

# Loading both high-performing models
MODEL_A_WEIGHTS = "/content/drive/MyDrive/FER_Phase4_Pseudo/best_frit_weights_pseudo.pth"
MODEL_B_WEIGHTS = "/content/drive/MyDrive/FER_Phase4_Pseudo_MixUpDecay/best_frit_weights_decay.pth"

def evaluate_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ensemble TTA Evaluation on: {device}")
    
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize and load Model A
    print("Loading Model A (Pseudo-Labels)...")
    model_a = FRITNet(num_classes=7).to(device)
    model_a.load_state_dict(torch.load(MODEL_A_WEIGHTS, map_location=device))
    model_a.eval()
    
    # Initialize and load Model B
    print("Loading Model B (MixUp Decay)...")
    model_b = FRITNet(num_classes=7).to(device)
    model_b.load_state_dict(torch.load(MODEL_B_WEIGHTS, map_location=device))
    model_b.eval()
    
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running Dual-Model Flip-TTA"):
            images, labels = images.to(device), labels.to(device)
            flipped_images = torch.flip(images, dims=[3])
            
            # --- Model A Predictions ---
            logits_a_orig, _, _, _ = model_a(images)
            probs_a_orig = F.softmax(logits_a_orig, dim=1)
            
            logits_a_flip, _, _, _ = model_a(flipped_images)
            probs_a_flip = F.softmax(logits_a_flip, dim=1)
            
            # --- Model B Predictions ---
            logits_b_orig, _, _, _ = model_b(images)
            probs_b_orig = F.softmax(logits_b_orig, dim=1)
            
            logits_b_flip, _, _, _ = model_b(flipped_images)
            probs_b_flip = F.softmax(logits_b_flip, dim=1)
            
            # --- Final Ensemble (Average of all 4 probability distributions) ---
            final_probs = (probs_a_orig + probs_a_flip + probs_b_orig + probs_b_flip) / 4.0
            
            _, predicted = torch.max(final_probs, 1)
            val_total += labels.size(0)
            
            # RAF-DB labels are 1-indexed
            val_correct += (predicted == (labels - 1)).sum().item()
            
    final_acc = val_correct / val_total
    
    print("\n" + "="*50)
    print(f"Total Test Images: {val_total}")
    print(f"Correct Predictions: {val_correct}")
    print(f"Final Ensemble TTA Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_ensemble()
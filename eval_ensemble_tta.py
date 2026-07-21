import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
WEIGHTS_SAMPLER = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_regularized_sampler.pth"
WEIGHTS_MIXUP = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_curriculum_mixup.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

def run_ensemble_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nStarting Dual-Model Ensemble TTA (10-Pass Inference)\n{'='*70}")

    print(f"--> Loading Model 1: {os.path.basename(WEIGHTS_SAMPLER)}")
    model_1 = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model_1.load_state_dict(torch.load(WEIGHTS_SAMPLER, map_location=device))
    model_1.eval()

    print(f"--> Loading Model 2: {os.path.basename(WEIGHTS_MIXUP)}")
    model_2 = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model_2.load_state_dict(torch.load(WEIGHTS_MIXUP, map_location=device))
    model_2.eval()

    print("--> Initializing Validation Dataloader...")
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    # Batch size halved to prevent VRAM overflow during multi-pass tensor expansion
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers=2)

    val_correct = 0
    val_total = 0

    print("\n--> Executing 10-Pass Simultaneous Inference...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Ensemble TTA")
        
        for images_raw, labels in pbar:
            labels = labels.to(device)
            targets = labels - 1 
            
            batch_logits_1 = torch.zeros(labels.size(0), 7).to(device)
            batch_logits_2 = torch.zeros(labels.size(0), 7).to(device)
            
            images = images_raw.to(device)
            
            # --- TTA Variations ---
            images_flipped = torch.flip(images, dims=[3])
            images_zoomed = F.interpolate(images, scale_factor=1.05, mode='bilinear', align_corners=False)
            images_zoomed = transforms.CenterCrop(224)(images_zoomed)
            images_bright = torch.clamp(images * 1.2, 0, 1)
            images_dark = torch.clamp(images * 0.8, 0, 1)
            
            tta_batch = [images, images_flipped, images_zoomed, images_bright, images_dark]
            
            # Pass all 5 variations through Model 1
            for variant in tta_batch:
                logits, _, _, _ = model_1(variant)
                batch_logits_1 += F.softmax(logits, dim=1)
                
            # Pass all 5 variations through Model 2
            for variant in tta_batch:
                logits, _, _, _ = model_2(variant)
                batch_logits_2 += F.softmax(logits, dim=1)
            
            # Average the logits for each model independently, then sum them together
            final_probs = (batch_logits_1 / 5.0) + (batch_logits_2 / 5.0)
            _, predicted = torch.max(final_probs, 1)
            
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    accuracy = val_correct / val_total
    
    print("\n======================================================")
    print(f"Final Ensemble TTA Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("======================================================")

if __name__ == "__main__":
    run_ensemble_tta()
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 16 

BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Evaluating the highest performing weights
WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase4_Pseudo_MixUpDecay/best_frit_weights_decay.pth"

def evaluate_10crop_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Five-Crop (10-View) TTA on: {device}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    # Use the native RAFDBDataset to guarantee correct path resolution
    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    
    # Override the transform to output 10 crops instead of 1
    val_dataset.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) 
            for crop in crops
        ]))
    ])

    dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = FRITNet(num_classes=7).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    val_correct = 0
    val_total = 0
    
    print(f"\nEvaluating: {os.path.basename(WEIGHTS_PATH)}")
    
    with torch.no_grad():
        for crops, labels in tqdm(dataloader, desc="Evaluating"):
            # crops shape: [batch_size, 10, 3, 224, 224]
            bs, ncrops, c, h, w = crops.size()
            
            # Flatten batch for the forward pass
            crops = crops.view(-1, c, h, w).to(device)
            labels = labels.to(device)
            
            logits, _, _, _ = model(crops)
            
            # Reshape back to [batch_size, 10, 7]
            logits = logits.view(bs, ncrops, -1)
            
            # Average probabilities across the 10 views
            probs = F.softmax(logits, dim=2)
            avg_probs = probs.mean(dim=1)
            
            _, predicted = torch.max(avg_probs, 1)
            
            val_total += labels.size(0)
            
            # RAF-DB labels are 1-indexed, shift to 0-index for comparison
            val_correct += (predicted == (labels - 1)).sum().item()
            
    final_acc = val_correct / val_total
    
    print("\n" + "="*50)
    print(f"Total Test Images: {val_total}")
    print(f"Correct Predictions: {val_correct}")
    print(f"Final Five-Crop TTA Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_10crop_tta()
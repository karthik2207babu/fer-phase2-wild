import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

from model import FRITNet

# --- Configuration ---
# Reduced batch size because each image expands into 10 crops. 
# A batch of 16 means 160 images processed simultaneously by the GPU.
BATCH_SIZE = 16 

BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")

# Evaluating the highest performing weights
WEIGHTS_PATH = "/content/drive/MyDrive/FER_Phase4_Pseudo_MixUpDecay/best_frit_weights_decay.pth"

# --- Custom TenCrop Dataset ---
# We build a lightweight dataset class here to handle the 5D tensor generation
# without disrupting your existing dataset.py structure.
class RAFDBTenCropDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) 
                for crop in crops
            ]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Handle filename formatting
        img_name = str(self.data.iloc[idx, 0])
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += '.jpg'
            
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # RAF-DB labels are 1-indexed in the CSV, shifting to 0-indexed for PyTorch
        label = int(self.data.iloc[idx, 1]) - 1 
        
        crops = self.transform(image) # Returns a tensor of shape [10, 3, 224, 224]
        return crops, label

def evaluate_10crop_tta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Five-Crop (10-View) TTA on: {device}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    dataset = RAFDBTenCropDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = FRITNet(num_classes=7).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    val_correct = 0
    val_total = 0
    
    print(f"\nEvaluating: {os.path.basename(WEIGHTS_PATH)}")
    
    with torch.no_grad():
        for crops, labels in tqdm(dataloader, desc="Evaluating"):
            # crops shape: [batch_size, 10, 3, 224, 224]
            # labels shape: [batch_size]
            
            bs, ncrops, c, h, w = crops.size()
            
            # Flatten the batch to feed all crops through the network
            # Resulting shape: [batch_size * 10, 3, 224, 224]
            crops = crops.view(-1, c, h, w).to(device)
            labels = labels.to(device)
            
            logits, _, _, _ = model(crops)
            
            # Reshape logits back to group by the original image
            # Resulting shape: [batch_size, 10, 7]
            logits = logits.view(bs, ncrops, -1)
            
            # Average the probabilities across all 10 views
            probs = F.softmax(logits, dim=2)
            avg_probs = probs.mean(dim=1)
            
            _, predicted = torch.max(avg_probs, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    final_acc = val_correct / val_total
    
    print("\n" + "="*50)
    print(f"Total Test Images: {val_total}")
    print(f"Correct Predictions: {val_correct}")
    print(f"Final Five-Crop TTA Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_10crop_tta()
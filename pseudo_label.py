import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import FRITNet

# --- Configuration ---
BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.95

# Paths
ZIP_PATH = "/content/drive/MyDrive/affectnet.zip"
EXTRACT_PATH = "/content/data"
AFFECTNET_DIR = os.path.join(EXTRACT_PATH, "affectnet/affectnet/Train") 

# UPDATED: Pointing to the MixUp weights that match the current model.py architecture
RAF_WEIGHTS = "/content/drive/MyDrive/FER_Phase3_Results/best_frit_weights_mixup.pth" 
OUTPUT_CSV = "/content/drive/MyDrive/pseudo_labeled_affectnet.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Pseudo-Labeling on: {device}")

# =========================================================
# ZIP EXTRACTION
# =========================================================
if not os.path.exists(AFFECTNET_DIR):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    print("\nExtracting AffectNet ZIP...")
    os.system(f'unzip -q -n "{ZIP_PATH}" -d "{EXTRACT_PATH}"')
    print("Extraction Complete")
else:
    print("\nDataset already extracted.")

# --- Transforms (Strictly Inference/Validation format) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Unlabeled Dataset Loader ---
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        
        # Recursively grab all images from the AffectNet folders
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def generate_pseudo_labels():
    dataset = UnlabeledDataset(root_dir=AFFECTNET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Found {len(dataset)} unlabeled images to process.")

    model = FRITNet(num_classes=7).to(device)
    model.load_state_dict(torch.load(RAF_WEIGHTS, map_location=device))
    model.eval()

    pseudo_data = []
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Generating Labels"):
            images = images.to(device)
            
            logits, _, _, _ = model(images)
            probabilities = F.softmax(logits, dim=1)
            
            max_probs, predicted_classes = torch.max(probabilities, dim=1)
            
            # Filter by confidence threshold
            for i in range(len(max_probs)):
                if max_probs[i].item() >= CONFIDENCE_THRESHOLD:
                    # RAF-DB labels are 1-indexed (1 to 7)
                    raf_label = predicted_classes[i].item() + 1
                    pseudo_data.append({
                        "file_path": paths[i],
                        "label": raf_label,
                        "confidence": max_probs[i].item()
                    })

    df = pd.DataFrame(pseudo_data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n===================================")
    print(f"Pseudo-Labeling Complete.")
    print(f"Total High-Confidence Images Kept: {len(df)} / {len(dataset)}")
    print(f"Saved to: {OUTPUT_CSV}")
    print("===================================")

if __name__ == "__main__":
    generate_pseudo_labels()
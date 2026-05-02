import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RAFDBDataset(Dataset):
    def __init__(self, csv_file, root_dir, phase='train'):
        self.root_dir = root_dir
        self.phase = phase
        
        # Load CSV: [image_name, label]
        self.annotations = pd.read_csv(csv_file)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = str(self.annotations.iloc[idx, 0])
        label = int(self.annotations.iloc[idx, 1]) # This is 1, 2, 3, etc.
        
        # FIX: Look inside the subfolder named after the label
        # Example: .../DATASET/train/5/train_05103_aligned.jpg
        img_path = os.path.join(self.root_dir, str(label), img_name)
        
        if not os.path.exists(img_path):
            # Fallback check: sometimes CSVs use 0-indexing or the label is different
            raise FileNotFoundError(f"Image not found at {img_path}. Check if folder '{label}' exists.")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
            
        return image, label
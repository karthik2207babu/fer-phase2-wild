import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# ============================
# Controlled Random Masking
# ============================
class RandomMasking:
    def __init__(self, min_area=0.04, max_area=0.2):
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        rand_val = torch.rand(1).item()
        
        # 20% chance of NO mask (0.8 to 1.0)
        if rand_val > 0.8:
            return img

        C, H, W = img.shape
        area = H * W

        mask_area = torch.empty(1).uniform_(self.min_area, self.max_area).item() * area
        aspect_ratio = torch.empty(1).uniform_(0.3, 3.3).item()

        h = int((mask_area * aspect_ratio) ** 0.5)
        w = int((mask_area / aspect_ratio) ** 0.5)

        if h >= H // 2 or w >= W:
            return img

        # Random horizontal placement
        left = torch.randint(0, W - w, (1,)).item()

        if rand_val <= 0.4:
            # 40% chance: Upper region (0 to H/2)
            top = torch.randint(0, max(1, (H // 2) - h), (1,)).item()
        else:
            # 40% chance: Lower region (H/2 to H)
            top = torch.randint(H // 2, max((H // 2) + 1, H - h), (1,)).item()

        img[:, top:top+h, left:left+w] = 0
        return img


class RAFDBDataset(Dataset):
    def __init__(self, csv_file, root_dir, phase='train'):

        self.root_dir = root_dir
        self.phase = phase
        self.annotations = pd.read_csv(csv_file)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if phase == 'train':

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                
                # Kaggle-inspired Augmentations
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), # Translate images 10% in x and y
                    scale=(0.9, 1.1)      # Zoom in/out by 50%
                ),
                transforms.ColorJitter(
                    contrast=0.1          # Adjust contrast
                ),

                transforms.ToTensor(),

                # 40% upper, 40% lower, 20% none
                RandomMasking(min_area=0.04, max_area=0.2),

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
        label = int(self.annotations.iloc[idx, 1])

        img_path = os.path.join(
            self.root_dir,
            str(label),
            img_name
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label
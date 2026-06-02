import os
import zipfile
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

def prepare_ferplus_data(zip_path="/content/drive/MyDrive/FERPLUS.zip", extract_path="/content/ferplus_extracted"):
    """Extracts FERPlus zip from Drive if not already present."""
    if not os.path.exists(extract_path):
        print(f"--> Extracting {zip_path} to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("--> Extraction complete.")
    else:
        print("--> Dataset already extracted.")
    return extract_path

def get_ferplus_dataloaders(root_dir, batch_size=64):
    # Find the train/val/test splits dynamically inside the extracted folder
    splits = os.listdir(root_dir)
    train_dir, val_dir = None, None
    
    for split in splits:
        low = split.lower()
        if 'train' in low:
            train_dir = os.path.join(root_dir, split)
        elif 'val' in low or 'test' in low:
            val_dir = os.path.join(root_dir, split)
            
    if not train_dir or not val_dir:
        # Fallback if names are completely arbitrary
        subdirs = [os.path.join(root_dir, d) for d in splits if os.path.isdir(os.path.join(root_dir, d))]
        if len(subdirs) == 0:
            raise RuntimeError(f"No subdirectories found in root_dir={root_dir}")
        train_dir = subdirs[0]
        val_dir = subdirs[1] if len(subdirs) > 1 else subdirs[0]

    # Transform pipeline: Upsample 112x112 to 224x224
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"--> Loading Train from: {train_dir}")
    print(f"--> Loading Val from: {val_dir}")

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
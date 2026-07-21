import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from dataset import RAFDBDataset
from model import FRITNet

# --- Configuration ---
WEIGHTS_A = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_regularized_sampler.pth"
WEIGHTS_B = "/content/drive/MyDrive/RAFDB_Results/best_rafdb_curriculum_mixup.pth"
SAVE_PATH = "/content/drive/MyDrive/RAFDB_Results/averaged_models_init.pth"

BASE_PATH = "/content/data/Datasets/RAF-DB"
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")
BATCH_SIZE = 64

def average_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nExecuting Weight Averaging (Model Soup)\n{'='*60}")

    # 1. Load both state dicts
    print("--> Loading Model A (Sampler) and Model B (MixUp)...")
    state_a = torch.load(WEIGHTS_A, map_location='cpu')
    state_b = torch.load(WEIGHTS_B, map_location='cpu')

    # 2. Average the weights
    print("--> Mathematically averaging parameter matrices...")
    averaged_state = {}
    for key in state_a.keys():
        if key in state_b:
            # Add tensors and divide by 2
            averaged_state[key] = (state_a[key] + state_b[key]) / 2.0
        else:
            averaged_state[key] = state_a[key]

    # 3. Save the averaged weights
    torch.save(averaged_state, SAVE_PATH)
    print(f"--> Saved averaged weights to {os.path.basename(SAVE_PATH)}")

    # 4. Load into a fresh model for validation
    print("\n--> Initializing Student Model with Averaged Weights...")
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)
    model.load_state_dict(averaged_state)
    model.eval()

    val_dataset = RAFDBDataset(csv_file=VAL_CSV, root_dir=VAL_ROOT, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    val_correct = 0
    val_total = 0

    print("--> Executing Zero-Shot Validation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Testing Averaged Model"):
            images, labels = images.to(device), labels.to(device)
            targets = labels - 1
            
            logits, _, _, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    accuracy = val_correct / val_total
    print("\n======================================================")
    print(f"Averaged Model Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("======================================================")

if __name__ == "__main__":
    average_and_test()
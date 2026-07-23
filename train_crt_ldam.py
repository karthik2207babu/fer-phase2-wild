import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Import your custom FRITNet model
from model import FRITNet 

# ==========================================
# 1. LDAM Loss Implementation
# ==========================================
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super(LDAMLoss, self).__init__()
        # Calculate margins inversely proportional to class frequency (N_j ^ 1/4)
        m_list = 1.0 / np.power(cls_num_list, 0.25)
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s

    def forward(self, logits, target):
        # Note: Logits are already scaled by s (30.0) inside our L2NormLinear head
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        
        m_list = self.m_list.to(logits.device)
        batch_m = m_list[target] 
        
        # Subtract the scaled margin strictly from the ground-truth logits
        logits_m = logits - (batch_m.view(-1, 1) * self.s)
        
        # Combine the margin-penalized ground truth with the unmodified logits
        output = torch.where(index, logits_m, logits)
        
        return F.cross_entropy(output, target)

# ==========================================
# 2. Main Training Function
# ==========================================
def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15
    batch_size = 64
    learning_rate = 0.01
    
    # EXACT RAF-DB Class Frequencies 
    # [Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral]
    # Verify these match your dataloader's class-to-index mapping!
    cls_num_list = [1290, 281, 717, 4772, 1982, 705, 2524] 
    
    # Paths
    data_dir = '/content/data/Datasets/RAF-DB/DATASET'
    pretrained_weights_path = 'best_rafdb_averaged.pth' # Your 88.07% Model Soup
    save_path = 'best_rafdb_crt_ldam.pth'               # UNIQUE weight save name

    # --- Data Loading ---
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Model Initialization & Freezing ---
    print("Initializing FRITNet with L2-Normalized Head...")
    model = FRITNet(num_classes=7, transformer_depth=2).to(device)

    # Load previously averaged weights. strict=False is REQUIRED because 
    # the old standard nn.Linear weights will be ignored and the new head will initialize fresh.
    if os.path.exists(pretrained_weights_path):
        print(f"Loading base weights from {pretrained_weights_path}...")
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device), strict=False)
    else:
        print("WARNING: Base weights not found. Starting from scratch.")

    # Apply the Hard Freeze for cRT
    print("Freezing upstream modules. Unfreezing ONLY the L2-Normalized Classifier...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # --- Loss and Optimizer ---
    criterion = LDAMLoss(cls_num_list=cls_num_list, s=30.0).to(device)
    
    # Pass ONLY the classifier parameters to the optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- Training Loop ---
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits, _, _, _ = model(inputs)
            
            # Calculate LDAM Loss
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                logits, _, _, _ = model(inputs)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save the unique weights
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"--> New Best Accuracy: {best_acc:.2f}%. Saving uniquely as {save_path}")
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining Complete. Final Best cRT LDAM Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
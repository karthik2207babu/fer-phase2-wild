import torch
import torch.nn as nn

# Import the custom modules we just wrote
from backbone import TruncatedFaceNet
from lfa import LFAModule
from multiscale import MultiScaleGlobalConv
from safm import SAFM
from transformer import FRITTransformer

class FRITNet(nn.Module):
    def __init__(self, num_classes=7):
        super(FRITNet, self).__init__()
        
        # 1. Backbone: Extracts deep spatial features
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        
        # 2. Local Feature Augmentation: Upsamples and extracts geometric contours
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        
        # 3. Multi-Scale Global Convolution: Handles varying facial sizes
        self.multiscale = MultiScaleGlobalConv(in_channels=128)
        
        # 4. Spatial Attention Feature Module: Focuses on emotional regions
        self.safm = SAFM(kernel_size=7)
        
        # 5. Transformer: Global correlation and final classification
        self.transformer = FRITTransformer(
            embed_dim=128, 
            num_heads=8,      # Increased capacity
            num_layers=4,     # Deepened
            num_classes=num_classes, 
            dropout=0.6       # Heavier dropout for regularization
        )

    def forward(self, x):
        # Input shape expected: (Batch, 3, 224, 224)
        x = self.backbone(x)     # Output: (Batch, 1792, 7, 7)
        x = self.lfa(x)          # Output: (Batch, 128, 28, 28)
        x = self.multiscale(x)   # Output: (Batch, 128, 28, 28)
        x = self.safm(x)         # Output: (Batch, 128, 28, 28)
        
        # Returns both for the CombinedFERLoss (logits for CE, features for Cluster)
        logits, features = self.transformer(x) 
        
        return logits, features

if __name__ == "__main__":
    model = FRITNet()
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, features = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
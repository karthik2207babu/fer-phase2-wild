import torch
import torch.nn as nn

from backbone import TruncatedFaceNet
from lfa import LFAModule
from safm import SAFM
from transformer import FRITTransformer

class FRITNet(nn.Module):
    def __init__(self, num_classes=7, transformer_depth=6):
        super(FRITNet, self).__init__()
        
        # 1. Feature Extraction & Spatial Upscaling
        self.backbone = TruncatedFaceNet(pretrained='vggface2', freeze_early_layers=True)
        self.lfa = LFAModule(in_channels=1792, out_channels=128)
        
        # 2. Spatial Noise Gate
        self.safm = SAFM(kernel_size=7)
        
        # 3. Cross-Attention Relation Transformer
        self.transformer = FRITTransformer(
            embed_dim=128, 
            num_heads=8,      
            num_local_layers=transformer_depth,  
            num_classes=num_classes, 
            dropout=0.5       
        )

    def forward(self, x):
        x = self.backbone(x)     
        x = self.lfa(x)          
        x = self.safm(x)         
        
        logits, features, aux_global, aux_local = self.transformer(x) 
        
        return logits, features, aux_global, aux_local

if __name__ == "__main__":
    model = FRITNet(transformer_depth=6)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, features, aux_global, aux_local = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
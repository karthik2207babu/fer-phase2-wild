import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import TruncatedFaceNet
from lfa import LFAModule
from safm import SAFM
from transformer import FRITTransformer

# Custom L2-Normalized Linear Layer for LDAM
class L2NormLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0):
        super(L2NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        
        # Define the weight matrix as a trainable parameter
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # L2 Normalize the incoming feature vectors and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity and apply the scale factor
        logits = self.scale * F.linear(x_norm, w_norm)
        
        return logits

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
        
        # 4. New L2-Normalized Classification Head
        self.classifier = L2NormLinear(in_features=128, out_features=num_classes, scale=30.0)

    def forward(self, x):
        x = self.backbone(x)     
        x = self.lfa(x)          
        x = self.safm(x)         
        
        # Bypass the transformer's internal logits (using '_') and grab the features
        _, features, aux_global, aux_local = self.transformer(x) 
        
        # Generate the new logits using the L2-Normalized head
        logits = self.classifier(features)
        
        return logits, features, aux_global, aux_local

if __name__ == "__main__":
    model = FRITNet(transformer_depth=6)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, features, aux_global, aux_local = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
import torch
import torch.nn as nn

class SAFM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAFM, self).__init__()
        
        # Convolution to generate the spatial attention map
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (Batch, 128, 28, 28)
        
        # 1. Average and Max pooling across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. Concatenate the two pooled maps (Result: 2 channels)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        
        # 3. Pass through conv and sigmoid to get attention weights [0, 1]
        attention_map = self.sigmoid(self.conv(x_concat))
        
        # 4. Multiply original input by the attention weights
        out = x * attention_map
        
        return out

if __name__ == "__main__":
    # Test tensor: (Batch, 128, 28, 28)
    dummy_input = torch.randn(2, 128, 28, 28) 
    safm = SAFM()
    output = safm(dummy_input)
    
    print(f"SAFM Input shape: {dummy_input.shape}")
    print(f"SAFM Output shape: {output.shape}") 
    # Expected: torch.Size([2, 128, 28, 28])
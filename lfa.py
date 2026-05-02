import torch
import torch.nn as nn
import torch.nn.functional as F

class LFAModule(nn.Module):
    def __init__(self, in_channels=1792, out_channels=128):
        super(LFAModule, self).__init__()
        
        # 1. Bridge: Compress channels from FaceNet (1792) to LFA expectation (128)
        self.bridge = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_bridge = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Sequential Asymmetric Convolutions
        # Horizontal focus (1x3)
        self.conv1x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Spatial expansion (3x3)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Vertical focus (3x1)
        self.conv3x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def _process_region(self, region):
        # Pass through the sequential sequence to extract geometric contours
        x = self.relu(self.bn1(self.conv1x3(region)))
        x = self.relu(self.bn2(self.conv3x3(x)))
        x = self.relu(self.bn3(self.conv3x1(x)))
        
        # Residual connection
        return x + region

    def forward(self, x):
        """
        Input x shape: (Batch, 1792, 7, 7)
        Output shape: (Batch, 128, 28, 28)
        """
        # 1. Bridge & Upsample
        x = self.relu(self.bn_bridge(self.bridge(x))) 
        # Expand 7x7 grid to 28x28
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False) 
        
        # 2. Spatial Split (4 Quadrants of 14x14)
        top_left = x[:, :, :14, :14]
        top_right = x[:, :, :14, 14:]
        bottom_left = x[:, :, 14:, :14]
        bottom_right = x[:, :, 14:, 14:]
        
        # 3. Sequential Convolutions on each region
        tl = self._process_region(top_left)
        tr = self._process_region(top_right)
        bl = self._process_region(bottom_left)
        br = self._process_region(bottom_right)
        
        # 4. Region Fusion (Stitch back to 28x28)
        top_half = torch.cat((tl, tr), dim=3)    # Concat along width
        bottom_half = torch.cat((bl, br), dim=3) # Concat along width
        fused_map = torch.cat((top_half, bottom_half), dim=2) # Concat along height
        
        return fused_map

if __name__ == "__main__":
    # Local test block
    lfa = LFAModule()
    dummy_input = torch.randn(2, 1792, 7, 7) 
    output = lfa(dummy_input)
    
    print(f"LFA Input shape: {dummy_input.shape}")
    print(f"LFA Output shape: {output.shape}") 
    # Expected: torch.Size([2, 128, 28, 28])
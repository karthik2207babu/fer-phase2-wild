import torch
import torch.nn as nn

class MultiScaleGlobalConv(nn.Module):
    def __init__(self, in_channels=128):
        super(MultiScaleGlobalConv, self).__init__()
        
        # Split channels evenly (128 / 4 = 32)
        split_channels = in_channels // 4
        
        # Branch 1: 1x1 Convolution (Fine details)
        self.branch1 = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 7x7 Convolution (Wide structural context)
        self.branch4 = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: (Batch, 128, 28, 28)
        
        # 1. Split into 4 chunks of 32 channels
        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)
        
        # 2. Parallel processing
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        y3 = self.branch3(x3)
        y4 = self.branch4(x4)

        # # =========================================
        # # Reduce dominance of 7x7 branch
        # # =========================================
        # y4 = 0.7 * y4
        
        # 3. Concatenate back to 128 channels
        out = torch.cat((y1, y2, y3, y4), dim=1)
        
        # 4. Residual connection
        return out + x

if __name__ == "__main__":
    ms_conv = MultiScaleGlobalConv()
    dummy_input = torch.randn(2, 128, 28, 28) 
    output = ms_conv(dummy_input)
    print(f"Output shape: {output.shape}")
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class TruncatedFaceNet(nn.Module):
    def __init__(self, pretrained='vggface2', freeze_early_layers=True):
        super(TruncatedFaceNet, self).__init__()
        
        # Load FaceNet pre-trained on VGGFace2[cite: 1]
        facenet = InceptionResnetV1(pretrained=pretrained)
        
        # Extract blocks, intercepting before AdaptiveAvgPool2d and Flatten
        self.features = nn.Sequential(
            facenet.conv2d_1a,
            facenet.conv2d_2a,
            facenet.conv2d_2b,
            facenet.maxpool_3a,
            facenet.conv2d_3b,
            facenet.conv2d_4a,
            facenet.conv2d_4b,
            facenet.repeat_1,
            facenet.mixed_6a,
            facenet.repeat_2,
            facenet.mixed_7a,
            facenet.repeat_3,
            facenet.block8
        )
        
        # Freeze early layers to protect pre-trained weights[cite: 1]
        if freeze_early_layers:
            for param in self.features[:-4].parameters():
                param.requires_grad = False

    def forward(self, x):
        # Input: Aligned Face Tensor (Batch, 3, 224, 224)
        # Output: Deep spatial feature map (Batch, 1792, 7, 7)
        return self.features(x)

if __name__ == "__main__":
    # Local test block
    model = TruncatedFaceNet(pretrained=None) # pretrained=None for local shape testing
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [2, 1792, 7, 7]
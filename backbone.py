import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class TruncatedFaceNet(nn.Module):
    def __init__(self, pretrained='vggface2', freeze_early_layers=True):
        super(TruncatedFaceNet, self).__init__()
        
        facenet = InceptionResnetV1(pretrained=pretrained)
        
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
        
        # Freeze early layers only
        if freeze_early_layers:
            for param in self.features[:10].parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)

if __name__ == "__main__":
    model = TruncatedFaceNet(pretrained=None)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TruncatedResNetV27(nn.Module):
    def __init__(self, freeze_early_layers=True):
        super().__init__()

        net = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4
        )

        self.adapter = nn.Sequential(
            nn.Conv2d(
                2048,
                1792,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(1792),
            nn.ReLU(inplace=True)
        )

        if freeze_early_layers:
            for p in self.features[:-1].parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.adapter(x)
        return x
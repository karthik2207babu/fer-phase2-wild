import torch
import torch.nn as nn

from backbone_v27 import TruncatedResNetV27
from lfa import LFAModule
from safm import SAFM
from transformer import FRITTransformer


class FRITNetV27(nn.Module):
    def __init__(
        self,
        num_classes=8,
        transformer_depth=2
    ):
        super().__init__()

        self.backbone = TruncatedResNetV27(
            freeze_early_layers=True
        )

        self.lfa = LFAModule(
            in_channels=1792,
            out_channels=128
        )

        self.safm = SAFM(
            kernel_size=7
        )

        self.transformer = FRITTransformer(
            embed_dim=128,
            num_heads=8,
            num_local_layers=transformer_depth,
            num_classes=num_classes,
            dropout=0.5
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.lfa(x)
        x = self.safm(x)

        _, features, aux_global, aux_local = (
            self.transformer(x)
        )

        logits = self.classifier(
            features
        )

        return (
            logits,
            features,
            aux_global,
            aux_local
        )
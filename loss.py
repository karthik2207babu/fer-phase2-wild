import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=7):
        super(CombinedFERLoss, self).__init__()
        self.feat_dim = feat_dim
        
        weights = torch.tensor([
            0.2178,
            1.0000,
            0.3919,
            0.0589,
            0.1418,
            0.3986,
            0.1113
        ])
        
        self.register_buffer('class_weights', weights)

    def forward(self, logits, features, labels):
        target = labels - 1
        
        # 👇 ONLY CHANGE: label_smoothing added
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=0.1
        )
        
        return ce_loss
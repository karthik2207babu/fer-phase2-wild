import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=7):
        super(CombinedFERLoss, self).__init__()
        self.feat_dim = feat_dim
        
        # Using your script-calculated weights for RAF-DB
        # Mapping: Class 1, 2, 3, 4, 5, 6, 7
        weights = torch.tensor([
            0.2178, # Class 1
            1.0000, # Class 2
            0.3919, # Class 3
            0.0589, # Class 4
            0.1418, # Class 5
            0.3986, # Class 6
            0.1113  # Class 7
        ])
        
        # This ensures the weights move to the GPU automatically
        self.register_buffer('class_weights', weights)

    def forward(self, logits, features, labels):
        # Shift labels 1-7 to 0-6 for PyTorch indexing
        target = labels - 1
        
        # Standard Cross Entropy with your custom penalty weights
        ce_loss = F.cross_entropy(logits, target, weight=self.class_weights)
        
        # Batch Weights for the Feature Clustering (optional expansion)
        # self.class_weights[target] now pulls the exact penalty for each image
        return ce_loss
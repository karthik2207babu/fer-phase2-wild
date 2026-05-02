import torch
import torch.nn as nn

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim=128, lambda_c=1.0):
        super(CombinedFERLoss, self).__init__()
        
        # 1. Your calculated weights for RAF-DB
        # Sequence: 1:Surprise, 2:Fear, 3:Disgust, 4:Happy, 5:Sad, 6:Anger, 7:Neutral
        self.weights = torch.tensor([0.2178, 1.0000, 0.3919, 0.0589, 0.1418, 0.3986, 0.1113])
        
        # Weighted-Softmax (Cross Entropy)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        
        # 2. Weighted Cluster Loss Parameters
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(7, feat_dim))

    def forward(self, logits, features, labels):
        # Labels are 1-7 in RAF-DB, PyTorch expects 0-6
        target = labels - 1
        
        # Weighted-Softmax component[cite: 1]
        l_ws = self.ce_loss(logits, target)
        
        # Weighted-Cluster component[cite: 1]
        batch_size = features.size(0)
        batch_weights = self.weights[target].to(features.device)
        
        # Intra-class distance (pulling features to centers)[cite: 1]
        centers_batch = self.centers[target]
        dist_to_center = torch.pow(features - centers_batch, 2).sum(dim=1)
        
        # Inter-class distance (pushing centers apart)[cite: 1]
        dist_matrix = torch.cdist(self.centers, self.centers, p=2)
        inter_class_dist = dist_matrix.sum(dim=1) - dist_matrix.diag()
        
        # Final Cluster calculation[cite: 1]
        l_wc = batch_weights * (dist_to_center / (inter_class_dist[target] + 1.0))
        
        return l_ws + (self.lambda_c * l_wc.mean())
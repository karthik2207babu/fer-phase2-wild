import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        loss = -mean_log_prob_pos.mean()
        return loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, weights=None):
        if weights is not None:
            weights = weights.to(inputs.device)
            
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=weights, 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=7, alpha=0.2):
        super(CombinedFERLoss, self).__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        self.supcon = SupConLoss(temperature=0.07)
        self.focal = WeightedFocalLoss(gamma=2.0, label_smoothing=0.1) 
        
        weights = torch.tensor([
            0.2178, 1.0000, 0.3919, 0.0589, 0.1418, 0.3986, 0.1113
        ])
        self.register_buffer('class_weights', weights)

    def forward(self, logits, features, labels, aux_global=None, aux_local=None):
        target = labels - 1
        
        # 1. Main Classification Loss
        main_loss = self.focal(logits, target, self.class_weights)
        
        # 2. Supervised Contrastive Loss 
        total_loss = main_loss
        if self.alpha > 0.0:
            supcon_loss = self.supcon(features, labels)
            total_loss += (self.alpha * supcon_loss)
            
        # 3. Joint Optimization
        if aux_global is not None:
            total_loss += 0.5 * self.focal(aux_global, target, self.class_weights)
            
        if aux_local is not None:
            total_loss += 0.5 * self.focal(aux_local, target, self.class_weights)

        return total_loss
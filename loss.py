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

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=7, alpha=0.1):
        super(CombinedFERLoss, self).__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.supcon = SupConLoss(temperature=0.07)
        
        weights = torch.tensor([
            0.2178, 1.0000, 0.3919, 0.0589, 0.1418, 0.3986, 0.1113
        ])
        self.register_buffer('class_weights', weights)

    def forward(self, logits, features, labels, aux_global=None, aux_local=None):
        target = labels - 1
        
        # AGGRESSIVE LABEL SMOOTHING = 0.25
        ce_loss = F.cross_entropy(logits, target, weight=self.class_weights, label_smoothing=0.25)
        supcon_loss = self.supcon(features, target)
        
        total_loss = ce_loss + (self.alpha * supcon_loss)

        if aux_global is not None and aux_local is not None:
            loss_global = F.cross_entropy(aux_global, target, weight=self.class_weights, label_smoothing=0.25)
            loss_local = F.cross_entropy(aux_local, target, weight=self.class_weights, label_smoothing=0.25)
            total_loss = total_loss + loss_global + loss_local
            
        return total_loss
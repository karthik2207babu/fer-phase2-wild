import torch
import torch.nn as nn
import torch.nn.functional as F

class FERPlusMRANLoss(nn.Module):
    def __init__(self):
        super(FERPlusMRANLoss, self).__init__()

    def forward(self, logits, aux_global, aux_local, targets):
        # Added label_smoothing=0.2 to prevent hard-label memorization
        loss_main = F.cross_entropy(logits, targets, label_smoothing=0.2)
        loss_global = F.cross_entropy(aux_global, targets, label_smoothing=0.2)
        loss_local = F.cross_entropy(aux_local, targets, label_smoothing=0.2)
        
        total_loss = loss_main + (0.5 * loss_global) + (0.5 * loss_local)
        return total_loss
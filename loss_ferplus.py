import torch
import torch.nn as nn
import torch.nn.functional as F

class FERPlusMRANLoss(nn.Module):
    def __init__(self):
        super(FERPlusMRANLoss, self).__init__()

    def forward(self, logits, aux_global, aux_local, targets):
        # ImageFolder labels are 0-indexed [0, 7] out of the box
        loss_main = F.cross_entropy(logits, targets)
        loss_global = F.cross_entropy(aux_global, targets)
        loss_local = F.cross_entropy(aux_local, targets)
        
        # Joint Optimization formula following the MRAN blueprint
        total_loss = loss_main + (0.5 * loss_global) + (0.5 * loss_local)
        return total_loss
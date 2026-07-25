import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        # Set to 0.0. We need hard targets to crush gradients on easy examples.
        self.label_smoothing = label_smoothing 

    def forward(self, inputs, targets, weights=None):
        if weights is not None:
            weights = weights.to(inputs.device)
            
        # CRITICAL FIX: reduction='none' computes loss per image in the batch
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=weights, 
            label_smoothing=self.label_smoothing,
            reduction='none' 
        )
        
        # Now pt is a tensor of probabilities for each individual image
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

class CombinedFERLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=7, alpha=0.2):
        super(CombinedFERLoss, self).__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        self.supcon = SupConLoss(temperature=0.07)
        # Initialize the fixed Focal Loss
        self.focal = WeightedFocalLoss(gamma=2.0, label_smoothing=0.0) 
        
        # Bypassed static weights. Focal loss natively balances classes dynamically.
        self.class_weights = None

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

# --- ADDED: Clean Joint Optimization Loss For FERPlus ---
class FERPlusMRANLoss(nn.Module):
    def __init__(self, smoothing=0.25, gamma=2.0):
        super(FERPlusMRANLoss, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, features, labels, aux_global=None, aux_local=None):
        # Dynamic Focal Loss calculation 
        def compute_focal(inputs, targets):
            # Calculate raw cross entropy without reducing it immediately
            ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.smoothing, reduction='none')
            # pt is the predicted probability of the true class
            pt = torch.exp(-ce_loss)
            # Apply the focal scaling factor to down-weight easy examples
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

        total_loss = compute_focal(logits, labels)
        
        # Joint Optimization tracking
        if aux_global is not None:
            total_loss += (0.5 * compute_focal(aux_global, labels))
            
        if aux_local is not None:
            total_loss += (0.5 * compute_focal(aux_local, labels))

        return total_loss

# --- ADDED: NLA Framework Loss (Append to bottom of loss.py) ---


class NAWLoss(nn.Module):
    def __init__(self, num_classes=7, total_epochs=50, lambda_param=0.5):
        super(NAWLoss, self).__init__()
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.lambda_param = lambda_param
        self.current_epoch = 0

        # Mean vectors for True and False predictions
        self.mu_true = torch.tensor([0.5, 0.5])
        self.mu_false = torch.tensor([0.3, 0.15])

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _compute_naw_weights(self, p_gt, p_nn, device):
        # Dynamic Covariance Scheduler
        e = float(self.current_epoch)
        E = float(max(1, self.total_epochs))
        cs = max(1.0 - math.exp(-10.0 * (e / E)), 1e-4)

        cov_true = torch.tensor([[0.8 * cs, 0.2 * cs], [0.2 * cs, 0.4 * cs]], device=device)
        cov_false = torch.tensor([[0.8, 0.1], [0.1, 0.15]], device=device)

        inv_cov_true = torch.linalg.pinv(cov_true)
        inv_cov_false = torch.linalg.pinv(cov_false)

        points = torch.stack([p_gt, p_nn], dim=1) 
        is_true = (p_gt >= p_nn)

        weights = torch.zeros_like(p_gt)
        mu_t, mu_f = self.mu_true.to(device), self.mu_false.to(device)

        if is_true.any():
            diff_t = points[is_true] - mu_t
            dist_t = torch.sum((diff_t @ inv_cov_true) * diff_t, dim=1)
            weights[is_true] = torch.exp(-0.5 * dist_t)

        if (~is_true).any():
            diff_f = points[~is_true] - mu_f
            dist_f = torch.sum((diff_f @ inv_cov_false) * diff_f, dim=1)
            weights[~is_true] = torch.exp(-0.5 * dist_f)

        return weights

    def compute_jsd(self, logits1, logits2):
        p1, p2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
        m = 0.5 * (p1 + p2)
        kl1 = F.kl_div(F.log_softmax(logits1, dim=1), m, reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(logits2, dim=1), m, reduction='batchmean')
        return 0.5 * (kl1 + kl2)

    def forward(self, logits, targets, aux_global=None, aux_local=None, logits_aug=None):
        probs = F.softmax(logits, dim=1)
        targets_idx = targets - 1 if targets.min() >= 1 else targets

        p_gt = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, targets_idx.unsqueeze(1), False)
        p_nn = probs[mask].view(probs.size(0), self.num_classes - 1).max(dim=1)[0]

        w_star = self._compute_naw_weights(p_gt, p_nn, logits.device)
        ce_loss = F.cross_entropy(logits, targets_idx, reduction='none')
        l_naw_ce = ((1.0 + w_star) * ce_loss).mean()

        if aux_global is not None:
            l_naw_ce += 0.2 * F.cross_entropy(aux_global, targets_idx)
        if aux_local is not None:
            l_naw_ce += 0.2 * F.cross_entropy(aux_local, targets_idx)

        if logits_aug is not None:
            total_loss = (self.lambda_param * l_naw_ce) + ((1.0 - self.lambda_param) * self.compute_jsd(logits, logits_aug))
        else:
            total_loss = l_naw_ce

        return total_loss
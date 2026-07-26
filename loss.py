import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NAWLoss(nn.Module):
    def __init__(self, num_classes=7, total_epochs=50, lambda_param=0.5):
        super(NAWLoss, self).__init__()
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.lambda_param = lambda_param
        self.current_epoch = 0

        # RECALIBRATED for L2Norm + s=30 head (was [0.5,0.5] / [0.3,0.15])
        # True branch: confidently-correct samples sit near (0.8-0.9, 0.1-0.2), not (0.5,0.5)
        self.mu_true = torch.tensor([0.8, 0.15])
        # False branch: confidently-wrong samples are the mirror — high p_NN, low p_GT
        self.mu_false = torch.tensor([0.15, 0.8])

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _compute_naw_weights(self, p_gt, p_nn, device):
        e = float(self.current_epoch)
        E = float(max(1, self.total_epochs))
        cs = max(1.0 - math.exp(-10.0 * (e / E)), 1e-4)

        # RECALIBRATED: tighter diagonal (0.15 vs paper's 0.8/0.4) so the bump doesn't
        # spread across the whole saturated 0-1 range and swallow easy examples too.
        cov_true = torch.tensor([
            [0.15, 0.05 * cs],
            [0.05 * cs, 0.15]
        ], device=device)

        cov_false = torch.tensor([
            [0.15, 0.05],
            [0.05, 0.15]
        ], device=device)

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
        targets_idx = targets

        p_gt = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, targets_idx.unsqueeze(1), False)
        p_nn = probs[mask].view(probs.size(0), self.num_classes - 1).max(dim=1)[0]

        w_star = self._compute_naw_weights(p_gt, p_nn, logits.device)

        # DEBUG: confirm the head's real distribution + whether the Gaussian is now
        # actually firing. Remove once confirmed.
        if self.current_epoch == 0:
            print(f"[DEBUG] p_gt mean={p_gt.mean().item():.3f} p_nn mean={p_nn.mean().item():.3f} "
                  f"w_star mean={w_star.mean().item():.3f} w_star max={w_star.max().item():.3f}")

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
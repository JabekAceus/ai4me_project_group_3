
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, Tensor
from utils import simplex, sset, one_hot




class CrossEntropy(nn.Module):
    """
    A wrapper for the standard `torch.nn.CrossEntropyLoss` that adapts it
    to the project's pipeline, which provides probabilities and one-hot targets.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, pred_probs, true_one_hot, **kwargs):
        true_classes = torch.argmax(true_one_hot, dim=1)
        pred_logits = torch.log(pred_probs + 1e-9)

        return self.ce_loss(pred_logits, true_classes)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred_probs, true_one_hot, **kwargs): 
        true_one_hot = true_one_hot.float()
        pred = pred_probs[:, 1:, ...]
        true = true_one_hot[:, 1:, ...]
        intersection = torch.sum(pred * true, dim=(1,2,3))
        cardinality = torch.sum(pred + true, dim=(1,2,3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred_probs, true_one_hot, **kwargs): 
        ce_loss = F.binary_cross_entropy(pred_probs, true_one_hot.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred_probs, true_one_hot, **kwargs): 
        true_one_hot = true_one_hot.float()
        pred = pred_probs[:, 1:, ...]
        true = true_one_hot[:, 1:, ...]
        
        tp = torch.sum(pred * true, dim=(1,2,3))
        fp = torch.sum(pred * (1-true), dim=(1,2,3))
        fn = torch.sum((1-pred) * true, dim=(1,2,3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky.mean()



class BoundaryLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred_probs: Tensor, true_one_hot: Tensor, dist_maps: Tensor, **kwargs) -> Tensor:
        """
        Computes a boundary loss that penalizes errors at segmentation boundaries.
        The distance map is a signed distance transform, where the boundary is at 0.
        """
        
        gt = true_one_hot[:, 1:, ...].float()
        pc = pred_probs[:, 1:, ...].float()
        
        
        error = torch.abs(gt - pc)
        
        
        
        
        boundary_weights = torch.exp(-torch.abs(dist_maps))
        
        
        weighted_error = boundary_weights * error
        
        return weighted_error.mean()

class CompoundLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    def forward(self, pred_probs, true_one_hot, **kwargs): 
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            
            total_loss += weight * loss(pred_probs, true_one_hot, **kwargs)
        return total_loss
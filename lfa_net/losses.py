"""Loss functions for LFA-Net segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Compute Dice loss for binary segmentation.

    Args:
        pred: Predictions [B, 1, H, W] (logits or probabilities)
        target: Ground truth [B, 1, H, W] (0 or 1)
        smooth: Smoothing factor to avoid division by zero
        from_logits: If True, apply sigmoid to pred

    Returns:
        Scalar Dice loss (1 - Dice coefficient)
    """
    if from_logits:
        pred = torch.sigmoid(pred)

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss for stable training.

    Args:
        pred: Predictions [B, 1, H, W] (logits)
        target: Ground truth [B, 1, H, W] (0 or 1)
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss

    Returns:
        Weighted combined loss
    """
    # BCE with logits (numerically stable)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    # Dice loss
    dice = dice_loss(pred, target, from_logits=True)

    return bce_weight * bce + dice_weight * dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss as nn.Module."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bce_dice_loss(pred, target, self.bce_weight, self.dice_weight)


# =============================================================================
# Multi-class loss functions (for artery/vein segmentation)
# =============================================================================


def multiclass_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Compute Dice loss averaged over multiple classes.
    
    Args:
        pred: Predictions [B, C, H, W] (logits or probabilities)
        target: Ground truth [B, C, H, W] (one-hot per channel)
        smooth: Smoothing factor
        from_logits: If True, apply sigmoid to pred
        
    Returns:
        Scalar Dice loss averaged over all classes
    """
    if from_logits:
        pred = torch.sigmoid(pred)
    
    num_classes = pred.shape[1]
    total_loss = 0.0
    
    for c in range(num_classes):
        pred_c = pred[:, c:c+1]
        target_c = target[:, c:c+1]
        
        pred_flat = pred_c.view(pred_c.size(0), -1)
        target_flat = target_c.view(target_c.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        total_loss += (1.0 - dice.mean())
    
    return total_loss / num_classes


def multiclass_bce_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss for multi-class segmentation.
    
    Args:
        pred: Predictions [B, C, H, W] (logits)
        target: Ground truth [B, C, H, W] (one-hot per channel)
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        
    Returns:
        Weighted combined loss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = multiclass_dice_loss(pred, target, from_logits=True)
    return bce_weight * bce + dice_weight * dice


def per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    from_logits: bool = True,
) -> list[float]:
    """
    Compute per-class Dice scores.
    
    Args:
        pred: Predictions [B, C, H, W]
        target: Ground truth [B, C, H, W]
        smooth: Smoothing factor
        from_logits: If True, apply sigmoid to pred
        
    Returns:
        List of Dice scores for each class
    """
    if from_logits:
        pred = torch.sigmoid(pred)
    
    num_classes = pred.shape[1]
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = pred[:, c:c+1]
        target_c = target[:, c:c+1]
        
        pred_flat = pred_c.view(pred_c.size(0), -1)
        target_flat = target_c.view(target_c.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean().item())
    
    return dice_scores


class MulticlassBCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for multi-class as nn.Module."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return multiclass_bce_dice_loss(pred, target, self.bce_weight, self.dice_weight)

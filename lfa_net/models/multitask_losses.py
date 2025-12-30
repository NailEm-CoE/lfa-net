"""Multi-Task Loss Functions with Missing Label Handling.

Key features:
- Per-channel segmentation loss (BCE + Dice)
- Safe mean for batches with missing labels (no NaN gradients)
- Per-task loss breakdown for logging
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_mean(
    loss: torch.Tensor, 
    valid_mask: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute mean loss over valid samples only.
    
    Returns 0 instead of NaN when no valid samples exist.
    
    Args:
        loss: Loss values [B] or [B, ...]
        valid_mask: Binary mask [B] indicating valid samples
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Scalar mean loss (0 if no valid samples)
    """
    n_valid = valid_mask.sum()
    if n_valid < eps:
        return torch.zeros(1, device=loss.device, dtype=loss.dtype).squeeze()
    
    # Flatten loss to [B, ...] and mask
    if loss.dim() > 1:
        # Average over non-batch dimensions first
        loss = loss.flatten(1).mean(dim=1)  # [B]
    
    valid_mask = valid_mask.float()
    return (loss * valid_mask).sum() / n_valid


def dice_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice loss per sample.
    
    Args:
        pred: Predictions [B, H, W] (logits, before sigmoid)
        target: Targets [B, H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice loss per sample [B]
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    pred_flat = pred.flatten(1)  # [B, H*W]
    target_flat = target.flatten(1)  # [B, H*W]
    
    intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice  # [B]


def bce_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute BCE loss per sample.
    
    Args:
        pred: Predictions [B, H, W] (logits, before sigmoid)
        target: Targets [B, H, W]
        
    Returns:
        BCE loss per sample [B]
    """
    # Use with_logits version for mixed precision compatibility
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # [B, H, W]
    return bce.flatten(1).mean(dim=1)  # [B]


def dice_bce_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined BCE + Dice loss per sample.
    
    Args:
        pred: Predictions [B, H, W] (logits, before sigmoid)
        target: Targets [B, H, W]
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        
    Returns:
        Combined loss per sample [B]
    """
    bce = bce_loss_per_sample(pred, target)
    dice = dice_loss_per_sample(pred, target)
    return bce_weight * bce + dice_weight * dice


class MultitaskLoss(nn.Module):
    """Multi-Task Loss with missing label handling.
    
    Handles:
    - Per-channel segmentation loss (BCE + Dice)
    - Fovea regression loss (SmoothL1)
    - Disease classification loss (BCE)
    
    All losses handle missing labels gracefully:
    - Returns 0 loss for tasks with no valid samples in batch
    - No NaN gradients
    
    Args:
        seg_channel_names: Names for segmentation channels
        disease_names: Names for disease classes
        seg_weights: Per-channel weights for segmentation
        fovea_weight: Weight for fovea loss
        disease_weight: Weight for disease loss
        bce_weight: BCE weight in BCE+Dice combo
        dice_weight: Dice weight in BCE+Dice combo
    """
    
    DEFAULT_SEG_NAMES = ["disc", "cup", "artery", "vein", "vessel"]
    DEFAULT_DISEASE_NAMES = ["dr", "amd", "glaucoma"]
    
    def __init__(
        self,
        seg_channel_names: Optional[list[str]] = None,
        disease_names: Optional[list[str]] = None,
        seg_weights: Optional[dict[str, float]] = None,
        fovea_weight: float = 0.5,
        disease_weight: float = 0.5,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        
        self.seg_channel_names = seg_channel_names or self.DEFAULT_SEG_NAMES
        self.disease_names = disease_names or self.DEFAULT_DISEASE_NAMES
        self.n_seg_channels = len(self.seg_channel_names)
        self.n_disease_classes = len(self.disease_names)
        
        # Default segmentation weights
        default_seg_weights = {name: 1.0 for name in self.seg_channel_names}
        default_seg_weights["vein"] = 0.5  # Sparse data
        self.seg_weights = seg_weights or default_seg_weights
        
        self.fovea_weight = fovea_weight
        self.disease_weight = disease_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # SmoothL1 for fovea (handles out-of-bounds better than MSE)
        self.fovea_loss_fn = nn.SmoothL1Loss(reduction='none')
    
    def forward(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss.
        
        Args:
            preds: Model predictions
                - "segmentation": [B, 5, H, W]
                - "fovea": [B, 2]
                - "disease": [B, N]
            batch: Batch data
                - "seg_masks": [B, 5, H, W]
                - "seg_valid": [B, 5] per-channel validity
                - "fovea": [B, 2] normalized coords
                - "fovea_valid": [B] validity flag
                - "disease_labels": [B, N]
                - "disease_valid": [B, N] per-class validity
                
        Returns:
            Dict with:
                - "loss": Total loss (scalar)
                - "seg_loss": Total segmentation loss
                - "seg_<name>": Per-channel segmentation loss
                - "fovea_loss": Fovea regression loss
                - "disease_loss": Disease classification loss
                - "n_valid_seg": [5] valid count per channel
                - "n_valid_fovea": Valid fovea count
                - "n_valid_disease": Valid disease count
        """
        device = preds["segmentation"].device
        dtype = preds["segmentation"].dtype
        
        total_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        loss_dict = {}
        
        # =====================================================================
        # Segmentation Loss (per-channel)
        # =====================================================================
        seg_pred = preds["segmentation"]  # [B, 5, H, W]
        seg_target = batch["seg_masks"]  # [B, 5, H, W]
        seg_valid = batch["seg_valid"]  # [B, 5]
        
        seg_total_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        n_valid_seg = torch.zeros(self.n_seg_channels, device=device)
        
        for ch_idx, ch_name in enumerate(self.seg_channel_names):
            ch_valid = seg_valid[:, ch_idx]  # [B]
            n_valid = ch_valid.sum()
            n_valid_seg[ch_idx] = n_valid
            
            if n_valid > 0:
                ch_pred = seg_pred[:, ch_idx]  # [B, H, W]
                ch_target = seg_target[:, ch_idx]  # [B, H, W]
                
                # BCE + Dice per sample
                ch_loss_per_sample = dice_bce_loss_per_sample(
                    ch_pred, ch_target, 
                    self.bce_weight, self.dice_weight
                )  # [B]
                
                # Safe mean over valid samples
                ch_weight = self.seg_weights.get(ch_name, 1.0)
                ch_loss = safe_mean(ch_loss_per_sample, ch_valid) * ch_weight
                
                seg_total_loss = seg_total_loss + ch_loss
                loss_dict[f"seg_{ch_name}"] = ch_loss
            else:
                loss_dict[f"seg_{ch_name}"] = torch.zeros(1, device=device).squeeze()
        
        total_loss = total_loss + seg_total_loss
        loss_dict["seg_loss"] = seg_total_loss
        loss_dict["n_valid_seg"] = n_valid_seg
        
        # =====================================================================
        # Fovea Loss
        # =====================================================================
        fovea_pred = preds["fovea"]  # [B, 2]
        fovea_target = batch["fovea"]  # [B, 2]
        fovea_valid = batch["fovea_valid"]  # [B]
        
        n_valid_fovea = fovea_valid.sum()
        
        if n_valid_fovea > 0:
            fovea_loss_per_sample = self.fovea_loss_fn(
                fovea_pred, fovea_target
            ).mean(dim=1)  # [B]
            
            fovea_loss = safe_mean(fovea_loss_per_sample, fovea_valid) * self.fovea_weight
        else:
            fovea_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        
        total_loss = total_loss + fovea_loss
        loss_dict["fovea_loss"] = fovea_loss
        loss_dict["n_valid_fovea"] = n_valid_fovea
        
        # =====================================================================
        # Disease Loss
        # =====================================================================
        disease_pred = preds["disease"]  # [B, N]
        disease_target = batch["disease_labels"]  # [B, N]
        disease_valid = batch["disease_valid"]  # [B, N]
        
        # Check if any disease labels are valid
        n_valid_disease = disease_valid.sum()
        
        if n_valid_disease > 0:
            # BCE loss per sample per class
            disease_loss_per_element = F.binary_cross_entropy_with_logits(
                disease_pred, disease_target.float(), reduction='none'
            )  # [B, N]
            
            # Mask invalid elements
            disease_loss_per_element = disease_loss_per_element * disease_valid.float()
            
            # Average over valid elements
            disease_loss = disease_loss_per_element.sum() / n_valid_disease * self.disease_weight
        else:
            disease_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        
        total_loss = total_loss + disease_loss
        loss_dict["disease_loss"] = disease_loss
        loss_dict["n_valid_disease"] = n_valid_disease
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        loss_dict["loss"] = total_loss
        
        return loss_dict

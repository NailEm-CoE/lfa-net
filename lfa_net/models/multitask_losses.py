"""Multi-Task Loss Functions with Missing Label Handling.

Key features:
- Per-channel segmentation loss (BCE + Squared Dice from LFA-Net paper)
- Anatomical constraint losses (Cup-in-Disc, Vessel=A∪V)
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


def squared_dice_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Squared Dice loss per sample (from LFA-Net paper Equation 27).
    
    Uses S^2 + G^2 in denominator instead of S + G.
    Better for thin structures like vessels due to squared penalty.
    
    Formula: 1 - (2 * intersection) / (pred^2 + target^2 + smooth)
    
    Args:
        pred: Predictions [B, H, W] (after sigmoid)
        target: Targets [B, H, W]
        smooth: Smoothing factor
        
    Returns:
        Squared Dice loss per sample [B]
    """
    pred_flat = pred.flatten(1)  # [B, H*W]
    target_flat = target.flatten(1)  # [B, H*W]
    
    # Intersection: sum of element-wise product
    intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
    
    # Squared sums (key difference from standard Dice)
    pred_sq_sum = (pred_flat ** 2).sum(dim=1)  # [B]
    target_sq_sum = (target_flat ** 2).sum(dim=1)  # [B]
    
    # Squared Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sq_sum + target_sq_sum + smooth)
    
    return 1.0 - dice  # [B]


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


class CupInDiscLoss(nn.Module):
    """Self-consistency loss: predicted cup should be inside predicted disc.
    
    Anatomical constraint: optic cup ⊆ optic disc
    Loss = 1 - Dice(cup_pred, cup_pred ∩ disc_pred)
    
    If cup is fully inside disc: cup ∩ disc = cup → Dice = 1.0 → Loss = 0
    If cup is partially outside: cup ∩ disc < cup → Dice < 1.0 → Loss > 0
    
    Note: This uses predictions only (no GT required), computed on ALL samples.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        disc_pred: torch.Tensor,  # [B, H, W] sigmoid probabilities
        cup_pred: torch.Tensor,   # [B, H, W] sigmoid probabilities
    ) -> torch.Tensor:
        """Compute cup-in-disc consistency loss.
        
        Args:
            disc_pred: Disc segmentation (after sigmoid), [B, H, W]
            cup_pred: Cup segmentation (after sigmoid), [B, H, W]
        
        Returns:
            Scalar loss (mean over batch)
        """
        # Intersection of cup and disc predictions
        cup_in_disc = cup_pred * disc_pred  # [B, H, W]
        
        # Flatten for computation
        cup_flat = cup_pred.flatten(1)  # [B, N]
        cup_in_disc_flat = cup_in_disc.flatten(1)  # [B, N]
        
        # Squared Dice between cup and (cup ∩ disc)
        # If cup ⊆ disc, then cup_in_disc = cup → Dice = 1
        intersection = (cup_flat * cup_in_disc_flat).sum(dim=1)  # [B]
        cup_sq = (cup_flat ** 2).sum(dim=1)  # [B]
        cup_in_disc_sq = (cup_in_disc_flat ** 2).sum(dim=1)  # [B]
        
        dice = (2 * intersection + self.smooth) / (cup_sq + cup_in_disc_sq + self.smooth)  # [B]
        
        # Loss = 1 - Dice (0 when cup is fully inside disc)
        return (1 - dice).mean()


class VesselAVConsistencyLoss(nn.Module):
    """Self-consistency loss: vessel should equal union of artery and vein.
    
    Anatomical constraint: vessel = artery ∪ vein
    Loss = 1 - Dice(vessel_pred, artery_pred ∪ vein_pred)
    
    If vessel = A ∪ V: Dice = 1.0 → Loss = 0
    If mismatch: Dice < 1.0 → Loss > 0
    
    Note: This uses predictions only (no GT required), computed on ALL samples.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        artery_pred: torch.Tensor,  # [B, H, W] sigmoid probabilities
        vein_pred: torch.Tensor,    # [B, H, W] sigmoid probabilities
        vessel_pred: torch.Tensor,  # [B, H, W] sigmoid probabilities
    ) -> torch.Tensor:
        """Compute vessel = artery ∪ vein consistency loss.
        
        Args:
            artery_pred: Artery segmentation (after sigmoid), [B, H, W]
            vein_pred: Vein segmentation (after sigmoid), [B, H, W]
            vessel_pred: Vessel segmentation (after sigmoid), [B, H, W]
        
        Returns:
            Scalar loss (mean over batch)
        """
        # Union of artery and vein (soft OR = clamp(a + b))
        av_union = torch.clamp(artery_pred + vein_pred, 0, 1)  # [B, H, W]
        
        # Flatten for computation
        vessel_flat = vessel_pred.flatten(1)  # [B, N]
        av_union_flat = av_union.flatten(1)  # [B, N]
        
        # Squared Dice between vessel and (artery ∪ vein)
        intersection = (vessel_flat * av_union_flat).sum(dim=1)  # [B]
        vessel_sq = (vessel_flat ** 2).sum(dim=1)  # [B]
        av_union_sq = (av_union_flat ** 2).sum(dim=1)  # [B]
        
        dice = (2 * intersection + self.smooth) / (vessel_sq + av_union_sq + self.smooth)  # [B]
        
        # Loss = 1 - Dice (0 when vessel = artery ∪ vein)
        return (1 - dice).mean()


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
    use_squared_dice: bool = True,
) -> torch.Tensor:
    """Combined BCE + Dice loss per sample.
    
    Args:
        pred: Predictions [B, H, W] (logits, before sigmoid)
        target: Targets [B, H, W]
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        use_squared_dice: If True, use Squared Dice (LFA-Net paper Eq. 27)
        
    Returns:
        Combined loss per sample [B]
    """
    bce = bce_loss_per_sample(pred, target)
    
    # Apply sigmoid for dice computation
    pred_sigmoid = torch.sigmoid(pred)
    
    if use_squared_dice:
        dice = squared_dice_loss_per_sample(pred_sigmoid, target)
    else:
        dice = dice_loss_per_sample(pred, target)
    
    return bce_weight * bce + dice_weight * dice


class MultitaskLoss(nn.Module):
    """Multi-Task Loss with missing label handling.
    
    Handles:
    - Per-channel segmentation loss (BCE + Squared Dice from LFA-Net paper)
    - Anatomical constraint losses (Cup-in-Disc, Vessel=A∪V)
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
        cup_in_disc_weight: Weight for cup-in-disc anatomical constraint
        vessel_av_weight: Weight for vessel=A∪V anatomical constraint
        bce_weight: BCE weight in BCE+Dice combo
        dice_weight: Dice weight in BCE+Dice combo
        use_squared_dice: Use Squared Dice loss from LFA-Net paper
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
        cup_in_disc_weight: float = 0.25,
        vessel_av_weight: float = 0.25,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        use_squared_dice: bool = True,
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
        self.use_squared_dice = use_squared_dice
        
        # Anatomical constraint losses
        self.cup_in_disc_weight = cup_in_disc_weight
        self.vessel_av_weight = vessel_av_weight
        self.cup_in_disc_loss = CupInDiscLoss()
        self.vessel_av_loss = VesselAVConsistencyLoss()
        
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
                    self.bce_weight, self.dice_weight,
                    use_squared_dice=self.use_squared_dice,
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
        # Anatomical Constraint Losses (predictions only, no GT required)
        # =====================================================================
        # Get sigmoid predictions for constraint losses
        seg_pred_sigmoid = torch.sigmoid(seg_pred)  # [B, 5, H, W]
        
        # Channel indices (based on DEFAULT_SEG_NAMES order)
        disc_idx = self.seg_channel_names.index("disc") if "disc" in self.seg_channel_names else None
        cup_idx = self.seg_channel_names.index("cup") if "cup" in self.seg_channel_names else None
        artery_idx = self.seg_channel_names.index("artery") if "artery" in self.seg_channel_names else None
        vein_idx = self.seg_channel_names.index("vein") if "vein" in self.seg_channel_names else None
        vessel_idx = self.seg_channel_names.index("vessel") if "vessel" in self.seg_channel_names else None
        
        # Cup-in-Disc constraint: cup ⊆ disc
        if disc_idx is not None and cup_idx is not None and self.cup_in_disc_weight > 0:
            disc_pred = seg_pred_sigmoid[:, disc_idx]  # [B, H, W]
            cup_pred = seg_pred_sigmoid[:, cup_idx]    # [B, H, W]
            cup_in_disc_loss = self.cup_in_disc_loss(disc_pred, cup_pred) * self.cup_in_disc_weight
        else:
            cup_in_disc_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        
        total_loss = total_loss + cup_in_disc_loss
        loss_dict["cup_in_disc_loss"] = cup_in_disc_loss
        
        # Vessel = Artery ∪ Vein constraint
        if artery_idx is not None and vein_idx is not None and vessel_idx is not None and self.vessel_av_weight > 0:
            artery_pred = seg_pred_sigmoid[:, artery_idx]  # [B, H, W]
            vein_pred = seg_pred_sigmoid[:, vein_idx]      # [B, H, W]
            vessel_pred = seg_pred_sigmoid[:, vessel_idx]  # [B, H, W]
            vessel_av_loss = self.vessel_av_loss(artery_pred, vein_pred, vessel_pred) * self.vessel_av_weight
        else:
            vessel_av_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        
        total_loss = total_loss + vessel_av_loss
        loss_dict["vessel_av_loss"] = vessel_av_loss
        
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

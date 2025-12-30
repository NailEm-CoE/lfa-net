"""PyTorch Lightning Module for MultitaskLFANet.

Features:
- Multi-task training with missing label handling
- Per-task metrics (Dice, MAE, AUROC)
- Conditional logging based on valid samples
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import BinaryF1Score, BinaryAUROC

from .multitask import MultitaskLFANet, count_parameters
from .multitask_losses import MultitaskLoss


class MultitaskLFANetLightning(pl.LightningModule):
    """PyTorch Lightning module for MultitaskLFANet.
    
    Handles:
    - Multi-task training with partial labels
    - Per-task metric computation with missing label handling
    - Conditional logging (only log if valid samples exist)
    
    Args:
        in_channels: Input image channels
        encoder_channels: Encoder channel progression
        decoder_channels: Decoder channel progression
        latent_channels: Skip connection latent channels
        skip_spatial: Skip connection spatial size
        seg_out_channels: Segmentation output channels
        n_disease_classes: Number of disease classes
        seg_channel_names: Names for segmentation channels
        disease_names: Names for disease classes
        seg_weights: Per-channel weights for segmentation loss
        fovea_weight: Weight for fovea loss
        disease_weight: Weight for disease loss
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        dropout: Dropout rate
    """
    
    DEFAULT_SEG_NAMES = ["disc", "cup", "artery", "vein", "vessel"]
    DEFAULT_DISEASE_NAMES = ["dr", "amd", "glaucoma"]
    
    def __init__(
        self,
        # Model architecture
        in_channels: int = 3,
        encoder_channels: Optional[list[int]] = None,
        decoder_channels: Optional[list[int]] = None,
        latent_channels: int = 8,
        skip_spatial: int = 64,
        seg_out_channels: int = 5,
        n_disease_classes: int = 3,
        # Task names
        seg_channel_names: Optional[list[str]] = None,
        disease_names: Optional[list[str]] = None,
        # Loss weights
        seg_weights: Optional[dict[str, float]] = None,
        fovea_weight: float = 0.5,
        disease_weight: float = 0.5,
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Names
        self.seg_channel_names = seg_channel_names or self.DEFAULT_SEG_NAMES
        self.disease_names = disease_names or self.DEFAULT_DISEASE_NAMES
        
        # Build model
        self.model = MultitaskLFANet(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            latent_channels=latent_channels,
            skip_spatial=skip_spatial,
            seg_out_channels=seg_out_channels,
            n_disease_classes=n_disease_classes,
            dropout=dropout,
        )
        
        # Loss function
        self.loss_fn = MultitaskLoss(
            seg_channel_names=self.seg_channel_names,
            disease_names=self.disease_names,
            seg_weights=seg_weights,
            fovea_weight=fovea_weight,
            disease_weight=disease_weight,
        )
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Count parameters
        self.n_params = count_parameters(self.model)
        
        # Create metrics
        self._create_metrics()
    
    def _create_metrics(self):
        """Create per-task metrics."""
        # Segmentation: Per-channel Dice (F1)
        self.train_seg_dice = nn.ModuleDict({
            name: BinaryF1Score() for name in self.seg_channel_names
        })
        self.val_seg_dice = nn.ModuleDict({
            name: BinaryF1Score() for name in self.seg_channel_names
        })
        
        # Track valid sample counts for averaging
        self.train_seg_count = {name: 0 for name in self.seg_channel_names}
        self.val_seg_count = {name: 0 for name in self.seg_channel_names}
        
        # Fovea: MAE
        self.train_fovea_mae = MeanAbsoluteError()
        self.val_fovea_mae = MeanAbsoluteError()
        self.train_fovea_count = 0
        self.val_fovea_count = 0
        
        # Disease: Per-class AUROC (only computed at epoch end)
        self.val_disease_preds = {name: [] for name in self.disease_names}
        self.val_disease_targets = {name: [] for name in self.disease_names}
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)
    
    def _shared_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Shared step for train/val."""
        preds = self(batch["image"])
        loss_dict = self.loss_fn(preds, batch)
        return preds, loss_dict
    
    def _update_seg_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        split: str,
    ):
        """Update segmentation metrics for valid samples."""
        metrics = self.train_seg_dice if split == "train" else self.val_seg_dice
        counts = self.train_seg_count if split == "train" else self.val_seg_count
        
        probs = torch.sigmoid(preds)
        
        for ch_idx, name in enumerate(self.seg_channel_names):
            ch_valid = valid[:, ch_idx].bool()
            n_valid = ch_valid.sum().item()
            
            if n_valid > 0:
                ch_pred = probs[:, ch_idx][ch_valid]
                ch_target = targets[:, ch_idx][ch_valid]
                
                metrics[name].update(
                    ch_pred.flatten(),
                    ch_target.flatten().int()
                )
                counts[name] += n_valid
    
    def _update_fovea_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        split: str,
    ):
        """Update fovea metrics for valid samples."""
        metric = self.train_fovea_mae if split == "train" else self.val_fovea_mae
        
        valid_mask = valid.bool()
        n_valid = valid_mask.sum().item()
        
        if n_valid > 0:
            valid_preds = preds[valid_mask]
            valid_targets = targets[valid_mask]
            
            metric.update(valid_preds, valid_targets)
            
            if split == "train":
                self.train_fovea_count += n_valid
            else:
                self.val_fovea_count += n_valid
    
    def _update_disease_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        split: str,
    ):
        """Accumulate disease predictions for AUROC computation."""
        if split != "val":
            return
        
        probs = torch.sigmoid(preds)
        
        for cls_idx, name in enumerate(self.disease_names):
            cls_valid = valid[:, cls_idx].bool()
            
            if cls_valid.sum() > 0:
                self.val_disease_preds[name].append(probs[:, cls_idx][cls_valid])
                self.val_disease_targets[name].append(targets[:, cls_idx][cls_valid])
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        preds, loss_dict = self._shared_step(batch)
        
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/seg_loss", loss_dict["seg_loss"], on_step=True, on_epoch=True)
        self.log("train/fovea_loss", loss_dict["fovea_loss"], on_step=True, on_epoch=True)
        self.log("train/disease_loss", loss_dict["disease_loss"], on_step=True, on_epoch=True)
        
        for name in self.seg_channel_names:
            key = f"seg_{name}"
            if key in loss_dict:
                self.log(f"train/{key}", loss_dict[key], on_step=True, on_epoch=True)
        
        self._update_seg_metrics(
            preds["segmentation"],
            batch["seg_masks"],
            batch["seg_valid"],
            "train",
        )
        self._update_fovea_metrics(
            preds["fovea"],
            batch["fovea"],
            batch["fovea_valid"],
            "train",
        )
        
        return loss_dict["loss"]
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        dice_scores = []
        for name in self.seg_channel_names:
            if self.train_seg_count[name] > 0:
                dice = self.train_seg_dice[name].compute()
                self.log(f"train/dice_{name}", dice)
                dice_scores.append(dice)
                self.train_seg_dice[name].reset()
            else:
                self.log(f"train/dice_{name}", 0.0)
            self.train_seg_count[name] = 0
        
        if dice_scores:
            avg_dice = sum(dice_scores) / len(dice_scores)
            self.log("train/dice_mean", avg_dice)
        
        if self.train_fovea_count > 0:
            fovea_mae = self.train_fovea_mae.compute()
            self.log("train/fovea_mae", fovea_mae)
            self.train_fovea_mae.reset()
        self.train_fovea_count = 0
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        preds, loss_dict = self._shared_step(batch)
        
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/seg_loss", loss_dict["seg_loss"], on_step=False, on_epoch=True)
        self.log("val/fovea_loss", loss_dict["fovea_loss"], on_step=False, on_epoch=True)
        self.log("val/disease_loss", loss_dict["disease_loss"], on_step=False, on_epoch=True)
        
        self._update_seg_metrics(
            preds["segmentation"],
            batch["seg_masks"],
            batch["seg_valid"],
            "val",
        )
        self._update_fovea_metrics(
            preds["fovea"],
            batch["fovea"],
            batch["fovea_valid"],
            "val",
        )
        self._update_disease_metrics(
            preds["disease"],
            batch["disease_labels"],
            batch["disease_valid"],
            "val",
        )
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        dice_scores = []
        for name in self.seg_channel_names:
            if self.val_seg_count[name] > 0:
                dice = self.val_seg_dice[name].compute()
                self.log(f"val/dice_{name}", dice)
                dice_scores.append(dice)
                self.val_seg_dice[name].reset()
            else:
                self.log(f"val/dice_{name}", 0.0)
            self.val_seg_count[name] = 0
        
        if dice_scores:
            avg_dice = sum(dice_scores) / len(dice_scores)
            self.log("val/dice_mean", avg_dice, prog_bar=True)
        else:
            self.log("val/dice_mean", 0.0, prog_bar=True)
        
        if self.val_fovea_count > 0:
            fovea_mae = self.val_fovea_mae.compute()
            self.log("val/fovea_mae", fovea_mae, prog_bar=True)
            self.val_fovea_mae.reset()
        self.val_fovea_count = 0
        
        for name in self.disease_names:
            if len(self.val_disease_preds[name]) > 0:
                preds = torch.cat(self.val_disease_preds[name])
                targets = torch.cat(self.val_disease_targets[name])
                
                unique_targets = targets.unique()
                if len(unique_targets) >= 2:
                    auroc = BinaryAUROC()
                    auroc_score = auroc(preds, targets.int())
                    self.log(f"val/auroc_{name}", auroc_score)
            
            self.val_disease_preds[name] = []
            self.val_disease_targets[name] = []
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
    
    def on_fit_start(self):
        """Log model info at training start."""
        if self.logger:
            self.logger.log_hyperparams({
                "model/params": self.n_params,
                "model/encoder_channels": self.hparams.encoder_channels,
                "model/latent_channels": self.hparams.latent_channels,
            })

"""PyTorch Lightning Modules for LFA-Net training.

Provides:
- LFANetLightning: Original binary segmentation (backward compatible)
- LFABlockNetLightning: Flexible architecture with Hydra config support
- MulticlassLFANetLightning: Multi-class (artery/vein) segmentation
- BottleneckLFABlockNetLightning: Bottleneck architecture for multi-class
"""

from typing import Any, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryRecall,
    BinarySpecificity,
)

from .losses import bce_dice_loss, multiclass_bce_dice_loss
from .model import LFABlockNet, LFANet
from .models import BottleneckLFABlockNet


class LFANetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for LFA-Net.

    Original 3-level architecture for backward compatibility.
    For flexible architecture, use LFABlockNetLightning.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        feature_scale: int = 2,
        dropout: float = 0.5,
        ra_k: int = 16,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        learning_rate: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            feature_scale: Divisor for filter counts
            dropout: Dropout rate
            ra_k: RA attention k parameter
            focal_gamma: Focal modulation gamma
            focal_alpha: Focal modulation alpha
            learning_rate: Learning rate for Adam optimizer
            bce_weight: Weight for BCE in combined loss
            dice_weight: Weight for Dice in combined loss
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = LFANet(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_scale=feature_scale,
            dropout=dropout,
            ra_k=ra_k,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

        # Loss weights
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.learning_rate = learning_rate

        # Metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup binary metrics."""
        metrics = MetricCollection({
            "dice": BinaryF1Score(),
            "iou": BinaryJaccardIndex(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
        })

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for train/val/test."""
        images, masks = batch
        logits = self(images)
        loss = bce_dice_loss(logits, masks, self.bce_weight, self.dice_weight)
        preds = torch.sigmoid(logits)
        return loss, preds, masks

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(preds, masks.int())
        return loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images, masks = batch
        loss, preds, masks = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(preds, masks.int())

        if batch_idx == 0 and self.logger is not None:
            self._log_val_images(images, masks, preds)

        return loss

    def _log_val_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        num_samples: int = 4,
    ) -> None:
        """Log validation images to TensorBoard."""
        import torchvision

        num_samples = min(num_samples, images.size(0))
        imgs = images[:num_samples].cpu()
        gts = masks[:num_samples].cpu()
        pred_masks = (preds[:num_samples] > 0.5).float().cpu()

        gts_rgb = gts.repeat(1, 3, 1, 1)
        preds_rgb = pred_masks.repeat(1, 3, 1, 1)
        combined = torch.cat([imgs, gts_rgb, preds_rgb], dim=3)
        grid = torchvision.utils.make_grid(combined, nrow=1, normalize=False)

        self.logger.experiment.add_image("val/predictions", grid, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_metrics.update(preds, masks.int())
        return loss

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class LFABlockNetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for LFABlockNet.

    Supports configurable encoder depth and Hydra configuration.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_filters: Optional[list[int]] = None,
        bottleneck_filters: Optional[int] = None,
        dropout: float = 0.5,
        ra_k: int = 16,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        learning_rate: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        log_val_image_n_epoch: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (1 for binary)
            encoder_filters: List of filter counts for each encoder level
            bottleneck_filters: Filter count for bottleneck
            dropout: Dropout rate
            ra_k: RA attention k parameter
            focal_gamma: Focal modulation gamma
            focal_alpha: Focal modulation alpha
            learning_rate: Learning rate for optimizer
            bce_weight: Weight for BCE in combined loss
            dice_weight: Weight for Dice in combined loss
            log_val_image_n_epoch: Log validation images every N epochs
        """
        super().__init__()
        self.save_hyperparameters()

        if encoder_filters is None:
            encoder_filters = [8, 16, 32]

        self.log_val_image_n_epoch = log_val_image_n_epoch

        self.model = LFABlockNet(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_filters=list(encoder_filters),
            bottleneck_filters=bottleneck_filters,
            dropout=dropout,
            ra_k=ra_k,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.learning_rate = learning_rate

        self._setup_metrics()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "LFABlockNetLightning":
        """Create instance from Hydra config."""
        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            encoder_filters=list(cfg.model.encoder_filters),
            bottleneck_filters=cfg.model.get("bottleneck_filters"),
            dropout=cfg.model.dropout,
            ra_k=cfg.model.ra_k,
            focal_gamma=cfg.model.focal_gamma,
            focal_alpha=cfg.model.focal_alpha,
            learning_rate=cfg.train.learning_rate,
            bce_weight=cfg.model.bce_weight,
            dice_weight=cfg.model.dice_weight,
            log_val_image_n_epoch=cfg.train.get("log_val_image_n_epoch", 3),
        )

    def _setup_metrics(self) -> None:
        """Setup binary metrics."""
        metrics = MetricCollection({
            "dice": BinaryF1Score(),
            "iou": BinaryJaccardIndex(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
        })

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for train/val/test."""
        images, masks = batch
        logits = self(images)
        loss = bce_dice_loss(logits, masks, self.bce_weight, self.dice_weight)
        preds = torch.sigmoid(logits)
        return loss, preds, masks

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(preds, masks.int())
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images, masks = batch
        loss, preds, masks = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(preds, masks.int())

        # Log validation images periodically (starting from epoch 0)
        if batch_idx == 0 and self.logger is not None:
            if self.current_epoch % self.log_val_image_n_epoch == 0:
                self._log_val_images(images, masks, preds)

        return loss

    def _log_val_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        num_samples: int = 4,
    ) -> None:
        """Log validation images to TensorBoard."""
        import torchvision

        num_samples = min(num_samples, images.size(0))
        imgs = images[:num_samples].cpu()
        gts = masks[:num_samples].cpu()
        pred_masks = (preds[:num_samples] > 0.5).float().cpu()

        # Green channel for vessel mask
        gts_rgb = torch.zeros(num_samples, 3, gts.size(2), gts.size(3))
        gts_rgb[:, 1:2] = gts  # Green
        preds_rgb = torch.zeros(num_samples, 3, pred_masks.size(2), pred_masks.size(3))
        preds_rgb[:, 1:2] = pred_masks  # Green

        combined = torch.cat([imgs, gts_rgb, preds_rgb], dim=3)
        grid = torchvision.utils.make_grid(combined, nrow=1, normalize=False)

        self.logger.experiment.add_image("val/predictions", grid, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_metrics.update(preds, masks.int())
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class MulticlassLFANetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for multi-class segmentation.

    Supports artery/vein segmentation with per-class metrics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        encoder_filters: Optional[list[int]] = None,
        bottleneck_filters: Optional[int] = None,
        dropout: float = 0.5,
        ra_k: int = 16,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        learning_rate: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_names: Optional[list[str]] = None,
        log_val_image_n_epoch: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (2 for artery/vein)
            encoder_filters: List of filter counts for each encoder level
            bottleneck_filters: Filter count for bottleneck
            dropout: Dropout rate
            ra_k: RA attention k parameter
            focal_gamma: Focal modulation gamma
            focal_alpha: Focal modulation alpha
            learning_rate: Learning rate for optimizer
            bce_weight: Weight for BCE in combined loss
            dice_weight: Weight for Dice in combined loss
            class_names: Names for each class (e.g., ["artery", "vein"])
            log_val_image_n_epoch: Log validation images every N epochs
        """
        super().__init__()
        self.save_hyperparameters()

        if encoder_filters is None:
            encoder_filters = [16, 32, 64]
        if class_names is None:
            class_names = ["artery", "vein"]

        self.class_names = class_names
        self.out_channels = out_channels

        self.model = LFABlockNet(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_filters=list(encoder_filters),
            bottleneck_filters=bottleneck_filters,
            dropout=dropout,
            ra_k=ra_k,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.learning_rate = learning_rate
        self.log_val_image_n_epoch = log_val_image_n_epoch

        self._setup_metrics()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MulticlassLFANetLightning":
        """Create instance from Hydra config."""
        class_names = list(cfg.data.get("class_names", ["artery", "vein"]))
        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            encoder_filters=list(cfg.model.encoder_filters),
            bottleneck_filters=cfg.model.get("bottleneck_filters"),
            dropout=cfg.model.dropout,
            ra_k=cfg.model.ra_k,
            focal_gamma=cfg.model.focal_gamma,
            focal_alpha=cfg.model.focal_alpha,
            learning_rate=cfg.train.learning_rate,
            bce_weight=cfg.model.bce_weight,
            dice_weight=cfg.model.dice_weight,
            class_names=class_names,
            log_val_image_n_epoch=cfg.train.get("log_val_image_n_epoch", 3),
        )

    def _setup_metrics(self) -> None:
        """Setup per-class metrics."""
        for split in ["train", "val", "test"]:
            # Per-class metrics
            for i, name in enumerate(self.class_names):
                setattr(self, f"{split}_{name}_dice", BinaryF1Score())
                setattr(self, f"{split}_{name}_iou", BinaryJaccardIndex())
            # Average metrics
            setattr(self, f"{split}_avg_dice", BinaryF1Score())
            setattr(self, f"{split}_avg_iou", BinaryJaccardIndex())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for train/val/test."""
        images, masks = batch
        logits = self(images)
        loss = multiclass_bce_dice_loss(
            logits, masks, self.bce_weight, self.dice_weight
        )
        preds = torch.sigmoid(logits)
        return loss, preds, masks

    def _update_metrics(self, preds: torch.Tensor, masks: torch.Tensor, split: str) -> None:
        """Update metrics for a split."""
        # Per-class metrics
        for i, name in enumerate(self.class_names):
            pred_c = preds[:, i:i+1]
            mask_c = masks[:, i:i+1]
            getattr(self, f"{split}_{name}_dice").update(pred_c, mask_c.int())
            getattr(self, f"{split}_{name}_iou").update(pred_c, mask_c.int())

        # Average (flatten all classes)
        getattr(self, f"{split}_avg_dice").update(preds, masks.int())
        getattr(self, f"{split}_avg_iou").update(preds, masks.int())

    def _log_metrics(self, split: str, prog_bar: bool = False) -> None:
        """Log and reset metrics for a split."""
        metrics = {}
        for name in self.class_names:
            dice_metric = getattr(self, f"{split}_{name}_dice")
            iou_metric = getattr(self, f"{split}_{name}_iou")
            metrics[f"{split}/{name}_dice"] = dice_metric.compute()
            metrics[f"{split}/{name}_iou"] = iou_metric.compute()
            dice_metric.reset()
            iou_metric.reset()

        avg_dice = getattr(self, f"{split}_avg_dice")
        avg_iou = getattr(self, f"{split}_avg_iou")
        metrics[f"{split}/avg_dice"] = avg_dice.compute()
        metrics[f"{split}/avg_iou"] = avg_iou.compute()
        avg_dice.reset()
        avg_iou.reset()

        self.log_dict(metrics, prog_bar=prog_bar)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, masks, "train")
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train", prog_bar=False)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images, masks = batch
        loss, preds, masks = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, masks, "val")

        # Log validation images periodically (starting from epoch 0)
        if batch_idx == 0 and self.logger is not None:
            if self.current_epoch % self.log_val_image_n_epoch == 0:
                self._log_val_images(images, masks, preds)

        return loss

    def _log_val_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        num_samples: int = 4,
    ) -> None:
        """Log validation images to TensorBoard (multi-class with color coding).

        Colors: artery=red, vein=blue
        """
        import torchvision

        num_samples = min(num_samples, images.size(0))
        imgs = images[:num_samples].cpu()

        # Artery (red), Vein (blue) - combined in single RGB image
        gt_artery = masks[:num_samples, 0:1].cpu()
        gt_vein = masks[:num_samples, 1:2].cpu()
        pred_artery = (preds[:num_samples, 0:1] > 0.5).float().cpu()
        pred_vein = (preds[:num_samples, 1:2] > 0.5).float().cpu()

        # Ground truth: R=artery, G=0, B=vein
        gts_rgb = torch.zeros(num_samples, 3, gt_artery.size(2), gt_artery.size(3))
        gts_rgb[:, 0:1] = gt_artery  # Red channel
        gts_rgb[:, 2:3] = gt_vein    # Blue channel

        # Predictions: R=artery, G=0, B=vein
        preds_rgb = torch.zeros(num_samples, 3, pred_artery.size(2), pred_artery.size(3))
        preds_rgb[:, 0:1] = pred_artery  # Red channel
        preds_rgb[:, 2:3] = pred_vein    # Blue channel

        combined = torch.cat([imgs, gts_rgb, preds_rgb], dim=3)
        grid = torchvision.utils.make_grid(combined, nrow=1, normalize=False)
        self.logger.experiment.add_image("val/predictions", grid, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._update_metrics(preds, masks, "test")
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", prog_bar=False)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/avg_dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class BottleneckLFABlockNetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for BottleneckLFABlockNet.

    Supports multi-class segmentation (artery/vein) with latent skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        encoder_channels: Optional[list[int]] = None,
        decoder_channels: Optional[list[int]] = None,
        latent_channels: int = 8,
        skip_spatial: int = 32,
        n_classes: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_names: Optional[list[str]] = None,
        log_val_image_n_epoch: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (2 for artery/vein)
            encoder_channels: Encoder channel progression
            decoder_channels: Decoder channel progression (auto-derived if None)
            latent_channels: Skip projection latent channels
            skip_spatial: Skip projection spatial size
            n_classes: Number of classes for RAA attention
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            bce_weight: Weight for BCE in combined loss
            dice_weight: Weight for Dice in combined loss
            class_names: Names for each class (e.g., ["artery", "vein"])
            log_val_image_n_epoch: Log validation images every N epochs
        """
        super().__init__()
        self.save_hyperparameters()

        if encoder_channels is None:
            encoder_channels = [32, 48, 72, 144]
        if class_names is None:
            class_names = ["artery", "vein"]

        self.class_names = class_names
        self.out_channels = out_channels

        self.model = BottleneckLFABlockNet(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=list(encoder_channels),
            decoder_channels=list(decoder_channels) if decoder_channels else None,
            n_classes=n_classes,
            latent_channels=latent_channels,
            skip_spatial=skip_spatial,
            use_sigmoid=False,  # Use logits for BCE loss
            dropout=dropout,
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.learning_rate = learning_rate
        self.log_val_image_n_epoch = log_val_image_n_epoch

        self._setup_metrics()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "BottleneckLFABlockNetLightning":
        """Create instance from Hydra config."""
        class_names = list(cfg.data.get("class_names", ["artery", "vein"]))
        
        # Handle decoder_channels (can be null in yaml)
        decoder_channels = cfg.model.get("decoder_channels")
        if decoder_channels is not None:
            decoder_channels = list(decoder_channels)
        
        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            encoder_channels=list(cfg.model.encoder_channels),
            decoder_channels=decoder_channels,
            latent_channels=cfg.model.get("latent_channels", 8),
            skip_spatial=cfg.model.get("skip_spatial", 32),
            n_classes=cfg.model.get("n_classes", 2),
            dropout=cfg.model.get("dropout", 0.1),
            learning_rate=cfg.train.learning_rate,
            bce_weight=cfg.model.get("bce_weight", 0.5),
            dice_weight=cfg.model.get("dice_weight", 0.5),
            class_names=class_names,
            log_val_image_n_epoch=cfg.train.get("log_val_image_n_epoch", 3),
        )

    def _setup_metrics(self) -> None:
        """Setup per-class metrics."""
        for split in ["train", "val", "test"]:
            for i, name in enumerate(self.class_names):
                setattr(self, f"{split}_{name}_dice", BinaryF1Score())
                setattr(self, f"{split}_{name}_iou", BinaryJaccardIndex())
            setattr(self, f"{split}_avg_dice", BinaryF1Score())
            setattr(self, f"{split}_avg_iou", BinaryJaccardIndex())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for train/val/test."""
        images, masks = batch
        logits = self(images)
        loss = multiclass_bce_dice_loss(
            logits, masks, self.bce_weight, self.dice_weight
        )
        preds = torch.sigmoid(logits)
        return loss, preds, masks

    def _update_metrics(self, preds: torch.Tensor, masks: torch.Tensor, split: str) -> None:
        """Update metrics for a split."""
        for i, name in enumerate(self.class_names):
            pred_c = preds[:, i:i+1]
            mask_c = masks[:, i:i+1]
            getattr(self, f"{split}_{name}_dice").update(pred_c, mask_c.int())
            getattr(self, f"{split}_{name}_iou").update(pred_c, mask_c.int())
        
        # Average across classes
        getattr(self, f"{split}_avg_dice").update(preds, masks.int())
        getattr(self, f"{split}_avg_iou").update(preds, masks.int())

    def _log_metrics(self, split: str, prog_bar: bool = False) -> None:
        """Log metrics for a split."""
        for name in self.class_names:
            dice = getattr(self, f"{split}_{name}_dice").compute()
            iou = getattr(self, f"{split}_{name}_iou").compute()
            self.log(f"{split}/{name}_dice", dice, prog_bar=False)
            self.log(f"{split}/{name}_iou", iou, prog_bar=False)
            getattr(self, f"{split}_{name}_dice").reset()
            getattr(self, f"{split}_{name}_iou").reset()
        
        avg_dice = getattr(self, f"{split}_avg_dice").compute()
        avg_iou = getattr(self, f"{split}_avg_iou").compute()
        self.log(f"{split}/avg_dice", avg_dice, prog_bar=prog_bar)
        self.log(f"{split}/avg_iou", avg_iou, prog_bar=False)
        getattr(self, f"{split}_avg_dice").reset()
        getattr(self, f"{split}_avg_iou").reset()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, masks, "train")
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self._update_metrics(preds, masks, "val")
        
        # Log comparison images on first batch
        if batch_idx == 0 and (self.current_epoch + 1) % self.log_val_image_n_epoch == 0:
            self._log_comparison_images(batch[0], preds, masks)
        
        return loss

    def _log_comparison_images(
        self, 
        images: torch.Tensor, 
        preds: torch.Tensor, 
        masks: torch.Tensor,
        n_images: int = 4,
    ) -> None:
        """Log side-by-side comparison of predictions vs ground truth."""
        import torchvision
        
        n = min(n_images, images.shape[0])
        
        # For each class, create comparison grid
        for i, name in enumerate(self.class_names):
            pred_c = (preds[:n, i:i+1] > 0.5).float()  # [N, 1, H, W]
            mask_c = masks[:n, i:i+1].float()  # [N, 1, H, W]
            
            # Create side-by-side: [pred, gt] for each sample
            # Stack horizontally: pred | gt
            comparison = torch.cat([pred_c, mask_c], dim=3)  # [N, 1, H, W*2]
            
            # Make grid
            grid = torchvision.utils.make_grid(comparison, nrow=1, padding=2, normalize=False)
            
            # Log to tensorboard
            if self.logger is not None:
                self.logger.experiment.add_image(
                    f"val/{name}_pred_vs_gt",
                    grid,
                    self.current_epoch,
                )

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, masks = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._update_metrics(preds, masks, "test")
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", prog_bar=False)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

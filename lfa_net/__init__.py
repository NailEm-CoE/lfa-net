"""LFA-Net: Local Feature Aggregation Network for retinal vessel segmentation.

This package provides a PyTorch implementation of LFA-Net for medical image
segmentation, specifically designed for retinal vessel segmentation tasks.

Example:
    >>> from lfa_net import LFANet, LFANetLightning
    >>> from lfa_net.data import HFVesselDataModule
    >>> 
    >>> # Create model
    >>> model = LFANet()
    >>> 
    >>> # Or use Lightning wrapper for training
    >>> lit_model = LFANetLightning(learning_rate=1e-3)
    >>>
    >>> # Flexible architecture with Hydra config
    >>> from lfa_net import LFABlockNet, LFABlockNetLightning
    >>> model = LFABlockNet(encoder_filters=[16, 32, 64, 128])
    >>>
    >>> # Bottleneck architecture (uniform skip projections)
    >>> from lfa_net import BottleneckLFABlockNet
    >>> model = BottleneckLFABlockNet(encoder_channels=[32, 48, 72, 144])
    >>>
    >>> # Multi-class artery/vein segmentation
    >>> from lfa_net import MulticlassLFANetLightning
    >>> from lfa_net.data import AVVesselDataModule
    >>>
    >>> # Multi-task fundus analysis (segmentation + fovea + disease)
    >>> from lfa_net import MultitaskLFANet, MultitaskLFANetLightning
    >>> from lfa_net.data import MultitaskDataModule
    >>> model = MultitaskLFANet()
    >>> out = model(torch.randn(1, 3, 512, 512))
    >>> # out["segmentation"], out["fovea"], out["disease"]
"""

from .layers import (
    ContextAggregation,
    FocalModulation,
    LiteFusionAttention,
    MultiScaleConvBlock,
    RAAttentionBlock,
    SEMAttentionBlock,
    VisionMambaInspired,
)
from .lightning_module import (
    LFABlockNetLightning,
    BottleneckLFABlockNetLightning,
    LFANetLightning,
    MulticlassLFANetLightning,
)
from .losses import (
    BCEDiceLoss,
    MulticlassBCEDiceLoss,
    bce_dice_loss,
    dice_loss,
    multiclass_bce_dice_loss,
    multiclass_dice_loss,
    per_class_dice,
)
from .metrics import get_metrics
from .model import LFABlockNet, LFANet, count_parameters
from .models import (
    BottleneckLFABlockNet,
    MultitaskLFANet,
    MultitaskLFANetLightning,
    MultitaskLoss,
    # Enhanced heads from Step 18
    DualSegmentationHead,
    EnhancedFoveaHead,
    EnhancedDiseaseHead,
    LFAFusionBlock,
    # Fovea-related heads (predict_od_center mode)
    FoveaRelatedHead,
    EnhancedFoveaRelatedHead,
    MultiScaleFeatureFusion,
)

__version__ = "0.5.0"

__all__ = [
    # Model
    "LFANet",
    "LFABlockNet",
    "BottleneckLFABlockNet",
    "MultitaskLFANet",
    "LFANetLightning",
    "LFABlockNetLightning",
    "MulticlassLFANetLightning",
    "BottleneckLFABlockNetLightning",
    "MultitaskLFANetLightning",
    "count_parameters",
    # Layers
    "MultiScaleConvBlock",
    "FocalModulation",
    "ContextAggregation",
    "VisionMambaInspired",
    "LiteFusionAttention",
    "RAAttentionBlock",
    "SEMAttentionBlock",
    # Losses
    "dice_loss",
    "bce_dice_loss",
    "BCEDiceLoss",
    "multiclass_dice_loss",
    "multiclass_bce_dice_loss",
    "per_class_dice",
    "MulticlassBCEDiceLoss",
    # Multitask
    "MultitaskLoss",
    # Enhanced multitask heads (Step 18)
    "DualSegmentationHead",
    "EnhancedFoveaHead",
    "EnhancedDiseaseHead",
    "LFAFusionBlock",
    # Fovea-related heads (predict_od_center mode)
    "FoveaRelatedHead",
    "EnhancedFoveaRelatedHead",
    "MultiScaleFeatureFusion",
    # Metrics
    "get_metrics",
]

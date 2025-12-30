"""LFA-Net model variants.

This module provides different LFA-Net architectures:
- BottleneckLFABlockNet: Bottleneck architecture with latent skip connections (from research/lfa_net_v3)
"""

from .latent_skip import (
    BottleneckLFABlockNet,
    LFABlock,
    LFADecoder,
    LFADecoderBlock,
    LFAEncoder,
    LiteFusionAttention,
    RegionAwareAttention,
    SkipProjection,
    SkipUnprojection,
)
from .multitask import (
    MultitaskLFANet,
    AsinhLeakySigmoid,
    SegmentationHead,
    FoveaHead,
    DiseaseHead,
    count_parameters,
)
from .multitask_losses import (
    MultitaskLoss,
    safe_mean,
    dice_loss_per_sample,
    bce_loss_per_sample,
    dice_bce_loss_per_sample,
)
from .multitask_lightning import MultitaskLFANetLightning

__all__ = [
    # Main models
    "BottleneckLFABlockNet",
    "MultitaskLFANet",
    "MultitaskLFANetLightning",
    # Encoder components
    "LFABlock",
    "LFAEncoder",
    "LiteFusionAttention",
    "RegionAwareAttention",
    "SkipProjection",
    # Decoder components
    "LFADecoderBlock",
    "LFADecoder",
    "SkipUnprojection",
    # Multitask heads
    "AsinhLeakySigmoid",
    "SegmentationHead",
    "FoveaHead",
    "DiseaseHead",
    # Multitask losses
    "MultitaskLoss",
    "safe_mean",
    "dice_loss_per_sample",
    "bce_loss_per_sample",
    "dice_bce_loss_per_sample",
    # Utils
    "count_parameters",
]

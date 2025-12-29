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

__all__ = [
    # Main model
    "BottleneckLFABlockNet",
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
]

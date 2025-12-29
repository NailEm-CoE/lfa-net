"""LFA-Net model variants.

This module provides different LFA-Net architectures:
- LatentSkipLFANet: Latent skip connection architecture (from research/lfa_net_v3)
"""

from .latent_skip import (
    LatentSkipLFANet,
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
    "LatentSkipLFANet",
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

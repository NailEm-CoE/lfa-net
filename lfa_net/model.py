"""LFA-Net model architecture using PyTorch.

This module provides both:
- LFANet: Original 3-level architecture (backward compatible)
- FlexibleLFANet: Configurable depth architecture (3-6 levels)
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import (
    LiteFusionAttention,
    MultiScaleConvBlock,
    RAAttentionBlock,
)


class LFANet(nn.Module):
    """
    LFA-Net: Local Feature Aggregation Network for retinal vessel segmentation.
    
    This is the original 3-level architecture for backward compatibility.
    For flexible depth, use FlexibleLFANet.
    
    Architecture:
        - Encoder: 3 stages with MultiScaleConvBlock + MaxPool + BN
        - Bottleneck: LiteFusionAttention
        - Decoder: ConvTranspose + RA attention skip connections
        - Output: logits (no sigmoid - use BCEWithLogitsLoss)
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
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for binary)
            feature_scale: Divisor for filter counts (2 = [8, 16, 32])
            dropout: Dropout rate
            ra_k: RA attention k parameter
            focal_gamma: Focal modulation gamma
            focal_alpha: Focal modulation alpha
        """
        super().__init__()
        
        # Filter sizes: [16, 32, 64] / feature_scale
        filters = [16 // feature_scale, 32 // feature_scale, 64 // feature_scale]
        self.filters = filters
        
        # Encoder
        self.enc1 = MultiScaleConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.enc2 = MultiScaleConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.enc3 = MultiScaleConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        # Bottleneck - LiteFusion Attention
        bottleneck_filters = 32  # Fixed as per paper
        self.bottleneck = LiteFusionAttention(
            in_channels=filters[2],
            filters=bottleneck_filters,
            gamma=focal_gamma,
            alpha=focal_alpha,
            dropout=dropout,
        )
        
        # RA Attention for skip connections
        self.ra_bottleneck = RAAttentionBlock(bottleneck_filters, n_classes=num_classes, k=ra_k)
        self.ra_enc2 = RAAttentionBlock(filters[1], n_classes=num_classes, k=ra_k)
        self.ra_enc1 = RAAttentionBlock(filters[0], n_classes=num_classes, k=ra_k)
        
        # Decoder
        # After bottleneck concat: bottleneck_filters + bottleneck_filters = 64
        self.up1 = nn.ConvTranspose2d(bottleneck_filters * 2, filters[2], 3, stride=2, padding=1, output_padding=1)
        # After concat with ra_enc2: filters[2] + filters[1]
        self.dec1_conv = nn.Conv2d(filters[2] + filters[1], filters[2], 3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[2], 3, stride=2, padding=1, output_padding=1)
        # After concat with ra_enc1: filters[2] + filters[0]
        self.dec2_conv = nn.Conv2d(filters[2] + filters[0], filters[2], 3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(filters[2], filters[0], 3, stride=2, padding=1, output_padding=1)
        self.dec3_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        
        # Output - logits (no sigmoid)
        self.output = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Logits tensor [B, 1, H, W]
        """
        # Encoder
        e1 = self.bn1(self.pool1(self.enc1(x)))      # [B, 8, H/2, W/2]
        e2 = self.bn2(self.pool2(self.enc2(e1)))     # [B, 16, H/4, W/4]
        e3 = self.bn3(self.pool3(self.enc3(e2)))     # [B, 32, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(e3)                       # [B, 32, H/8, W/8]
        
        # RA attention on bottleneck
        b_att = self.ra_bottleneck(b)                 # [B, 32, H/8, W/8]
        b_fused = torch.cat([b_att, b], dim=1)        # [B, 64, H/8, W/8]
        
        # Decoder stage 1
        d1 = nn.functional.relu(self.up1(b_fused))    # [B, 32, H/4, W/4]
        e2_att = self.ra_enc2(e2)                     # [B, 16, H/4, W/4]
        d1 = torch.cat([e2_att, d1], dim=1)           # [B, 48, H/4, W/4]
        d1 = nn.functional.relu(self.dec1_conv(d1))   # [B, 32, H/4, W/4]
        
        # Decoder stage 2
        d2 = nn.functional.relu(self.up2(d1))         # [B, 32, H/2, W/2]
        e1_att = self.ra_enc1(e1)                     # [B, 8, H/2, W/2]
        d2 = torch.cat([e1_att, d2], dim=1)           # [B, 40, H/2, W/2]
        d2 = nn.functional.relu(self.dec2_conv(d2))   # [B, 32, H/2, W/2]
        
        # Decoder stage 3
        d3 = nn.functional.relu(self.up3(d2))         # [B, 8, H, W]
        d3 = nn.functional.relu(self.dec3_conv(d3))   # [B, 8, H, W]
        
        # Output
        out = self.output(d3)                         # [B, 1, H, W]
        
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FlexibleLFANet(nn.Module):
    """
    Flexible-depth LFA-Net with configurable encoder/decoder levels.
    
    Supports 3-6 levels with individually configurable filter counts.
    
    Architecture:
        - Encoder: N stages with MultiScaleConvBlock + MaxPool + BN
        - Bottleneck: LiteFusionAttention
        - Decoder: ConvTranspose + RA attention skip connections
        - Output: logits (no sigmoid - use BCEWithLogitsLoss)
    
    Example configurations:
        - Paper default: [8, 16, 32] (3 levels, 0.097M params)
        - Light: [16, 32, 64] (3 levels, 0.384M params)
        - Deep: [16, 32, 64, 128] (4 levels, 1.45M params)
        - Very deep: [8, 16, 32, 64, 128] (5 levels, 1.46M params)
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
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary, 2 for AV)
            encoder_filters: List of filter counts for each encoder level.
                           Defaults to [8, 16, 32] (paper configuration).
                           Supports 3-6 levels.
            bottleneck_filters: Filter count for bottleneck. Defaults to last encoder filter.
            dropout: Dropout rate
            ra_k: RA attention k parameter
            focal_gamma: Focal modulation gamma
            focal_alpha: Focal modulation alpha
        """
        super().__init__()
        
        # Default filters matching paper
        if encoder_filters is None:
            encoder_filters = [8, 16, 32]
        
        # Validate levels
        num_levels = len(encoder_filters)
        if not 3 <= num_levels <= 6:
            raise ValueError(f"encoder_filters must have 3-6 levels, got {num_levels}")
        
        self.encoder_filters = list(encoder_filters)
        self.num_levels = num_levels
        
        # Bottleneck defaults to last encoder filter
        if bottleneck_filters is None:
            bottleneck_filters = encoder_filters[-1]
        self.bottleneck_filters = bottleneck_filters
        
        # Build encoder dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_ch = in_channels
        for i, out_ch in enumerate(encoder_filters):
            self.encoders.append(MultiScaleConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        
        # Bottleneck - LiteFusion Attention
        self.bottleneck = LiteFusionAttention(
            in_channels=encoder_filters[-1],
            filters=bottleneck_filters,
            gamma=focal_gamma,
            alpha=focal_alpha,
            dropout=dropout,
        )
        
        # RA Attention for skip connections
        self.ra_bottleneck = RAAttentionBlock(bottleneck_filters, n_classes=out_channels, k=ra_k)
        self.ra_encoders = nn.ModuleList([
            RAAttentionBlock(f, n_classes=out_channels, k=ra_k)
            for f in encoder_filters[:-1]  # Skip last (connected to bottleneck)
        ])
        
        # Build decoder dynamically (reverse of encoder)
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        
        # First decoder: from bottleneck (doubled due to RA concat)
        prev_ch = bottleneck_filters * 2
        
        for i in range(num_levels - 1, -1, -1):
            out_ch = encoder_filters[i]
            self.ups.append(
                nn.ConvTranspose2d(prev_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
            )
            
            if i > 0:
                # Concat with RA-processed skip from encoder[i-1]
                skip_ch = encoder_filters[i - 1]
                self.dec_convs.append(nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1))
                prev_ch = out_ch
            else:
                # Last decoder stage, no skip
                self.dec_convs.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
                prev_ch = out_ch
        
        # Output layer
        self.output = nn.Conv2d(encoder_filters[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Logits tensor [B, out_channels, H, W]
        """
        # Encoder: collect features at each level
        encoder_features = []
        feat = x
        for enc, pool, bn in zip(self.encoders, self.pools, self.bns):
            feat = bn(pool(enc(feat)))
            encoder_features.append(feat)
        
        # Bottleneck
        b = self.bottleneck(encoder_features[-1])
        
        # RA attention on bottleneck and concat
        b_att = self.ra_bottleneck(b)
        d = torch.cat([b_att, b], dim=1)  # [B, 2*bottleneck_filters, H, W]
        
        # Decoder: work backwards through encoder features
        for i, (up, dec_conv) in enumerate(zip(self.ups, self.dec_convs)):
            d = nn.functional.relu(up(d))
            
            # Skip connection (except last stage)
            enc_idx = self.num_levels - 2 - i  # Index into encoder_features (excluding last)
            if enc_idx >= 0:
                skip = self.ra_encoders[enc_idx](encoder_features[enc_idx])
                d = torch.cat([skip, d], dim=1)
            
            d = nn.functional.relu(dec_conv(d))
        
        # Output
        return self.output(d)


if __name__ == "__main__":
    # Quick test
    print("Testing LFANet (original):")
    model = LFANet()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {count_parameters(model) / 1e6:.3f}M")
    
    print("\nTesting FlexibleLFANet configurations:")
    
    configs = [
        ("Paper [8,16,32]", [8, 16, 32]),
        ("Light [16,32,64]", [16, 32, 64]),
        ("Deep [16,32,64,128]", [16, 32, 64, 128]),
        ("Very deep [8,16,32,64,128]", [8, 16, 32, 64, 128]),
    ]
    
    for name, filters in configs:
        model = FlexibleLFANet(encoder_filters=filters)
        x = torch.randn(1, 3, 512, 512)
        y = model(x)
        print(f"  {name}: {y.shape}, {count_parameters(model) / 1e6:.3f}M params")

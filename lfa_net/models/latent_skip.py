"""LFA-Net with Latent Skip Connections.

This module provides the latent skip connection architecture where all skip
connections are projected to a uniform latent space [B, latent_ch, S, S]
before the decoder, enabling more flexible and efficient skip connection handling.

Architecture:
- Encoder: LFABlocks with Multi-Scale Conv + LiteFusion + RAA
- All skip connections projected to uniform [B, latent_ch, S, S] space
- Decoder: LFADecoderBlocks with Skip Unprojection + RAA + Refinement

Example:
    >>> from lfa_net.models import BottleneckLFABlockNet
    >>> model = BottleneckLFABlockNet(out_channels=1)  # Binary segmentation
    >>> x = torch.randn(1, 3, 512, 512)
    >>> pred = model(x)  # [1, 1, 512, 512]
    
    >>> # Multi-class (artery/vein)
    >>> model = BottleneckLFABlockNet(out_channels=2, use_sigmoid=True)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Sub-Components
# =============================================================================


class FocalModulation(nn.Module):
    """Adaptive channel attention with learnable scaling.

    Operation:
        modulation = sigmoid(conv1x1((max_pool - avg_pool) * alpha))
        output = input * modulation * scale

    Args:
        channels: Number of input/output channels
        alpha: Modulation strength factor
    """

    def __init__(self, channels: int, alpha: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.modulation_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply focal modulation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Modulated tensor [B, C, H, W]
        """
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        diff = (max_pool - avg_pool) * self.alpha
        modulation = torch.sigmoid(self.modulation_conv(diff))

        return x * modulation * self.scale


class ContextAggregation(nn.Module):
    """Combine local and global context features.

    Operation:
        local = conv3x3(input)
        global_ctx = sigmoid(conv1x1(GAP(conv1x1(input))))
        fused = local * global_ctx
        output = concat(local, focal_mod(fused))

    Args:
        in_channels: Input channels
        out_channels: Output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.local_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.global_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.global_gate = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.focal_mod = FocalModulation(out_channels)
        self.out_proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply context aggregation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Context-aggregated tensor [B, out_channels, H, W]
        """
        local = self.local_conv(x)

        global_feat = self.global_proj(x)
        global_ctx = F.adaptive_avg_pool2d(global_feat, 1)
        global_ctx = torch.sigmoid(self.global_gate(global_ctx))

        fused = local * global_ctx
        fused_modulated = self.focal_mod(fused)

        out = torch.cat([local, fused_modulated], dim=1)
        return self.out_proj(out)


class TokenMixer(nn.Module):
    """Spatial token mixing using depthwise convolution.

    Operation:
        LayerNorm → DWConv → GELU → Dropout → Residual

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + shortcut


class ChannelMixer(nn.Module):
    """Channel mixing with expansion factor.

    Operation:
        LayerNorm → Dense(4x) → GELU → Dropout → Dense → Residual

    Args:
        channels: Number of channels
        expansion: Hidden layer expansion factor
        dropout: Dropout rate
    """

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + shortcut


class VisionMambaBlock(nn.Module):
    """Vision Mamba-inspired block for efficient token/channel mixing.

    Components:
        - TokenMixer: Depthwise spatial mixing
        - ChannelMixer: Dense channel mixing

    Args:
        channels: Number of channels
        expansion: Channel mixer expansion factor
        dropout: Dropout rate
    """

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.token_mixer = TokenMixer(channels, dropout=dropout)
        self.channel_mixer = ChannelMixer(channels, expansion=expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


class LiteFusionAttention(nn.Module):
    """LiteFusion Attention Module (Bottleneck).

    Components:
        1. Channel projection (1x1 conv)
        2. LayerNorm + 3x3 conv
        3. Focal Modulation + Context Aggregation
        4. Back projection (1x1 conv)
        5. Residual add
        6. Vision Mamba Block

    Args:
        in_channels: Input channels
        out_channels: Output channels (defaults to in_channels)
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.in_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.focal_mod = FocalModulation(out_channels)
        self.context_agg = ContextAggregation(out_channels, out_channels)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.skip_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.mamba_block = VisionMambaBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_proj(x)

        x = self.in_proj(x)
        x = self.norm(x)
        x = self.conv(x)
        x = self.focal_mod(x)
        x = self.context_agg(x)
        x = self.out_proj(x)
        x = x + identity
        x = self.mamba_block(x)

        return x


class RegionAwareAttention(nn.Module):
    """Region-Aware Attention (RAA) for skip connections.

    Operation:
        F = conv3x3(input)
        x1 = GAP(F), x2 = GMP(F)
        S = mean(reshape(x1 * x2, [n_classes, k]))
        M = mean(reshape(F, [H, W, n_classes, k]) * S)
        output = input * sigmoid(M)

    Args:
        channels: Input channels
        n_classes: Number of output classes (1 for binary)
        k: Attention dimension
    """

    def __init__(self, channels: int, n_classes: int = 1, k: int = 16):
        super().__init__()
        self.n_classes = n_classes
        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(channels, k * n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(k * n_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x

        feat = self.conv(x)

        x1 = F.adaptive_max_pool2d(feat, 1)
        x2 = F.adaptive_avg_pool2d(feat, 1)

        S = x1 * x2
        S = S.view(B, self.n_classes, self.k).mean(dim=2, keepdim=True)
        feat_reshaped = feat.view(B, self.n_classes, self.k, H, W)
        S = S.unsqueeze(-1).unsqueeze(-1)
        M = (feat_reshaped * S).mean(dim=2).mean(dim=1, keepdim=True)
        M = torch.sigmoid(M)

        return identity * M


# =============================================================================
# Multi-Scale Convolution Block
# =============================================================================


class MultiScaleConv(nn.Module):
    """Multi-scale feature extraction using parallel convolutions.

    Branches:
        - 1x1 conv: Point-wise features
        - 3x3 conv: Local features
        - 3x3 dilated conv (d=2): Contextual features

    Args:
        in_channels: Input channels
        out_channels: Output channels
        dilation: Dilation rate for the dilated convolution
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - branch_channels * 3

        self.conv1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv3x3_dilated = nn.Conv2d(
            in_channels,
            branch_channels + remainder,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)

        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv3x3_dilated(x)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn(out)
        out = self.act(out + identity)

        return out


# =============================================================================
# Skip Projection Components
# =============================================================================


class SkipProjection(nn.Module):
    """Projects features to fixed skip connection dimensions.

    Converts [B, C, H, W] -> [B, latent_channels, spatial, spatial]

    Args:
        in_channels: Input channels
        latent_channels: Output channels for latent space
        spatial_size: Output spatial size
        mode: Interpolation mode for resizing
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 8,
        spatial_size: int = 32,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.mode = mode

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.spatial_size or x.shape[3] != self.spatial_size:
            x = F.interpolate(
                x,
                size=(self.spatial_size, self.spatial_size),
                mode=self.mode,
                align_corners=False if self.mode != "nearest" else None,
            )
        return self.proj(x)


class SkipUnprojection(nn.Module):
    """Unprojects skip connection from latent space back to decoder dimensions.

    Converts [B, latent_channels, skip_spatial, skip_spatial]
          -> [B, out_channels, target_H, target_W]

    Args:
        latent_channels: Input latent channels
        out_channels: Output channels
        mode: Interpolation mode for resizing
    """

    def __init__(
        self,
        latent_channels: int = 8,
        out_channels: int = 64,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.mode = mode

        self.unproj = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels * 4, kernel_size=1),
            nn.BatchNorm2d(latent_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_channels * 4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Unproject skip connection.

        Args:
            x: Skip tensor [B, latent_channels, S, S]
            target_size: (H, W) to resize to

        Returns:
            Unprojected tensor [B, out_channels, H, W]
        """
        if x.shape[2] != target_size[0] or x.shape[3] != target_size[1]:
            x = F.interpolate(
                x,
                size=target_size,
                mode=self.mode,
                align_corners=False if self.mode != "nearest" else None,
            )
        return self.unproj(x)


# =============================================================================
# LFA Block (Encoder)
# =============================================================================


class LFABlock(nn.Module):
    """LFA Block combining Multi-Scale Conv, LiteFusion, and RAA.

    Outputs:
        - skip: For skip connections [B, latent_channels, skip_spatial, skip_spatial]
        - out: For next block [B, out_channels, H//2, W//2]

    Args:
        in_channels: Input channels
        out_channels: Output channels
        latent_channels: Skip connection output channels
        skip_spatial: Skip connection spatial size
        n_classes: Number of segmentation classes (for RAA)
        raa_k: Attention dimension for RAA
        use_litefusion: Whether to apply LiteFusion Attention
        dropout: Dropout rate
        downsample_mode: 'maxpool', 'avgpool', or 'conv'
        skip_resize_mode: Interpolation mode for skip projection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int = 8,
        skip_spatial: int = 32,
        n_classes: int = 1,
        raa_k: int = 16,
        use_litefusion: bool = True,
        dropout: float = 0.1,
        downsample_mode: str = "maxpool",
        skip_resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.use_litefusion = use_litefusion

        self.multiscale_conv = MultiScaleConv(in_channels, out_channels)
        self.litefusion = (
            LiteFusionAttention(out_channels, dropout=dropout)
            if use_litefusion
            else nn.Identity()
        )
        self.raa = RegionAwareAttention(out_channels, n_classes=n_classes, k=raa_k)
        self.skip_proj = SkipProjection(
            in_channels=out_channels,
            latent_channels=latent_channels,
            spatial_size=skip_spatial,
            mode=skip_resize_mode,
        )

        if downsample_mode == "maxpool":
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == "avgpool":
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        elif downsample_mode == "conv":
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            skip: [B, latent_channels, skip_spatial, skip_spatial]
            out: [B, out_channels, H//2, W//2]
        """
        x = self.multiscale_conv(x)
        x = self.litefusion(x)

        skip = self.raa(x)
        skip = self.skip_proj(skip)

        out = self.downsample(x)

        return skip, out


# =============================================================================
# Encoder
# =============================================================================


class LFAEncoder(nn.Module):
    """Encoder using stacked LFABlocks.

    All skip connections have uniform shape: [B, latent_channels, skip_spatial, skip_spatial]

    Args:
        in_channels: Input image channels
        channels: Channel progression (e.g., [32, 48, 72, 144])
        n_classes: Number of segmentation classes
        latent_channels: Skip connection latent channels
        skip_spatial: Skip connection spatial size
        dropout: Dropout rate
    """

    DEFAULT_CHANNELS = [32, 48, 72, 144]

    def __init__(
        self,
        in_channels: int = 3,
        channels: Optional[list[int]] = None,
        n_classes: int = 1,
        latent_channels: int = 8,
        skip_spatial: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels or self.DEFAULT_CHANNELS
        self.n_levels = len(self.channels)
        self.latent_channels = latent_channels
        self.skip_spatial = skip_spatial

        self.blocks = nn.ModuleList()
        ch_in = in_channels

        for ch_out in self.channels:
            self.blocks.append(
                LFABlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    latent_channels=latent_channels,
                    skip_spatial=skip_spatial,
                    n_classes=n_classes,
                    use_litefusion=True,
                    dropout=dropout,
                )
            )
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input [B, C, H, W]

        Returns:
            out: Final encoded features
            skips: Dict of skip connections {'level1': tensor, ...}
        """
        skips = {}

        for i, block in enumerate(self.blocks):
            skip, x = block(x)
            skips[f"level{i + 1}"] = skip

        return x, skips


# =============================================================================
# Decoder Block
# =============================================================================


class LFADecoderBlock(nn.Module):
    """LFA Decoder Block with skip unprojection and refinement.

    Components:
        1. Upsamples input from previous decoder layer
        2. Unprojects skip connection from latent space
        3. Applies RAA to skip
        4. Merges upsampled features with skip
        5. Applies convolution refinement

    Args:
        in_channels: Input channels from previous decoder layer
        out_channels: Output channels
        latent_channels: Skip connection latent channels
        n_classes: Number of segmentation classes
        raa_k: Attention dimension for RAA
        upsample_mode: 'convtranspose', 'bilinear', or 'nearest'
        merge_mode: 'add' or 'concat'
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int = 8,
        n_classes: int = 1,
        raa_k: int = 16,
        upsample_mode: str = "convtranspose",
        merge_mode: str = "add",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.merge_mode = merge_mode

        if upsample_mode == "convtranspose":
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        elif upsample_mode == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        elif upsample_mode == "nearest":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self.skip_unproj = SkipUnprojection(
            latent_channels=latent_channels, out_channels=out_channels
        )
        self.raa = RegionAwareAttention(out_channels, n_classes=n_classes, k=raa_k)

        if merge_mode == "concat":
            self.merge_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.merge_conv = nn.Identity()

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input from previous decoder [B, in_ch, H, W]
            skip: Skip connection [B, latent_channels, S, S]

        Returns:
            Output tensor [B, out_channels, H*2, W*2]
        """
        x = self.upsample(x)

        target_size = (x.shape[2], x.shape[3])
        skip = self.skip_unproj(skip, target_size)
        skip = self.raa(skip)

        if self.merge_mode == "concat":
            x = torch.cat([x, skip], dim=1)
            x = self.merge_conv(x)
        else:
            x = x + skip

        x = self.refine(x)

        return x


# =============================================================================
# Decoder
# =============================================================================


class LFADecoder(nn.Module):
    """LFA Decoder using stacked LFADecoderBlocks.

    Args:
        in_channels: Bottleneck input channels
        decoder_channels: Channel list for decoder levels
        n_levels: Number of decoder levels
        out_channels: Number of output channels
        latent_channels: Skip connection latent channels
        n_classes: Number of classes for RAA
        use_sigmoid: Whether to apply sigmoid at output
        dropout: Dropout rate
    """

    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]

    def __init__(
        self,
        in_channels: int = 144,
        decoder_channels: Optional[list[int]] = None,
        n_levels: int = 4,
        out_channels: int = 1,
        latent_channels: int = 8,
        n_classes: int = 1,
        use_sigmoid: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        self.n_levels = n_levels
        self.use_sigmoid = use_sigmoid

        if len(self.decoder_channels) != n_levels:
            raise ValueError(
                f"decoder_channels length ({len(self.decoder_channels)}) "
                f"must match n_levels ({n_levels})"
            )

        self.blocks = nn.ModuleList()
        ch_in = in_channels

        for ch_out in self.decoder_channels:
            self.blocks.append(
                LFADecoderBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    latent_channels=latent_channels,
                    n_classes=n_classes,
                    dropout=dropout,
                )
            )
            ch_in = ch_out

        final_ch = ch_in
        self.output_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final_ch, out_channels, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Bottleneck features [B, C, H, W]
            skips: Dict of skip connections from encoder

        Returns:
            Output tensor [B, out_channels, H, W]
        """
        for i, block in enumerate(self.blocks):
            skip_key = f"level{self.n_levels - i}"
            skip = skips.get(skip_key)

            if skip is not None:
                x = block(x, skip)
            else:
                x = block.upsample(x)
                x = block.refine(x)

        x = self.output_head(x)

        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x


# =============================================================================
# Complete LFA-Net with Latent Skip Connections
# =============================================================================


class BottleneckLFABlockNet(nn.Module):
    """LFA-Net with Latent Skip Connections (Bottleneck Architecture).

    All skip connections are projected to a uniform latent space
    [B, latent_ch, S, S] before the decoder.

    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output channels (1 for binary, 2 for AV)
        encoder_channels: Encoder channel progression
        decoder_channels: Decoder channel progression (auto-derived if None)
        n_classes: Number of classes for RAA attention
        latent_channels: Skip connection latent channels
        skip_spatial: Skip connection spatial size
        use_sigmoid: Whether to apply sigmoid at output
        dropout: Dropout rate

    Example:
        >>> model = BottleneckLFABlockNet(out_channels=1)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> pred = model(x)  # [1, 1, 512, 512]
    """

    DEFAULT_ENCODER_CHANNELS = [32, 48, 72, 144]
    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_channels: Optional[list[int]] = None,
        decoder_channels: Optional[list[int]] = None,
        n_classes: int = 1,
        latent_channels: int = 8,
        skip_spatial: int = 32,
        use_sigmoid: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels or self.DEFAULT_ENCODER_CHANNELS

        if decoder_channels is None:
            self.decoder_channels = list(reversed(self.encoder_channels[:-1])) + [
                max(self.encoder_channels[0] // 2, 16)
            ]
        else:
            self.decoder_channels = decoder_channels

        n_levels = len(self.encoder_channels)

        if len(self.decoder_channels) != n_levels:
            raise ValueError(
                f"decoder_channels length ({len(self.decoder_channels)}) "
                f"must match encoder_channels length ({n_levels})"
            )

        self.encoder = LFAEncoder(
            in_channels=in_channels,
            channels=self.encoder_channels,
            n_classes=n_classes,
            latent_channels=latent_channels,
            skip_spatial=skip_spatial,
            dropout=dropout,
        )

        bottleneck_channels = self.encoder_channels[-1]

        self.decoder = LFADecoder(
            in_channels=bottleneck_channels,
            decoder_channels=self.decoder_channels,
            n_levels=n_levels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            n_classes=n_classes,
            use_sigmoid=use_sigmoid,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            Output [B, out_channels, H, W]
        """
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips)
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

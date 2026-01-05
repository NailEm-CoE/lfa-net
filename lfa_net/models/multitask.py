"""MultitaskLFANet: Multi-Task Learning for Fundus Analysis.

Architecture:
- Shared encoder: LFABlocks with Multi-Scale Conv + LiteFusion + RAA
- All skip connections projected to uniform latent space
- Task-specific heads with multi-scale feature fusion:
  - DualSegmentationHead: Separate decoders for disc/cup vs vessels
  - EnhancedFoveaHead: LFA blocks with skip+bottleneck fusion → (x, y)
  - EnhancedDiseaseHead: LFA blocks with skip+bottleneck fusion → N classes
  - ReconstructionHead (optional): Self-supervised RGB reconstruction

Example:
    >>> model = MultitaskLFANet()
    >>> x = torch.randn(2, 3, 512, 512)
    >>> out = model(x)
    >>> out["segmentation"].shape  # [2, 5, 512, 512]
    >>> out["fovea"].shape  # [2, 2]
    >>> out["disease"].shape  # [2, 3]
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_skip import (
    LFAEncoder,
    LFADecoder,
    LiteFusionAttention,
    MultiScaleConv,
)


# =============================================================================
# Task-Specific Heads
# =============================================================================


class AsinhLeakySigmoid(nn.Module):
    """Activation allowing predictions outside [0, 1] for out-of-bounds fovea.
    
    Combines sigmoid center with asinh tails for smooth extrapolation.
    Default output range: [-1, 2] to support out-of-bounds predictions.
    
    Formula:
        result = sigmoid(x) + leak * asinh(x)
        result = clamp(result, min_val, max_val)
    
    Args:
        min_val: Minimum output value (default -1 for out-of-bounds)
        max_val: Maximum output value (default 2 for out-of-bounds)
        leak: Strength of asinh leakage (higher = more extrapolation)
    """
    
    def __init__(self, min_val: float = -1.0, max_val: float = 2.0, leak: float = 0.15):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.leak = leak
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply leaky sigmoid activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor with values in [min_val, max_val]
        """
        sig = torch.sigmoid(x)
        result = sig + self.leak * torch.asinh(x)
        return torch.clamp(result, self.min_val, self.max_val)


class SegmentationHead(nn.Module):
    """Decoder head for multi-channel segmentation.
    
    Uses LFADecoder with skip connections from encoder.
    Output: [B, out_channels, H, W] with sigmoid activation.
    
    Args:
        in_channels: Bottleneck input channels
        decoder_channels: Channel progression for decoder
        out_channels: Number of output segmentation channels
        latent_channels: Skip connection latent dimension
        n_levels: Number of decoder levels
        dropout: Dropout rate
    """
    
    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]
    
    def __init__(
        self,
        in_channels: int = 144,
        decoder_channels: Optional[list[int]] = None,
        out_channels: int = 5,  # disc, cup, artery, vein, vessel
        latent_channels: int = 8,
        n_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        
        self.decoder = LFADecoder(
            in_channels=in_channels,
            decoder_channels=self.decoder_channels,
            n_levels=n_levels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            n_classes=out_channels,  # For RAA
            use_sigmoid=False,  # Return logits for mixed precision compatibility
            dropout=dropout,
        )
    
    def forward(
        self, 
        bottleneck: torch.Tensor, 
        skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, C, H, W]
            skips: Dict of skip connections from encoder
            
        Returns:
            Segmentation masks [B, out_channels, H, W]
        """
        return self.decoder(bottleneck, skips)


class FoveaHead(nn.Module):
    """Regression head for fovea localization.
    
    Uses global average pooling + FC layers with AsinhLeakySigmoid.
    Output: [B, 2] normalized (x, y) coordinates in range [-1, 2].
    
    Fovea coordinates are normalized by 1024 (minimum expected image dimension).
    Range [-1, 2] supports out-of-bounds predictions:
        - [0, 1]: Inside image
        - [-1, 0) or (1, 2]: Outside image boundary
    
    Args:
        in_channels: Input channels from bottleneck
        hidden_dim: Hidden layer dimension
        min_val: Minimum output value
        max_val: Maximum output value
        leak: AsinhLeakySigmoid leak factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 144,
        hidden_dim: int = 256,
        min_val: float = -1.0,
        max_val: float = 2.0,
        leak: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # (x, y)
        )
        self.activation = AsinhLeakySigmoid(min_val=min_val, max_val=max_val, leak=leak)
    
    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, C, H, W]
            
        Returns:
            Fovea coordinates [B, 2] in range [min_val, max_val]
        """
        x = self.pool(bottleneck)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.fc(x)  # [B, 2]
        x = self.activation(x)  # Apply AsinhLeakySigmoid
        return x


class DiseaseHead(nn.Module):
    """Classification head for multi-label disease detection.
    
    Uses global average pooling + FC layers with sigmoid output.
    Output: [B, n_classes] logits (apply sigmoid for probabilities).
    
    Args:
        in_channels: Input channels from bottleneck
        hidden_dim: Hidden layer dimension
        n_classes: Number of disease classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 144,
        hidden_dim: int = 256,
        n_classes: int = 3,  # DR, AMD, Glaucoma
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
    
    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, C, H, W]
            
        Returns:
            Disease logits [B, n_classes]
        """
        x = self.pool(bottleneck)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.fc(x)  # [B, n_classes]
        return x  # Return logits, apply sigmoid in loss/inference


# =============================================================================
# Dual Decoder and Enhanced Heads
# =============================================================================


class DualSegmentationHead(nn.Module):
    """Dual decoder head with separate decoders for disc/cup and vessels.
    
    Separates the segmentation task into two specialized decoders:
    - Disc/Cup decoder: 2 channels (disc, cup) - larger structures
    - Vessel decoder: 3 channels (artery, vein, vessel) - fine structures
    
    This separation allows each decoder to specialize in its task
    without interference from conflicting gradients.
    
    Args:
        in_channels: Bottleneck input channels
        decoder_channels: Channel progression for decoders
        latent_channels: Skip connection latent dimension
        n_levels: Number of decoder levels
        dropout: Dropout rate
    """
    
    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]
    
    def __init__(
        self,
        in_channels: int = 144,
        decoder_channels: Optional[list[int]] = None,
        latent_channels: int = 8,
        n_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        
        # Disc/Cup decoder (2 channels: disc, cup)
        self.disc_cup_decoder = LFADecoder(
            in_channels=in_channels,
            decoder_channels=self.decoder_channels,
            n_levels=n_levels,
            out_channels=2,
            latent_channels=latent_channels,
            n_classes=2,  # For RAA
            use_sigmoid=False,
            dropout=dropout,
        )
        
        # Vessel decoder (3 channels: artery, vein, vessel)
        self.vessel_decoder = LFADecoder(
            in_channels=in_channels,
            decoder_channels=self.decoder_channels,
            n_levels=n_levels,
            out_channels=3,
            latent_channels=latent_channels,
            n_classes=3,  # For RAA
            use_sigmoid=False,
            dropout=dropout,
        )
    
    def forward(
        self, 
        bottleneck: torch.Tensor, 
        skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through both decoders.
        
        Args:
            bottleneck: Bottleneck features [B, C, H, W]
            skips: Dict of skip connections from encoder
            
        Returns:
            Segmentation masks [B, 5, H, W] (disc, cup, artery, vein, vessel)
        """
        disc_cup = self.disc_cup_decoder(bottleneck, skips)  # [B, 2, H, W]
        vessels = self.vessel_decoder(bottleneck, skips)  # [B, 3, H, W]
        
        # Concatenate: [disc, cup, artery, vein, vessel]
        return torch.cat([disc_cup, vessels], dim=1)  # [B, 5, H, W]


class LFAFusionBlock(nn.Module):
    """LFA Block for feature fusion (no skip projection or downsampling).
    
    Components: MultiScaleConv + LiteFusion (like LFABlock but simplified)
    Used for fusing multi-scale features before classification/regression.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.multiscale_conv = MultiScaleConv(in_channels, out_channels)
        self.litefusion = LiteFusionAttention(out_channels, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        x = self.multiscale_conv(x)
        x = self.litefusion(x)
        return x


class LFAClassificationBlock(nn.Module):
    """LFA Block with pooling for classification/regression heads.
    
    Applies LFAFusionBlock then pools to reduce spatial dimensions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        pool_size: Output spatial size after pooling (None = no pooling)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = LFAFusionBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(pool_size) if pool_size else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return x


class EnhancedFoveaHead(nn.Module):
    """Enhanced fovea head using LFA blocks with progressive pooling.
    
    Uses skip connections + bottleneck for multi-scale feature fusion.
    
    Architecture:
        Skip+Bottleneck concat [B, 176, 32, 32]
        → LFABlock → [B, 64, 16, 16]  (pool)
        → LFABlock → [B, 64, 8, 8]    (pool)
        → LFABlock → [B, 128, 4, 4]   (pool)
        → GAP → [B, 128]
        → FC → [B, 2]
    
    Args:
        n_skip_levels: Number of encoder skip connections
        latent_channels: Skip connection channels
        bottleneck_channels: Bottleneck channels
        skip_spatial: Skip connection spatial size
        block_channels: Channel progression for LFA blocks
        min_val: AsinhLeakySigmoid minimum
        max_val: AsinhLeakySigmoid maximum
        leak: AsinhLeakySigmoid leak factor
        dropout: Dropout rate
    """
    
    DEFAULT_BLOCK_CHANNELS = [64, 64, 128]
    
    def __init__(
        self,
        n_skip_levels: int = 4,
        latent_channels: int = 8,
        bottleneck_channels: int = 144,
        skip_spatial: int = 32,
        block_channels: list[int] | None = None,
        min_val: float = -1.0,
        max_val: float = 2.0,
        leak: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_skip_levels = n_skip_levels
        self.skip_spatial = skip_spatial
        
        # Pool bottleneck to skip spatial size
        self.bottleneck_pool = nn.AdaptiveAvgPool2d(skip_spatial)
        
        # Total input channels: skips + bottleneck
        in_channels = n_skip_levels * latent_channels + bottleneck_channels  # 176
        
        # LFA blocks with progressive pooling
        channels = block_channels or self.DEFAULT_BLOCK_CHANNELS
        pool_sizes = [16, 8, 4]  # Progressive spatial reduction
        
        self.blocks = nn.ModuleList()
        for i, (out_ch, pool_sz) in enumerate(zip(channels, pool_sizes)):
            self.blocks.append(
                LFAClassificationBlock(in_channels, out_ch, pool_size=pool_sz, dropout=dropout)
            )
            in_channels = out_ch
        
        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 2)
        self.activation = AsinhLeakySigmoid(min_val=min_val, max_val=max_val, leak=leak)
    
    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, 144, H, W]
            skips: Skip connections from encoder {'level1': ..., 'level4': ...}
            
        Returns:
            Fovea coordinates [B, 2]
        """
        # Collect skip connections
        skip_tensors = [skips[f"level{i+1}"] for i in range(self.n_skip_levels)]
        
        # Pool bottleneck to skip spatial size
        bottleneck_pooled = self.bottleneck_pool(bottleneck)
        
        # Concatenate all features
        x = torch.cat(skip_tensors + [bottleneck_pooled], dim=1)  # [B, 176, 32, 32]
        
        # Apply LFA blocks with pooling
        for block in self.blocks:
            x = block(x)
        
        # GAP + FC
        x = self.gap(x).flatten(1)  # [B, 128]
        x = self.fc(x)  # [B, 2]
        
        return self.activation(x)


class EnhancedDiseaseHead(nn.Module):
    """Enhanced disease classification head using LFA blocks with progressive pooling.
    
    Uses skip connections + bottleneck for multi-scale feature fusion.
    
    Architecture:
        Skip+Bottleneck concat [B, 176, 32, 32]
        → LFABlock → [B, 64, 16, 16]  (pool)
        → LFABlock → [B, 64, 8, 8]    (pool)
        → LFABlock → [B, 128, 4, 4]   (pool)
        → GAP → [B, 128]
        → FC → [B, n_classes]
    
    Args:
        n_skip_levels: Number of encoder skip connections
        latent_channels: Skip connection channels
        bottleneck_channels: Bottleneck channels
        skip_spatial: Skip connection spatial size
        block_channels: Channel progression for LFA blocks
        n_classes: Number of disease classes
        dropout: Dropout rate
    """
    
    DEFAULT_BLOCK_CHANNELS = [64, 64, 128]
    
    def __init__(
        self,
        n_skip_levels: int = 4,
        latent_channels: int = 8,
        bottleneck_channels: int = 144,
        skip_spatial: int = 32,
        block_channels: list[int] | None = None,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_skip_levels = n_skip_levels
        self.skip_spatial = skip_spatial
        
        # Pool bottleneck to skip spatial size
        self.bottleneck_pool = nn.AdaptiveAvgPool2d(skip_spatial)
        
        # Total input channels: skips + bottleneck
        in_channels = n_skip_levels * latent_channels + bottleneck_channels  # 176
        
        # LFA blocks with progressive pooling
        channels = block_channels or self.DEFAULT_BLOCK_CHANNELS
        pool_sizes = [16, 8, 4]  # Progressive spatial reduction
        
        self.blocks = nn.ModuleList()
        for i, (out_ch, pool_sz) in enumerate(zip(channels, pool_sizes)):
            self.blocks.append(
                LFAClassificationBlock(in_channels, out_ch, pool_size=pool_sz, dropout=dropout)
            )
            in_channels = out_ch
        
        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], n_classes)
    
    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, 144, H, W]
            skips: Skip connections from encoder {'level1': ..., 'level4': ...}
            
        Returns:
            Disease logits [B, n_classes]
        """
        # Collect skip connections
        skip_tensors = [skips[f"level{i+1}"] for i in range(self.n_skip_levels)]
        
        # Pool bottleneck to skip spatial size
        bottleneck_pooled = self.bottleneck_pool(bottleneck)
        
        # Concatenate all features
        x = torch.cat(skip_tensors + [bottleneck_pooled], dim=1)  # [B, 176, 32, 32]
        
        # Apply LFA blocks with pooling
        for block in self.blocks:
            x = block(x)
        
        # GAP + FC
        x = self.gap(x).flatten(1)  # [B, 128]
        return self.fc(x)  # [B, n_classes]


class ReconstructionHead(nn.Module):
    """Decoder head for RGB reconstruction (self-supervised auxiliary task).
    
    Uses LFADecoder with skip connections to reconstruct input RGB image.
    Output: [B, 3, H, W] reconstructed RGB values.
    
    Args:
        in_channels: Bottleneck input channels
        decoder_channels: Channel progression for decoder
        latent_channels: Skip connection latent dimension
        n_levels: Number of decoder levels
        dropout: Dropout rate
    """
    
    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]
    
    def __init__(
        self,
        in_channels: int = 144,
        decoder_channels: Optional[list[int]] = None,
        latent_channels: int = 8,
        n_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        
        self.decoder = LFADecoder(
            in_channels=in_channels,
            decoder_channels=self.decoder_channels,
            n_levels=n_levels,
            out_channels=3,  # RGB output
            latent_channels=latent_channels,
            n_classes=3,  # For RAA
            use_sigmoid=False,  # Return raw values, apply sigmoid in loss/inference
            dropout=dropout,
        )
    
    def forward(
        self, 
        bottleneck: torch.Tensor, 
        skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features [B, C, H, W]
            skips: Dict of skip connections from encoder
            
        Returns:
            Reconstructed RGB [B, 3, H, W]
        """
        return self.decoder(bottleneck, skips)


# =============================================================================
# MultitaskLFANet
# =============================================================================


class MultitaskLFANet(nn.Module):
    """Multi-Task LFA-Net with multi-scale feature fusion for fundus analysis.
    
    Features:
    - Dual decoder: Separate decoders for disc/cup (large structures) and 
      vessels (fine structures) to prevent gradient interference
    - Enhanced fovea/disease heads: Use skip connections + bottleneck with
      LFA blocks for multi-scale feature fusion
    - Optional reconstruction head: Self-supervised RGB reconstruction
    
    Architecture:
        Input → Shared Encoder → Bottleneck
                     ↓               ↓
                   Skips        Bottleneck
                     ↓               ↓
                     └──────┬────────┘
                            ↓
                    DualSegmentationHead → 5ch masks (disc, cup, artery, vein, vessel)
                            ↓
                    EnhancedFoveaHead → (x, y) coords  [uses skips + bottleneck]
                            ↓
                    EnhancedDiseaseHead → N class logits  [uses skips + bottleneck]
                            ↓
                    ReconstructionHead → RGB (optional)
    
    Args:
        in_channels: Input image channels (3 for RGB)
        encoder_channels: Encoder channel progression
        decoder_channels: Decoder channel progression (for segmentation)
        latent_channels: Skip connection latent dimension
        skip_spatial: Skip connection spatial size
        seg_out_channels: Segmentation output channels
        n_disease_classes: Number of disease classes
        head_block_channels: LFA block channels for fovea/disease heads
        fovea_min_val: Fovea output minimum (-1 for out-of-bounds)
        fovea_max_val: Fovea output maximum (2 for out-of-bounds)
        fovea_leak: AsinhLeakySigmoid leak factor
        dropout: Dropout rate
        use_reconstruction: Enable RGB reconstruction decoder
        use_dual_decoder: Use separate decoders for disc/cup vs vessels
    
    Example:
        >>> model = MultitaskLFANet()
        >>> x = torch.randn(2, 3, 512, 512)
        >>> out = model(x)
        >>> out["segmentation"].shape  # [2, 5, 512, 512]
        >>> out["fovea"].shape  # [2, 2]
        >>> out["disease"].shape  # [2, 3]
    """
    
    DEFAULT_ENCODER_CHANNELS = [32, 48, 72, 144]
    DEFAULT_DECODER_CHANNELS = [72, 48, 32, 16]
    SEG_CHANNEL_NAMES = ["disc", "cup", "artery", "vein", "vessel"]
    DISEASE_NAMES = ["dr", "amd", "glaucoma"]
    
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: Optional[list[int]] = None,
        decoder_channels: Optional[list[int]] = None,
        latent_channels: int = 8,
        skip_spatial: int = 32,
        seg_out_channels: int = 5,
        n_disease_classes: int = 3,
        head_block_channels: Optional[list[int]] = None,  # [64, 64, 128] default
        fovea_min_val: float = -1.0,
        fovea_max_val: float = 2.0,
        fovea_leak: float = 0.15,
        dropout: float = 0.1,
        use_reconstruction: bool = False,
        use_dual_decoder: bool = True,  # Separate decoders for disc/cup vs vessels
    ):
        super().__init__()
        
        self.encoder_channels = encoder_channels or self.DEFAULT_ENCODER_CHANNELS
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        self.seg_out_channels = seg_out_channels
        self.n_disease_classes = n_disease_classes
        self.latent_channels = latent_channels
        self.skip_spatial = skip_spatial
        self.use_reconstruction = use_reconstruction
        self.use_dual_decoder = use_dual_decoder
        
        n_levels = len(self.encoder_channels)
        bottleneck_channels = self.encoder_channels[-1]
        
        # Shared encoder
        self.encoder = LFAEncoder(
            in_channels=in_channels,
            channels=self.encoder_channels,
            n_classes=seg_out_channels,  # For RAA
            latent_channels=latent_channels,
            skip_spatial=skip_spatial,
            dropout=dropout,
        )
        
        # Bottleneck (LiteFusion attention)
        self.bottleneck = LiteFusionAttention(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            dropout=dropout,
        )
        
        # Segmentation head: dual decoder (separate disc/cup vs vessels) or single shared
        if use_dual_decoder:
            self.seg_head = DualSegmentationHead(
                in_channels=bottleneck_channels,
                decoder_channels=self.decoder_channels,
                latent_channels=latent_channels,
                n_levels=n_levels,
                dropout=dropout,
            )
        else:
            self.seg_head = SegmentationHead(
                in_channels=bottleneck_channels,
                decoder_channels=self.decoder_channels,
                out_channels=seg_out_channels,
                latent_channels=latent_channels,
                n_levels=n_levels,
                dropout=dropout,
            )
        
        # Enhanced fovea head with LFA blocks (progressive pooling)
        self.fovea_head = EnhancedFoveaHead(
            n_skip_levels=n_levels,
            latent_channels=latent_channels,
            bottleneck_channels=bottleneck_channels,
            skip_spatial=skip_spatial,
            block_channels=head_block_channels,
            min_val=fovea_min_val,
            max_val=fovea_max_val,
            leak=fovea_leak,
            dropout=dropout,
        )
        
        # Enhanced disease head with LFA blocks (progressive pooling)
        self.disease_head = EnhancedDiseaseHead(
            n_skip_levels=n_levels,
            latent_channels=latent_channels,
            bottleneck_channels=bottleneck_channels,
            skip_spatial=skip_spatial,
            block_channels=head_block_channels,
            n_classes=n_disease_classes,
            dropout=dropout,
        )
        
        # Optional reconstruction head for self-supervised auxiliary task
        self.recon_head: Optional[ReconstructionHead] = None
        if use_reconstruction:
            self.recon_head = ReconstructionHead(
                in_channels=bottleneck_channels,
                decoder_channels=self.decoder_channels,
                latent_channels=latent_channels,
                n_levels=n_levels,
                dropout=dropout,
            )
        
        # Temporarily disable skip connections for ablation study
        # Set to list of level names to zero out, e.g., ["level1"] or ["level1", "level2"]
        self.disabled_skips: list[str] = []  # Empty = all skips enabled
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all task heads.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Dict with keys:
                - "segmentation": [B, 5, H, W] masks (logits)
                - "fovea": [B, 2] normalized (x, y) coordinates
                - "disease": [B, N] disease logits
                - "reconstruction": [B, 3, H, W] reconstructed RGB (if use_reconstruction=True)
        """
        # Shared encoder
        features, skips = self.encoder(x)
        # skips: {'level1': [B,8,32,32], ..., 'level4': [B,8,32,32]}
        # features: [B, 144, H/16, W/16]
        
        # Ablation: zero out disabled skip connections
        for skip_name in self.disabled_skips:
            if skip_name in skips:
                skips[skip_name] = torch.zeros_like(skips[skip_name])
        
        # Bottleneck
        bottleneck = self.bottleneck(features)  # [B, 144, H/16, W/16]
        
        # Segmentation uses bottleneck + skips
        segmentation = self.seg_head(bottleneck, skips)  # [B, 5, H, W]
        
        # Fovea/Disease use multi-scale fusion with bottleneck + skips
        fovea = self.fovea_head(bottleneck, skips)  # [B, 2]
        disease = self.disease_head(bottleneck, skips)  # [B, N]
        
        result = {
            "segmentation": segmentation,
            "fovea": fovea,
            "disease": disease,
        }
        
        # Optional: RGB reconstruction for self-supervised learning
        if self.recon_head is not None:
            result["reconstruction"] = self.recon_head(bottleneck, skips)  # [B, 3, H, W]
        
        return result
    
    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through encoder only (for feature extraction).
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            bottleneck: [B, C, H/16, W/16]
            skips: Dict of skip connections
        """
        features, skips = self.encoder(x)
        bottleneck = self.bottleneck(features)
        return bottleneck, skips


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("=" * 50)
    print("Testing MultitaskLFANet")
    print("=" * 50)
    model = MultitaskLFANet()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    
    print(f"Segmentation: {out['segmentation'].shape}")  # [2, 5, 512, 512]
    print(f"Fovea: {out['fovea'].shape}")  # [2, 2]
    print(f"Disease: {out['disease'].shape}")  # [2, 3]
    print(f"Fovea range: [{out['fovea'].min():.3f}, {out['fovea'].max():.3f}]")
    print(f"Has reconstruction: {'reconstruction' in out}")
    
    # Test with reconstruction
    print("\n" + "=" * 50)
    print("Testing MultitaskLFANet (With Reconstruction)")
    print("=" * 50)
    model_recon = MultitaskLFANet(use_reconstruction=True)
    print(f"Parameters: {count_parameters(model_recon):,}")
    
    out_recon = model_recon(x)
    print(f"Reconstruction: {out_recon['reconstruction'].shape}")  # [2, 3, 512, 512]
    
    # Test with single decoder
    print("\n" + "=" * 50)
    print("Testing MultitaskLFANet (Single Decoder)")
    print("=" * 50)
    model_single = MultitaskLFANet(use_dual_decoder=False)
    print(f"Parameters: {count_parameters(model_single):,}")
    
    out_single = model_single(x)
    print(f"Segmentation: {out_single['segmentation'].shape}")
    
    print("\nDone!")

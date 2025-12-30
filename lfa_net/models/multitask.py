"""MultitaskLFANet: Multi-Task Learning for Fundus Analysis.

Architecture:
- Shared encoder: LFABlocks with Multi-Scale Conv + LiteFusion + RAA
- All skip connections projected to uniform latent space
- Task-specific heads:
  - SegmentationHead: Decoder → 5ch (disc, cup, artery, vein, vessel)
  - FoveaHead: GAP → FC → 2 values with AsinhLeakySigmoid
  - DiseaseHead: GAP → FC → N classes (sigmoid multi-label)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_skip import (
    LFAEncoder,
    LFADecoder,
    LiteFusionAttention,
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
# MultitaskLFANet
# =============================================================================


class MultitaskLFANet(nn.Module):
    """Multi-Task LFA-Net for comprehensive fundus analysis.
    
    Architecture:
        Input → Shared Encoder → Bottleneck
                                    ├→ SegmentationHead → 5ch masks
                                    ├→ FoveaHead → (x, y) coords
                                    └→ DiseaseHead → N class logits
    
    Args:
        in_channels: Input image channels (3 for RGB)
        encoder_channels: Encoder channel progression
        decoder_channels: Decoder channel progression (for segmentation)
        latent_channels: Skip connection latent dimension
        skip_spatial: Skip connection spatial size
        seg_out_channels: Segmentation output channels
        n_disease_classes: Number of disease classes
        fovea_hidden_dim: Fovea head hidden dimension
        fovea_min_val: Fovea output minimum (-1 for out-of-bounds)
        fovea_max_val: Fovea output maximum (2 for out-of-bounds)
        fovea_leak: AsinhLeakySigmoid leak factor
        dropout: Dropout rate
    
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
        skip_spatial: int = 64,
        seg_out_channels: int = 5,
        n_disease_classes: int = 3,
        fovea_hidden_dim: int = 256,
        fovea_min_val: float = -1.0,
        fovea_max_val: float = 2.0,
        fovea_leak: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_channels = encoder_channels or self.DEFAULT_ENCODER_CHANNELS
        self.decoder_channels = decoder_channels or self.DEFAULT_DECODER_CHANNELS
        self.seg_out_channels = seg_out_channels
        self.n_disease_classes = n_disease_classes
        
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
        
        # Task-specific heads
        self.seg_head = SegmentationHead(
            in_channels=bottleneck_channels,
            decoder_channels=self.decoder_channels,
            out_channels=seg_out_channels,
            latent_channels=latent_channels,
            n_levels=n_levels,
            dropout=dropout,
        )
        
        self.fovea_head = FoveaHead(
            in_channels=bottleneck_channels,
            hidden_dim=fovea_hidden_dim,
            min_val=fovea_min_val,
            max_val=fovea_max_val,
            leak=fovea_leak,
            dropout=dropout,
        )
        
        self.disease_head = DiseaseHead(
            in_channels=bottleneck_channels,
            hidden_dim=fovea_hidden_dim,  # Reuse same hidden dim
            n_classes=n_disease_classes,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all task heads.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Dict with keys:
                - "segmentation": [B, 5, H, W] masks (sigmoid activated)
                - "fovea": [B, 2] normalized (x, y) coordinates
                - "disease": [B, N] disease logits
        """
        # Shared encoder
        features, skips = self.encoder(x)  # features: [B, C, H/16, W/16]
        
        # Bottleneck
        bottleneck = self.bottleneck(features)  # [B, C, H/16, W/16]
        
        # Task heads
        segmentation = self.seg_head(bottleneck, skips)  # [B, 5, H, W]
        fovea = self.fovea_head(bottleneck)  # [B, 2]
        disease = self.disease_head(bottleneck)  # [B, N]
        
        return {
            "segmentation": segmentation,
            "fovea": fovea,
            "disease": disease,
        }
    
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
    model = MultitaskLFANet()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    
    print(f"Segmentation: {out['segmentation'].shape}")  # [2, 5, 512, 512]
    print(f"Fovea: {out['fovea'].shape}")  # [2, 2]
    print(f"Disease: {out['disease'].shape}")  # [2, 3]
    
    # Check fovea range
    print(f"Fovea range: [{out['fovea'].min():.3f}, {out['fovea'].max():.3f}]")

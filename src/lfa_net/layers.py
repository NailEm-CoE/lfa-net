"""LFA-Net layer implementations using PyTorch nn.Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block with 1x1, 3x3, and dilated 3x3 convolutions.
    
    Outputs are summed and passed through LeakyReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv3x3_dilated = nn.Conv2d(
            in_channels, out_channels, 3, padding=2, dilation=2, bias=False
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, H, W]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        out = self.conv1x1(x) + self.conv3x3(x) + self.conv3x3_dilated(x)
        return self.activation(out)


class FocalModulation(nn.Module):
    """
    Focal Modulation block for channel attention.
    
    Computes modulation from gap/gmp difference and applies scaling.
    """

    def __init__(self, channels: int, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            channels: Number of input/output channels
            gamma: Power scaling factor (unused in simplified version)
            alpha: Modulation strength
        """
        super().__init__()
        self.alpha = alpha
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Modulated tensor [B, C, H, W]
        """
        # Global pooling
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        max_val = x.amax(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Modulation
        modulation = (max_val - mean) * self.alpha
        modulation = torch.sigmoid(self.conv(modulation))
        
        # Apply modulation (simplified - no gamma power to avoid gradient issues)
        return x * modulation


class ContextAggregation(nn.Module):
    """
    Context Aggregation block combining local and global features with focal modulation.
    """

    def __init__(self, in_channels: int, out_channels: int, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            gamma: Focal modulation gamma
            alpha: Focal modulation alpha
        """
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.global_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.focal_mod = FocalModulation(out_channels, gamma, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, H, W]
            
        Returns:
            Concatenated tensor [B, C_out*2, H, W]
        """
        # Local features
        conv1 = F.relu(self.conv3x3(x))
        
        # Global context
        conv2 = F.relu(self.conv1x1(x))
        global_ctx = conv2.mean(dim=(2, 3), keepdim=True)  # GAP
        global_ctx = torch.sigmoid(self.global_conv(global_ctx))
        global_ctx = conv1 * global_ctx
        
        # Focal modulation
        fm = self.focal_mod(global_ctx)
        
        # Concatenate local and focal-modulated features
        return torch.cat([conv1, fm], dim=1)


class VisionMambaInspired(nn.Module):
    """
    Vision Mamba-inspired block with token and channel mixing.
    
    Uses depthwise conv for token mixing and MLP for channel mixing.
    """

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        """
        Args:
            dim: Feature dimension (channels)
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Token mixing with residual
        shortcut = x
        # LayerNorm expects [B, H, W, C]
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.token_mixer(x_norm) + shortcut
        
        # Channel mixing with residual
        shortcut = x
        # Permute for LayerNorm and Linear: [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm)
        x_mixed = self.channel_mixer(x_norm)
        x = x_mixed.permute(0, 3, 1, 2) + shortcut
        
        return x


class LiteFusionAttention(nn.Module):
    """
    LiteFusion Attention bottleneck block.
    
    Combines context aggregation with focal modulation and Vision Mamba.
    """

    def __init__(self, in_channels: int, filters: int, gamma: float = 2.0, alpha: float = 0.25, dropout: float = 0.1):
        """
        Args:
            in_channels: Number of input channels
            filters: Number of output filters
            gamma: Focal modulation gamma
            alpha: Focal modulation alpha
            dropout: Dropout rate for Vision Mamba
        """
        super().__init__()
        # Project input
        self.proj_in = nn.Conv2d(in_channels, filters, 1)
        self.norm = nn.LayerNorm(filters)
        self.conv = nn.Conv2d(filters, filters, 3, padding=1)
        
        # Context aggregation (outputs 2*filters channels)
        self.context_agg = ContextAggregation(filters, filters, gamma, alpha)
        
        # Project back
        self.proj_out = nn.Conv2d(filters * 2, filters, 1)
        
        # Vision Mamba
        self.mamba = VisionMambaInspired(filters, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, H, W]
            
        Returns:
            Output tensor [B, filters, H, W]
        """
        # Project input
        identity = self.proj_in(x)
        
        # LayerNorm + Conv
        x_norm = self.norm(identity.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_conv = self.conv(x_norm)
        
        # Context aggregation with focal modulation
        x_ctx = self.context_agg(x_conv)
        
        # Project back and residual
        x_proj = self.proj_out(x_ctx)
        out = x_proj + identity
        
        # Vision Mamba
        out = self.mamba(out)
        
        return out


class RAAttentionBlock(nn.Module):
    """
    Region-Aware Attention Block for skip connections.
    
    Computes semantic attention using global pooling and channel reduction.
    """

    def __init__(self, in_channels: int, n_classes: int = 1, k: int = 16):
        """
        Args:
            in_channels: Number of input channels
            n_classes: Number of output classes
            k: Reduction factor for attention
        """
        super().__init__()
        self.n_classes = n_classes
        self.k = k
        
        self.conv = nn.Conv2d(in_channels, k * n_classes, 3, padding=1)
        self.bn = nn.BatchNorm2d(k * n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Conv + BN + ReLU
        f = F.relu(self.bn(self.conv(x)))  # [B, k*n_classes, H, W]
        
        # Global pooling
        gmp = f.amax(dim=(2, 3))  # [B, k*n_classes]
        gap = f.mean(dim=(2, 3))  # [B, k*n_classes]
        pooled = gmp * gap  # [B, k*n_classes]
        
        # Reshape and mean over k
        pooled = pooled.view(B, self.n_classes, self.k)  # [B, n_classes, k]
        S = pooled.mean(dim=-1, keepdim=True)  # [B, n_classes, 1]
        
        # Spatial features
        f_spatial = f.view(B, self.n_classes, self.k, H, W)  # [B, n_classes, k, H, W]
        f_spatial = f_spatial.mean(dim=2)  # [B, n_classes, H, W]
        
        # Apply attention
        S_expanded = S.unsqueeze(-1)  # [B, n_classes, 1, 1]
        attended = f_spatial * S_expanded  # [B, n_classes, H, W]
        
        # Mean over classes to get mask
        M = attended.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply to input
        return x * M


class SEMAttentionBlock(nn.Module):
    """
    Semantic Attention Block using max and average pooling fusion.
    """

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels * 2, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gap_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attention-enhanced tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # F2 branch
        f2 = F.relu(self.bn1(self.conv1(x)))
        
        # Global pooling
        x1 = F.adaptive_max_pool2d(f2, 1)  # [B, C, 1, 1]
        x2 = F.adaptive_avg_pool2d(f2, 1)  # [B, C, 1, 1]
        
        # Concatenate and process
        con = torch.cat([x1, x2], dim=1)  # [B, 2C, 1, 1]
        f3 = F.relu(self.bn2(self.conv2(con)))  # [B, C, 1, 1]
        
        # Global attention
        xa = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        xa = torch.sigmoid(self.gap_conv(xa))  # [B, C, 1, 1]
        
        # Apply attention
        xm = f3 * xa
        
        return x + xm

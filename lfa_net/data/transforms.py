"""Kornia GPU augmentations for training."""

import torch
import kornia.augmentation as K
import torch.nn as nn
import torch.nn.functional as F
from kornia.constants import DataKey, Resample


class RandomResizedCropPair(nn.Module):
    """Random resized crop that handles image and mask separately.
    
    Works around Kornia's mask handling issues with align_corners.
    """
    
    def __init__(self, size: int, scale: tuple = (0.8, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, C, H, W]
            mask: [B, 1, H, W]
        """
        B, C, H, W = image.shape
        
        # Random scale for each sample in batch
        scale = torch.empty(B).uniform_(self.scale[0], self.scale[1]).to(image.device)
        
        # Compute crop size (square)
        crop_size = (scale * min(H, W)).int()
        
        # Random crop position for each sample
        out_images = []
        out_masks = []
        
        for i in range(B):
            cs = crop_size[i].item()
            max_y = H - cs
            max_x = W - cs
            
            if max_y > 0:
                top = torch.randint(0, max_y, (1,)).item()
            else:
                top = 0
            if max_x > 0:
                left = torch.randint(0, max_x, (1,)).item()
            else:
                left = 0
            
            # Crop
            img_crop = image[i:i+1, :, top:top+cs, left:left+cs]
            mask_crop = mask[i:i+1, :, top:top+cs, left:left+cs]
            
            # Resize to target size
            img_resized = F.interpolate(img_crop, size=self.size, mode='bilinear', align_corners=False)
            mask_resized = F.interpolate(mask_crop, size=self.size, mode='nearest')
            
            out_images.append(img_resized)
            out_masks.append(mask_resized)
        
        return torch.cat(out_images, dim=0), torch.cat(out_masks, dim=0)


def get_train_augmentations(img_size: int = 512) -> nn.Module:
    """
    Get Kornia augmentation pipeline for training.
    
    Augmentations applied:
    - Random crop: square crop with scale 0.8-1.0, resize to img_size
    - Geometric: flip, rotation, affine
    - Photometric: brightness, contrast, saturation, hue
    - Noise: Gaussian noise, random erasing (blackout)
    - Blur: Gaussian blur
    
    Args:
        img_size: Target output size after random crop and resize
    
    Returns:
        nn.Module: Kornia augmentation container
    """
    # Custom random crop that handles mask correctly (avoids Kornia align_corners issues)
    crop_aug = RandomResizedCropPair(size=img_size, scale=(0.8, 1.0))
    
    # Geometric augmentations (applied to both image and mask)
    # Note: extra_args sets align_corners=True for mask resampling to avoid PyTorch warning
    geometric_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=15, p=0.3, align_corners=True),
        K.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            p=0.3,
            align_corners=True,
        ),
        data_keys=["input", "mask"],
        same_on_batch=False,
        extra_args={DataKey.MASK: {"align_corners": True, "resample": Resample.NEAREST}},
    )
    
    # Photometric augmentations (applied to image only)
    photometric_augs = K.AugmentationSequential(
        K.RandomBrightness(brightness=(0.8, 1.2), p=0.3),
        K.RandomContrast(contrast=(0.8, 1.2), p=0.3),
        K.RandomSaturation(saturation=(0.8, 1.2), p=0.3),
        K.RandomHue(hue=(-0.1, 0.1), p=0.2),
        K.RandomGamma(gamma=(0.8, 1.2), p=0.2),
        data_keys=["input"],
        same_on_batch=False,
    )
    
    # Noise and degradation (applied to image only)
    noise_augs = K.AugmentationSequential(
        K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.3),
        K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0), p=0.2),
        K.RandomErasing(
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0.0,
            p=0.2,
        ),
        data_keys=["input"],
        same_on_batch=False,
    )
    
    return CombinedAugmentation(crop_aug, geometric_augs, photometric_augs, noise_augs)


class CombinedAugmentation(nn.Module):
    """Combine crop, geometric (image+mask) and photometric (image only) augmentations."""
    
    def __init__(
        self,
        crop: nn.Module,
        geometric: nn.Module,
        photometric: nn.Module,
        noise: nn.Module,
    ):
        super().__init__()
        self.crop = crop
        self.geometric = geometric
        self.photometric = photometric
        self.noise = noise
    
    def forward(self, image, mask):
        """
        Apply augmentations.
        
        Args:
            image: [B, C, H, W] tensor
            mask: [B, 1, H, W] tensor
            
        Returns:
            Augmented image and mask
        """
        # Random crop and resize (both)
        image, mask = self.crop(image, mask)
        
        # Geometric augmentations (both)
        image, mask = self.geometric(image, mask)
        
        # Photometric augmentations (image only)
        image = self.photometric(image)
        
        # Noise augmentations (image only)
        image = self.noise(image)
        
        # Clamp image to valid range
        image = image.clamp(0.0, 1.0)
        
        return image, mask

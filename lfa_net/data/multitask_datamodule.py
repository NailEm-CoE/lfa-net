"""Multi-Task DataModule for Fundus Analysis.

Loads from HuggingFace dataset with:
- 5-channel segmentation masks (disc, cup, artery, vein, vessel)
- Fovea coordinates (normalized by 1024, range [-1, 2] for out-of-bounds)
- Disease labels (multi-label)
- Per-sample validity flags

Fovea Normalization:
- Coordinates normalized by 1024 (minimum expected image dimension)
- Output range: [-1, 2] to support out-of-bounds predictions
- Augmentation uses RandomCrop/ShiftScaleRotate to push fovea out ~30%
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Fovea normalization constant (minimum image dimension)
FOVEA_NORM_SIZE = 1024


class MultitaskFundusDataset(Dataset):
    """Multi-task fundus dataset.
    
    Segmentation channels (5):
        0: Optic Disc
        1: Optic Cup
        2: Artery
        3: Vein
        4: Vessel (binary vessel mask)
    
    Disease classes (3):
        0: DR (Diabetic Retinopathy)
        1: AMD (Age-related Macular Degeneration)
        2: Glaucoma
    
    Args:
        hf_dataset: HuggingFace dataset split
        img_size: Target image size
        transform: Optional transform function
        return_metadata: Whether to return sample metadata
    """
    
    SEG_CHANNEL_NAMES = ["disc", "cup", "artery", "vein", "vessel"]
    DISEASE_NAMES = ["dr", "amd", "glaucoma"]
    
    def __init__(
        self,
        hf_dataset,
        img_size: int = 512,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        self.dataset = hf_dataset
        self.img_size = img_size
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Mask column mappings
        self.mask_cols = {
            "disc": ("od_mask", "has_od_mask"),
            "cup": ("oc_mask", "has_oc_mask"),
            "artery": ("artery_mask", "has_artery_mask"),
            "vein": ("vein_mask", "has_vein_mask"),
            "vessel": ("vessel_mask", "has_vessel_mask"),
        }
        
        # Disease column mappings
        self.disease_cols = {
            "dr": "has_dr",
            "amd": "has_amd",
            "glaucoma": "has_glaucoma",
        }
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.
        
        Returns:
            Dict with:
                - "image": [3, H, W]
                - "seg_masks": [5, H, W]
                - "seg_valid": [5]
                - "fovea": [2]
                - "fovea_valid": scalar
                - "disease_labels": [3]
                - "disease_valid": [3]
        """
        sample = self.dataset[idx]
        
        # Load and preprocess image
        image = self._load_image(sample["image"])
        
        # Load segmentation masks
        seg_masks, seg_valid = self._load_seg_masks(sample)
        
        # Load fovea coordinates
        fovea, fovea_valid = self._load_fovea(sample, image.shape[1:])
        
        # Load disease labels
        disease_labels, disease_valid = self._load_disease(sample)
        
        # Resize to target size
        image = self._resize(image, mode="bilinear")
        seg_masks = self._resize(seg_masks, mode="nearest")
        
        # Apply transforms (if any)
        if self.transform is not None:
            image, seg_masks, fovea = self.transform(image, seg_masks, fovea)
        
        result = {
            "image": image,
            "seg_masks": seg_masks,
            "seg_valid": seg_valid,
            "fovea": fovea,
            "fovea_valid": fovea_valid,
            "disease_labels": disease_labels,
            "disease_valid": disease_valid,
        }
        
        if self.return_metadata:
            result["hash_id"] = sample["hash_id"]
            result["dataset"] = sample["dataset"]
        
        return result
    
    def _load_image(self, pil_image) -> torch.Tensor:
        """Load and normalize image."""
        img = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    
    def _load_seg_masks(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Load segmentation masks and validity flags."""
        masks = []
        valid = []
        
        for ch_name in self.SEG_CHANNEL_NAMES:
            mask_col, has_col = self.mask_cols[ch_name]
            has_mask = sample.get(has_col, 0) == 1
            
            if has_mask and sample.get(mask_col) is not None:
                mask_img = sample[mask_col]
                mask = np.array(mask_img.convert("L"), dtype=np.float32) / 255.0
                mask = (mask > 0.5).astype(np.float32)  # Binarize
                masks.append(torch.from_numpy(mask))
                valid.append(1.0)
            else:
                masks.append(torch.zeros(1, 1))
                valid.append(0.0)
        
        h, w = None, None
        for i, (m, v) in enumerate(zip(masks, valid)):
            if v == 1.0:
                h, w = m.shape
                break
        
        if h is None:
            h, w = 512, 512
        
        final_masks = []
        for i, (m, v) in enumerate(zip(masks, valid)):
            if v == 1.0:
                final_masks.append(m)
            else:
                final_masks.append(torch.zeros(h, w))
        
        seg_masks = torch.stack(final_masks, dim=0)  # [5, H, W]
        seg_valid = torch.tensor(valid)  # [5]
        
        return seg_masks, seg_valid
    
    def _load_fovea(self, sample, img_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Load fovea coordinates (normalized by FOVEA_NORM_SIZE=1024)."""
        fovea_x = sample.get("fovea_x", -1)
        fovea_y = sample.get("fovea_y", -1)
        
        if fovea_x >= 0 and fovea_y >= 0:
            x_norm = fovea_x / FOVEA_NORM_SIZE
            y_norm = fovea_y / FOVEA_NORM_SIZE
            
            fovea = torch.tensor([x_norm, y_norm], dtype=torch.float32)
            fovea_valid = torch.tensor(1.0)
        else:
            fovea = torch.tensor([-1.0, -1.0], dtype=torch.float32)
            fovea_valid = torch.tensor(0.0)
        
        return fovea, fovea_valid
    
    def _load_disease(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Load disease labels and validity."""
        labels = []
        valid = []
        
        for disease_name in self.DISEASE_NAMES:
            col = self.disease_cols[disease_name]
            value = sample.get(col, -1)
            
            if value >= 0:
                labels.append(float(value))
                valid.append(1.0)
            else:
                labels.append(0.0)
                valid.append(0.0)
        
        disease_labels = torch.tensor(labels, dtype=torch.float32)
        disease_valid = torch.tensor(valid, dtype=torch.float32)
        
        return disease_labels, disease_valid
    
    def _resize(self, tensor: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """Resize tensor to target size."""
        if tensor.shape[-2:] == (self.img_size, self.img_size):
            return tensor
        
        align_corners = False if mode == "bilinear" else None
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode=mode,
            align_corners=align_corners,
        ).squeeze(0)


class MultitaskDataModule(pl.LightningDataModule):
    """DataModule for multi-task fundus analysis.
    
    Supports loading from:
    - Local HuggingFace dataset path (load_from_disk)
    - HuggingFace Hub dataset name (load_dataset)
    
    Args:
        dataset_path: Path to local dataset or HuggingFace Hub name
        img_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_transform: Optional training augmentation
        val_transform: Optional validation transform
        use_hub: If True, load from HuggingFace Hub instead of disk
    """
    
    def __init__(
        self,
        dataset_path: str = "kapong/mtl",
        img_size: int = 512,
        batch_size: int = 4,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        use_hub: bool = True,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.use_hub = use_hub
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_vasx_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Load datasets."""
        from datasets import load_from_disk, load_dataset
        
        # Load dataset
        if self.use_hub:
            ds = load_dataset(self.dataset_path)
        else:
            ds = load_from_disk(self.dataset_path)
        
        if stage == "fit" or stage is None:
            self.train_dataset = MultitaskFundusDataset(
                ds["train"],
                img_size=self.img_size,
                transform=self.train_transform,
            )
            self.val_dataset = MultitaskFundusDataset(
                ds["val"],
                img_size=self.img_size,
                transform=self.val_transform,
            )
        
        if stage == "test" or stage is None:
            if "test" in ds:
                self.test_dataset = MultitaskFundusDataset(
                    ds["test"],
                    img_size=self.img_size,
                    transform=self.val_transform,
                    return_metadata=True,
                )
            if "test_vasx" in ds:
                self.test_vasx_dataset = MultitaskFundusDataset(
                    ds["test_vasx"],
                    img_size=self.img_size,
                    transform=self.val_transform,
                    return_metadata=True,
                )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def test_dataloader(self) -> list[DataLoader]:
        """Return both test and test_vasx dataloaders."""
        loaders = []
        if self.test_dataset is not None:
            loaders.append(DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            ))
        if self.test_vasx_dataset is not None:
            loaders.append(DataLoader(
                self.test_vasx_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            ))
        return loaders


def get_basic_augmentations(img_size: int = 512, p_out_of_bounds: float = 0.3):
    """Get augmentation transforms using albumentations.
    
    Augmentations allow fovea to go out of bounds ~30% of the time.
    Uses remove_invisible=False to keep out-of-bounds keypoints.
    
    Args:
        img_size: Target image size
        p_out_of_bounds: Probability of transforms that can push fovea out
        
    Returns:
        Transform function (image, masks, fovea) -> (image, masks, fovea)
    """
    import albumentations as A
    
    geometric_transform = A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=p_out_of_bounds,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=p_out_of_bounds,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3, border_mode=0),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False,
    ))
    
    color_transform = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3,
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
    ])
    
    def transform(
        image: torch.Tensor, 
        masks: torch.Tensor, 
        fovea: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentations."""
        img_np = image.permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]
        
        masks_np = masks.permute(1, 2, 0).numpy()
        
        fovea_np = fovea.numpy()
        fovea_x = fovea_np[0] * w
        fovea_y = fovea_np[1] * h
        
        keypoints = [(fovea_x, fovea_y)]
        
        transformed = geometric_transform(
            image=img_np,
            mask=masks_np,
            keypoints=keypoints,
        )
        
        img_np = transformed["image"]
        masks_np = transformed["mask"]
        keypoints = transformed["keypoints"]
        
        color_transformed = color_transform(image=(img_np * 255).astype(np.uint8))
        img_np = color_transformed["image"].astype(np.float32) / 255.0
        
        image = torch.from_numpy(img_np).permute(2, 0, 1)
        masks = torch.from_numpy(masks_np).permute(2, 0, 1)
        
        new_h, new_w = image.shape[1:]
        
        if len(keypoints) > 0:
            kp_x, kp_y = keypoints[0]
            fovea_x_norm = kp_x / new_w
            fovea_y_norm = kp_y / new_h
            fovea_x_norm = np.clip(fovea_x_norm, -1.0, 2.0)
            fovea_y_norm = np.clip(fovea_y_norm, -1.0, 2.0)
            fovea = torch.tensor([fovea_x_norm, fovea_y_norm], dtype=torch.float32)
        else:
            fovea = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        
        return image, masks, fovea
    
    return transform


def get_val_transform(img_size: int = 512):
    """Get validation transform (resize only, no augmentation)."""
    
    def transform(
        image: torch.Tensor, 
        masks: torch.Tensor, 
        fovea: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Just pass through (resize handled separately)."""
        return image, masks, fovea
    
    return transform

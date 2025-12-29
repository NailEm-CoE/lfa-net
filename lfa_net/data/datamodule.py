"""PyTorch Lightning DataModules for vessel segmentation.

Provides:
- BaseVesselDataset: Abstract base for mask loading
- BinaryVesselDataset: Binary vessel/background (backward compatible)
- AVVesselDataset: Artery/vein multi-class
- HFVesselDataModule: Original module (backward compatible alias)
- BinaryVesselDataModule: Binary segmentation DataModule
- AVVesselDataModule: Multi-class artery/vein DataModule
- CSVVesselDataset: CSV-based vessel dataset
- CSVVesselDataModule: CSV-based DataModule for VASX format
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import get_train_augmentations


# =============================================================================
# Dataset Classes
# =============================================================================


class BaseVesselDataset(Dataset, ABC):
    """Abstract base class for vessel segmentation datasets."""

    def __init__(
        self,
        hf_dataset,
        img_size: int = 512,
        return_metadata: bool = False,
        apply_crop: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            hf_dataset: HuggingFace dataset split
            img_size: Target image size
            return_metadata: Whether to return item_id
            apply_crop: If True, center crop to img_size
        """
        self.dataset = hf_dataset
        self.img_size = img_size
        self.return_metadata = return_metadata
        self.apply_crop = apply_crop
        self.train_size = int(img_size * 1.25)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample. Returns (image [3, H, W], mask [C, H, W])."""
        sample = self.dataset[idx]
        
        # Image processing
        image = np.array(sample["image"])
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Mask processing (subclass-specific)
        mask = np.array(sample["mask"])
        mask_t = self._process_mask(mask)
        
        # Resize
        if self.apply_crop:
            image_t, mask_t = self._resize_square(image_t, mask_t, self.img_size)
        else:
            image_t, mask_t = self._resize_square(image_t, mask_t, self.train_size)
        
        if self.return_metadata:
            return image_t, mask_t, sample["item_id"]
        return image_t, mask_t

    @abstractmethod
    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Process raw mask to tensor. Override in subclasses."""
        pass

    def _resize_square(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize to square target size."""
        image = F.interpolate(
            image.unsqueeze(0), size=(target, target),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0), size=(target, target),
            mode="nearest"
        ).squeeze(0)
        return image, mask


class BinaryVesselDataset(BaseVesselDataset):
    """Binary vessel/background segmentation dataset."""

    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Convert to binary mask [1, H, W]."""
        return torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)


class AVVesselDataset(BaseVesselDataset):
    """
    Artery/Vein segmentation dataset.
    
    Mask encoding:
        0: Background
        1: Artery (red)
        2: Vein (blue)
        3: Crossing/Uncertain
        
    Returns [2, H, W] mask with:
        Channel 0: Artery (1 where artery or crossing)
        Channel 1: Vein (1 where vein or crossing)
    """

    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Convert to 2-channel artery/vein mask [2, H, W]."""
        # Artery: class 1 or crossing (3)
        artery = ((mask == 1) | (mask == 3)).astype(np.float32)
        # Vein: class 2 or crossing (3)
        vein = ((mask == 2) | (mask == 3)).astype(np.float32)
        
        # Stack to [2, H, W]
        av_mask = np.stack([artery, vein], axis=0)
        return torch.from_numpy(av_mask)


# Legacy alias for backward compatibility
HFVesselDataset = BinaryVesselDataset


# =============================================================================
# DataModule Classes
# =============================================================================


class BaseVesselDataModule(pl.LightningDataModule):
    """Base DataModule for vessel segmentation."""

    # Override in subclasses
    dataset_class = BinaryVesselDataset

    def __init__(
        self,
        dataset_path: str = "data/fundus_vessels/dataset",
        img_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
        subset_fraction: float = 1.0,
    ):
        """
        Initialize DataModule.
        
        Args:
            dataset_path: Path to HuggingFace dataset
            img_size: Target image size
            batch_size: Batch size
            num_workers: DataLoader workers
            augment: Whether to apply augmentations
            subset_fraction: Fraction of data to use (1.0 = all, 0.2 = 20%)
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.subset_fraction = subset_fraction
        
        self.train_dataset: Optional[BaseVesselDataset] = None
        self.val_dataset: Optional[BaseVesselDataset] = None
        self.test_dataset: Optional[BaseVesselDataset] = None
        self.train_augment: Optional[torch.nn.Module] = None

    def _subset_hf_dataset(self, hf_split):
        """Subset HuggingFace dataset split by fraction."""
        if self.subset_fraction >= 1.0:
            return hf_split
        n_samples = max(1, int(len(hf_split) * self.subset_fraction))
        return hf_split.select(range(n_samples))

    def setup(self, stage: Optional[str] = None) -> None:
        """Load HuggingFace dataset."""
        hf_dataset = load_from_disk(self.dataset_path)
        
        if stage == "fit" or stage is None:
            train_split = self._subset_hf_dataset(hf_dataset["train"])
            val_split = self._subset_hf_dataset(hf_dataset["validation"])
            
            self.train_dataset = self.dataset_class(
                train_split, self.img_size, apply_crop=not self.augment
            )
            self.val_dataset = self.dataset_class(
                val_split, self.img_size, apply_crop=True
            )
            if self.augment:
                self.train_augment = get_train_augmentations(self.img_size)
            
            subset_info = f" (subset {self.subset_fraction:.0%})" if self.subset_fraction < 1.0 else ""
            print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}{subset_info}")
        
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                hf_dataset["test"], self.img_size, apply_crop=True
            )
            print(f"Test: {len(self.test_dataset)}")

    def on_after_batch_transfer(
        self, batch: tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply GPU augmentations after batch transfer."""
        images, masks = batch
        
        if self.trainer.training and self.train_augment is not None:
            images, masks = self.train_augment(images, masks)
        
        return images, masks

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class BinaryVesselDataModule(BaseVesselDataModule):
    """DataModule for binary vessel segmentation."""
    dataset_class = BinaryVesselDataset


class AVVesselDataModule(BaseVesselDataModule):
    """DataModule for artery/vein segmentation."""
    dataset_class = AVVesselDataset


# Legacy alias for backward compatibility
HFVesselDataModule = BinaryVesselDataModule


# =============================================================================
# CSV-based Dataset Classes (for VASX and similar formats)
# =============================================================================


class CSVVesselDataset(Dataset):
    """
    CSV-based vessel dataset for artery/vein segmentation.
    
    Expects CSV with columns: item_id, org_path, av_path
    Where av_path points to grayscale masks with:
        - 0: background
        - 1: artery
        - 2: vein
        - 3: crossing (mapped to both artery and vein)
    
    Returns [2, H, W] mask with:
        Channel 0: Artery (1 where artery or crossing)
        Channel 1: Vein (1 where vein or crossing)
    
    Skips truncated/corrupted images automatically.
    """

    def __init__(
        self,
        csv_path: str,
        base_dir: str,
        img_size: int = 512,
        apply_crop: bool = True,
        validate_images: bool = False,  # Disabled by default (slow for large datasets)
    ):
        """
        Initialize CSV-based dataset.
        
        Args:
            csv_path: Path to CSV file with columns: item_id, org_path, av_path
            base_dir: Base directory for resolving relative paths
            img_size: Target image size
            apply_crop: Whether to apply center crop
            validate_images: Whether to validate images on init (slow, disabled by default)
        """
        import logging
        import pandas as pd
        
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.img_size = img_size
        self.apply_crop = apply_crop
        self.logger = logging.getLogger(__name__)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Verify required columns
        required_cols = {"item_id", "org_path", "av_path"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must have columns: {required_cols}, got: {set(df.columns)}")
        
        # Validate images if requested (skip truncated/corrupted)
        if validate_images:
            valid_indices = []
            skipped = 0
            for i, row in df.iterrows():
                try:
                    img_path = os.path.join(base_dir, row["org_path"])
                    mask_path = os.path.join(base_dir, row["av_path"])
                    # Try to open and verify images
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(mask_path) as mask:
                        mask.verify()
                    valid_indices.append(i)
                except Exception as e:
                    skipped += 1
                    self.logger.warning(f"Skipping {row['item_id']}: {e}")
            
            self.df = df.iloc[valid_indices].reset_index(drop=True)
            if skipped > 0:
                self.logger.info(f"Skipped {skipped} truncated/corrupted images")
        else:
            self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        try:
            # Load image
            img_path = os.path.join(self.base_dir, row["org_path"])
            image = Image.open(img_path).convert("RGB")
            
            # Load AV mask (grayscale: 0=bg, 1=artery, 2=vein, 3=crossing)
            mask_path = os.path.join(self.base_dir, row["av_path"])
            mask = Image.open(mask_path).convert("L")  # Grayscale
        except Exception as e:
            # Return blank tensors if image is corrupted (should be rare after validation)
            self.logger.warning(f"Error loading {row['item_id']}: {e}")
            return (
                torch.zeros(3, self.img_size, self.img_size),
                torch.zeros(2, self.img_size, self.img_size),
            )
        
        # Apply center crop BEFORE resize for better quality
        if self.apply_crop:
            w, h = image.size
            crop_size = min(h, w)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            mask = mask.crop((left, top, left + crop_size, top + crop_size))
        
        # Resize to target size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to numpy
        image_np = np.array(image, dtype=np.float32) / 255.0  # [H, W, 3]
        mask_np = np.array(mask)  # [H, W], values 0-3
        
        # Extract artery and vein channels (crossing = both)
        # Artery: class 1 or crossing (3)
        artery = ((mask_np == 1) | (mask_np == 3)).astype(np.float32)
        # Vein: class 2 or crossing (3)
        vein = ((mask_np == 2) | (mask_np == 3)).astype(np.float32)
        
        # Convert to tensors [C, H, W]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.stack([
            torch.from_numpy(artery),
            torch.from_numpy(vein)
        ], dim=0)  # [2, H, W]
        
        return image_tensor, mask_tensor


class CSVVesselDataModule(pl.LightningDataModule):
    """
    DataModule for CSV-based vessel datasets.
    
    Supports VASX and similar formats with CSV index files.
    """

    def __init__(
        self,
        csv_path: str,
        base_dir: str,
        img_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize CSV DataModule.
        
        Args:
            csv_path: Path to CSV file
            base_dir: Base directory for image paths
            img_size: Target image size
            batch_size: Batch size
            num_workers: DataLoader workers
            augment: Whether to apply augmentations
            train_split: Fraction for training
            val_split: Fraction for validation (rest is test)
            seed: Random seed for splitting
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        
        self.train_dataset: Optional[CSVVesselDataset] = None
        self.val_dataset: Optional[CSVVesselDataset] = None
        self.test_dataset: Optional[CSVVesselDataset] = None
        self.train_augment: Optional[torch.nn.Module] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Split CSV and create datasets."""
        import pandas as pd
        
        # Load and shuffle CSV
        df = pd.read_csv(self.csv_path)
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)
        
        # Save split CSVs to temp files
        import tempfile
        self._temp_dir = tempfile.mkdtemp()
        
        train_csv = os.path.join(self._temp_dir, "train.csv")
        val_csv = os.path.join(self._temp_dir, "val.csv")
        test_csv = os.path.join(self._temp_dir, "test.csv")
        
        df.iloc[:train_end].to_csv(train_csv, index=False)
        df.iloc[train_end:val_end].to_csv(val_csv, index=False)
        df.iloc[val_end:].to_csv(test_csv, index=False)
        
        if stage == "fit" or stage is None:
            self.train_dataset = CSVVesselDataset(
                train_csv, self.base_dir, self.img_size, apply_crop=not self.augment
            )
            self.val_dataset = CSVVesselDataset(
                val_csv, self.base_dir, self.img_size, apply_crop=True
            )
            if self.augment:
                self.train_augment = get_train_augmentations(self.img_size)
            
            print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            self.test_dataset = CSVVesselDataset(
                test_csv, self.base_dir, self.img_size, apply_crop=True
            )
            print(f"Test: {len(self.test_dataset)}")

    def on_after_batch_transfer(
        self, batch: tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply GPU augmentations after batch transfer."""
        images, masks = batch
        
        if self.trainer.training and self.train_augment is not None:
            images, masks = self.train_augment(images, masks)
        
        return images, masks

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

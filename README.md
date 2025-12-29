# LFA-Net: Local Feature Aggregation Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NailEm-CoE/lfa-net/blob/main/notebooks/train_colab.ipynb)

A PyTorch implementation of LFA-Net for retinal vessel segmentation.

## Features

- **LFA-Net Architecture**: Lightweight encoder-decoder with LiteFusion Attention bottleneck
- **Latent Skip Architecture**: Uniform skip projections to latent space for efficient memory usage
- **Flexible Depth**: Configurable encoder filters for 3-6 level architectures
- **Multi-class Support**: Binary or artery/vein segmentation
- **Hydra Configuration**: Full hyperparameter management with TensorBoard logging
- **PyTorch Lightning**: Clean training/validation loops with logging
- **Kornia Augmentations**: GPU-accelerated data augmentation pipeline
- **HuggingFace Datasets**: Efficient PyArrow-backed data loading

## Performance

| Configuration | Dice | IoU | Parameters |
|--------------|------|-----|------------|
| Paper [8,16,32] | 0.832 | 0.712 | 0.097M |
| Light [16,32,64] | - | - | 0.367M |
| Deep [16,32,64,128] | - | - | 1.45M |
| **LatentSkip** [32,48,72,144] | - | - | **1.92M** |

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/NailEm-CoE/lfa-net.git

# Or clone and install locally
git clone https://github.com/NailEm-CoE/lfa-net.git
cd lfa-net
pip install -e .
```

### With GPU Support (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/NailEm-CoE/lfa-net.git
```

### Optional Dependencies

```bash
# For Hydra configuration
pip install hydra-core omegaconf

# For hyperparameter tuning
pip install optuna optuna-dashboard
```

## Quick Start

### Load Dataset from Hugging Face

```python
from datasets import load_dataset

# Load retinal vessel dataset
dataset = load_dataset("kapong/fundus-vessel-segmentation")
print(dataset)
# DatasetDict({
#     'train': Dataset({features: ['image', 'mask'], num_rows: 1211}),
#     'validation': Dataset({features: ['image', 'mask'], num_rows: 800}),
#     'test': Dataset({features: ['image', 'mask'], num_rows: 903})
# })
```

## Usage

### Training with Hydra (Recommended)

```bash
# Default training (512×512, binary)
python scripts/train_hydra.py

# 1024×1024 with deeper model
python scripts/train_hydra.py model=lfa_net_1024 data=fundus_1024 train=train_1024

# Custom encoder filters
python scripts/train_hydra.py 'model.encoder_filters=[16, 32, 64, 128]'

# Multi-class artery/vein segmentation
python scripts/train_hydra.py model=lfa_net_av data=fundus_av

# Override hyperparameters
python scripts/train_hydra.py train.learning_rate=5e-4 train.max_epochs=300
```

### Legacy Training

```bash
python scripts/train.py \
    --data_dir data/fundus_vessels/dataset \
    --max_epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint_path outputs/checkpoints/best.ckpt \
    --data_dir data/fundus_vessels/dataset
```

### Python API

```python
import torch
from lfa_net import LFANet, FlexibleLFANet, LatentSkipLFANet, count_parameters

# Original model (backward compatible)
model = LFANet(in_channels=3, num_classes=1, feature_scale=2)

# Flexible architecture with custom depth
model = FlexibleLFANet(
    encoder_filters=[16, 32, 64, 128],  # 4 levels
    out_channels=1,
)
print(f"Parameters: {count_parameters(model) / 1e6:.3f}M")

# Latent Skip architecture (uniform skip projections)
model = LatentSkipLFANet(
    encoder_channels=[32, 48, 72, 144],  # 4 levels
    out_channels=1,
    latent_channels=8,   # Skip projection channels
    skip_spatial=32,     # Skip spatial size
)
print(f"LatentSkip Parameters: {count_parameters(model) / 1e6:.3f}M")

# Inference
x = torch.randn(1, 3, 512, 512)
pred = model(x)  # [1, 1, 512, 512]

# Multi-class (artery/vein)
model = LatentSkipLFANet(
    encoder_channels=[32, 48, 72, 144],
    out_channels=2,  # artery + vein
)

# With Lightning module
from lfa_net import FlexibleLFANetLightning, MulticlassLFANetLightning

# Binary segmentation
lightning_model = FlexibleLFANetLightning(
    encoder_filters=[16, 32, 64],
    learning_rate=1e-4,
)

# Multi-class
multiclass_model = MulticlassLFANetLightning(
    encoder_filters=[16, 32, 64],
    out_channels=2,
    class_names=["artery", "vein"],
)
```

## Architecture

```
Input (512×512×3)
    ↓
[Encoder Block 1] → MaxPool → BN → [filters[0]]
    ↓
[Encoder Block 2] → MaxPool → BN → [filters[1]]
    ↓
[Encoder Block N] → MaxPool → BN → [filters[N-1]]
    ↓
[LiteFusion Attention Bottleneck]
  ├── Focal Modulation
  ├── Context Aggregation
  └── Vision Mamba Inspired
    ↓
[RA Attention] + Skip Connections
    ↓
[Decoder Blocks with Upsampling]
    ↓
Output (512×512×C)  # C=1 binary, C=2 artery/vein
```

### Model Configurations

| Name | encoder_filters | Parameters | Use Case |
|------|-----------------|------------|----------|
| Paper | [8, 16, 32] | 0.097M | Baseline |
| Light | [16, 32, 64] | 0.367M | Better accuracy |
| Deep | [16, 32, 64, 128] | 1.45M | 1024×1024 |
| Very Deep | [8, 16, 32, 64, 128] | 1.46M | Maximum depth |
| **LatentSkip** | [32, 48, 72, 144] | **1.92M** | Uniform skip projections |

### Latent Skip Architecture

The `LatentSkipLFANet` projects all skip connections to a uniform latent space `[B, latent_ch, S, S]`:

```
Input (512×512×3)
    ↓
[LFABlock 1] → Skip1 → SkipProjection → [B, 8, 32, 32]
    ↓ (downsample)
[LFABlock 2] → Skip2 → SkipProjection → [B, 8, 32, 32]
    ↓
[LFABlock 3] → Skip3 → SkipProjection → [B, 8, 32, 32]
    ↓
[LFABlock 4] → Skip4 → SkipProjection → [B, 8, 32, 32]
    ↓
[Bottleneck]
    ↓
[Decoder] ← SkipUnprojection ← Uniform skips
    ↓
Output (512×512×C)
```

**Benefits**:
- Uniform representation: All skips have same shape
- Memory efficiency: Smaller latent space vs full-resolution skips
- Flexibility: Easy to add/remove encoder levels

## Data Augmentation

Training pipeline includes:
- RandomResizedCrop (scale 0.8-1.0)
- Random horizontal/vertical flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Gaussian noise

## Project Structure

```
lfa-net/
├── src/lfa_net/              # Core library
│   ├── config/               # Hydra config files
│   │   ├── config.yaml       # Main config
│   │   ├── model/            # Model configs
│   │   ├── data/             # Data configs
│   │   └── train/            # Training configs
│   ├── data/                 # DataModule & transforms
│   │   ├── datamodule.py     # Binary & AV DataModules
│   │   └── transforms.py     # Kornia augmentations
│   ├── models/               # Model architectures
│   │   └── latent_skip.py    # LatentSkipLFANet
│   ├── layers.py             # Custom layers
│   ├── model.py              # LFANet & FlexibleLFANet
│   ├── lightning_module.py   # Lightning wrappers
│   ├── losses.py             # Dice, BCE+Dice, multiclass
│   └── metrics.py            # Segmentation metrics
├── scripts/                  # CLI tools
│   ├── train_hydra.py        # Hydra training (recommended)
│   ├── train.py              # Legacy argparse training
│   └── evaluate.py           # Evaluation
├── tests/                    # Unit tests
├── research/                 # Experiments
└── plan/                     # Development plans
```

## Citation

Based on the LFA-Net paper for retinal vessel segmentation.

## License

MIT License - see [LICENSE](LICENSE) for details.

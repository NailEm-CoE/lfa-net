# LFA-Net

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Local Feature Aggregation Network** for retinal vessel segmentation.

## Installation

```bash
pip install git+https://github.com/NailEm-CoE/lfa-net.git
```

## Quick Start

```python
import torch
from lfa_net import BottleneckLFABlockNet, count_parameters

# Create model
model = BottleneckLFABlockNet(
    encoder_channels=[32, 48, 72, 144],
    out_channels=2,  # artery + vein
    latent_channels=8,
    skip_spatial=64,
)
print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")

# Inference
x = torch.randn(1, 3, 1024, 1024)
pred = model(x)  # [1, 2, 1024, 1024]
```

## Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("kapong/fundus-vessel-segmentation")
print(dataset)
# DatasetDict({
#     'train': Dataset({num_rows: 1211}),
#     'validation': Dataset({num_rows: 800}),
#     'test': Dataset({num_rows: 903})
# })
```

## Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `LFANet` | Original architecture | ~0.1M |
| `LFABlockNet` | Configurable encoder depth | 0.3-1.5M |
| `BottleneckLFABlockNet` | Uniform skip projections | ~2M |

## Training with PyTorch Lightning

```python
from lfa_net import BottleneckLFABlockNetLightning
from lfa_net.data import AVVesselDataModule
import pytorch_lightning as pl

# DataModule
datamodule = AVVesselDataModule(
    dataset_path="kapong/fundus-vessel-segmentation",
    image_size=1024,
    batch_size=4,
)

# Model
model = BottleneckLFABlockNetLightning(
    encoder_channels=[32, 48, 72, 144],
    out_channels=2,
    learning_rate=1e-4,
)

# Train
trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, datamodule)
```

## License

MIT

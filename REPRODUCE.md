# Complete Reproduction Guide

*Everything needed to recreate latentspacewalker from scratch*

---

## Requirements

### Hardware
- NVIDIA GPU with at least 6GB VRAM (tested on GTX 1660 Super)
- 32GB RAM recommended
- SSD for faster data loading

### Software
- Python 3.10+
- PyTorch 2.0+ with CUDA
- Linux (tested on Ubuntu 22.04)

---

## Step 1: Environment Setup

```bash
# Create project directory
mkdir latentspacewalker
cd latentspacewalker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy pillow
```

---

## Step 2: Prepare Dataset

You need a dataset of images. The model was trained on ~7,800 personal photographs at 256x256 resolution.

```bash
# Create dataset directory
mkdir dataset

# Your images should be placed here as .png or .jpg files
# The model will resize them during training
```

**Dataset requirements:**
- Minimum ~5,000 images recommended
- Any resolution (will be resized progressively)
- Diverse content works better than uniform

---

## Step 3: Model Architecture

Create `progressive_gan_smooth.py`:

```python
"""
Progressive Growing GAN with Smooth Upsampling

Based on: "Progressive Growing of GANs" (Karras et al., 2018)
With fix from: "Deconvolution and Checkerboard Artifacts" (Odena et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = np.sqrt(2.0 / in_features)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class PixelNorm(nn.Module):
    """Pixel-wise normalization."""
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation - helps against mode collapse."""
    def forward(self, x):
        batch_size = x.size(0)
        std = torch.std(x, dim=0, keepdim=True)
        mean_std = torch.mean(std)
        mean_std = mean_std.expand(batch_size, 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


class SmoothGeneratorBlock(nn.Module):
    """Generator block with BILINEAR upsampling instead of nearest."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.pixel_norm = PixelNorm()
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.leaky(self.pixel_norm(self.conv1(x)))
        x = self.leaky(self.pixel_norm(self.conv2(x)))
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator downsampling block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.downsample(x)
        return x


class SmoothProgressiveGenerator(nn.Module):
    """
    Progressive Generator with smooth (bilinear) upsampling.

    Resolutions: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
    Levels:      0    1     2     3     4      5      6
    Channels:  256  256   256   128    64     32     16
    """
    def __init__(self, latent_dim=256, base_channels=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial block (4x4)
        self.initial = nn.Sequential(
            EqualizedLinear(latent_dim, base_channels * 4 * 4),
            nn.Unflatten(1, (base_channels, 4, 4)),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base_channels, base_channels, 3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        )

        # Channel progression
        channels = [base_channels, base_channels, 256, 128, 64, 32, 32]

        # Progressive blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(SmoothGeneratorBlock(channels[i], channels[i+1]))

        # toRGB layers for each resolution
        self.to_rgb = nn.ModuleList()
        for ch in channels:
            self.to_rgb.append(EqualizedConv2d(ch, 3, 1))

        self.current_level = 0
        self.alpha = 1.0

    def forward(self, z):
        x = self.initial(z)

        if self.current_level == 0:
            return torch.tanh(self.to_rgb[0](x))

        for i in range(self.current_level):
            x_prev = x
            x = self.blocks[i](x)

        if self.alpha < 1.0:
            x_prev_up = F.interpolate(x_prev, scale_factor=2, mode='bilinear', align_corners=False)
            rgb_prev = self.to_rgb[self.current_level - 1](x_prev_up)
            rgb_curr = self.to_rgb[self.current_level](x)
            out = self.alpha * rgb_curr + (1 - self.alpha) * rgb_prev
        else:
            out = self.to_rgb[self.current_level](x)

        return torch.tanh(out)

    def grow(self):
        if self.current_level < len(self.blocks):
            self.current_level += 1
            self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = min(1.0, self.alpha + delta)


class SmoothProgressiveDiscriminator(nn.Module):
    """Progressive Discriminator (mirror structure to Generator)."""
    def __init__(self, base_channels=256):
        super().__init__()
        self.base_channels = base_channels

        self.channels = [base_channels, base_channels, 256, 128, 64, 32, 32]

        # fromRGB layers
        self.from_rgb = nn.ModuleList()
        for ch in self.channels:
            self.from_rgb.append(nn.Sequential(
                EqualizedConv2d(3, ch, 1),
                nn.LeakyReLU(0.2)
            ))

        # Progressive blocks
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            in_ch = self.channels[i + 1]
            out_ch = self.channels[i]
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch))

        # Final block
        self.final = nn.Sequential(
            MinibatchStdDev(),
            EqualizedConv2d(base_channels + 1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base_channels, base_channels, 4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            EqualizedLinear(base_channels, 1)
        )

        self.current_level = 0
        self.alpha = 1.0

    def forward(self, x):
        if self.current_level == 0:
            x = self.from_rgb[0](x)
        else:
            if self.alpha < 1.0:
                x_down = F.avg_pool2d(x, 2)
                x_prev = self.from_rgb[self.current_level - 1](x_down)

                x_curr = self.from_rgb[self.current_level](x)
                x_curr = self.blocks[self.current_level - 1](x_curr)

                x = self.alpha * x_curr + (1 - self.alpha) * x_prev
            else:
                x = self.from_rgb[self.current_level](x)
                x = self.blocks[self.current_level - 1](x)

            for i in range(self.current_level - 2, -1, -1):
                x = self.blocks[i](x)

        x = self.final(x)
        return x.view(-1)

    def grow(self):
        if self.current_level < len(self.blocks):
            self.current_level += 1
            self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = min(1.0, self.alpha + delta)


class ProgressiveDataset(Dataset):
    """Dataset that returns images at different resolutions."""
    def __init__(self, image_dir, current_resolution=4):
        self.image_dir = Path(image_dir)
        self.current_resolution = current_resolution
        self.image_files = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg'))
        print(f"Found {len(self.image_files)} images")

    def set_resolution(self, resolution):
        self.current_resolution = resolution

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        img = img.resize((self.current_resolution, self.current_resolution), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img


def get_resolution_for_level(level):
    """Returns resolution for a given level."""
    return 4 * (2 ** level)
```

---

## Step 4: Training Script

Create `train_progressive.py`:

```python
"""
Progressive GAN Training Script

Trains from 4x4 up to 128x128 (Level 5).
WGAN-GP loss for stable training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from progressive_gan_smooth import (
    SmoothProgressiveGenerator,
    SmoothProgressiveDiscriminator,
    ProgressiveDataset,
    get_resolution_for_level
)


def gradient_penalty(D, real, fake, device):
    """WGAN-GP gradient penalty."""
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


def train_level(G, D, dataset, level, config, device):
    """Train at a specific resolution level."""
    resolution = get_resolution_for_level(level)
    dataset.set_resolution(resolution)

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    opt_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(0.0, 0.99))

    # Set current level
    G.current_level = level
    D.current_level = level
    G.alpha = 1.0
    D.alpha = 1.0

    print(f"\n=== Training Level {level} ({resolution}x{resolution}) ===")

    for epoch in range(config['epochs_per_level']):
        total_d_loss = 0
        total_g_loss = 0

        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)

            # Train Discriminator
            for _ in range(config['d_steps']):
                z = torch.randn(batch_size, G.latent_dim, device=device)
                fake = G(z).detach()

                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake, device)

                d_loss = d_fake - d_real + config['gp_lambda'] * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # Train Generator
            z = torch.randn(batch_size, G.latent_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d = total_d_loss / len(dataloader)
        avg_g = total_g_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs_per_level']} | D: {avg_d:.4f} | G: {avg_g:.4f}")

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint(G, D, level, epoch + 1, config['checkpoint_dir'])


def save_checkpoint(G, D, level, epoch, checkpoint_dir):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'level': level,
        'epoch': epoch
    }, f"{checkpoint_dir}/level{level}_epoch{epoch:04d}.pt")
    print(f"Saved checkpoint: level{level}_epoch{epoch:04d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to image dataset')
    parser.add_argument('--max_level', type=int, default=5, help='Maximum level (5=128x128)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs_per_level', type=int, default=50, help='Epochs per level')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = {
        'batch_size': args.batch_size,
        'lr_g': 0.001,
        'lr_d': 0.001,
        'd_steps': 1,
        'gp_lambda': 10.0,
        'epochs_per_level': args.epochs_per_level,
        'save_every': 5,
        'checkpoint_dir': args.checkpoint_dir
    }

    # Initialize models
    G = SmoothProgressiveGenerator(latent_dim=256).to(device)
    D = SmoothProgressiveDiscriminator().to(device)

    # Load checkpoint if resuming
    start_level = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G_state_dict'])
        D.load_state_dict(ckpt['D_state_dict'])
        start_level = ckpt['level']
        print(f"Resumed from level {start_level}")

    # Load dataset
    dataset = ProgressiveDataset(args.dataset)

    # Train progressively
    for level in range(start_level, args.max_level + 1):
        train_level(G, D, dataset, level, config, device)
        if level < args.max_level:
            G.grow()
            D.grow()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
```

---

## Step 5: Training

```bash
# Activate environment
source venv/bin/activate

# Train from scratch (takes ~20 hours on GTX 1660 Super)
python train_progressive.py --dataset dataset/ --max_level 5 --batch_size 12

# Or resume from checkpoint
python train_progressive.py --dataset dataset/ --resume checkpoints/level3_epoch0050.pt
```

**Training schedule:**
- Level 0 (4x4): ~10 minutes
- Level 1 (8x8): ~15 minutes
- Level 2 (16x16): ~30 minutes
- Level 3 (32x32): ~1 hour
- Level 4 (64x64): ~3 hours
- Level 5 (128x128): ~15 hours

**Total: ~20 hours**

---

## Step 6: Generate Images

Create `generate.py`:

```python
"""Generate images from trained model."""

import torch
from torchvision.utils import save_image
from pathlib import Path
import argparse

from progressive_gan_smooth import SmoothProgressiveGenerator


def generate(checkpoint_path, num_images=16, seed=None, output_dir='generated'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    G = SmoothProgressiveGenerator(latent_dim=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.current_level = 5  # 128x128
    G.alpha = 1.0
    G.eval()

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Generate
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(num_images, 256, device=device)
        images = G(z)
        images = (images + 1) / 2  # Scale to [0, 1]

        for i, img in enumerate(images):
            save_image(img, f"{output_dir}/generated_{i:03d}.png")

    print(f"Generated {num_images} images in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num', type=int, default=16)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default='generated')
    args = parser.parse_args()

    generate(args.checkpoint, args.num, args.seed, args.output)
```

---

## Step 7: Latent Space Exploration

See `code/latent_explorer.py`, `code/stepwise_walker.py`, and `code/gradient_walker.py` for exploration tools.

Basic usage:

```bash
# Random walk (16 frames as grid)
python code/latent_explorer.py --walk 123 --checkpoint checkpoints/final.pt

# Step-by-step walk (individual images)
python code/stepwise_walker.py 123 --steps 32 --checkpoint checkpoints/final.pt

# Interpolation between two seeds
python code/gradient_walker.py --interpolate 42 666 --checkpoint checkpoints/final.pt
```

---

## Summary

1. **Setup:** Python + PyTorch + CUDA
2. **Dataset:** ~5,000+ images, any content
3. **Architecture:** Progressive GAN with bilinear upsampling
4. **Training:** ~20 hours to reach 128x128
5. **Output:** 256-dimensional latent space, deterministic generation

The key insight: Progressive training learns coarse structure before fine details, which works better than training directly at high resolution.

---

## Troubleshooting

### Out of Memory
- Reduce batch_size (min: 4)
- Train fewer epochs per level
- Use mixed precision (add `torch.cuda.amp`)

### Mode Collapse
- Increase gradient penalty (gp_lambda)
- Train discriminator more steps (d_steps)
- Check dataset diversity

### Blurry Results
- Train longer at each level
- Check that bilinear upsampling is used (not nearest)
- Verify alpha blending during level transitions

---

*latentspacewalker - 2026*

"""
Progressive Growing GAN - SMOOTH VERSION

Fixar banding-artefakter genom bilinear upsampling istallet for nearest.

Kompatibel med original progressive_gan.py checkpoints (levels 0-4).
Lagger till Level 5 (128x128) med 32 kanaler.

Baserat pa: "Progressive Growing of GANs" (Karras et al., 2018)
Med fix fran: "Deconvolution and Checkerboard Artifacts" (Odena et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image


class EqualizedLinear(nn.Module):
    """Linear layer med equalized learning rate."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = np.sqrt(2.0 / in_features)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """Conv2d med equalized learning rate."""
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
    """Minibatch standard deviation - hjalper mot mode collapse."""
    def forward(self, x):
        batch_size = x.size(0)
        std = torch.std(x, dim=0, keepdim=True)
        mean_std = torch.mean(std)
        mean_std = mean_std.expand(batch_size, 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


class SmoothGeneratorBlock(nn.Module):
    """
    Generator block med BILINEAR upsampling istallet for nearest.
    Detta eliminerar blockiga artefakter vid hog upplosning.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.pixel_norm = PixelNorm()
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        # BILINEAR istallet for NEAREST - kritisk andring!
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.leaky(self.pixel_norm(self.conv1(x)))
        x = self.leaky(self.pixel_norm(self.conv2(x)))
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator downsampling block (oforandrad)."""
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
    Progressive Generator med smooth (bilinear) upsampling.

    VIKTIG: Kanaler matchar original progressive_gan.py for levels 0-4!
    Level 5 (128x128) och Level 6 (256x256) tillagda.

    Resolutions: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
    Levels:      0    1     2     3     4      5      6
    Channels:  256  256   256   128    64     32     16
    """
    def __init__(self, latent_dim=256, base_channels=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial block (4x4) - EXAKT som original
        self.initial = nn.Sequential(
            EqualizedLinear(latent_dim, base_channels * 4 * 4),
            nn.Unflatten(1, (base_channels, 4, 4)),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base_channels, base_channels, 3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        )

        # Channel progression - levels 0-4 matchar original, level 5-6 tillagda
        # Level:                 0      1      2      3     4     5      6
        # Resolution:           4x4   8x8   16x16  32x32 64x64 128x128 256x256
        # OBS: Level 6 far 32 kanaler (samma som Level 5) for stabilitet
        channels = [base_channels, base_channels, 256, 128, 64, 32, 32]

        # Progressive blocks - anvander SmoothGeneratorBlock
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
            # BILINEAR fade-in ocksa
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
    """
    Progressive Discriminator (spegelbar struktur till Generator).

    VIKTIG: Kanaler matchar original progressive_gan.py for levels 0-4!
    Level 5 (128x128) och Level 6 (256x256) tillagda.
    """
    def __init__(self, base_channels=256):
        super().__init__()
        self.base_channels = base_channels

        # Kanaler for varje upplosning - matchar Generator
        # Level 6 (256x256) har 32 kanaler (samma som Level 5 for stabilitet)
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
    """Dataset som kan returnera bilder i olika upplosningar."""
    def __init__(self, image_dir, current_resolution=4):
        self.image_dir = Path(image_dir)
        self.current_resolution = current_resolution
        self.image_files = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg'))
        print(f"Hittade {len(self.image_files)} bilder")

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
    """Returnerar upplosning for en given level."""
    return 4 * (2 ** level)

"""
Stepwise Walker - Random walk with each step as separate image

Saves each step as its own file so you can browse through
step by step and see the transformation gradually.
"""

import torch
from torchvision.utils import save_image
from pathlib import Path
import argparse

from progressive_gan_smooth import SmoothProgressiveGenerator

# Configuration - adjust these paths for your setup
CHECKPOINT = 'checkpoints/generator.pt'  # Path to trained generator
OUTPUT_DIR = Path('gradient_walks')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 256


def load_generator(checkpoint_path=CHECKPOINT):
    """Load trained Progressive GAN generator."""
    print(f"Loading generator from {checkpoint_path}...")
    G = SmoothProgressiveGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt['G_state_dict'], strict=False)
    G.current_level = 5  # 128x128 resolution
    G.alpha = 1.0
    G.eval()
    return G


def stepwise_walk(G, seed, n_steps=32, step_size=0.3, output_dir=OUTPUT_DIR):
    """
    Random walk where each step is saved as a separate image.

    Args:
        G: Generator model
        seed: Starting point seed (for reproducibility)
        n_steps: Number of steps to take
        step_size: How far to move each step (0.1=subtle, 0.3=visible, 1.0=large)
        output_dir: Where to save images
    """
    walk_dir = Path(output_dir) / f'walk_{seed}'
    walk_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    z = torch.randn(1, LATENT_DIM, device=DEVICE)

    print(f"Generating {n_steps} steps from seed {seed}...")

    with torch.no_grad():
        for i in range(n_steps):
            # Generate image
            img = G(z)
            img = (img + 1) / 2  # Scale from [-1,1] to [0,1]

            # Save step
            filename = f'step_{i:02d}.png'
            save_image(img, walk_dir / filename)

            # Take a step in random direction
            dz = torch.randn(1, LATENT_DIM, device=DEVICE) * step_size
            z = z + dz

            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{n_steps}")

    print(f"\nDone! {n_steps} images saved to:")
    print(f"  {walk_dir}")

    return walk_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stepwise Walker - Random walk through latent space')
    parser.add_argument('seed', type=int, help='Starting point seed')
    parser.add_argument('--steps', type=int, default=32, help='Number of steps')
    parser.add_argument('--step_size', type=float, default=0.3, help='Step size (0.1-1.0)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='Path to generator checkpoint')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')

    args = parser.parse_args()

    G = load_generator(args.checkpoint)
    stepwise_walk(G, args.seed, args.steps, args.step_size, args.output)

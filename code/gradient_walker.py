"""
Gradient Walker - Directional traversal through latent space

Unlike random walk (random direction each step), gradient walk
chooses ONE direction and follows it consistently.

This gives a gradual, controlled transformation along an "axis"
in the 256-dimensional space.
"""

import torch
from torchvision.utils import save_image
from pathlib import Path
import numpy as np
import argparse

from progressive_gan_smooth import SmoothProgressiveGenerator

# Configuration - adjust these paths for your setup
CHECKPOINT = 'checkpoints/generator.pt'
OUTPUT_DIR = Path('gradient_walks')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 256


def load_generator(checkpoint_path=CHECKPOINT):
    """Load trained Progressive GAN generator."""
    print(f"Loading generator from {checkpoint_path}...")
    G = SmoothProgressiveGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt['G_state_dict'], strict=False)
    G.current_level = 5
    G.alpha = 1.0
    G.eval()
    return G


def gradient_walk(G, start_seed, direction_seed, n_steps=16, step_size=0.5, output_dir=OUTPUT_DIR):
    """
    Walk in a FIXED direction from a starting point.

    Args:
        start_seed: Seed for starting point
        direction_seed: Seed for direction vector
        n_steps: Number of steps
        step_size: How far each step goes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create starting point
    torch.manual_seed(start_seed)
    z_start = torch.randn(1, LATENT_DIM, device=DEVICE)

    # Create direction (normalized to unit length)
    torch.manual_seed(direction_seed)
    direction = torch.randn(1, LATENT_DIM, device=DEVICE)
    direction = direction / direction.norm()

    images = []
    z = z_start.clone()

    with torch.no_grad():
        for i in range(n_steps):
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

            # Move in SAME direction each step
            z = z + step_size * direction

    grid = torch.cat(images, dim=0)
    filename = f'gradient_start{start_seed}_dir{direction_seed}.png'
    save_image(grid, output_dir / filename, nrow=4)
    print(f"Saved: {output_dir / filename}")

    return filename


def axis_walk(G, start_seed, axis, n_steps=16, output_dir=OUTPUT_DIR):
    """
    Walk along a SPECIFIC DIMENSION (0-255) in latent space.

    This shows what each individual dimension "controls".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(start_seed)
    z_start = torch.randn(1, LATENT_DIM, device=DEVICE)

    # Create direction along a single axis
    direction = torch.zeros(1, LATENT_DIM, device=DEVICE)
    direction[0, axis] = 1.0

    images = []

    # Walk from -range to +range
    with torch.no_grad():
        for i, alpha in enumerate(np.linspace(-3, 3, n_steps)):
            z = z_start + alpha * direction
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

    grid = torch.cat(images, dim=0)
    filename = f'axis{axis}_start{start_seed}.png'
    save_image(grid, output_dir / filename, nrow=4)
    print(f"Saved: {output_dir / filename}")

    return filename


def interpolate_walk(G, seed_a, seed_b, n_steps=16, output_dir=OUTPUT_DIR):
    """
    Interpolate between two points - the straightest path.

    Linear interpolation: z(t) = (1-t)*z_A + t*z_B
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed_a)
    z_a = torch.randn(1, LATENT_DIM, device=DEVICE)

    torch.manual_seed(seed_b)
    z_b = torch.randn(1, LATENT_DIM, device=DEVICE)

    images = []

    with torch.no_grad():
        for t in np.linspace(0, 1, n_steps):
            z = (1 - t) * z_a + t * z_b
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

    grid = torch.cat(images, dim=0)
    filename = f'interp_{seed_a}_to_{seed_b}.png'
    save_image(grid, output_dir / filename, nrow=4)
    print(f"Saved: {output_dir / filename}")

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient Walker - Directional traversal')
    parser.add_argument('--gradient', nargs=2, type=int, metavar=('START', 'DIR'),
                       help='Gradient walk: start_seed direction_seed')
    parser.add_argument('--axis', nargs=2, type=int, metavar=('START', 'AXIS'),
                       help='Axis walk: start_seed axis_number(0-255)')
    parser.add_argument('--interpolate', nargs=2, type=int, metavar=('A', 'B'),
                       help='Interpolate: seed_a seed_b')
    parser.add_argument('--steps', type=int, default=16, help='Number of steps')
    parser.add_argument('--step_size', type=float, default=0.5, help='Step size for gradient walk')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='Generator checkpoint')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')

    args = parser.parse_args()

    G = load_generator(args.checkpoint)

    if args.gradient:
        gradient_walk(G, args.gradient[0], args.gradient[1],
                     args.steps, args.step_size, args.output)
    elif args.axis:
        axis_walk(G, args.axis[0], args.axis[1], args.steps, args.output)
    elif args.interpolate:
        interpolate_walk(G, args.interpolate[0], args.interpolate[1], args.steps, args.output)
    else:
        print("Gradient Walker - directional traversal through latent space")
        print("=" * 55)
        print()
        print("Usage:")
        print("  --gradient START DIR   Walk in direction DIR from START")
        print("  --axis START N         Walk along dimension N (0-255)")
        print("  --interpolate A B      Interpolate from A to B")
        print()
        print("Examples:")
        print("  python gradient_walker.py --gradient 42 123")
        print("  python gradient_walker.py --axis 42 0")
        print("  python gradient_walker.py --interpolate 42 999")

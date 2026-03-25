"""
Latent Space Explorer

Tools for exploring the latent space of a trained Progressive GAN:
1. Generate samples with random z-vectors
2. Train direction classifier (requires labeling)
3. Explore along learned directions
4. Interpolate between points
5. Random walks through the space
"""

import torch
import json
from torchvision.utils import save_image, make_grid
from pathlib import Path
import argparse

from progressive_gan_smooth import SmoothProgressiveGenerator

# Configuration - adjust for your setup
CHECKPOINT = 'checkpoints/generator.pt'
OUTPUT_DIR = Path('exploration')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 256


def load_generator(checkpoint_path=CHECKPOINT):
    """Load trained generator from checkpoint."""
    print(f"Loading generator from {checkpoint_path}")
    G = SmoothProgressiveGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt['G_state_dict'], strict=False)
    G.current_level = 5  # 128x128
    G.alpha = 1.0
    G.eval()
    print(f"Generator loaded (epoch {ckpt.get('epoch', '?')})")
    return G


def generate_samples(G, n_samples=100, output_dir=OUTPUT_DIR):
    """
    Generate N images with random z-vectors.
    Saves both images and z-vectors for later analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_vectors = []

    print(f"Generating {n_samples} samples...")

    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            img = G(z)
            img = (img + 1) / 2

            save_image(img, output_dir / f'sample_{i:04d}.png')
            z_vectors.append(z.cpu().numpy().flatten().tolist())

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_samples}")

    # Save z-vectors for reproducibility
    with open(output_dir / 'z_vectors.json', 'w') as f:
        json.dump(z_vectors, f)

    # Create empty labels file
    labels = {f'sample_{i:04d}.png': None for i in range(n_samples)}
    with open(output_dir / 'labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone! Samples saved to {output_dir}")
    print(f"\nTo train a direction classifier:")
    print(f"   - Set 1 for good/desired images")
    print(f"   - Set 0 for bad/undesired images")
    print(f"   - Edit {output_dir / 'labels.json'}")


def train_direction(output_dir=OUTPUT_DIR):
    """
    Train linear classifier to find direction in latent space.
    Requires labels.json to be filled in.
    """
    output_dir = Path(output_dir)

    with open(output_dir / 'z_vectors.json') as f:
        z_vectors = json.load(f)
    with open(output_dir / 'labels.json') as f:
        labels = json.load(f)

    # Filter labeled samples
    X, y = [], []
    for i, (filename, label) in enumerate(labels.items()):
        if label is not None:
            X.append(z_vectors[i])
            y.append(label)

    if len(X) < 10:
        print(f"Need at least 10 labeled samples. Found: {len(X)}")
        print(f"Edit {output_dir / 'labels.json'} and label more images.")
        return None

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Simple linear classifier
    direction = torch.randn(LATENT_DIM, 1, requires_grad=True)
    optimizer = torch.optim.Adam([direction], lr=0.01)

    for epoch in range(1000):
        pred = torch.sigmoid(X @ direction)
        loss = torch.nn.functional.binary_cross_entropy(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Normalize direction
    direction = direction.detach() / direction.norm()

    # Save direction
    torch.save(direction, output_dir / 'direction.pt')
    print(f"Direction saved to {output_dir / 'direction.pt'}")

    return direction


def explore_direction(G, direction, start_seed=42, n_steps=16, step_size=0.5, output_dir=OUTPUT_DIR):
    """
    Generate images along a learned direction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(start_seed)
    z_start = torch.randn(1, LATENT_DIM, device=DEVICE)

    direction = direction.to(DEVICE).view(1, -1)

    images = []
    with torch.no_grad():
        for i, alpha in enumerate(torch.linspace(-3, 3, n_steps)):
            z = z_start + alpha * direction
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

    grid = make_grid(torch.cat(images), nrow=4)
    save_image(grid, output_dir / f'direction_exploration_seed{start_seed}.png')
    print(f"Saved: {output_dir / f'direction_exploration_seed{start_seed}.png'}")


def interpolate(G, seed_a, seed_b, n_steps=16, output_dir=OUTPUT_DIR):
    """
    Linear interpolation between two latent points.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed_a)
    z_a = torch.randn(1, LATENT_DIM, device=DEVICE)

    torch.manual_seed(seed_b)
    z_b = torch.randn(1, LATENT_DIM, device=DEVICE)

    images = []
    with torch.no_grad():
        for t in torch.linspace(0, 1, n_steps):
            z = (1 - t) * z_a + t * z_b
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

    grid = make_grid(torch.cat(images), nrow=4)
    save_image(grid, output_dir / f'interp_{seed_a}_to_{seed_b}.png')
    print(f"Saved: {output_dir / f'interp_{seed_a}_to_{seed_b}.png'}")


def random_walk(G, seed, n_steps=16, step_size=0.3, output_dir=OUTPUT_DIR):
    """
    Random walk through latent space.
    Each step moves in a random direction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    z = torch.randn(1, LATENT_DIM, device=DEVICE)

    images = []
    with torch.no_grad():
        for i in range(n_steps):
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

            # Random step
            dz = torch.randn(1, LATENT_DIM, device=DEVICE) * step_size
            z = z + dz

    # Save as grid
    grid = make_grid(torch.cat(images), nrow=4)
    save_image(grid, output_dir / f'walk_seed{seed}.png')
    print(f"Saved: {output_dir / f'walk_seed{seed}.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Space Explorer')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='Path to generator checkpoint')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')

    # Actions
    parser.add_argument('--generate', type=int, metavar='N', help='Generate N samples for labeling')
    parser.add_argument('--train', action='store_true', help='Train direction from labels')
    parser.add_argument('--explore', type=int, metavar='SEED', help='Explore along trained direction')
    parser.add_argument('--interpolate', nargs=2, type=int, metavar=('A', 'B'), help='Interpolate between seeds')
    parser.add_argument('--walk', type=int, metavar='SEED', help='Random walk from seed')

    # Parameters
    parser.add_argument('--steps', type=int, default=16, help='Number of steps')
    parser.add_argument('--step_size', type=float, default=0.3, help='Step size for walk')

    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)
    CHECKPOINT = args.checkpoint

    if args.generate:
        G = load_generator(args.checkpoint)
        generate_samples(G, args.generate, OUTPUT_DIR)

    elif args.train:
        train_direction(OUTPUT_DIR)

    elif args.explore is not None:
        G = load_generator(args.checkpoint)
        direction = torch.load(OUTPUT_DIR / 'direction.pt')
        explore_direction(G, direction, args.explore, args.steps, output_dir=OUTPUT_DIR)

    elif args.interpolate:
        G = load_generator(args.checkpoint)
        interpolate(G, args.interpolate[0], args.interpolate[1], args.steps, OUTPUT_DIR)

    elif args.walk is not None:
        G = load_generator(args.checkpoint)
        random_walk(G, args.walk, args.steps, args.step_size, OUTPUT_DIR)

    else:
        print("Latent Space Explorer")
        print("=" * 40)
        print()
        print("Usage:")
        print("  --generate N       Generate N samples for labeling")
        print("  --train            Train direction classifier")
        print("  --explore SEED     Explore along direction from SEED")
        print("  --interpolate A B  Interpolate between seeds A and B")
        print("  --walk SEED        Random walk from SEED")
        print()
        print("Examples:")
        print("  python latent_explorer.py --walk 123 --checkpoint model.pt")
        print("  python latent_explorer.py --interpolate 42 666")

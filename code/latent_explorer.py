"""
Latent Space Explorer for Progressive GAN

Hitta riktningar i latent space som styr mot specifika egenskaper
(t.ex. skulptur, glas, organisk).

Arbetsflode:
1. Generera bilder med slumpmassiga z-vektorer
2. Labella manuellt (skulptur/ej skulptur)
3. Trana linjar klassificerare
4. Extrahera riktningsvektor
5. Generera bilder langs riktningen

Koppling till teori:
Latent space ar det "preindividuella" - ett potentialfalt dar alla
mojliga bilder existerar virtuellt. Att navigera detta rum ar individuation.
"""

import sys
sys.path.insert(0, '/home/biffy/examen/min_bild_ai')

import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
import json
import numpy as np
from PIL import Image

from progressive_gan_smooth import SmoothProgressiveGenerator

# Konfig
CHECKPOINT = '/home/biffy/examen/min_bild_ai/experiments/progressive_level5_continued/checkpoints/epoch_0100.pt'
OUTPUT_DIR = Path('/home/biffy/examen/min_bild_ai/latent_exploration')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 256


def load_generator():
    """Ladda tranad generator"""
    print(f"Laddar generator fran {CHECKPOINT}")
    G = SmoothProgressiveGenerator(latent_dim=LATENT_DIM).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt['G_state_dict'], strict=False)
    G.current_level = 5
    G.alpha = 1.0
    G.eval()

    print(f"Generator laddad (epoch {ckpt.get('epoch', '?')})")
    return G


def generate_samples(G, n_samples=100, seed=42):
    """
    Generera N bilder med slumpmassiga z-vektorer.
    Spara bade bilder och z-vektorer for senare analys.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = OUTPUT_DIR / 'samples_to_label'
    samples_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)

    z_vectors = []

    print(f"Genererar {n_samples} samples...")

    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            img = G(z)
            img = (img + 1) / 2  # [-1,1] -> [0,1]

            # Spara bild
            save_image(img, samples_dir / f'sample_{i:04d}.png')

            # Spara z-vektor
            z_vectors.append(z.cpu().numpy().flatten().tolist())

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_samples}")

    # Spara alla z-vektorer
    with open(OUTPUT_DIR / 'z_vectors.json', 'w') as f:
        json.dump(z_vectors, f)

    # Skapa tom labels-fil
    labels = {f'sample_{i:04d}.png': None for i in range(n_samples)}
    with open(OUTPUT_DIR / 'labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\nKlart! Samples sparade i: {samples_dir}")
    print(f"Z-vektorer sparade i: {OUTPUT_DIR / 'z_vectors.json'}")
    print(f"\nNASTA STEG:")
    print(f"1. Oppna {samples_dir}")
    print(f"2. Titta igenom bilderna")
    print(f"3. Redigera {OUTPUT_DIR / 'labels.json'}")
    print(f"   - Satt 1 for skulptur/bra")
    print(f"   - Satt 0 for ej skulptur/dalig")
    print(f"4. Kor: python latent_explorer.py --train")


def train_direction(min_labels=20):
    """
    Trana linjar klassificerare for att hitta riktning i latent space.
    Kraver att labels.json ar ifylld.
    """
    # Ladda z-vektorer
    with open(OUTPUT_DIR / 'z_vectors.json', 'r') as f:
        z_vectors = json.load(f)

    # Ladda labels
    with open(OUTPUT_DIR / 'labels.json', 'r') as f:
        labels = json.load(f)

    # Filtrera labellade samples
    X = []
    y = []

    for i, (filename, label) in enumerate(labels.items()):
        if label is not None:
            X.append(z_vectors[i])
            y.append(label)

    if len(X) < min_labels:
        print(f"For fa labels! Har {len(X)}, behover minst {min_labels}.")
        print(f"Redigera {OUTPUT_DIR / 'labels.json'} och labella fler bilder.")
        return None

    X = np.array(X)
    y = np.array(y)

    print(f"Tranar pa {len(X)} labellade samples")
    print(f"  Positiva (skulptur): {sum(y)}")
    print(f"  Negativa (ej skulptur): {len(y) - sum(y)}")

    # Enkel linjar klassificerare (logistisk regression)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Riktningsvektorn ar klassificerarens vikter
    direction = clf.coef_[0]
    direction = direction / np.linalg.norm(direction)  # Normalisera

    # Spara riktning
    np.save(OUTPUT_DIR / 'sculpture_direction.npy', direction)

    print(f"\nRiktningsvektor sparad: {OUTPUT_DIR / 'sculpture_direction.npy'}")
    print(f"Accuracy pa traningsdata: {clf.score(X, y):.2%}")

    return direction


def generate_along_direction(G, direction, n_samples=5, alpha_range=(-3, 3)):
    """
    Generera bilder langs en riktning i latent space.
    """
    direction = torch.tensor(direction, dtype=torch.float32, device=DEVICE)

    output_dir = OUTPUT_DIR / 'direction_samples'
    output_dir.mkdir(exist_ok=True)

    # Generera fran flera startpunkter
    for seed in range(3):
        torch.manual_seed(seed + 100)
        z_base = torch.randn(1, LATENT_DIM, device=DEVICE)

        images = []
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_samples)

        with torch.no_grad():
            for alpha in alphas:
                z = z_base + alpha * direction.unsqueeze(0)
                img = G(z)
                img = (img + 1) / 2
                images.append(img)

        # Spara som grid
        grid = torch.cat(images, dim=0)
        save_image(grid, output_dir / f'direction_seed{seed}.png', nrow=n_samples)

    print(f"Riktningssamples sparade i: {output_dir}")


def interpolate(G, seed1, seed2, n_steps=8):
    """
    Interpolera mellan tva punkter i latent space.
    """
    output_dir = OUTPUT_DIR / 'interpolations'
    output_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed1)
    z1 = torch.randn(1, LATENT_DIM, device=DEVICE)

    torch.manual_seed(seed2)
    z2 = torch.randn(1, LATENT_DIM, device=DEVICE)

    images = []

    with torch.no_grad():
        for t in np.linspace(0, 1, n_steps):
            z = (1 - t) * z1 + t * z2
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

    grid = torch.cat(images, dim=0)
    save_image(grid, output_dir / f'interp_{seed1}_to_{seed2}.png', nrow=n_steps)

    print(f"Interpolation sparad: {output_dir / f'interp_{seed1}_to_{seed2}.png'}")


def random_walk(G, seed=42, n_steps=16, step_size=0.3):
    """
    Slumpmassig vandring i latent space.
    """
    output_dir = OUTPUT_DIR / 'random_walks'
    output_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    z = torch.randn(1, LATENT_DIM, device=DEVICE)

    images = []

    with torch.no_grad():
        for i in range(n_steps):
            img = G(z)
            img = (img + 1) / 2
            images.append(img)

            # Ta ett steg i slumpmassig riktning
            dz = torch.randn(1, LATENT_DIM, device=DEVICE) * step_size
            z = z + dz

    grid = torch.cat(images, dim=0)
    save_image(grid, output_dir / f'walk_seed{seed}.png', nrow=4)

    print(f"Random walk sparad: {output_dir / f'walk_seed{seed}.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Latent Space Explorer')
    parser.add_argument('--generate', type=int, default=0,
                       help='Generera N samples for labeling')
    parser.add_argument('--train', action='store_true',
                       help='Trana riktningsvektor fran labels')
    parser.add_argument('--explore', action='store_true',
                       help='Generera samples langs riktningen')
    parser.add_argument('--interpolate', nargs=2, type=int,
                       help='Interpolera mellan tva seeds')
    parser.add_argument('--walk', type=int, default=0,
                       help='Random walk fran seed')

    args = parser.parse_args()

    G = load_generator()

    if args.generate > 0:
        generate_samples(G, n_samples=args.generate)

    elif args.train:
        direction = train_direction()
        if direction is not None:
            generate_along_direction(G, direction)

    elif args.explore:
        direction = np.load(OUTPUT_DIR / 'sculpture_direction.npy')
        generate_along_direction(G, direction)

    elif args.interpolate:
        interpolate(G, args.interpolate[0], args.interpolate[1])

    elif args.walk > 0:
        random_walk(G, seed=args.walk)

    else:
        # Visa hjalp
        print("Latent Space Explorer")
        print("=" * 50)
        print("\nAnvandning:")
        print("  python latent_explorer.py --generate 100   # Generera 100 samples")
        print("  python latent_explorer.py --train          # Trana riktning")
        print("  python latent_explorer.py --explore        # Generera langs riktning")
        print("  python latent_explorer.py --interpolate 42 123")
        print("  python latent_explorer.py --walk 42        # Random walk")
        print("\nArbetsflode:")
        print("  1. --generate 100")
        print("  2. Labella i labels.json (1=skulptur, 0=ej)")
        print("  3. --train")
        print("  4. --explore")


if __name__ == '__main__':
    main()

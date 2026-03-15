# latentspacewalker

A Progressive GAN trained from scratch on personal photographs. No pretrained model, no external dataset, no CLIP, no Stable Diffusion.

The model learned to generate images by studying one person's visual world: photographs accumulated over several years.

Everything the model "knows" comes from that single source.

## Examples

### Random Walk Grid (16 frames)

![Walk Seed 123](examples/walk_seed123.png)

*Seed 123: organic forms moving through latent space*

### Smooth Walk Sequence

Gradual transformation through the space:

| Step 0 | Step 8 | Step 16 | Step 24 | Step 32 | Step 40 |
|--------|--------|---------|---------|---------|---------|
| ![](examples/smooth_sequence/step_00.png) | ![](examples/smooth_sequence/step_08.png) | ![](examples/smooth_sequence/step_16.png) | ![](examples/smooth_sequence/step_24.png) | ![](examples/smooth_sequence/step_32.png) | ![](examples/smooth_sequence/step_40.png) |

### More walks

| Seed 42 | Seed 666 |
|---------|----------|
| ![](examples/walk_seed42.png) | ![](examples/walk_seed666.png) |

## What is this?

A GAN compresses visual information into a latent space: a mathematical manifold where each point corresponds to a possible image. This project explores that space by walking through it, step by step, watching how images change.

## Latent space

A point in latent space is a list of 256 numbers:

```
z = [0.42, -1.23, 0.87, 0.01, ..., -0.56]
```

Feed it to the generator, get an image:

```
image = Generator(z)
```

256 dimensions. Each captures some aspect of visual variation, but not in a way you can interpret directly. Dimension 47 doesn't mean "brightness". The meaning is distributed, entangled.

The space is continuous. No walls, no gaps. Every point has neighbors. Move slightly in any direction, get a slightly different image. That's what makes walking possible.

## Three ways to walk

### Random Walk

```
z_next = z_current + random_direction * step_size
```

Random direction each step. Like walking a city without a map.

### Gradient Walk

```
z_next = z_current + FIXED_direction * step_size
```

One direction, all the way. Systematic transformation along an axis.

### Interpolation

```
z = (1-t) * z_A + t * z_B
```

Straight line between two points. Morphing.

## Step size

| Step size | Effect |
|-----------|--------|
| 0.05 | Barely visible change |
| 0.15 | Soft gradient, good for animation |
| 0.30 | Clear transformation |
| 0.50 | Large jumps |
| 1.00 | Risk of discontinuity |

## Geography

After hundreds of walks: patterns. The space has regions.

**Document**: horizontal lines, text-like patterns, light backgrounds.
**Organic**: centered forms, soft gradients, skin-like textures.
**Glass**: high contrast, transparency, reflections.
**Atmospheric**: silhouettes, fog, horizon lines.
**Material**: rust, stone, metal, corrosion.

No hard boundaries. A walk can start in organic territory and drift into glass.

## Architecture

Progressive GAN. Training starts at 4x4 and increases:

```
4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
```

Learns coarse structures first, then details. WGAN-GP loss.

Generator: 256-dimensional latent vector in, 128x128 RGB image out.

## Usage

```bash
# Grid walk
python code/latent_explorer.py --walk 123

# Step-by-step
python code/stepwise_walker.py 123 --steps 32 --step_size 0.3

# Smooth
python code/stepwise_walker.py 123 --steps 48 --step_size 0.15

# Interpolation
python code/latent_explorer.py --interpolate 42 666
```

Requires trained model checkpoint (not included due to size).

## From scratch

No ImageNet. No CLIP. No Stable Diffusion. No external knowledge.

Everything was learned from the source photographs. The constraint is the point.

## Documentation

- [HISTORY.txt](HISTORY.txt): the full journey, what failed and why
- [REPRODUCE.txt](REPRODUCE.txt): step-by-step guide
- [REFERENCES.txt](REFERENCES.txt): papers and attribution
- [TECHNICAL_GUIDE.txt](TECHNICAL_GUIDE.txt): tensors, vectors, GAN theory

## License

Images and samples are original output from a personally trained model. Code provided for educational purposes.

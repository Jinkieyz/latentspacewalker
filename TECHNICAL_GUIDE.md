# Technical Guide: Understanding Generative Models

*From tensors to latent spaces - everything you need to know*

---

## Table of Contents

1. [Fundamental Concepts](#part-1-fundamental-concepts)
2. [Why NOT Diffusion](#part-2-why-not-diffusion)
3. [Why Progressive GAN Works](#part-3-why-progressive-gan-works)
4. [The Complete Experiment History](#part-4-the-complete-experiment-history)
5. [Architectural Decisions](#part-5-architectural-decisions)

---

# Part 1: Fundamental Concepts

## Tensors

A **tensor** is a multi-dimensional array of numbers. It's the basic data structure in deep learning.

### Dimensions (Ranks)

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | Single number | `()` |
| 1 | Vector | List of numbers | `(256,)` |
| 2 | Matrix | 2D grid | `(128, 128)` |
| 3 | 3D Tensor | Image (H x W x C) | `(128, 128, 3)` |
| 4 | 4D Tensor | Batch of images | `(12, 128, 128, 3)` |

### In This Project

```
Latent vector z:     shape (256,)      - 256 numbers
Generated image:     shape (128, 128, 3)  - 128x128 RGB
Batch of images:     shape (12, 128, 128, 3)  - 12 images
```

### Why Tensors Matter

Neural networks transform tensors into other tensors. The generator transforms a 256-dimensional vector into a 128x128x3 image tensor.

---

## Vectors

A **vector** is a 1D tensor - a list of numbers that can represent a point in space.

### The Latent Vector

Our latent vector `z` has 256 dimensions:

```
z = [z_0, z_1, z_2, ..., z_255]
```

Each `z_i` is a floating-point number, typically sampled from a standard normal distribution N(0, 1).

### Why 256?

The dimension is a design choice balancing:
- **Expressiveness:** More dimensions = more possible outputs
- **Trainability:** More dimensions = harder to learn
- **Data size:** ~8,000 images can meaningfully fill ~256 dimensions

For comparison:
- StyleGAN uses 512 dimensions (trained on millions of images)
- Simple autoencoders might use 32-128 dimensions

---

## Latent Space

**Latent space** is the abstract mathematical space where latent vectors live.

### Properties

1. **High-dimensional:** 256 dimensions (impossible to visualize directly)
2. **Continuous:** No gaps - every point is valid
3. **Smooth:** Nearby points produce similar images
4. **Learned:** The structure emerges from training

### The Mapping

```
Latent Space (256D) --[Generator]--> Image Space (128x128x3)
```

The generator is a function G: R^256 -> R^(128*128*3)

### Sampling

We sample from a Gaussian distribution because:
- Training data was encoded near this distribution
- Random uniform sampling would hit "empty" regions
- The model expects inputs with certain statistical properties

```python
z = torch.randn(1, 256)  # Sample from N(0, I)
```

---

## Neural Network Layers

### Convolutional Layer (Conv2d)

Applies a sliding filter across an image to detect features.

```
Input: (batch, channels_in, height, width)
Output: (batch, channels_out, height', width')
```

**What it learns:**
- Early layers: Edges, simple patterns
- Middle layers: Textures, shapes
- Late layers: Complex structures

### Transposed Convolution (Upsampling)

The opposite of convolution - increases spatial resolution.

```
Input: 64x64 -> Transposed Conv -> Output: 128x128
```

**Problem:** Can cause "checkerboard artifacts" (grid-like patterns).

**Solution:** Use bilinear upsampling + regular convolution instead.

### Fully Connected (Linear)

Every input connected to every output. Used for:
- Mapping latent vector to initial feature map
- Final classification in discriminator

```
Input: (batch, 256)
Output: (batch, 4*4*256 = 4096)
Reshape: (batch, 256, 4, 4)
```

---

## Normalization Techniques

### Batch Normalization

Normalizes across the batch dimension. **Not used in WGAN** because it interferes with the Lipschitz constraint.

### Pixel Normalization

Normalizes each pixel independently across channels:

```python
x = x / sqrt(mean(x^2, dim=channels) + epsilon)
```

Used in Progressive GAN generator to stabilize training.

### Equalized Learning Rate

Instead of normalizing activations, normalize weights at runtime:

```python
weight_normalized = weight * sqrt(2 / fan_in)
```

This ensures all layers learn at similar rates regardless of depth.

---

## Loss Functions

### Mean Squared Error (MSE)

```
L = mean((prediction - target)^2)
```

**Problem for generation:** Encourages averaging, produces blurry results.

### Adversarial Loss (GAN)

Two networks compete:
- Generator G tries to fool discriminator
- Discriminator D tries to distinguish real from fake

**Original GAN:**
```
L_D = -log(D(real)) - log(1 - D(G(z)))
L_G = -log(D(G(z)))
```

**Problem:** Mode collapse, training instability.

### Wasserstein Loss (WGAN)

Uses Earth Mover's Distance instead of JS divergence:

```
L_D = D(G(z)) - D(real)  # Minimize
L_G = -D(G(z))           # Minimize
```

**Requires:** Discriminator must be 1-Lipschitz (output changes slowly).

### Gradient Penalty (WGAN-GP)

Enforces Lipschitz constraint by penalizing gradient magnitude:

```
L_GP = lambda * (||grad_D(interpolated)||_2 - 1)^2
```

Where `interpolated` is a random mix of real and fake images.

**This is what we use.** It's stable and works well on small datasets.

---

# Part 2: Why NOT Diffusion

## What is Diffusion?

Diffusion models work by:
1. **Forward process:** Gradually add noise to images until pure noise
2. **Reverse process:** Learn to remove noise step by step

```
Image -> Noisy -> Noisier -> ... -> Pure Noise
                    ↑
            Train network to reverse this
```

### The Math

Forward process (add noise):
```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
```

Reverse process (predict noise):
```
predicted_noise = Network(x_t, t)
x_{t-1} = (x_t - predicted_noise) / sqrt(alpha_t) + noise
```

## Why Diffusion Failed for This Project

### Problem 1: High Dimensionality

Pixel-space diffusion operates on:
```
256 x 256 x 3 = 196,608 dimensions
```

With only ~8,000 training images, the model can't learn the full distribution. It learns the **average** instead.

### Problem 2: MSE Loss

Diffusion training minimizes:
```
L = MSE(predicted_noise, actual_noise)
```

MSE encourages predicting the mean. When data is diverse, the mean is blurry.

### Problem 3: No Structure Learning

Diffusion learns pixel distributions, not semantic structure. It doesn't understand "object" vs "background" - just pixel statistics.

### What We Got

After 100 epochs of pixel diffusion:
- Atmospheric textures (correct color distributions)
- No defined objects
- No clear contours
- "Mood generator" - pleasing but meaningless

### When Diffusion DOES Work

Diffusion excels when:
- Dataset is large (millions of images)
- Data is uniform (all faces, all bedrooms, etc.)
- Pretrained models provide structure (CLIP, etc.)
- Latent diffusion reduces dimensionality

None of these applied to our project.

---

## Latent Diffusion (Why We Didn't Use It)

Latent diffusion compresses images first:

```
Image (256x256x3) -> Encoder -> Latent (32x32x8) -> Diffusion -> Decoder -> Image
```

**Why we considered it:**
- 64x smaller latent space
- We had a trained VAE
- Faster training

**Why it still wouldn't work:**
- Still learns distributions, not structure
- Our VAE latent was 8x32x32 = 8,192 dims (still too high for 8k images)
- Conditioning required (we wanted unconditional)

---

# Part 3: Why Progressive GAN Works

## The Key Insight

Progressive GAN learns **hierarchically**:

1. First: Global structure (4x4)
2. Then: Rough shapes (8x8, 16x16)
3. Then: Details (32x32, 64x64)
4. Finally: Fine textures (128x128)

This matches how images are organized - coarse to fine.

## The Architecture

### Generator

```
z (256) -> Dense -> 4x4x256
        -> Upsample + Conv -> 8x8x256
        -> Upsample + Conv -> 16x16x256
        -> Upsample + Conv -> 32x32x128
        -> Upsample + Conv -> 64x64x64
        -> Upsample + Conv -> 128x128x32
        -> Conv 1x1 -> 128x128x3 (RGB)
```

### Discriminator

Mirror structure, but downsampling:

```
Image (128x128x3) -> Conv -> 128x128x32
                  -> Conv + Pool -> 64x64x64
                  -> Conv + Pool -> 32x32x128
                  -> Conv + Pool -> 16x16x256
                  -> Conv + Pool -> 8x8x256
                  -> Conv + Pool -> 4x4x256
                  -> Dense -> 1 (real/fake score)
```

## Why It Works on Small Datasets

### 1. Progressive Learning

By starting small, the model learns coarse structure before worrying about details. This is easier than learning everything at once.

### 2. Adversarial Loss

Unlike MSE, adversarial loss doesn't encourage averaging. The discriminator punishes blurry outputs.

### 3. WGAN-GP Stability

Gradient penalty prevents mode collapse (generating only one type of image).

### 4. Minibatch Std Dev

Adding minibatch statistics to the discriminator encourages diversity.

## What We Get

After training:
- Centered forms (figure-ground separation)
- Defined edges (not blurry)
- Diverse outputs (no mode collapse)
- Smooth latent space (interpolation works)

---

# Part 4: The Complete Experiment History

## Timeline: ~7000 Hours, 46 Experiments (June 2025 - March 2026)

297 days of continuous development, training, monitoring, and iteration.

**What those hours include:**
- GPU training running (often overnight/multi-day)
- Monitoring and adjusting hyperparameters
- Analyzing failed experiments
- Code development and debugging
- Research and paper reading
- Dataset preparation and curation
- Restarting crashed training runs
- Waiting for results, iterating

### Phase 1: StyleGAN2 (EXP-001 to EXP-017)

**Duration:** ~150 hours
**Result:** Mode collapse

StyleGAN2 is designed for uniform datasets (faces, cars). Our heterogeneous photos caused:
- Discriminator dominance
- Generator collapse to single mode
- No meaningful outputs

**Lesson:** State-of-the-art isn't always best for your problem.

### Phase 2: Diffusion (EXP-018 to EXP-022)

**Duration:** ~80 hours
**Result:** Blurry averages

| Experiment | Approach | Result |
|------------|----------|--------|
| EXP-018 | Pixel Diffusion | Atmospheric blur |
| EXP-019 | CLIP Diffusion | Violated from-scratch |
| EXP-020 | Custom Embeddings | Still blurry |
| EXP-021 | SlotDiffusion | Wrong data type |
| EXP-022 | Edge-First | GAN collapse |

**Lesson:** Diffusion learns distributions, not structure.

### Phase 3: Progressive GAN (EXP-023 to EXP-026)

**Duration:** ~15 hours
**Result:** SUCCESS

| Experiment | Resolution | Result |
|------------|------------|--------|
| EXP-023 | 4x4 to 64x64 | First forms appeared! |
| EXP-024 | With ADA | Abandoned (unnecessary) |
| EXP-025 | 128x128 | **Best results** |
| EXP-026 | 256x256 | Quality degraded |

**Lesson:** Progressive training + WGAN-GP works.

### Phase 4: Conditioning Attempts (EXP-027 to EXP-035)

**Duration:** ~35 hours
**Result:** Categories merge

Tried to add control:
- Class labels (is_sculpture)
- 32-feature vectors
- AC-GAN architecture

All failed because discriminator couldn't separate categories with limited data.

**Lesson:** Conditioning requires abundant labeled data.

### Phase 5: Token-Based (EXP-036 to EXP-044)

**Duration:** ~90 hours
**Result:** Various failures

| Experiment | Approach | Result |
|------------|----------|--------|
| EXP-036-038 | VQ-VAE | Tokenization worked |
| EXP-039-040 | Token Transformer | Raster bias |
| EXP-041-044 | MaskGIT | Mode collapse |

**Lesson:** Token methods amplify dataset imbalances.

### Phase 6: Return to ProGAN (EXP-045 to EXP-046)

**Duration:** ~5 hours
**Result:** Confirmed best

After exhausting alternatives, returned to Progressive GAN. Extended training to epoch 100 at Level 5 (128x128).

**Final model:** This is what latentspacewalker uses.

---

## The Central Insight

After 600+ hours:

> **"Architecture does NOT solve data problems."**

With 7,000 heterogeneous images + from-scratch training + unconditional generation, the problem is mathematically underdetermined.

Progressive GAN works not because it's "better" but because:
1. It learns hierarchically (coarse to fine)
2. It doesn't try to separate what can't be separated
3. It creates a smooth space where everything blends

---

# Part 5: Architectural Decisions

## Decision: No Pretrained Models (DEC-001)

**Why:**
- Pretrained models carry bias from their training data
- We wanted outputs that reflect only this specific photo archive
- Learning from scratch reveals what the data actually contains

**Trade-off:**
- Much harder to train
- Lower quality than fine-tuned models
- But: authentic to the source data

## Decision: Progressive Training (EXP-023+)

**Why:**
- Learns global structure before local details
- More stable than direct high-resolution training
- Better gradient flow through shallow early networks

**Implementation:**
```
Level 0: 4x4    (tiny, fast)
Level 1: 8x8    (still fast)
Level 2: 16x16  (shapes emerge)
Level 3: 32x32  (forms defined)
Level 4: 64x64  (details appear)
Level 5: 128x128 (final resolution)
```

## Decision: WGAN-GP Loss

**Why:**
- More stable than vanilla GAN loss
- Gradient penalty prevents mode collapse
- Works well on small datasets

**Implementation:**
```python
# Discriminator loss
d_loss = D(fake) - D(real) + 10 * gradient_penalty

# Generator loss
g_loss = -D(fake)
```

## Decision: Bilinear Upsampling

**Why:**
- Nearest-neighbor creates checkerboard artifacts
- Transposed convolution has similar problems
- Bilinear is smooth and artifact-free

**Implementation:**
```python
# Instead of:
x = nn.ConvTranspose2d(...)(x)

# We use:
x = F.interpolate(x, scale_factor=2, mode='bilinear')
x = nn.Conv2d(...)(x)
```

## Decision: 256D Latent Space

**Why:**
- Enough dimensions for visual diversity
- Not so many that training fails
- Standard choice for this scale of data

**Trade-off:**
- 512D might capture more variation
- But would be harder to train with limited data

## Decision: Stop at 128x128

**Why:**
- Quality degrades at 256x256 (Level 6)
- Dataset probably lacks sufficient high-frequency detail
- 128x128 is enough for the artistic purpose

**Trade-off:**
- Lower resolution than modern models
- But: stable, reliable quality

---

## What We Could NOT Do

### Native High Resolution

**Why not:** Dataset of ~8,000 images can't meaningfully fill the space needed for 512x512 or 1024x1024 generation.

### Text Conditioning

**Why not:** Would require CLIP or similar pretrained model, violating from-scratch principle.

### Category Control

**Why not:** Dataset too heterogeneous and small. Categories overlap too much for the discriminator to separate.

### Diffusion-Based Generation

**Why not:** Diffusion learns distributions, not structure. Results in blurry averages on diverse datasets.

### Real-Time Generation

**Why not:** Model requires GPU. Could potentially be optimized but wasn't a priority.

---

## Summary

This project demonstrates that:

1. **Simple architectures can work** - Progressive GAN from 2018 outperformed complex modern approaches
2. **Data determines outcomes** - Architecture can't fix data limitations
3. **From-scratch is hard** - But produces authentic results
4. **Latent space is the product** - The trained space is more valuable than individual outputs

The 256-dimensional latent space contains all possible images this model can generate. Walking through it reveals the visual vocabulary learned from one person's photograph archive.

---

*latentspacewalker - 2026*

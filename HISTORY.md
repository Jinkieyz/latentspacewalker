# The Complete History of min_bild_ai

*From 400 hours of failures to a working latent space explorer*

---

## Overview

This document chronicles the complete journey of building a generative image model from scratch, without pretrained weights, trained entirely on a personal photograph archive of ~7,800 images.

**Total project time:** ~7000 hours across 46 experiments (June 2025 - March 2026)

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

**Final working model:** Progressive GAN at 128x128 resolution
**Key insight:** Architecture does NOT solve data problems

---

## The Goal

Generate novel images that capture the visual essence of a personal photo archive. Not copies, not collages - new configurations of learned visual patterns.

**Core constraints:**
- No pretrained models (no ImageNet weights, no CLIP, no Stable Diffusion)
- Train from scratch on personal data only
- Run on consumer GPU (GTX 1660 Super, 6GB VRAM)

---

## Phase 1: StyleGAN2 Attempts (EXP-001 to EXP-017)

**Duration:** ~150 hours
**Result:** Complete failure (mode collapse)

### What we tried:
- StyleGAN2 with various learning rates
- Different batch sizes (4, 8, 12)
- R1 regularization tuning
- Path length regularization
- Discriminator/generator learning rate ratios

### Why it failed:
StyleGAN2 requires tens of thousands of images with consistent structure. Our dataset of ~7,800 heterogeneous photos (documents, faces, objects, landscapes) caused the discriminator to dominate, leading to mode collapse.

**Lesson:** StyleGAN2 is designed for uniform datasets (faces, cars, etc.), not heterogeneous personal archives.

---

## Phase 2: Diffusion Models (EXP-018 to EXP-022)

**Duration:** ~80 hours
**Result:** "Mood generators" - blurry averages without structure

### EXP-018: Pixel Diffusion
- Standard DDPM in pixel space
- Result: MSE loss produces mean images
- Everything converged to brownish blobs

### EXP-019: CLIP-guided Diffusion
- Added CLIP embeddings for guidance
- Result: Violated from-scratch principle
- CLIP's pretrained knowledge dominated

### EXP-020: NadjaEMBED
- Custom embedding network trained on our data
- Result: Still blurry, no object structure

### EXP-021: SlotDiffusion
- Slot attention + diffusion for object discovery
- Result: Designed for synthetic data with clear objects
- Failed on real-world heterogeneous photos

### EXP-022: Edge-First Training
- Train on edge maps first, then color
- Result: Sparse edge data + GAN collapse

**Lesson:** Diffusion models learn pixel distributions, not semantic structure. With heterogeneous data, they average everything into mush.

---

## Phase 3: The Breakthrough - Progressive GAN (EXP-023 to EXP-026)

**Duration:** ~15 hours
**Result:** SUCCESS - coherent forms at 128x128

### EXP-023: Progressive GAN Levels 0-4
First attempt at progressive training:
- Start at 4x4 resolution
- Gradually add layers up to 64x64
- Result: FORMS APPEARED for the first time

### EXP-025: Progressive GAN Level 5 (128x128)
Extended to full resolution:
- Continued training to 128x128
- 100 epochs, ~4.5 hours
- Result: Centered forms, defined edges, figure-ground separation

**Why it worked:**
1. Progressive training learns coarse structure before fine details
2. WGAN-GP loss is more stable than vanilla GAN
3. Curriculum learning matches how visual information is organized

### Architecture that worked:

```
Latent (256D) -> Dense -> 4x4x256
              -> Upsample + Conv -> 8x8x256
              -> Upsample + Conv -> 16x16x128
              -> Upsample + Conv -> 32x32x64
              -> Upsample + Conv -> 64x64x32
              -> Upsample + Conv -> 128x128x3
```

---

## Phase 4: Conditioning Attempts (EXP-027 to EXP-035)

**Duration:** ~35 hours
**Result:** Categories merge together

### What we tried:
- Conditional GAN with class labels
- AC-GAN (auxiliary classifier)
- Binary conditioning (is_sculpture: 0/1)
- 32-feature conditioning vectors

### Why it failed:
With heterogeneous data, the discriminator couldn't learn to distinguish categories reliably. All conditions converged to similar outputs.

**Lesson:** Conditioning requires either (a) many examples per category, or (b) pretrained knowledge. We had neither.

---

## Phase 5: Token-based Methods (EXP-036 to EXP-044)

**Duration:** ~90 hours
**Result:** Various failures (mode collapse, raster patterns)

### EXP-036-038: VQ-VAE
- Vector quantization for image tokenization
- Result: WORKED for encoding/decoding
- Created vocabulary of 1024 visual tokens

### EXP-039-040: Token Transformer
- GPT-style generation of image tokens
- Result: Raster bias - model learned scanning patterns
- Horizontal/vertical stripes instead of objects

### EXP-041-044: MaskGIT
- Masked token prediction (like BERT for images)
- Result: Mode collapse to gray, dots, or patterns
- Unconditional generation failed completely

**Lesson:** Token-based methods require balanced, consistent data. They amplify dataset imbalances rather than averaging them.

---

## Phase 6: Return to Progressive GAN (EXP-045 to EXP-046)

**Duration:** ~5 hours
**Result:** Confirmed as best approach

After exhausting alternatives, we returned to Progressive GAN and confirmed it remains the only architecture that produces coherent forms from our heterogeneous dataset.

**Final model:**
- Progressive GAN Level 5
- 128x128 resolution
- 256-dimensional latent space
- ~20 hours total training
- Checkpoint: epoch_0100.pt

---

## The Central Insight

> "Architecture does NOT solve data problems."

With 7,000 heterogeneous images + from-scratch training + unconditional generation, the problem is mathematically underdetermined.

**What Progressive GAN does differently:**
- Learns hierarchically (coarse to fine)
- Doesn't try to separate categories
- Embraces the heterogeneity as a feature
- Creates a smooth latent space where ALL visual patterns blend

---

## What "Sculpture" Means

The original goal was to generate "sculptures." Through 400 hours of experimentation, we refined the definition:

> "A sculpture is anything centered in the frame, with defined edges, showing figure-ground separation."

This is a STRUCTURAL definition, not semantic. The model doesn't know what a "sculpture" is - it learned to create centered forms with contrast against backgrounds.

---

## Latent Space Exploration

The final contribution is not just generating images, but EXPLORING the latent space:

### Random Walk
Move in random directions through latent space:
```
z_next = z_current + random_direction * step_size
```

### Gradient Walk
Move in a fixed direction:
```
z_next = z_current + FIXED_direction * step_size
```

### Interpolation
Straight line between two points:
```
z(t) = (1-t) * z_A + t * z_B
```

These tools reveal the topology of what the model learned - regions of organic forms, glass-like structures, document textures, and the smooth transitions between them.

---

## Timeline Summary

| Date | Experiment | Hours | Result |
|------|------------|-------|--------|
| 2025-11 to 2026-02 | StyleGAN2 (17 exps) | 150 | Mode collapse |
| 2026-02-06 to 02-09 | Diffusion (5 exps) | 80 | Blurry averages |
| 2026-02-14 to 02-16 | Progressive GAN | 15 | **SUCCESS** |
| 2026-02-18 to 02-22 | Conditioning (9 exps) | 35 | Categories merge |
| 2026-02-22 to 03-15 | Tokens (9 exps) | 90 | Mode collapse |
| 2026-03-24 to 03-25 | Return to ProgGAN | 5 | Confirmed best |

**Total:** ~400 hours, 46 experiments

---

## Hardware Used

- **GPU:** NVIDIA GTX 1660 Super (6GB VRAM)
- **CPU:** AMD Ryzen (details omitted)
- **RAM:** 32GB
- **Storage:** SSD

**Training constraints:**
- Batch size: max 12 (VRAM limit)
- Save checkpoints every 5 epochs (prevent data loss on crash)
- Monitor GPU temperature (stop if >78C)

---

## Lessons Learned

1. **Start simple:** Progressive GAN worked before complex architectures
2. **Respect data heterogeneity:** Don't force structure that isn't there
3. **From-scratch is hard:** 7,800 images is tiny by modern standards
4. **Exploration > Generation:** The latent space is more interesting than individual outputs
5. **Architecture is not magic:** The same model can fail or succeed depending on training approach

---

## Acknowledgments

400 hours of GPU time, countless freezes, and many experiments that taught more through failure than success.

---

*latentspacewalker - 2026*

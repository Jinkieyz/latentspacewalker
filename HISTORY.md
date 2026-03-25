# The Complete History of min_bild_ai

*From 7128 hours of development to a working latent space explorer*

---

## Overview

This document chronicles the complete journey of building a generative image model from scratch, without pretrained weights, trained entirely on a personal photograph archive of ~7,800 images.

**Total project time:** 7128 hours across 46 experiments (June 2025 - March 2026)

297 days x 24 hours of continuous development, training, monitoring, and iteration.

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

## Phase 0: The Undocumented Era (June 2025 - January 2026)

**Duration:** ~7 months
**Result:** 15+ versions, philosophy shift from pretrained to from-scratch
**Evidence:** Reconstructed from archived files and old documentation

Before the numbered experiments began, there was a long period of undocumented exploration. This phase saw a fundamental philosophical shift: from using pretrained models (CLIP, SAM, Qwen) to the final "from scratch" principle.

---

### Early Architecture (with pretrained models)

**Source:** `KOMPLETT_DOKUMENTATION.txt` (dated 2026-01-17)

The original project used multiple pretrained components:

```
EXTERNAL MODELS USED (later abandoned):
- CLIP (OpenAI)     : Text-to-image embedding
- Qwen2-VL          : Vision-Language model for captioning
- SAM (Meta)        : Segment Anything Model for masks
```

**Original pipeline:**
```
[Images] → SAM masks → Qwen captions → CLIP embeddings → StyleGAN2 → [Generated images]
```

This was eventually rejected because pretrained models carry knowledge from billions of external images, defeating the purpose of training on personal data alone.

---

### Version History (V1-V15)

**Source:** `MIN_BILD_AI_V15_KOMPLETT.md` (dated 2026-01-26)

> "15 versions since June 2025. Thousands of GPU hours on a GTX 1660 Super."

| Version | Period | Focus | Result |
|---------|--------|-------|--------|
| V1-V3 | Jun-Jul 2025 | Basic PyTorch learning | Understanding tensors |
| V4-V6 | Aug 2025 | First GAN attempts | Mode collapse |
| V7-V9 | Sep 2025 | CLIP conditioning | Pretrained bias |
| V10-V12 | Oct-Nov 2025 | SAM segmentation | Complex pipeline |
| V13-V14 | Dec 2025 | Feature engineering | 32 features extracted |
| V15 | Jan 2026 | 8-feature conditioning | Training when documented |

---

### Key Files from This Era

**Source:** File modification dates on disk

| File | Date | Purpose | Location |
|------|------|---------|----------|
| `stylegan2_generator.py` | Dec 22, 2025 | First from-scratch generator | `src/models/` |
| `stylegan2_discriminator.py` | Dec 22, 2025 | Discriminator architecture | `src/models/` |
| `stylegan2_ada.py` | Dec 22, 2025 | Adaptive augmentation | `src/models/` |
| `conditional_stylegan2.py` | Dec 22, 2025 | Added feature conditioning | `src/models/` |
| `evaluate_model.py` | Dec 20, 2025 | FID/IS metrics | `scripts/old_scripts/` |
| `analyze_contour_data.py` | Dec 20, 2025 | Contour analysis | `scripts/old_scripts/` |
| `tag_with_qwen.py` | Jan 2, 2026 | Dataset labeling with Qwen | `scripts/old_scripts/` |
| `tag_fast.py` | Jan 2, 2026 | Fast tagging variant | `scripts/old_scripts/` |
| `tag_turbo.py` | Jan 2, 2026 | Optimized tagging | `scripts/old_scripts/` |
| `tag_batch.py` | Jan 2, 2026 | Batch processing | `scripts/old_scripts/` |

---

### Abandoned Experiments

**Source:** `/experiments/` directory

| Experiment | Date | Approach | Why Abandoned |
|------------|------|----------|---------------|
| `genesis/` | Feb 11 | Early latent learning | Predates numbering system |
| `genesis_full/` | Feb 11 | Extended genesis | 516 MB latents.pt preserved |
| `simple_ae/` | Feb 11 | Basic autoencoder | Too simple for the task |
| `simple_gan/` | Feb 12 | Minimal GAN | Mode collapse |
| `simple_gan_v2/` | Feb 12 | Improved GAN | Still collapsed |

---

### The Philosophy Shift

**Source:** Comparison of `KOMPLETT_DOKUMENTATION.txt` vs final `CLAUDE.md`

**January 2026 (pretrained approach):**
> "The project uses pretrained models for text understanding"
> "CLIP (OpenAI): Text-to-image embedding"

**February 2026 (from-scratch principle):**
> "NO pretrained bias - everything trained from scratch"
> "NEVER SUGGEST: Fine-tune Stable Diffusion, LoRA/DreamBooth"

This shift was driven by realizing that pretrained models dominated the output with their learned biases, making the "personal archive" aspect meaningless.

---

### Evidence of Extensive Work

**27 Python files in `old_scripts/` alone:**
- 8 different tagging scripts (tag_*.py)
- 5 analysis scripts (analyze_*.py)
- 3 evaluation scripts (evaluate_*.py)
- Dashboard creation
- Visualization tools

**Source code comments (from `vaeana.py`):**
```python
# "Brushes not dipped in others' colors"
```

This metaphor captures why pretrained models were eventually rejected - the latent space should be shaped only by personal data, not by knowledge from billions of external images.

---

### Why Documentation Was Lost

1. **Learning phase:** Focus on understanding, not recording
2. **Rapid iteration:** Many quick experiments discarded
3. **Philosophy change:** Early work invalidated by the shift to from-scratch
4. **System crashes:** `RESCUE_2026-02-07` directory shows recovery from failure

**Lesson:** The documented 46 experiments (EXP-001 to EXP-046) represent only the organized phase after the from-scratch principle was established. The true experiment count including Phase 0 is likely **70-100+**.

---

### Educational Materials Created

**Source:** `vetroal.se/examenbody/projects/minbildai/`

An entire educational website was built to document and teach the concepts:

| Resource | Content |
|----------|---------|
| `index.html` | Project overview and architecture |
| `quiz.html` | 156 multiple-choice questions |
| `flashcards.html` | Concept review cards |
| `code.html` | Annotated source code |

**Quiz categories:**
- VQ-VAE: 40 questions
- Transformer: 35 questions
- Loss Functions: 25 questions
- Training: 30 questions
- Experiments: 26 questions

---

### Loss Functions Developed

**Source:** `src/analysis/losses.py`, `train_vqvae_64_v2.py`

Multiple specialized loss functions were implemented:

```python
# SSIM Loss - Structural Similarity
def ssim_loss(img1, img2):
    # Captures contrast, luminance, structure

# Sobel Edge Loss - Contour awareness
class SobelEdgeDetector:
    # Detects edges using Sobel filters
    # Based on DiffusionEdge (arXiv 2401.02032)

# Contour-Weighted Loss
class ContourAwareLoss:
    # Weights pixel loss by edge magnitude
    # Based on arXiv 2601.14338

# Perceptual Loss
# From arXiv 2401.00110
```

**Why these failed:** MSE-based losses (including SSIM) encourage averaging. Even edge-aware losses couldn't overcome the fundamental problem of learning distributions rather than structure.

---

### Scripts Created During Development

**Source:** `/home/biffy/examen/min_bild_ai/*.py` (100+ files)

| Category | Examples | Purpose |
|----------|----------|---------|
| Training | `train_vqvae_64_v2.py`, `train_sculpturenet.py` | Model training |
| Continuation | `continue_contour_vqvae.py`, `continue_token_gentle.py` | Resume training |
| Dataset | `curate_dataset.py`, `filter_aggressive.py` | Data preparation |
| Analysis | `evaluate_structured_ae.py`, `compare_sculpturenet_versions.py` | Evaluation |
| Generation | `create_interpolated_images.py`, `create_contact_sheet.py` | Image creation |
| Utilities | `disk_guard.py`, `find_trash_images.py` | System tools |

---

### Alternative Architectures Attempted

**Source:** `src/models/` directory

| Architecture | Files | Result |
|--------------|-------|--------|
| SculptureNet | `sculpturenet.py`, `train_sculpturenet.py` | 1% improvement over baseline |
| Structured AE | `structured_autoencoder.py` | Lost spatial coherence |
| Depth Network | `depth_network.py` | Added complexity without benefit |
| Slot Attention | `slot_attention.py`, `slot_diffusion.py` | Designed for synthetic data |
| Edge Diffusion | `edge_diffusion.py`, `edge_diffusion_v2.py` | Sparse data caused collapse |
| NadjaEmbed | `nadja_embed.py` | Custom embedding, still blurry |
| Latent Diffusion | `latent_diffusion.py`, `latent_diffusion_clip.py` | Distribution learning, not structure |

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

The original goal was to generate "sculptures." Through months of experimentation, we refined the definition:

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

**Total:** 7128 hours, 46 experiments

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

7128 hours of development, countless freezes, and many experiments that taught more through failure than success.

---

*latentspacewalker - 2026*

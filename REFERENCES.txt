# References and Attribution

*Sources, papers, and inspiration for this project*

---

## Core Architecture

### Progressive Growing of GANs

The generator and discriminator architecture is based on:

**Paper:** "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
**Authors:** Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (NVIDIA)
**Published:** ICLR 2018
**arXiv:** https://arxiv.org/abs/1710.10196

**Key concepts used:**
- Progressive training from 4x4 to higher resolutions
- Smooth fade-in of new layers (alpha blending)
- Equalized learning rate
- Pixel normalization
- Minibatch standard deviation

**Official implementation:** https://github.com/tkarras/progressive_growing_of_gans

---

### WGAN-GP Loss

Training uses Wasserstein GAN with Gradient Penalty:

**Paper:** "Improved Training of Wasserstein GANs"
**Authors:** Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
**Published:** NeurIPS 2017
**arXiv:** https://arxiv.org/abs/1704.00028

**Key concepts used:**
- Wasserstein distance instead of JS divergence
- Gradient penalty for 1-Lipschitz constraint
- No batch normalization in discriminator
- Critic trained more than generator

---

### Bilinear Upsampling Fix

The "smooth" variant uses bilinear upsampling to avoid checkerboard artifacts:

**Paper:** "Deconvolution and Checkerboard Artifacts"
**Authors:** Augustus Odena, Vincent Dumoulin, Chris Olah
**Published:** Distill 2016
**URL:** https://distill.pub/2016/deconv-checkerboard/

**Key insight:** Nearest-neighbor upsampling followed by convolution creates grid-like artifacts. Bilinear interpolation produces smoother results.

---

## Implementation Notes

### What was written from scratch

All code in this repository was written from scratch based on the papers above. No code was copied from existing implementations.

The implementation process:
1. Read and understood the original papers
2. Implemented each component (equalized conv, pixel norm, etc.)
3. Debugged using the paper's descriptions and figures
4. Iterated based on training results

### Differences from original ProGAN

| Aspect | Original ProGAN | This Implementation |
|--------|-----------------|---------------------|
| Upsampling | Nearest neighbor | Bilinear (smoother) |
| Max resolution | 1024x1024 | 128x128 |
| Training data | CelebA-HQ (30k) | Personal archive (~8k) |
| Hardware | Multiple V100s | Single GTX 1660 Super |
| Training time | Days | ~20 hours |

---

---

## Data Augmentation Techniques (Tested)

### ADA - Adaptive Discriminator Augmentation

**Paper:** "Training Generative Adversarial Networks with Limited Data"
**Authors:** Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila
**Published:** NeurIPS 2020
**arXiv:** https://arxiv.org/abs/2006.06676

Adaptive augmentation that adjusts intensity based on discriminator overfitting. Tested but not used in final model.

### DiffAugment

**Paper:** "Differentiable Augmentation for Data-Efficient GAN Training"
**Authors:** Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, Song Han (MIT Han Lab)
**Published:** NeurIPS 2020
**arXiv:** https://arxiv.org/abs/2006.10738
**Code:** https://github.com/mit-han-lab/data-efficient-gans

Differentiable augmentations (color, translation, cutout) that can be applied during training.

---

## Attention Mechanisms (Tested)

### Self-Attention GAN (SAGAN)

**Paper:** "Self-Attention Generative Adversarial Networks"
**Authors:** Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
**Published:** ICML 2019
**arXiv:** https://arxiv.org/abs/1805.08318

Self-attention layers to capture long-range dependencies in images. Tested but added complexity without improving results for our dataset.

### Spectral Normalization

**Paper:** "Spectral Normalization for Generative Adversarial Networks"
**Authors:** Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
**Published:** ICLR 2018
**arXiv:** https://arxiv.org/abs/1802.05957

Stabilizes GAN training by constraining discriminator Lipschitz constant. Used in some experiments.

---

## Transformer Architecture (Tested)

### Original Transformer

**Paper:** "Attention Is All You Need"
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**Published:** NeurIPS 2017
**arXiv:** https://arxiv.org/abs/1706.03762

Foundation for Token Transformer experiments. Sinusoidal positional encoding used.

---

## Classical Techniques

### Sobel Edge Detection

Classical image processing technique for edge detection using gradient approximation. Used in contour-aware loss experiments.

**Reference:** Sobel, I., Feldman, G. (1968). "A 3x3 Isotropic Gradient Operator for Image Processing"

### Lanczos Resampling

High-quality image resampling using sinc function approximation. Used for dataset preprocessing and upscaling.

**Reference:** Lanczos, C. (1964). "Evaluation of Noisy Data". Journal of SIAM Numerical Analysis.

---

## Failed Approaches (with references)

### StyleGAN2

**Paper:** "Analyzing and Improving the Image Quality of StyleGAN"
**Authors:** Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila
**Published:** CVPR 2020
**arXiv:** https://arxiv.org/abs/1912.04958

**Why it failed for us:** Requires larger, more uniform datasets. Mode collapsed on heterogeneous personal photos.

---

### Diffusion Models (DDPM)

**Paper:** "Denoising Diffusion Probabilistic Models"
**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
**Published:** NeurIPS 2020
**arXiv:** https://arxiv.org/abs/2006.11239

**Why it failed for us:** Learns pixel distributions, not semantic structure. Produced blurry "mood" images averaging the dataset.

---

### MaskGIT

**Paper:** "MaskGIT: Masked Generative Image Transformer"
**Authors:** Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman
**Published:** CVPR 2022
**arXiv:** https://arxiv.org/abs/2202.04200

**Why it failed for us:** Token-based methods amplify dataset imbalances. Mode collapsed to repetitive patterns.

---

### VQ-VAE

**Paper:** "Neural Discrete Representation Learning"
**Authors:** Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
**Published:** NeurIPS 2017
**arXiv:** https://arxiv.org/abs/1711.00937

**Status:** Successfully implemented for tokenization, but downstream generation failed.

---

## Latent Space Theory

### Understanding latent spaces

**Resource:** "Understanding Latent Space in Machine Learning"
**URL:** https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d

### Interpolation methods

**Paper:** "Sampling Generative Networks"
**Authors:** Tom White
**Published:** 2016
**arXiv:** https://arxiv.org/abs/1609.04468

**Concepts used:** Linear interpolation, spherical interpolation (slerp)

---

## Software Dependencies

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework | BSD |
| torchvision | 0.15+ | Image utilities | BSD |
| NumPy | 1.24+ | Numerical operations | BSD |
| Pillow | 9.0+ | Image loading/saving | HPND |

---

## Hardware

All training was performed on:
- **GPU:** NVIDIA GTX 1660 Super (6GB VRAM)
- **Framework:** PyTorch with CUDA 12.x

---

## Acknowledgments

This project would not be possible without:
- The researchers at NVIDIA who developed Progressive GAN
- The open publication of research papers (arXiv)
- The PyTorch team for the deep learning framework
- The Distill publication for clear explanations of neural network concepts

---

## License Note

This implementation is original code inspired by published research. The training data (personal photographs) is not included and cannot be redistributed.

The code is provided for educational purposes to demonstrate:
- How to implement Progressive GAN from papers
- How to train generative models on small datasets
- How to explore latent spaces

---

*latentspacewalker - 2026*

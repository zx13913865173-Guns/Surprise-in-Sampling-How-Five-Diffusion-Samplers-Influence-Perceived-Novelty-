# Surprise-in-Sampling-How-Five-Diffusion-Samplers-Influence-Perceived-Novelty-
Surprise in Sampling: How Five Diffusion Samplers Influence Perceived Novelty, Inspirational Value, Usability, Aesthetic Quality, and Creative Value in AI Art
 Surprise in Sampling

Reproduction code for:

**Surprise in Sampling: How Five Diffusion Samplers Influence Perceived Novelty, Inspirational Value, Usability, Aesthetic Quality, and Creative Value in AI Art**

Authors: Xin Zhang, Dmitry Galkin  
Tomsk State University

## 1. What this repository reproduces

- **Main experiment**: SDXL Base 1.0, 150 abstract prompts, 5 samplers
- **Samplers**: Euler a, DPM++ 2M Karras, LMS, Heun, DDIM
- **Total images**: 150 prompts × 5 samplers × 3 seeds = **2,250 images**
  - Main analysis: seed-42 subset (750 images)
  - Stability analysis: all three seeds (2,250 images)
- **Four-level progressive ablation**: L0–L4
- **Robustness**: SD1.5, SD3.5 with 50 prompts, CFG 3.0/7.5/15.0, steps 20/30/50
- **Objective metrics**: DreamSim diversity, Laplacian Variance, CLIP Score, LPIPS, FID, NIQE, BRISQUE
- **Subjective scales**: Surprise, Inspirational value, Usability, Aesthetic quality, Clarity/artifact
- **Expert panel**: 8 artists/curators, 7-point creative value
- **Statistics**: linear mixed models, Satterthwaite approximation, marginal/conditional R², Bonferroni, Benjamini–Hochberg

## 2. Hardware and software

| Component | Version |
|---|---|
| Main GPU | NVIDIA A100 40 GB |
| Validation GPU | NVIDIA RTX 4070 12 GB |
| CUDA | 12.1 |
| Python | 3.10.11 |
| PyTorch | 2.1.2 |
| Diffusers | 0.25.0 (SDXL, SD1.5); ≥ 0.31.0 (SD3.5) |
| Transformers | 4.36.0 |
| R | lme4, lmerTest, performance, emmeans |

**Note on low-VRAM devices**: For 8 GB GPUs, resolution is reduced to 768×768 and model CPU offload is enabled. This is recorded in the manifest and does not affect sampler ranking.

## 3. Install

### Python

```bash
conda create -n paper2 python=3.10 -y
conda activate paper2
pip install -r requirements.txt

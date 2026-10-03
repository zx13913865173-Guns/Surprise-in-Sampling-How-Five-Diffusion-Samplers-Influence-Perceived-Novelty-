# Surprise-in-Sampling-How-Five-Diffusion-Samplers-Influence-Perceived-Novelty-
Surprise in Sampling: How Five Diffusion Samplers Influence Perceived Novelty, Inspirational Value, Usability, Aesthetic Quality, and Creative Value in AI Art
# Paper2: Surprise in Sampling

Reproduction code for:

**Surprise in Sampling: How Five Diffusion Samplers Influence Perceived Novelty, Inspirational Value, Usability, Aesthetic Quality, and Creative Value in AI Art**

Authors: Xin Zhang, Dmitry Galkin  
Tomsk State University

## 1. What this repository reproduces

- Main experiment: SDXL Base 1.0, 150 abstract prompts, 5 samplers
- Samplers: Euler a, DPM++ 2M Karras, LMS, Heun, DDIM
- Main generation: 150 prompts × 5 samplers = 750 images
- Stability analysis: 3 seeds per condition = 2,250 images
- Four-level progressive ablation: L0–L4
- Robustness: SD1.5, SD3.5 with 50 prompts, CFG 3.0/7.5/15.0, steps 20/30/50
- Objective metrics: DreamSim diversity, Laplacian Variance, CLIP Score, LPIPS, FID, NIQE, BRISQUE
- Subjective scales: Surprise, Inspirational value, Usability, Aesthetic quality, Clarity/artifact
- Expert panel: 8 artists/curators, 7-point creative value
- Statistics: linear mixed models, Satterthwaite approximation, marginal/conditional R², Bonferroni, Benjamini–Hochberg

## 2. Hardware and software

Paper2 environment:

- OS: Windows 11 Pro 23H2
- GPU: NVIDIA A100 40 GB / RTX 3050 6 GB laptop for low-compute validation
- CUDA: 12.1 / 11.8
- Python: 3.10.11 / 3.10.12
- PyTorch: 2.1.2 / 2.0.1+cu118
- Diffusers: 0.25.0 / 0.24.0
- Transformers: 4.36.0
- R: lme4, lmerTest, performance, emmeans

For SD3.5, use `diffusers>=0.31.0`.

## 3. Install

```bash
conda create -n paper2 python=3.10 -y
conda activate paper2
pip install -r requirements.txt

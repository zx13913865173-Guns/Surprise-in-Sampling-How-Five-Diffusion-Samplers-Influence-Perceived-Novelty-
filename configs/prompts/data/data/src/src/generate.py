# -*- coding: utf-8 -*-
"""
Paper2 generation script.
Modes: main, ablation, robustness.

Reproduces:
- 150 prompts × 5 samplers × 3 seeds = 2,250 images
- L0–L4 ablation
- SD3.5 (50 prompts × 5 samplers × 3 seeds), CFG, steps
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
)

from utils import load_config, safe_name, get_vram_gb, resolve_resolution, ensure_dirs


# ------------------------------------------------------------
# Custom scheduler for L1: Euler a with injected noise scaled by 0.5
# ------------------------------------------------------------
class EulerAncestralNoiseScaleScheduler(EulerAncestralDiscreteScheduler):
    def __init__(self, noise_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = float(noise_scale)

    def step(self, model_output, timestep, sample,
             s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
             s_noise=1.0, generator=None, return_dict=True):
        return super().step(
            model_output, timestep, sample,
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax,
            s_noise=self.noise_scale,
            generator=generator, return_dict=return_dict,
        )


def load_prompts(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


# ------------------------------------------------------------
# Pipeline loading with low-VRAM adaptation
# ------------------------------------------------------------
def load_pipe(model_key, config):
    model_cfg = config["models"][model_key]
    model_id = model_cfg["id"]
    resolution = resolve_resolution(config, model_key)

    vram = get_vram_gb()
    use_offload = vram < 12
    use_slicing = vram < 16
    use_tiling = vram < 16
    dtype = torch.float16

    print(f"[load_pipe] model={model_key}, VRAM={vram:.1f} GB, "
          f"resolution={resolution}, offload={use_offload}, "
          f"slicing={use_slicing}, tiling={use_tiling}")

    if model_key == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype,
            variant=model_cfg.get("variant", "fp16"),
            use_safetensors=True,
        )
    elif model_key == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
        )
    elif model_key == "sd35":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    if use_slicing:
        pipe.enable_vae_slicing()
    if use_tiling:
        pipe.enable_vae_tiling()

    if use_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    if vram < 12:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)
    return pipe, resolution


def set_sampler(pipe, sampler_name):
    config = pipe.scheduler.config
    if sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
    elif sampler_name == "Euler a_noise_0.5":
        pipe.scheduler = EulerAncestralNoiseScaleScheduler.from_config(
            config, noise_scale=0.5)
    elif sampler_name == "Euler deterministic":
        pipe.scheduler = EulerDiscreteScheduler.from_config(config)
    elif sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    elif sampler_name == "LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(config)
    elif sampler_name == "Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(config)
    elif sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def generate_one(pipe, prompt, negative_prompt, cfg, steps, seed, resolution):
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=float(cfg),
        num_inference_steps=int(steps),
        width=resolution,
        height=resolution,
        generator=generator,
    )
    return out.images[0]


# ------------------------------------------------------------
# Modes
# ------------------------------------------------------------
def run_main(config):
    main_cfg = config["main"]
    model_key = main_cfg["model"]
    prompts = load_prompts(main_cfg["prompts"])
    output_dir = Path(main_cfg["output"])
    ensure_dirs(output_dir)

    pipe, resolution = load_pipe(model_key, config)
    manifest = []

    for sampler in config["samplers"]:
        set_sampler(pipe, sampler)
        sampler_dir = output_dir / safe_name(sampler)
        ensure_dirs(sampler_dir)

        for p_idx, prompt in enumerate(tqdm(prompts, desc=f"main {sampler}")):
            prompt_id = f"p{p_idx+1:03d}"
            for seed in main_cfg["seeds"]:
                img = generate_one(
                    pipe, prompt, main_cfg["negative_prompt"],
                    main_cfg["cfg"], main_cfg["steps"],
                    seed, resolution,
                )
                fname = f"{prompt_id}_{safe_name(sampler)}_seed{seed}.png"
                fpath = sampler_dir / fname
                img.save(fpath)
                manifest.append({
                    "image_id": fpath.stem,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sampler": sampler,
                    "seed": seed,
                    "is_main": int(seed == main_cfg["main_seed"]),
                    "model": model_key,
                    "cfg": main_cfg["cfg"],
                    "steps": main_cfg["steps"],
                    "resolution": resolution,
                    "path": str(fpath),
                })

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Main generation done. Images: {len(manifest)}")


def run_ablation(config):
    abl_cfg = config["ablation"]
    model_key = abl_cfg["model"]
    prompts = load_prompts(abl_cfg["prompts"])
    output_dir = Path(abl_cfg["output"])
    ensure_dirs(output_dir)

    pipe, resolution = load_pipe(model_key, config)
    manifest = []

    for level, sampler in abl_cfg["levels"].items():
        level_dir = output_dir / level
        ensure_dirs(level_dir)

        if level == "L4":
            for p_idx, prompt in enumerate(tqdm(prompts, desc="ablation L4")):
                prompt_id = f"p{p_idx+1:03d}"
                generator = torch.Generator(device="cpu").manual_seed(
                    int(abl_cfg["seed"]))
                latent = torch.randn((1, 4, 128, 128), generator=generator)
                ch = latent[0, :3].cpu().numpy()
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                rgb = (ch.transpose(1, 2, 0) * 255).astype(np.uint8)
                img = Image.fromarray(rgb).resize(
                    (resolution, resolution), Image.NEAREST)
                fpath = level_dir / f"{prompt_id}_L4_pure_noise.png"
                img.save(fpath)
                manifest.append({
                    "image_id": fpath.stem,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sampler": "L4_pure_noise",
                    "seed": abl_cfg["seed"],
                    "ablation_level": "L4",
                    "model": model_key,
                    "cfg": abl_cfg["cfg"],
                    "steps": abl_cfg["steps"],
                    "resolution": resolution,
                    "path": str(fpath),
                })
            continue

        set_sampler(pipe, sampler)
        for p_idx, prompt in enumerate(tqdm(prompts, desc=f"ablation {level}")):
            prompt_id = f"p{p_idx+1:03d}"
            img = generate_one(
                pipe, prompt, abl_cfg["negative_prompt"],
                abl_cfg["cfg"], abl_cfg["steps"],
                abl_cfg["seed"], resolution,
            )
            fname = f"{prompt_id}_{level}_{safe_name(sampler)}.png"
            fpath = level_dir / fname
            img.save(fpath)
            manifest.append({
                "image_id": fpath.stem,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "sampler": sampler,
                "seed": abl_cfg["seed"],
                "ablation_level": level,
                "model": model_key,
                "cfg": abl_cfg["cfg"],
                "steps": abl_cfg["steps"],
                "resolution": resolution,
                "path": str(fpath),
            })

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Ablation done. Images: {len(manifest)}")


def run_robustness(config):
    rb = config["robustness"]
    prompts = load_prompts(config["main"]["prompts"])
    output_dir = Path(rb["output"])
    ensure_dirs(output_dir)
    manifest = []

    # SD3.5: 50 prompts × 5 samplers × 3 seeds
    sd35_prompts = prompts[: rb["sd35_prompts"]]
    pipe, resolution = load_pipe("sd35", config)
    for sampler in config["samplers"]:
        set_sampler(pipe, sampler)
        sampler_dir = output_dir / "sd35" / safe_name(sampler)
        ensure_dirs(sampler_dir)
        for p_idx, prompt in enumerate(tqdm(sd35_prompts, desc=f"SD3.5 {sampler}")):
            prompt_id = f"p{p_idx+1:03d}"
            for seed in rb["sd35_seeds"]:
                img = generate_one(
                    pipe, prompt, rb["negative_prompt"],
                    rb["sd35_cfg"], rb["sd35_steps"], seed, resolution,
                )
                fpath = sampler_dir / f"{prompt_id}_{safe_name(sampler)}_seed{seed}.png"
                img.save(fpath)
                manifest.append({
                    "image_id": fpath.stem,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sampler": sampler,
                    "seed": seed,
                    "model": "sd35",
                    "cfg": rb["sd35_cfg"],
                    "steps": rb["sd35_steps"],
                    "resolution": resolution,
                    "path": str(fpath),
                })

    # SD1.5
    pipe, resolution = load_pipe("sd15", config)
    for sampler in config["samplers"]:
        set_sampler(pipe, sampler)
        sampler_dir = output_dir / "sd15" / safe_name(sampler)
        ensure_dirs(sampler_dir)
        for p_idx, prompt in enumerate(tqdm(prompts[:50], desc=f"SD1.5 {sampler}")):
            prompt_id = f"p{p_idx+1:03d}"
            img = generate_one(
                pipe, prompt, rb["negative_prompt"],
                7.5, 30, 42, resolution,
            )
            fpath = sampler_dir / f"{prompt_id}_{safe_name(sampler)}.png"
            img.save(fpath)
            manifest.append({
                "image_id": fpath.stem,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "sampler": sampler,
                "seed": 42,
                "model": "sd15",
                "cfg": 7.5,
                "steps": 30,
                "resolution": resolution,
                "path": str(fpath),
            })

    # CFG scale (SDXL)
    pipe, resolution = load_pipe("sdxl", config)
    for sampler in ["Euler a", "DDIM"]:
        set_sampler(pipe, sampler)
        for cfg in rb["cfg_values"]:
            cfg_dir = output_dir / "cfg" / f"cfg{cfg}" / safe_name(sampler)
            ensure_dirs(cfg_dir)
            for p_idx, prompt in enumerate(tqdm(prompts[:20], desc=f"CFG {cfg} {sampler}")):
                prompt_id = f"p{p_idx+1:03d}"
                img = generate_one(
                    pipe, prompt, rb["negative_prompt"],
                    cfg, 30, 42, resolution,
                )
                fpath = cfg_dir / f"{prompt_id}_{safe_name(sampler)}_cfg{cfg}.png"
                img.save(fpath)
                manifest.append({
                    "image_id": fpath.stem,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sampler": sampler,
                    "seed": 42,
                    "model": "sdxl",
                    "cfg": cfg,
                    "steps": 30,
                    "resolution": resolution,
                    "path": str(fpath),
                })

    # Steps (SDXL)
    for sampler in ["Euler a", "DDIM"]:
        set_sampler(pipe, sampler)
        for steps in rb["step_values"]:
            step_dir = output_dir / "steps" / f"steps{steps}" / safe_name(sampler)
            ensure_dirs(step_dir)
            for p_idx, prompt in enumerate(tqdm(prompts[:20], desc=f"steps {steps} {sampler}")):
                prompt_id = f"p{p_idx+1:03d}"
                img = generate_one(
                    pipe, prompt, rb["negative_prompt"],
                    7.5, steps, 42, resolution,
                )
                fpath = step_dir / f"{prompt_id}_{safe_name(sampler)}_steps{steps}.png"
                img.save(fpath)
                manifest.append({
                    "image_id": fpath.stem,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sampler": sampler,
                    "seed": 42,
                    "model": "sdxl",
                    "cfg": 7.5,
                    "steps": steps,
                    "resolution": resolution,
                    "path": str(fpath),
                })

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Robustness done. Images: {len(manifest)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["main", "ablation", "robustness"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "main":
        run_main(config)
    elif args.mode == "ablation":
        run_ablation(config)
    elif args.mode == "robustness":
        run_robustness(config)


if __name__ == "__main__":
    main()

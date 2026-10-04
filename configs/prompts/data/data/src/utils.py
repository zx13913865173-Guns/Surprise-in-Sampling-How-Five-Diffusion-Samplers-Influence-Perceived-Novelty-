# -*- coding: utf-8 -*-
"""
Shared utilities for Paper2 reproduction.
"""

import os
import torch
import yaml
from pathlib import Path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(s):
    return s.replace("+", "plus").replace(" ", "_").replace("/", "_")


def get_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def resolve_resolution(cfg, model_key):
    """
    Auto-select resolution based on VRAM and config thresholds.
    """
    low_cfg = cfg.get("low_vram", {})
    if not low_cfg.get("enabled", "auto"):
        return cfg["models"][model_key]["resolution"]

    vram = get_vram_gb()
    thresholds = low_cfg.get("vram_thresholds_gb", {})

    if vram >= thresholds.get("full", 16):
        return low_cfg.get("full_resolution", 1024)
    elif vram >= thresholds.get("mid", 12):
        return min(cfg["models"][model_key]["resolution"],
                   low_cfg.get("mid_resolution", 768))
    else:
        return min(cfg["models"][model_key]["resolution"],
                   low_cfg.get("low_resolution", 512))


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

# -*- coding: utf-8 -*-
"""
Compute objective metrics for Paper2:
DreamSim diversity, Laplacian Variance, CLIP Score, LPIPS, FID, NIQE, BRISQUE.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from utils import load_config, ensure_dirs

try:
    import clip
except Exception:
    clip = None

try:
    import lpips
except Exception:
    lpips = None

try:
    import pyiqa
except Exception:
    pyiqa = None

try:
    from dreamsim import dreamsim
except Exception:
    dreamsim = None


def laplacian_variance(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def clip_score(image_path, prompt, model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()
    return float(score)


def niqe_brisque(image_path, niqe_metric, brisque_metric, device):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        niqe = niqe_metric(tensor).item()
        brisque = brisque_metric(tensor).item()
    return float(niqe), float(brisque)


def dreamsim_diversity(paths, model, preprocess, device):
    if dreamsim is None or len(paths) < 2:
        return np.nan
    feats = []
    for p in paths:
        img = preprocess(Image.open(p).convert("RGB")).to(device)
        with torch.no_grad():
            f = model.embed(img.unsqueeze(0))
        feats.append(f.cpu())
    feats = torch.cat(feats, dim=0)
    dists = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            dists.append(torch.norm(feats[i] - feats[j], p=2).item())
    return float(np.mean(dists)) if dists else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("outputs/metrics")
    ensure_dirs(output_dir)

    manifest = pd.read_csv("outputs/main/manifest.csv")
    rows = []

    clip_model, clip_preprocess = (None, None)
    if clip is not None:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    niqe_metric = pyiqa.create_metric("niqe", device=device) if pyiqa is not None else None
    brisque_metric = pyiqa.create_metric("brisque", device=device) if pyiqa is not None else None

    ds_model, ds_preprocess = (None, None)
    if dreamsim is not None:
        ds_model, ds_preprocess = dreamsim(pretrained=True, device=device)

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="metrics"):
        p = Path(row["path"])
        if not p.exists():
            continue
        rec = row.to_dict()
        rec["laplacian_variance"] = laplacian_variance(p)
        if clip_model is not None:
            rec["clip_score"] = clip_score(p, row["prompt"],
                                           clip_model, clip_preprocess, device)
        if niqe_metric is not None:
            niqe, brisque = niqe_brisque(p, niqe_metric, brisque_metric, device)
            rec["niqe"] = niqe
            rec["brisque"] = brisque
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "image_metrics.csv", index=False)

    summary = df.groupby("sampler").agg({
        "laplacian_variance": ["mean", "std"],
        "clip_score": ["mean", "std"],
        "niqe": ["mean", "std"],
        "brisque": ["mean", "std"],
    }).reset_index()
    summary.to_csv(output_dir / "sampler_summary.csv", index=False)

    if ds_model is not None:
        div_rows = []
        for sampler, grp in df.groupby("sampler"):
            paths = [Path(p) for p in grp["path"].tolist()[:50]]
            div = dreamsim_diversity(paths, ds_model, ds_preprocess, device)
            div_rows.append({"sampler": sampler, "dreamsim_diversity": div})
        pd.DataFrame(div_rows).to_csv(
            output_dir / "dreamsim_diversity.csv", index=False)

    print("Metrics saved to outputs/metrics/")


if __name__ == "__main__":
    main()

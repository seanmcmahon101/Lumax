"""Stage 1c: Indirect GI dataset generation and serialization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from lumax.config import resolve_device, seed_everything
from lumax.pathtracer import trace_indirect_gi
from lumax.scene import build_cornell_scene


def generate_dataset(cfg: dict[str, Any]) -> Path:
    """Generate ground-truth (position, normal, light_pos) -> linear RGB indirect GI dataset."""
    seed = int(cfg["runtime"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    scene = build_cornell_scene(cfg["scene"], device=device)
    scfg = cfg["sampler"]
    num_samples = int(scfg["num_samples"])
    val_fraction = float(scfg.get("val_fraction", 0.1))
    spp = int(scfg.get("spp", 64))
    max_bounces = int(scfg.get("max_bounces", 3))
    ray_batch_size = int(scfg.get("ray_batch_size", 32768))
    rr_bounce = int(scfg.get("russian_roulette_bounce", 2))
    out_path = Path(scfg["dataset_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[stage 1: generate] device={device} | triangles={scene.num_triangles} | "
        f"samples={num_samples} | spp={spp} | bounces={max_bounces}"
    )
    t0 = time.perf_counter()

    gen = torch.Generator(device=device).manual_seed(seed)
    positions, normals, _albedos = scene.sample_surface_points(num_samples, generator=gen)
    light_pos = scene.sample_light_positions(num_samples, generator=gen)

    rgb_indirect = trace_indirect_gi(
        scene=scene,
        positions=positions,
        normals=normals,
        light_pos=light_pos,
        spp=spp,
        max_bounces=max_bounces,
        ray_batch_size=ray_batch_size,
        russian_roulette_bounce=rr_bounce,
        generator=gen,
    )

    # Split into train / val with deterministic permutation
    perm = torch.randperm(num_samples, device=device, generator=gen)
    n_val = max(1, int(round(num_samples * val_fraction)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    payload = {
        "train": {
            "positions": positions[train_idx].cpu(),
            "normals": normals[train_idx].cpu(),
            "light_pos": light_pos[train_idx].cpu(),
            "rgb": rgb_indirect[train_idx].cpu(),
        },
        "val": {
            "positions": positions[val_idx].cpu(),
            "normals": normals[val_idx].cpu(),
            "light_pos": light_pos[val_idx].cpu(),
            "rgb": rgb_indirect[val_idx].cpu(),
        },
        "meta": {
            "num_samples": num_samples,
            "num_train": int(train_idx.numel()),
            "num_val": int(val_idx.numel()),
            "spp": spp,
            "max_bounces": max_bounces,
            "seed": seed,
            "scene_bounds": scene.scene_bounds_dict(),
            "rgb_mean": rgb_indirect.mean(dim=0).cpu().tolist(),
            "rgb_max": rgb_indirect.amax(dim=0).cpu().tolist(),
        },
    }
    torch.save(payload, out_path)
    dt = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"[stage 1: generate] saved {out_path} ({size_mb:.2f} MB) in {dt:.2f}s | "
        f"train={payload['meta']['num_train']} val={payload['meta']['num_val']} | "
        f"mean_rgb={[round(x, 4) for x in payload['meta']['rgb_mean']]} "
        f"max_rgb={[round(x, 3) for x in payload['meta']['rgb_max']]}"
    )
    return out_path


def load_dataset(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load generated dataset tensors directly onto target device (tiny VRAM footprint)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    for split in ("train", "val"):
        for k in ("positions", "normals", "light_pos", "rgb"):
            data[split][k] = data[split][k].to(device=device, dtype=torch.float32)
    return data

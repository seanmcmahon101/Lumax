"""Configuration loading, deep merging, CLI overrides, and device/seed helpers."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply optional dot-notation key=value CLI overrides."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be in key=value format, got: {item!r}")
            key_path, raw_val = item.split("=", 1)
            val = yaml.safe_load(raw_val)
            set_nested(cfg, key_path.strip(), val)
    return cfg


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict merging patch into base recursively."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_nested(cfg: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested key using dot notation, e.g. 'train.epochs'."""
    parts = key_path.split(".")
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Resolve runtime device cleanly across CUDA GPUs, Apple MPS, and CPU."""
    spec = (device_spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        print("[lumax] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(spec)


def seed_everything(seed: int) -> None:
    """Set deterministic RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_smoke_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply a fast CPU/laptop-friendly preset while exercising every code path."""
    patch = {
        "sampler": {
            "num_samples": 4096,
            "val_fraction": 0.15,
            "spp": 16,
            "max_bounces": 3,
            "ray_batch_size": 8192,
        },
        "train": {
            "epochs": 25,
            "batch_size": 512,
            "log_every": 5,
        },
        "eval": {
            "width": 64,
            "height": 64,
            "gt_spp": 16,
            "ray_batch_size": 8192,
        },
        "sweep": {
            "epochs": 12,
            "eval_width": 48,
            "eval_height": 48,
            "gt_spp": 12,
            "variants": [
                {
                    "name": "pos3_2x16",
                    "model": {
                        "encoder": {"type": "positional", "num_bands_pos": 3, "num_bands_light": 2},
                        "mlp": {"hidden_layers": 2, "hidden_dim": 16, "activation": "relu"},
                    },
                },
                {
                    "name": "fourier16_2x24",
                    "model": {
                        "encoder": {"type": "fourier", "fourier_features": 16, "fourier_sigma_pos": 2.0},
                        "mlp": {"hidden_layers": 2, "hidden_dim": 24, "activation": "relu"},
                    },
                },
            ],
        },
    }
    return deep_merge(cfg, patch)

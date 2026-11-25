"""Stage 2b: Kilobyte-scale micro-MLP + exact parameter, fp16 KB, and per-pixel MAC accounting."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumax.encoders import SpatialEncoder, build_encoder

LOG1P_MAX_RAW = 8.0


def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "leaky_relu":
        return F.leaky_relu(x, negative_slope=0.05)
    if name == "silu":
        return F.silu(x)
    if name == "sin":
        return torch.sin(x)
    raise ValueError(f"Unsupported activation: {name!r}")


class NeuralGIModel(nn.Module):
    """Swappable SpatialEncoder + tiny intentionally-overfitted micro-MLP decoder."""

    def __init__(
        self,
        encoder: SpatialEncoder,
        hidden_layers: int = 3,
        hidden_dim: int = 32,
        activation: str = "relu",
        target_transform: str = "log1p",
        aabb_min: list[float] | tuple[float, float, float] = (-1.0, -1.0, -1.0),
        aabb_max: list[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if activation not in ("relu", "leaky_relu", "silu", "sin"):
            raise ValueError(f"Unsupported activation: {activation!r}")
        if target_transform not in ("log1p", "linear"):
            raise ValueError(f"Unsupported target_transform: {target_transform!r}")

        self.encoder = encoder
        self.hidden_layers = int(hidden_layers)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.target_transform = target_transform

        self.register_buffer("aabb_min", torch.tensor(aabb_min, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(aabb_max, dtype=torch.float32))

        dims = [encoder.output_dim] + [self.hidden_dim] * self.hidden_layers + [3]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if self.activation == "sin" and not is_last:
                bound = (1.0 / layer.in_features) if i == 0 else (6.0 / layer.in_features) ** 0.5
                nn.init.uniform_(layer.weight, -bound, bound)
            else:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        span = (self.aabb_max - self.aabb_min).clamp_min(1e-6)
        return 2.0 * (x - self.aabb_min) / span - 1.0

    def forward_raw(
        self, pos: torch.Tensor, normal: torch.Tensor, light_pos: torch.Tensor
    ) -> torch.Tensor:
        """Predict raw decoder output (in training target space before clamp/expm1)."""
        pos_n = self.normalize_coords(pos)
        lgt_n = self.normalize_coords(light_pos)
        h = self.encoder(pos_n, normal, lgt_n)
        for layer in self.layers[:-1]:
            h = _apply_activation(layer(h), self.activation)
        return self.layers[-1](h)

    def raw_to_linear_rgb(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw network output into non-negative linear RGB irradiance / pi."""
        pos_raw = F.relu(raw)
        if self.target_transform == "log1p":
            return torch.expm1(pos_raw.clamp_max(LOG1P_MAX_RAW))
        return pos_raw

    def linear_rgb_to_target(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear RGB ground truth into training target space."""
        clamped = rgb.clamp_min(0.0)
        if self.target_transform == "log1p":
            return torch.log1p(clamped)
        return clamped

    def forward(
        self, pos: torch.Tensor, normal: torch.Tensor, light_pos: torch.Tensor
    ) -> torch.Tensor:
        """Inference forward pass returning non-negative linear RGB indirect irradiance / pi."""
        return self.raw_to_linear_rgb(self.forward_raw(pos, normal, light_pos))

    def cost_report(self) -> dict[str, Any]:
        """Exact parameter count, fp16 kilobyte footprint, and per-pixel MACs."""
        enc_trainable = self.encoder.trainable_param_count()
        enc_buffers = self.encoder.runtime_buffer_count()
        mlp_params = sum(p.numel() for p in self.layers.parameters())

        trainable_total = enc_trainable + mlp_params
        runtime_total = trainable_total + enc_buffers
        fp16_bytes = runtime_total * 2
        fp16_kb = fp16_bytes / 1024.0

        enc_macs = self.encoder.macs_per_eval()
        layer_shapes: list[list[int]] = []
        mlp_macs = 0
        for layer in self.layers:
            layer_shapes.append([int(layer.in_features), int(layer.out_features)])
            mlp_macs += int(layer.in_features) * int(layer.out_features)

        total_macs = enc_macs + mlp_macs
        return {
            "encoder_type": self.encoder.encoder_type,
            "encoder_output_dim": self.encoder.output_dim,
            "hidden_layers": self.hidden_layers,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "target_transform": self.target_transform,
            "layer_shapes": layer_shapes,
            "encoder_trainable_params": enc_trainable,
            "encoder_buffer_params": enc_buffers,
            "mlp_params": mlp_params,
            "trainable_params": trainable_total,
            "runtime_params": runtime_total,
            "fp16_bytes": fp16_bytes,
            "fp16_kb": round(fp16_kb, 3),
            "encoder_macs": enc_macs,
            "mlp_macs": mlp_macs,
            "total_macs_per_pixel": total_macs,
        }

    def format_cost_line(self) -> str:
        c = self.cost_report()
        return (
            f"encoder={c['encoder_type']}(d={c['encoder_output_dim']}) | "
            f"mlp={c['hidden_layers']}x{c['hidden_dim']}({c['activation']}) | "
            f"params={c['runtime_params']:,} (trainable={c['trainable_params']:,}, "
            f"buf={c['encoder_buffer_params']:,}) | "
            f"fp16={c['fp16_kb']:.2f} KB ({c['fp16_bytes']} B) | "
            f"MACs/px={c['total_macs_per_pixel']:,} (enc={c['encoder_macs']:,}, mlp={c['mlp_macs']:,})"
        )


def build_model(
    model_cfg: dict[str, Any],
    scene_bounds: dict[str, list[float]] | None = None,
    seed: int = 42,
) -> NeuralGIModel:
    """Instantiate encoder + micro-MLP from config dictionary."""
    encoder = build_encoder(model_cfg.get("encoder", {}), seed=seed)
    mlp_cfg = model_cfg.get("mlp", {})
    bounds = scene_bounds or {}
    return NeuralGIModel(
        encoder=encoder,
        hidden_layers=int(mlp_cfg.get("hidden_layers", 3)),
        hidden_dim=int(mlp_cfg.get("hidden_dim", 32)),
        activation=str(mlp_cfg.get("activation", "relu")).lower(),
        target_transform=str(mlp_cfg.get("target_transform", "log1p")).lower(),
        aabb_min=bounds.get("aabb_min", [-1.0, -1.0, -1.0]),
        aabb_max=bounds.get("aabb_max", [1.0, 1.0, 1.0]),
    )


def save_checkpoint(
    path: str | Any,
    model: NeuralGIModel,
    model_cfg: dict[str, Any],
    scene_bounds: dict[str, list[float]],
    metrics: dict[str, Any],
    seed: int = 42,
) -> None:
    """Save self-contained checkpoint with architecture spec, bounds, cost report, and weights."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "scene_bounds": scene_bounds,
            "cost": model.cost_report(),
            "metrics": metrics,
            "seed": seed,
        },
        path,
    )


def load_checkpoint(path: str | Any, device: torch.device) -> tuple[NeuralGIModel, dict[str, Any]]:
    """Reconstruct NeuralGIModel + metadata from a saved checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        model_cfg=ckpt["model_cfg"],
        scene_bounds=ckpt["scene_bounds"],
        seed=int(ckpt.get("seed", 42)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt

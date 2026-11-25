"""Stage 2a: Swappable spatial encoders architected for micro-MLPs and future chunked feature grids."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module, ABC):
    """Base contract for swappable spatial encoders feeding a shared micro-MLP decoder.

    Inputs are always normalized world-space tensors:
        pos_norm:   (N, 3) in [-1, 1]
        normal:     (N, 3) unit vector in [-1, 1]
        light_norm: (N, 3) in [-1, 1]
    """

    encoder_type: str = "base"

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimensionality fed into the first MLP layer."""

    @abstractmethod
    def forward(
        self, pos_norm: torch.Tensor, normal: torch.Tensor, light_norm: torch.Tensor
    ) -> torch.Tensor:
        """Encode (pos_norm, normal, light_norm) -> (N, output_dim)."""

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def runtime_buffer_count(self) -> int:
        """Non-trainable static float buffers required at runtime (e.g. Fourier basis B)."""
        return 0

    def macs_per_eval(self) -> int:
        """Per-pixel multiply-add operations performed by this encoder."""
        return 0

    @abstractmethod
    def export_spec(self) -> dict[str, Any]:
        """Serializable metadata needed by the Stage 4 binary & GLSL exporter."""


class IdentityEncoder(SpatialEncoder):
    """Raw 9D input [pos_norm, normal, light_norm] baseline."""

    encoder_type = "identity"

    def __init__(self) -> None:
        super().__init__()
        self._out_dim = 9

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self, pos_norm: torch.Tensor, normal: torch.Tensor, light_norm: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([pos_norm, normal, light_norm], dim=-1)

    def export_spec(self) -> dict[str, Any]:
        return {"type": "identity", "output_dim": self._out_dim}


class PositionalEncoder(SpatialEncoder):
    """NeRF-style dyadic sin/cos multi-band encoder with per-group frequency budgets.

    Zero trainable parameters and zero static buffers -> ultra-compact VRAM & pure ALU GLSL.
    """

    encoder_type = "positional"

    def __init__(
        self,
        num_bands_pos: int = 4,
        num_bands_normal: int = 0,
        num_bands_light: int = 2,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.num_bands_pos = int(num_bands_pos)
        self.num_bands_normal = int(num_bands_normal)
        self.num_bands_light = int(num_bands_light)
        self.include_input = bool(include_input)

        # Register frequency multipliers [pi * 2^0, pi * 2^1, ...]
        self.register_buffer(
            "freqs_pos",
            math.pi * (2.0 ** torch.arange(self.num_bands_pos, dtype=torch.float32)),
            persistent=False,
        )
        self.register_buffer(
            "freqs_normal",
            math.pi * (2.0 ** torch.arange(self.num_bands_normal, dtype=torch.float32)),
            persistent=False,
        )
        self.register_buffer(
            "freqs_light",
            math.pi * (2.0 ** torch.arange(self.num_bands_light, dtype=torch.float32)),
            persistent=False,
        )

        dim = 9 if self.include_input else 0
        dim += 3 * 2 * (self.num_bands_pos + self.num_bands_normal + self.num_bands_light)
        self._out_dim = dim

    @property
    def output_dim(self) -> int:
        return self._out_dim

    @staticmethod
    def _encode_group(x: torch.Tensor, freqs: torch.Tensor) -> list[torch.Tensor]:
        if freqs.numel() == 0:
            return []
        # (N, 3, B) -> interleave sin/cos per band cleanly for simple GLSL matching
        feats: list[torch.Tensor] = []
        for f in freqs:
            arg = x * f
            feats.append(torch.sin(arg))
            feats.append(torch.cos(arg))
        return feats

    def forward(
        self, pos_norm: torch.Tensor, normal: torch.Tensor, light_norm: torch.Tensor
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.include_input:
            parts.extend([pos_norm, normal, light_norm])
        parts.extend(self._encode_group(pos_norm, self.freqs_pos))
        parts.extend(self._encode_group(normal, self.freqs_normal))
        parts.extend(self._encode_group(light_norm, self.freqs_light))
        return torch.cat(parts, dim=-1)

    def macs_per_eval(self) -> int:
        # 1 multiply per coordinate per frequency band (trig SFU ops counted separately in shader)
        return 3 * (self.num_bands_pos + self.num_bands_normal + self.num_bands_light)

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "positional",
            "include_input": self.include_input,
            "num_bands_pos": self.num_bands_pos,
            "num_bands_normal": self.num_bands_normal,
            "num_bands_light": self.num_bands_light,
            "output_dim": self._out_dim,
        }


class FourierEncoder(SpatialEncoder):
    """Random Gaussian Fourier feature mapping [sin(2*pi*x*B^T), cos(2*pi*x*B^T)].

    Uses separate frequency scales for position, normal, and light_pos.
    The projection matrix B is exported in fp16 and included in exact KB accounting.
    """

    encoder_type = "fourier"

    def __init__(
        self,
        fourier_features: int = 24,
        sigma_pos: float = 2.2,
        sigma_normal: float = 0.5,
        sigma_light: float = 1.0,
        include_input: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.fourier_features = int(fourier_features)
        self.include_input = bool(include_input)

        gen = torch.Generator().manual_seed(seed + 1337)
        b_pos = torch.randn((self.fourier_features, 3), generator=gen) * float(sigma_pos)
        b_nrm = torch.randn((self.fourier_features, 3), generator=gen) * float(sigma_normal)
        b_lgt = torch.randn((self.fourier_features, 3), generator=gen) * float(sigma_light)
        # Pre-multiply by 2*pi and snap to exact fp16 values so PyTorch training and
        # fp16 binary/GLSL export share identical Fourier phases with zero drift.
        basis = torch.cat([b_pos, b_nrm, b_lgt], dim=-1) * (2.0 * math.pi)
        basis = basis.to(torch.float16).to(torch.float32)
        self.register_buffer("basis", basis)  # (F, 9)

        self._out_dim = (9 if self.include_input else 0) + 2 * self.fourier_features

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self, pos_norm: torch.Tensor, normal: torch.Tensor, light_norm: torch.Tensor
    ) -> torch.Tensor:
        raw = torch.cat([pos_norm, normal, light_norm], dim=-1)  # (N, 9)
        proj = F.linear(raw, self.basis)  # (N, F)
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_input:
            return torch.cat([raw, enc], dim=-1)
        return enc

    def runtime_buffer_count(self) -> int:
        return int(self.basis.numel())

    def macs_per_eval(self) -> int:
        return 9 * self.fourier_features

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "fourier",
            "include_input": self.include_input,
            "fourier_features": self.fourier_features,
            "output_dim": self._out_dim,
        }


class ChunkedFeatureGridEncoder(SpatialEncoder):
    """Jointly trained chunked 3D feature grid encoder feeding one shared micro-MLP decoder.

    World space [-1, 1]^3 is partitioned into (Cx, Cy, Cz) spatial chunks, each holding a
    dense (Rx, Ry, Rz, F) trilinearly-interpolated feature grid. This enables future
    streaming/chunk-swapping of large scenes while sharing one tiny neural GI decoder.
    """

    encoder_type = "chunked_grid"

    def __init__(
        self,
        chunk_count: tuple[int, int, int] = (2, 2, 2),
        chunk_resolution: tuple[int, int, int] = (4, 4, 4),
        feature_dim: int = 8,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_count = tuple(int(c) for c in chunk_count)
        self.chunk_resolution = tuple(int(r) for r in chunk_resolution)
        self.feature_dim = int(feature_dim)
        self.include_input = bool(include_input)

        total_res = tuple(c * r for c, r in zip(self.chunk_count, self.chunk_resolution))
        self.total_resolution = total_res
        # Stored as a unified (1, F, D, H, W) tensor during training, cleanly sliceable per chunk
        grid = torch.empty((1, self.feature_dim, total_res[2], total_res[1], total_res[0]))
        nn.init.uniform_(grid, -1e-2, 1e-2)
        self.feature_grid = nn.Parameter(grid)

        self._out_dim = self.feature_dim + (9 if self.include_input else 6)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self, pos_norm: torch.Tensor, normal: torch.Tensor, light_norm: torch.Tensor
    ) -> torch.Tensor:
        # F.grid_sample expects (N, D_out, H_out, W_out, 3) in (x, y, z) order in [-1, 1]
        coords = pos_norm.view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(
            self.feature_grid,
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        feats = sampled.view(self.feature_dim, -1).transpose(0, 1)  # (N, F)
        if self.include_input:
            return torch.cat([pos_norm, normal, light_norm, feats], dim=-1)
        return torch.cat([normal, light_norm, feats], dim=-1)

    def macs_per_eval(self) -> int:
        # Trilinear interpolation across 8 corners per feature channel
        return 8 * self.feature_dim

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "chunked_grid",
            "chunk_count": list(self.chunk_count),
            "chunk_resolution": list(self.chunk_resolution),
            "feature_dim": self.feature_dim,
            "include_input": self.include_input,
            "output_dim": self._out_dim,
        }


def build_encoder(enc_cfg: dict[str, Any], seed: int = 42) -> SpatialEncoder:
    """Factory for swappable spatial encoders from config.yaml."""
    enc_type = str(enc_cfg.get("type", "positional")).lower()
    include_input = bool(enc_cfg.get("include_input", True))

    if enc_type == "identity":
        return IdentityEncoder()
    if enc_type == "positional":
        return PositionalEncoder(
            num_bands_pos=int(enc_cfg.get("num_bands_pos", 4)),
            num_bands_normal=int(enc_cfg.get("num_bands_normal", 0)),
            num_bands_light=int(enc_cfg.get("num_bands_light", 2)),
            include_input=include_input,
        )
    if enc_type == "fourier":
        return FourierEncoder(
            fourier_features=int(enc_cfg.get("fourier_features", 24)),
            sigma_pos=float(enc_cfg.get("fourier_sigma_pos", 2.2)),
            sigma_normal=float(enc_cfg.get("fourier_sigma_normal", 0.5)),
            sigma_light=float(enc_cfg.get("fourier_sigma_light", 1.0)),
            include_input=include_input,
            seed=seed,
        )
    if enc_type == "chunked_grid":
        gcfg = enc_cfg.get("chunked_grid", {})
        return ChunkedFeatureGridEncoder(
            chunk_count=tuple(gcfg.get("chunk_count", [2, 2, 2])),
            chunk_resolution=tuple(gcfg.get("chunk_resolution", [4, 4, 4])),
            feature_dim=int(gcfg.get("feature_dim", 8)),
            include_input=include_input,
        )
    raise ValueError(f"Unknown encoder type: {enc_type!r}")

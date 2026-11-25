"""Stage 1a: Procedural 3D Cornell Box with interior occluder wall and movable light."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SceneMesh:
    """GPU/CPU tensor representation of a procedural triangle mesh scene."""

    v0: torch.Tensor            # (T, 3) triangle vertex 0
    v1: torch.Tensor            # (T, 3) triangle vertex 1
    v2: torch.Tensor            # (T, 3) triangle vertex 2
    e1: torch.Tensor            # (T, 3) v1 - v0
    e2: torch.Tensor            # (T, 3) v2 - v0
    normals: torch.Tensor       # (T, 3) unit outward geometric normals
    albedos: torch.Tensor       # (T, 3) linear RGB diffuse albedos
    areas: torch.Tensor         # (T,)   triangle surface areas
    area_cdf: torch.Tensor      # (T,)   cumulative area distribution for stratified sampling
    aabb_min: torch.Tensor      # (3,)   world-space minimum bounds
    aabb_max: torch.Tensor      # (3,)   world-space maximum bounds
    light_color: torch.Tensor   # (3,)   linear RGB light tint
    light_intensity: float
    light_soft_radius: float
    light_bounds_min: torch.Tensor  # (3,)
    light_bounds_max: torch.Tensor  # (3,)
    occluder_min: torch.Tensor | None
    occluder_max: torch.Tensor | None
    wall_margin: float

    @property
    def num_triangles(self) -> int:
        return int(self.v0.shape[0])

    @property
    def device(self) -> torch.device:
        return self.v0.device

    def to(self, device: torch.device) -> "SceneMesh":
        return SceneMesh(
            v0=self.v0.to(device),
            v1=self.v1.to(device),
            v2=self.v2.to(device),
            e1=self.e1.to(device),
            e2=self.e2.to(device),
            normals=self.normals.to(device),
            albedos=self.albedos.to(device),
            areas=self.areas.to(device),
            area_cdf=self.area_cdf.to(device),
            aabb_min=self.aabb_min.to(device),
            aabb_max=self.aabb_max.to(device),
            light_color=self.light_color.to(device),
            light_intensity=self.light_intensity,
            light_soft_radius=self.light_soft_radius,
            light_bounds_min=self.light_bounds_min.to(device),
            light_bounds_max=self.light_bounds_max.to(device),
            occluder_min=self.occluder_min.to(device) if self.occluder_min is not None else None,
            occluder_max=self.occluder_max.to(device) if self.occluder_max is not None else None,
            wall_margin=self.wall_margin,
        )

    def sample_surface_points(
        self, num_points: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Area-weighted uniform surface sampling -> (positions, normals, albedos)."""
        u_tri = torch.rand(num_points, device=self.device, generator=generator)
        tri_idx = torch.searchsorted(self.area_cdf, u_tri).clamp(0, self.num_triangles - 1)

        # Uniform barycentric sampling on triangles
        r1 = torch.rand(num_points, 1, device=self.device, generator=generator).sqrt()
        r2 = torch.rand(num_points, 1, device=self.device, generator=generator)
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - r2)
        w2 = r1 * r2

        pos = w0 * self.v0[tri_idx] + w1 * self.v1[tri_idx] + w2 * self.v2[tri_idx]
        nrm = self.normals[tri_idx]
        alb = self.albedos[tri_idx]
        return pos, nrm, alb

    def sample_light_positions(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample valid movable light positions inside the box and outside the occluder wall."""
        span = self.light_bounds_max - self.light_bounds_min
        pts = self.light_bounds_min + torch.rand(
            (num_samples, 3), device=self.device, generator=generator
        ) * span

        if self.occluder_min is not None and self.occluder_max is not None:
            pad_min = self.occluder_min - self.wall_margin
            pad_max = self.occluder_max + self.wall_margin
            for _ in range(16):
                inside = ((pts >= pad_min) & (pts <= pad_max)).all(dim=-1)
                n_bad = int(inside.sum().item())
                if n_bad == 0:
                    break
                pts[inside] = self.light_bounds_min + torch.rand(
                    (n_bad, 3), device=self.device, generator=generator
                ) * span
        return pts

    def scene_bounds_dict(self) -> dict[str, list[float]]:
        return {
            "aabb_min": self.aabb_min.detach().cpu().tolist(),
            "aabb_max": self.aabb_max.detach().cpu().tolist(),
            "light_bounds_min": self.light_bounds_min.detach().cpu().tolist(),
            "light_bounds_max": self.light_bounds_max.detach().cpu().tolist(),
        }


def _add_quad(
    tris_v0: list[list[float]],
    tris_v1: list[list[float]],
    tris_v2: list[list[float]],
    tris_nrm: list[list[float]],
    tris_alb: list[list[float]],
    p0: list[float],
    p1: list[float],
    p2: list[float],
    p3: list[float],
    expected_normal: list[float],
    albedo: list[float],
) -> None:
    """Append two triangles for quad (p0, p1, p2, p3), ensuring winding matches expected_normal."""
    a = torch.tensor(p0, dtype=torch.float32)
    b = torch.tensor(p1, dtype=torch.float32)
    c = torch.tensor(p2, dtype=torch.float32)
    n_calc = torch.linalg.cross(b - a, c - a)
    n_exp = torch.tensor(expected_normal, dtype=torch.float32)
    n_exp = n_exp / n_exp.norm().clamp_min(1e-8)

    if float(torch.dot(n_calc, n_exp).item()) < 0.0:
        p1, p3 = p3, p1

    n_list = n_exp.tolist()
    # Tri 1: p0, p1, p2
    tris_v0.append(p0)
    tris_v1.append(p1)
    tris_v2.append(p2)
    tris_nrm.append(n_list)
    tris_alb.append(albedo)
    # Tri 2: p0, p2, p3
    tris_v0.append(p0)
    tris_v1.append(p2)
    tris_v2.append(p3)
    tris_nrm.append(n_list)
    tris_alb.append(albedo)


def _add_box(
    tris_v0: list[list[float]],
    tris_v1: list[list[float]],
    tris_v2: list[list[float]],
    tris_nrm: list[list[float]],
    tris_alb: list[list[float]],
    center: list[float],
    half_extents: list[float],
    albedo: list[float],
) -> tuple[list[float], list[float]]:
    """Append an outward-facing axis-aligned box (6 quads = 12 triangles)."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz

    # -X face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], albedo)
    # +X face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], albedo)
    # -Y face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], albedo)
    # +Y face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], albedo)
    # -Z face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], albedo)
    # +Z face
    _add_quad(tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
              [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], albedo)

    return [x0, y0, z0], [x1, y1, z1]


def build_cornell_scene(scene_cfg: dict[str, Any], device: torch.device) -> SceneMesh:
    """Build the procedural colored Cornell Box + interior occluder wall."""
    h = float(scene_cfg.get("half_size", 1.0))
    albedos_cfg = scene_cfg.get("albedos", {})
    open_front = bool(scene_cfg.get("open_front", True))

    tris_v0: list[list[float]] = []
    tris_v1: list[list[float]] = []
    tris_v2: list[list[float]] = []
    tris_nrm: list[list[float]] = []
    tris_alb: list[list[float]] = []

    # Floor (y = -h, normal +Y)
    _add_quad(
        tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
        [-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h],
        [0, 1, 0], albedos_cfg.get("floor", [0.74, 0.74, 0.72]),
    )
    # Ceiling (y = +h, normal -Y)
    _add_quad(
        tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
        [-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h],
        [0, -1, 0], albedos_cfg.get("ceiling", [0.76, 0.76, 0.76]),
    )
    # Back wall (z = -h, normal +Z)
    _add_quad(
        tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [0, 0, 1], albedos_cfg.get("back", [0.74, 0.74, 0.74]),
    )
    # Left wall (x = -h, normal +X, red)
    _add_quad(
        tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
        [-h, -h, h], [-h, -h, -h], [-h, h, -h], [-h, h, h],
        [1, 0, 0], albedos_cfg.get("left", [0.80, 0.12, 0.10]),
    )
    # Right wall (x = +h, normal -X, green)
    _add_quad(
        tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
        [h, -h, -h], [h, -h, h], [h, h, h], [h, h, -h],
        [-1, 0, 0], albedos_cfg.get("right", [0.12, 0.70, 0.18]),
    )
    if not open_front:
        _add_quad(
            tris_v0, tris_v1, tris_v2, tris_nrm, tris_alb,
            [h, -h, h], [-h, -h, h], [-h, h, h], [h, h, h],
            [0, 0, -1], albedos_cfg.get("front", [0.74, 0.74, 0.74]),
        )

    occ_min: torch.Tensor | None = None
    occ_max: torch.Tensor | None = None
    occ_cfg = scene_cfg.get("occluder_wall", {})
    if occ_cfg.get("enabled", True):
        omin, omax = _add_box(
            tris_v0,
            tris_v1,
            tris_v2,
            tris_nrm,
            tris_alb,
            center=list(occ_cfg.get("center", [0.0, -0.32, -0.05])),
            half_extents=list(occ_cfg.get("half_extents", [0.06, 0.68, 0.52])),
            albedo=list(occ_cfg.get("albedo", [0.76, 0.75, 0.73])),
        )
        occ_min = torch.tensor(omin, dtype=torch.float32)
        occ_max = torch.tensor(omax, dtype=torch.float32)

    v0 = torch.tensor(tris_v0, dtype=torch.float32)
    v1 = torch.tensor(tris_v1, dtype=torch.float32)
    v2 = torch.tensor(tris_v2, dtype=torch.float32)
    e1 = v1 - v0
    e2 = v2 - v0
    normals = torch.tensor(tris_nrm, dtype=torch.float32)
    albedos = torch.tensor(tris_alb, dtype=torch.float32)
    areas = 0.5 * torch.linalg.cross(e1, e2, dim=-1).norm(dim=-1)
    area_cdf = torch.cumsum(areas, dim=0) / areas.sum().clamp_min(1e-8)

    light_cfg = scene_cfg.get("light", {})
    mesh = SceneMesh(
        v0=v0,
        v1=v1,
        v2=v2,
        e1=e1,
        e2=e2,
        normals=normals,
        albedos=albedos,
        areas=areas,
        area_cdf=area_cdf,
        aabb_min=torch.tensor([-h, -h, -h], dtype=torch.float32),
        aabb_max=torch.tensor([h, h, h], dtype=torch.float32),
        light_color=torch.tensor(light_cfg.get("color", [1.0, 0.96, 0.90]), dtype=torch.float32),
        light_intensity=float(light_cfg.get("intensity", 6.5)),
        light_soft_radius=float(light_cfg.get("soft_radius", 0.16)),
        light_bounds_min=torch.tensor(light_cfg.get("bounds_min", [-0.82, -0.25, -0.82]), dtype=torch.float32),
        light_bounds_max=torch.tensor(light_cfg.get("bounds_max", [0.82, 0.85, 0.75]), dtype=torch.float32),
        occluder_min=occ_min,
        occluder_max=occ_max,
        wall_margin=float(light_cfg.get("wall_margin", 0.12)),
    )
    return mesh.to(device)

"""Stage 1b: Pure-PyTorch vectorized GPU/CPU path tracer for indirect-only GI."""

from __future__ import annotations

import math

import torch

from lumax.scene import SceneMesh

RAY_EPS = 1e-4
HIT_INF = 1e9


def intersect_rays(
    scene: SceneMesh,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    max_dist: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Moller-Trumbore ray-triangle intersection.

    Args:
        origins: (N, 3) ray origins
        dirs:    (N, 3) unit ray directions
        max_dist: optional (N,) maximum valid hit distance

    Returns:
        hit_mask: (N,) bool
        hit_t:    (N,) closest hit distance (HIT_INF if miss)
        hit_tri:  (N,) triangle index (0 if miss)
    """
    # Broadcast: rays (N, 1, 3) vs triangles (1, T, 3)
    ro = origins.unsqueeze(1)
    rd = dirs.unsqueeze(1)
    v0 = scene.v0.unsqueeze(0)
    e1 = scene.e1.unsqueeze(0)
    e2 = scene.e2.unsqueeze(0)

    h = torch.linalg.cross(rd, e2, dim=-1)      # (N, T, 3)
    det = (e1 * h).sum(dim=-1)                  # (N, T)

    # Single-sided/front-face or two-sided: allow both sides so interior shadow rays never leak
    valid = det.abs() > 1e-7
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv_det = 1.0 / safe_det

    s = ro - v0                                 # (N, T, 3)
    u = (s * h).sum(dim=-1) * inv_det           # (N, T)
    valid = valid & (u >= 0.0) & (u <= 1.0)

    q = torch.linalg.cross(s, e1, dim=-1)       # (N, T, 3)
    v = (rd * q).sum(dim=-1) * inv_det          # (N, T)
    valid = valid & (v >= 0.0) & ((u + v) <= 1.0)

    t = (e2 * q).sum(dim=-1) * inv_det          # (N, T)
    valid = valid & (t > RAY_EPS)
    if max_dist is not None:
        valid = valid & (t < max_dist.unsqueeze(1))

    t_masked = torch.where(valid, t, torch.full_like(t, HIT_INF))
    hit_t, hit_tri = t_masked.min(dim=-1)
    hit_mask = hit_t < (HIT_INF * 0.5)
    return hit_mask, hit_t, hit_tri


def is_occluded(
    scene: SceneMesh,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    max_dist: torch.Tensor,
) -> torch.Tensor:
    """Fast shadow-ray occlusion test returning (N,) bool."""
    ro = origins.unsqueeze(1)
    rd = dirs.unsqueeze(1)
    v0 = scene.v0.unsqueeze(0)
    e1 = scene.e1.unsqueeze(0)
    e2 = scene.e2.unsqueeze(0)

    h = torch.linalg.cross(rd, e2, dim=-1)
    det = (e1 * h).sum(dim=-1)
    valid = det.abs() > 1e-7
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv_det = 1.0 / safe_det

    s = ro - v0
    u = (s * h).sum(dim=-1) * inv_det
    valid = valid & (u >= 0.0) & (u <= 1.0)

    q = torch.linalg.cross(s, e1, dim=-1)
    v = (rd * q).sum(dim=-1) * inv_det
    valid = valid & (v >= 0.0) & ((u + v) <= 1.0)

    t = (e2 * q).sum(dim=-1) * inv_det
    valid = valid & (t > RAY_EPS) & (t < (max_dist.unsqueeze(1) - RAY_EPS))
    return valid.any(dim=-1)


def _build_onb(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Duff et al. (2017) branchless orthonormal basis from unit normals (N, 3)."""
    sign = torch.where(normals[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + normals[:, 2])
    b = normals[:, 0] * normals[:, 1] * a
    t1 = torch.stack(
        [1.0 + sign * normals[:, 0] * normals[:, 0] * a, sign * b, -sign * normals[:, 0]],
        dim=-1,
    )
    t2 = torch.stack(
        [b, sign + normals[:, 1] * normals[:, 1] * a, -normals[:, 1]],
        dim=-1,
    )
    return t1, t2


def sample_cosine_hemisphere(
    normals: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample cosine-weighted hemisphere directions around unit normals (N, 3)."""
    n = normals.shape[0]
    u1 = torch.rand(n, device=normals.device, generator=generator)
    u2 = torch.rand(n, device=normals.device, generator=generator)

    r = u1.sqrt()
    phi = (2.0 * math.pi) * u2
    x = r * phi.cos()
    y = r * phi.sin()
    z = (1.0 - u1).clamp_min(0.0).sqrt()

    t1, t2 = _build_onb(normals)
    dirs = t1 * x.unsqueeze(-1) + t2 * y.unsqueeze(-1) + normals * z.unsqueeze(-1)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _direct_irradiance_over_pi(
    scene: SceneMesh,
    points: torch.Tensor,
    normals: torch.Tensor,
    light_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute direct incident diffuse irradiance / pi at points from point light_pos."""
    to_light = light_pos - points
    dist_sq = (to_light * to_light).sum(dim=-1)
    dist = dist_sq.clamp_min(1e-8).sqrt()
    l_dir = to_light / dist.unsqueeze(-1)

    ndotl = (normals * l_dir).sum(dim=-1).clamp_min(0.0)
    lit_candidates = ndotl > 1e-6

    vis = torch.zeros_like(ndotl)
    if lit_candidates.any():
        idx = lit_candidates.nonzero(as_tuple=False).squeeze(-1)
        shadow_origins = points[idx] + normals[idx] * (RAY_EPS * 4.0)
        occluded = is_occluded(scene, shadow_origins, l_dir[idx], dist[idx])
        vis[idx] = (~occluded).to(points.dtype)

    attenuation = scene.light_intensity / (dist_sq + scene.light_soft_radius ** 2)
    scale = (vis * ndotl * attenuation / math.pi).unsqueeze(-1)
    return scale * scene.light_color.unsqueeze(0)


@torch.no_grad()
def trace_indirect_gi(
    scene: SceneMesh,
    positions: torch.Tensor,
    normals: torch.Tensor,
    light_pos: torch.Tensor,
    spp: int = 64,
    max_bounces: int = 3,
    ray_batch_size: int = 32768,
    russian_roulette_bounce: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute unbiased multi-bounce indirect-only diffuse irradiance / pi in linear RGB.

    Returns:
        (N, 3) linear RGB indirect lighting factor I_indirect such that final
        indirect surface radiance is `albedo(p) * I_indirect(p, n, light_pos)`.
        Direct (0-bounce) lighting is intentionally excluded.
    """
    n_points = positions.shape[0]
    device = scene.device
    positions = positions.to(device)
    normals = normals.to(device)
    light_pos = light_pos.to(device)

    out = torch.zeros((n_points, 3), dtype=torch.float32, device=device)
    pts_per_chunk = max(1, ray_batch_size // max(1, spp))

    for start in range(0, n_points, pts_per_chunk):
        end = min(n_points, start + pts_per_chunk)
        m = end - start

        # Expand each query point by spp rays -> (M * spp, 3)
        cur_p = positions[start:end].unsqueeze(1).expand(m, spp, 3).reshape(m * spp, 3)
        cur_n = normals[start:end].unsqueeze(1).expand(m, spp, 3).reshape(m * spp, 3)
        cur_l = light_pos[start:end].unsqueeze(1).expand(m, spp, 3).reshape(m * spp, 3)

        # First indirect bounce starts from shading point p along cosine-weighted hemisphere
        ray_o = cur_p + cur_n * (RAY_EPS * 4.0)
        ray_d = sample_cosine_hemisphere(cur_n, generator=generator)

        throughput = torch.ones((m * spp, 3), dtype=torch.float32, device=device)
        accum = torch.zeros((m * spp, 3), dtype=torch.float32, device=device)
        active = torch.ones(m * spp, dtype=torch.bool, device=device)

        for bounce in range(max_bounces):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=False).squeeze(-1)
            hit_mask, hit_t, hit_tri = intersect_rays(scene, ray_o[active_idx], ray_d[active_idx])

            if not hit_mask.any():
                break

            hit_global = active_idx[hit_mask]
            tri = hit_tri[hit_mask]
            hit_p = ray_o[hit_global] + ray_d[hit_global] * hit_t[hit_mask].unsqueeze(-1)
            hit_n = scene.normals[tri]
            # Ensure shading normal faces incoming ray
            flip = (hit_n * ray_d[hit_global]).sum(dim=-1, keepdim=True) > 0.0
            hit_n = torch.where(flip, -hit_n, hit_n)
            hit_albedo = scene.albedos[tri]

            # Update active set: only rays that hit continue
            next_active = torch.zeros_like(active)
            next_active[hit_global] = True
            active = next_active

            # Throughput accumulates surface albedo at hit bounce
            # Cosine-weighted hemisphere PDF (cos/pi) cancels the 1/pi of Lambertian BRDF.
            throughput[hit_global] = throughput[hit_global] * hit_albedo

            # Next Event Estimation (direct light arriving at bounce surface)
            direct_at_hit = _direct_irradiance_over_pi(scene, hit_p, hit_n, cur_l[hit_global])
            accum[hit_global] = accum[hit_global] + throughput[hit_global] * direct_at_hit

            if bounce + 1 < max_bounces:
                # Optional Russian Roulette termination after configured bounce
                if bounce + 1 >= russian_roulette_bounce:
                    p_survive = throughput[hit_global].amax(dim=-1).clamp(0.05, 0.95)
                    survive = torch.rand(
                        hit_global.shape[0], device=device, generator=generator
                    ) < p_survive
                    dead_global = hit_global[~survive]
                    active[dead_global] = False
                    surv_global = hit_global[survive]
                    if surv_global.numel() == 0:
                        break
                    throughput[surv_global] = throughput[surv_global] / p_survive[survive].unsqueeze(-1)
                    hit_p = hit_p[survive]
                    hit_n = hit_n[survive]
                    hit_global = surv_global

                ray_o[hit_global] = hit_p + hit_n * (RAY_EPS * 4.0)
                ray_d[hit_global] = sample_cosine_hemisphere(hit_n, generator=generator)

        out[start:end] = accum.reshape(m, spp, 3).mean(dim=1)

    return out


@torch.no_grad()
def render_gbuffer(
    scene: SceneMesh,
    eye: list[float],
    target: list[float],
    width: int,
    height: int,
    fov_deg: float = 48.0,
) -> dict[str, torch.Tensor]:
    """Rasterize primary camera rays into a G-buffer (hit_mask, pos, normal, albedo)."""
    device = scene.device
    eye_t = torch.tensor(eye, dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    up_t = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    forward = target_t - eye_t
    forward = forward / forward.norm().clamp_min(1e-8)
    right = torch.linalg.cross(forward, up_t)
    right = right / right.norm().clamp_min(1e-8)
    cam_up = torch.linalg.cross(right, forward)

    aspect = width / float(height)
    tan_half = math.tan(math.radians(fov_deg) * 0.5)

    ys = torch.linspace(1.0, -1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    px = grid_x * aspect * tan_half
    py = grid_y * tan_half
    dirs = (
        forward.view(1, 1, 3)
        + right.view(1, 1, 3) * px.unsqueeze(-1)
        + cam_up.view(1, 1, 3) * py.unsqueeze(-1)
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    n_pix = width * height
    origins = eye_t.view(1, 3).expand(n_pix, 3)
    dirs_flat = dirs.reshape(n_pix, 3)

    hit_mask, hit_t, hit_tri = intersect_rays(scene, origins, dirs_flat)
    hit_pos = origins + dirs_flat * hit_t.unsqueeze(-1)
    hit_nrm = scene.normals[hit_tri]
    flip = (hit_nrm * dirs_flat).sum(dim=-1, keepdim=True) > 0.0
    hit_nrm = torch.where(flip, -hit_nrm, hit_nrm)
    hit_alb = scene.albedos[hit_tri]

    return {
        "hit_mask": hit_mask.reshape(height, width),
        "positions": hit_pos.reshape(height, width, 3),
        "normals": hit_nrm.reshape(height, width, 3),
        "albedos": hit_alb.reshape(height, width, 3),
    }

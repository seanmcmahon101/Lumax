"""Stage 3: Unseen viewpoint/light visual evaluation and automated architecture sweep."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from lumax.config import deep_merge, resolve_device, seed_everything
from lumax.dataset import load_dataset
from lumax.model import NeuralGIModel, load_checkpoint
from lumax.pathtracer import render_gbuffer, trace_indirect_gi
from lumax.scene import SceneMesh, build_cornell_scene
from lumax.train import train_model

# Perceptual Inferno-like 6-stop heatmap LUT (black -> indigo -> magenta -> orange -> yellow -> white)
_HEATMAP_STOPS = np.array(
    [
        [0.00, 0.00, 0.02],
        [0.18, 0.05, 0.36],
        [0.58, 0.14, 0.42],
        [0.89, 0.35, 0.16],
        [0.99, 0.76, 0.14],
        [1.00, 0.99, 0.82],
    ],
    dtype=np.float32,
)


def tonemap_aces_srgb(linear_rgb: np.ndarray, exposure: float = 1.6) -> np.ndarray:
    """ACES filmic tone mapping + sRGB gamma transfer -> float32 in [0, 1]."""
    x = np.maximum(linear_rgb * float(exposure), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    # Standard sRGB transfer
    srgb = np.where(mapped <= 0.0031308, mapped * 12.92, 1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def error_to_heatmap(err_scalar: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """Map scalar absolute error (H, W) -> RGB float32 heatmap in [0, 1]."""
    v = np.clip(err_scalar * float(scale), 0.0, 1.0)
    n_stops = _HEATMAP_STOPS.shape[0] - 1
    pos = v * n_stops
    idx0 = np.floor(pos).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, n_stops)
    frac = (pos - idx0)[..., None]
    return (1.0 - frac) * _HEATMAP_STOPS[idx0] + frac * _HEATMAP_STOPS[idx1]


def _compute_ssim(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    """Windowed SSIM on sRGB images (H, W, 3) using pure NumPy separable box filter."""
    # Luminance for crisp perceptual structure comparison
    la = 0.2126 * img_a[..., 0] + 0.7152 * img_a[..., 1] + 0.0722 * img_a[..., 2]
    lb = 0.2126 * img_b[..., 0] + 0.7152 * img_b[..., 1] + 0.0722 * img_b[..., 2]

    def box_blur(arr: np.ndarray, k: int = 7) -> np.ndarray:
        pad = k // 2
        p = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
        cs = np.cumsum(p, axis=0)
        cs = np.pad(cs, ((1, 0), (0, 0)), mode="constant")
        h_blur = (cs[k:, :] - cs[:-k, :]) / k
        cs2 = np.cumsum(h_blur, axis=1)
        cs2 = np.pad(cs2, ((0, 0), (1, 0)), mode="constant")
        return (cs2[:, k:] - cs2[:, :-k]) / k

    mu_a = box_blur(la)
    mu_b = box_blur(lb)
    sig_a = np.maximum(box_blur(la * la) - mu_a * mu_a, 0.0)
    sig_b = np.maximum(box_blur(lb * lb) - mu_b * mu_b, 0.0)
    sig_ab = box_blur(la * lb) - mu_a * mu_b

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a + sig_b + c2)
    )
    if mask.any():
        return float(ssim_map[mask].mean())
    return float(ssim_map.mean())


def _compute_grad_mae(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    """Spatial gradient MAE on sRGB images — quantifies edge & crease sharpness."""
    gx_a = img_a[:, 1:, :] - img_a[:, :-1, :]
    gx_b = img_b[:, 1:, :] - img_b[:, :-1, :]
    gy_a = img_a[1:, :, :] - img_a[:-1, :, :]
    gy_b = img_b[1:, :, :] - img_b[:-1, :, :]

    mx = mask[:, 1:] & mask[:, :-1]
    my = mask[1:, :] & mask[:-1, :]
    err_x = float(np.abs(gx_a - gx_b)[mx].mean()) if mx.any() else 0.0
    err_y = float(np.abs(gy_a - gy_b)[my].mean()) if my.any() else 0.0
    return 0.5 * (err_x + err_y)


def _save_triptych(
    gt_srgb: np.ndarray,
    pred_srgb: np.ndarray,
    err_rgb: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    error_scale: float,
) -> None:
    """Composite a labeled side-by-side PNG: Ground Truth | Neural GI | Error Map."""
    h, w, _ = gt_srgb.shape
    scale = max(1, math.ceil(192 / max(1, w)))
    dw, dh = w * scale, h * scale
    gap = 8
    header_h = 36
    footer_h = 24
    total_w = dw * 3 + gap * 2
    total_h = header_h + dh + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 22))
    draw = ImageDraw.Draw(canvas)

    def to_pil(arr: np.ndarray) -> Image.Image:
        u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")
        return img.resize((dw, dh), Image.Resampling.NEAREST) if scale > 1 else img

    canvas.paste(to_pil(gt_srgb), (0, header_h))
    canvas.paste(to_pil(pred_srgb), (dw + gap, header_h))
    canvas.paste(to_pil(err_rgb), (2 * (dw + gap), header_h))

    draw.text((8, 4), title, fill=(235, 235, 240))
    draw.text((8, 20), subtitle, fill=(175, 195, 220))

    labels = [
        ("Path-Traced GT (Indirect)", 0),
        ("Neural GI (Zero RT)", dw + gap),
        (f"Abs Error (x{error_scale:.0f})", 2 * (dw + gap)),
    ]
    for label, x0 in labels:
        draw.text((x0 + 6, header_h + dh + 6), label, fill=(210, 210, 215))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@torch.no_grad()
def render_ground_truth_cases(
    scene: SceneMesh,
    test_cases: list[dict[str, Any]],
    width: int,
    height: int,
    gt_spp: int,
    max_bounces: int,
    ray_batch_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pre-render G-buffers and path-traced indirect-only ground truth for unseen test cases."""
    device = scene.device
    gen = torch.Generator(device=device).manual_seed(seed + 9001)
    cached: list[dict[str, Any]] = []

    for tc in test_cases:
        gbuf = render_gbuffer(
            scene=scene,
            eye=tc["eye"],
            target=tc["target"],
            width=width,
            height=height,
            fov_deg=float(tc.get("fov_deg", 48.0)),
        )
        hit_mask = gbuf["hit_mask"]
        hit_pos = gbuf["positions"][hit_mask]
        hit_nrm = gbuf["normals"][hit_mask]
        light_t = torch.tensor(tc["light_pos"], dtype=torch.float32, device=device)
        hit_lgt = light_t.unsqueeze(0).expand(hit_pos.shape[0], 3)

        gt_irrad_hits = trace_indirect_gi(
            scene=scene,
            positions=hit_pos,
            normals=hit_nrm,
            light_pos=hit_lgt,
            spp=gt_spp,
            max_bounces=max_bounces,
            ray_batch_size=ray_batch_size,
            generator=gen,
        )
        gt_irrad = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
        gt_irrad[hit_mask] = gt_irrad_hits

        cached.append(
            {
                "case": tc,
                "hit_mask": hit_mask,
                "positions": gbuf["positions"],
                "normals": gbuf["normals"],
                "albedos": gbuf["albedos"],
                "gt_irrad": gt_irrad,
            }
        )
    return cached


@torch.no_grad()
def evaluate_model_on_cases(
    model: NeuralGIModel,
    gt_cases: list[dict[str, Any]],
    out_dir: str | Path,
    render_mode: str = "indirect_radiance",
    exposure: float = 1.6,
    error_scale: float = 4.0,
    save_images: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained NeuralGIModel against cached GT cases and write side-by-side PNGs."""
    model.eval()
    device = next(model.parameters()).device
    out_path = Path(out_dir)
    if save_images:
        out_path.mkdir(parents=True, exist_ok=True)

    cost = model.cost_report()
    per_case: list[dict[str, Any]] = []
    total_infer_s = 0.0
    total_pixels = 0

    for item in gt_cases:
        tc = item["case"]
        hit_mask = item["hit_mask"].to(device)
        positions = item["positions"].to(device)
        normals = item["normals"].to(device)
        albedos = item["albedos"].to(device)
        gt_irrad = item["gt_irrad"].to(device)

        h, w = hit_mask.shape
        hit_pos = positions[hit_mask]
        hit_nrm = normals[hit_mask]
        light_t = torch.tensor(tc["light_pos"], dtype=torch.float32, device=device)
        hit_lgt = light_t.unsqueeze(0).expand(hit_pos.shape[0], 3)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_irrad_hits = model(hit_pos, hit_nrm, hit_lgt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_s += time.perf_counter() - t0
        total_pixels += int(hit_pos.shape[0])

        pred_irrad = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        pred_irrad[hit_mask] = pred_irrad_hits

        if render_mode == "indirect_radiance":
            gt_lin = gt_irrad * albedos
            pred_lin = pred_irrad * albedos
        else:
            gt_lin = gt_irrad
            pred_lin = pred_irrad

        mask_np = hit_mask.cpu().numpy()
        gt_np = gt_lin.cpu().numpy()
        pred_np = pred_lin.cpu().numpy()

        gt_srgb = tonemap_aces_srgb(gt_np, exposure=exposure)
        pred_srgb = tonemap_aces_srgb(pred_np, exposure=exposure)
        gt_srgb[~mask_np] = 0.04
        pred_srgb[~mask_np] = 0.04

        abs_err_srgb = np.abs(gt_srgb - pred_srgb).mean(axis=-1)
        abs_err_srgb[~mask_np] = 0.0
        err_map = error_to_heatmap(abs_err_srgb, scale=error_scale)
        err_map[~mask_np] = 0.04

        diff_lin = pred_np[mask_np] - gt_np[mask_np]
        lin_mae = float(np.abs(diff_lin).mean())
        lin_rmse = float(np.sqrt((diff_lin ** 2).mean()))

        diff_srgb = pred_srgb[mask_np] - gt_srgb[mask_np]
        srgb_mse = float((diff_srgb ** 2).mean())
        psnr = -10.0 * math.log10(max(srgb_mse, 1e-10))
        ssim = _compute_ssim(gt_srgb, pred_srgb, mask_np)
        grad_mae = _compute_grad_mae(gt_srgb, pred_srgb, mask_np)

        img_file = out_path / f"{tc['name']}.png"
        if save_images:
            title = f"{tc['name']} | PSNR {psnr:.2f}dB | SSIM {ssim:.3f} | GradMAE {grad_mae:.4f}"
            subtitle = (
                f"{cost['encoder_type']} {cost['hidden_layers']}x{cost['hidden_dim']} | "
                f"{cost['fp16_kb']:.2f} KB fp16 | {cost['total_macs_per_pixel']:,} MACs/px"
            )
            _save_triptych(gt_srgb, pred_srgb, err_map, img_file, title, subtitle, error_scale)

        per_case.append(
            {
                "name": tc["name"],
                "psnr_db": round(psnr, 3),
                "ssim": round(ssim, 4),
                "grad_mae": round(grad_mae, 5),
                "linear_mae": round(lin_mae, 5),
                "linear_rmse": round(lin_rmse, 5),
                "image": str(img_file) if save_images else None,
            }
        )

    mpix_per_sec = (total_pixels / max(total_infer_s, 1e-9)) / 1e6
    agg = {
        "mean_psnr_db": round(float(np.mean([c["psnr_db"] for c in per_case])), 3),
        "mean_ssim": round(float(np.mean([c["ssim"] for c in per_case])), 4),
        "mean_grad_mae": round(float(np.mean([c["grad_mae"] for c in per_case])), 5),
        "mean_linear_mae": round(float(np.mean([c["linear_mae"] for c in per_case])), 5),
        "mean_linear_rmse": round(float(np.mean([c["linear_rmse"] for c in per_case])), 5),
        "neural_mpix_per_sec": round(mpix_per_sec, 2),
        "cost": cost,
        "cases": per_case,
    }
    if save_images:
        (out_path / "eval_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg


def run_eval(cfg: dict[str, Any], checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Run Stage 3 single-model visual + quantitative evaluation on unseen viewpoints & lights."""
    seed = int(cfg["runtime"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    ckpt = Path(checkpoint_path or cfg["train"]["checkpoint_path"])
    model, _meta = load_checkpoint(ckpt, device=device)
    scene = build_cornell_scene(cfg["scene"], device=device)

    ecfg = cfg["eval"]
    width = int(ecfg.get("width", 256))
    height = int(ecfg.get("height", 256))
    gt_spp = int(ecfg.get("gt_spp", 96))
    max_bounces = int(ecfg.get("max_bounces", 3))
    ray_batch_size = int(ecfg.get("ray_batch_size", 32768))
    out_dir = Path(ecfg.get("output_dir", "outputs/eval"))

    print(
        f"[stage 3: eval] device={device} | {len(ecfg['test_cases'])} unseen views "
        f"({width}x{height}, gt_spp={gt_spp}) | ckpt={ckpt}"
    )
    t0 = time.perf_counter()
    gt_cases = render_ground_truth_cases(
        scene=scene,
        test_cases=ecfg["test_cases"],
        width=width,
        height=height,
        gt_spp=gt_spp,
        max_bounces=max_bounces,
        ray_batch_size=ray_batch_size,
        seed=seed,
    )
    results = evaluate_model_on_cases(
        model=model,
        gt_cases=gt_cases,
        out_dir=out_dir,
        render_mode=str(ecfg.get("render_mode", "indirect_radiance")),
        exposure=float(ecfg.get("exposure", 1.6)),
        error_scale=float(ecfg.get("error_scale", 4.0)),
        save_images=True,
    )
    dt = time.perf_counter() - t0
    print(
        f"[stage 3: eval] done in {dt:.2f}s | PSNR={results['mean_psnr_db']:.2f} dB | "
        f"SSIM={results['mean_ssim']:.4f} | GradMAE={results['mean_grad_mae']:.5f} | "
        f"neural={results['neural_mpix_per_sec']:.2f} MPix/s | saved {out_dir}"
    )
    return results


def run_sweep(cfg: dict[str, Any]) -> Path:
    """Run automated architecture/encoder sweep against shared unseen GT test views."""
    seed = int(cfg["runtime"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    scfg = cfg["sweep"]
    ecfg = cfg["eval"]
    out_dir = Path(scfg.get("output_dir", "outputs/sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg["sampler"]["dataset_path"], device=device)
    scene = build_cornell_scene(cfg["scene"], device=device)

    eval_w = int(scfg.get("eval_width", ecfg.get("width", 192)))
    eval_h = int(scfg.get("eval_height", ecfg.get("height", 192)))
    gt_spp = int(scfg.get("gt_spp", ecfg.get("gt_spp", 64)))
    max_bounces = int(ecfg.get("max_bounces", 3))
    ray_batch_size = int(ecfg.get("ray_batch_size", 32768))
    variants = scfg.get("variants", [])

    print(
        f"[stage 3: sweep] device={device} | pre-rendering {len(ecfg['test_cases'])} GT views "
        f"({eval_w}x{eval_h}, spp={gt_spp}) for {len(variants)} variants..."
    )
    gt_cases = render_ground_truth_cases(
        scene=scene,
        test_cases=ecfg["test_cases"],
        width=eval_w,
        height=eval_h,
        gt_spp=gt_spp,
        max_bounces=max_bounces,
        ray_batch_size=ray_batch_size,
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        vname = str(variant["name"])
        vcfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != "name"})
        if "epochs" in scfg and "train" not in variant:
            vcfg["train"]["epochs"] = int(scfg["epochs"])

        vdir = out_dir / vname
        vdir.mkdir(parents=True, exist_ok=True)
        ckpt_path = vdir / "model.pt"

        print(f"  -> variant '{vname}'...")
        model, train_summary, _ = train_model(
            vcfg, dataset_override=dataset, checkpoint_override=ckpt_path, quiet=True
        )
        eval_res = evaluate_model_on_cases(
            model=model,
            gt_cases=gt_cases,
            out_dir=vdir,
            render_mode=str(ecfg.get("render_mode", "indirect_radiance")),
            exposure=float(ecfg.get("exposure", 1.6)),
            error_scale=float(ecfg.get("error_scale", 4.0)),
            save_images=True,
        )
        cost = eval_res["cost"]
        row = {
            "variant": vname,
            "encoder": cost["encoder_type"],
            "enc_dim": cost["encoder_output_dim"],
            "mlp": f"{cost['hidden_layers']}x{cost['hidden_dim']}({cost['activation']})",
            "runtime_params": cost["runtime_params"],
            "fp16_kb": cost["fp16_kb"],
            "macs_per_px": cost["total_macs_per_pixel"],
            "psnr_db": eval_res["mean_psnr_db"],
            "ssim": eval_res["mean_ssim"],
            "grad_mae": eval_res["mean_grad_mae"],
            "linear_mae": eval_res["mean_linear_mae"],
            "train_s": train_summary["train_seconds"],
            "checkpoint": str(ckpt_path),
        }
        rows.append(row)
        print(
            f"     {vname:16s} | {cost['fp16_kb']:5.2f} KB | {cost['total_macs_per_pixel']:5,d} MACs/px | "
            f"PSNR={row['psnr_db']:5.2f} dB | SSIM={row['ssim']:.4f} | GradMAE={row['grad_mae']:.5f}"
        )

    rows.sort(key=lambda r: r["psnr_db"], reverse=True)

    csv_path = out_dir / "sweep_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_lines = [
        "# Lumax Neural GI Sweep Summary",
        "",
        "| Variant | Encoder | MLP | Params | fp16 KB | MACs/px | PSNR (dB) | SSIM | Grad-MAE (Sharpness) | Linear MAE | Train (s) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['variant']}` | `{r['encoder']}` (d={r['enc_dim']}) | `{r['mlp']}` | "
            f"{r['runtime_params']:,} | {r['fp16_kb']:.2f} | {r['macs_per_px']:,} | "
            f"**{r['psnr_db']:.2f}** | {r['ssim']:.4f} | {r['grad_mae']:.5f} | "
            f"{r['linear_mae']:.5f} | {r['train_s']:.2f} |"
        )
    md_path = out_dir / "sweep_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[stage 3: sweep] wrote {csv_path} and {md_path}")
    return md_path

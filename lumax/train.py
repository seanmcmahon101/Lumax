"""Stage 2c: Trainer for intentionally overfitted kilobyte-scale micro-MLPs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lumax.config import resolve_device, seed_everything
from lumax.dataset import load_dataset
from lumax.model import NeuralGIModel, build_model, save_checkpoint


def _compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "l1":
        return F.l1_loss(pred, target)
    if loss_name == "l2":
        return F.mse_loss(pred, target)
    if loss_name == "huber":
        return F.smooth_l1_loss(pred, target, beta=0.05)
    if loss_name == "relative_l2":
        denom = (target.detach().pow(2) + 1e-2)
        return ((pred - target).pow(2) / denom).mean()
    raise ValueError(f"Unsupported loss: {loss_name!r}")


@torch.no_grad()
def _eval_split(
    model: NeuralGIModel,
    split: dict[str, torch.Tensor],
    loss_name: str,
    batch_size: int = 16384,
) -> dict[str, float]:
    model.eval()
    n = split["positions"].shape[0]
    total_loss = 0.0
    total_l1_lin = 0.0
    total_mse_lin = 0.0
    for i in range(0, n, batch_size):
        p = split["positions"][i : i + batch_size]
        nrm = split["normals"][i : i + batch_size]
        lgt = split["light_pos"][i : i + batch_size]
        rgb = split["rgb"][i : i + batch_size]
        m = p.shape[0]

        raw = model.forward_raw(p, nrm, lgt)
        tgt = model.linear_rgb_to_target(rgb)
        loss = _compute_loss(raw, tgt, loss_name)

        pred_lin = model.raw_to_linear_rgb(raw)
        total_loss += float(loss.item()) * m
        total_l1_lin += float((pred_lin - rgb).abs().mean().item()) * m
        total_mse_lin += float((pred_lin - rgb).pow(2).mean().item()) * m

    mae = total_l1_lin / n
    mse = total_mse_lin / n
    psnr = -10.0 * math.log10(max(mse, 1e-10))
    return {
        "loss": total_loss / n,
        "linear_mae": mae,
        "linear_rmse": math.sqrt(mse),
        "linear_psnr": psnr,
    }


def train_model(
    cfg: dict[str, Any],
    dataset_override: dict[str, Any] | None = None,
    checkpoint_override: str | Path | None = None,
    quiet: bool = False,
) -> tuple[NeuralGIModel, dict[str, Any], Path]:
    """Train a micro-MLP neural GI model and save the best checkpoint."""
    seed = int(cfg["runtime"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    tcfg = cfg["train"]
    epochs = int(tcfg.get("epochs", 120))
    batch_size = int(tcfg.get("batch_size", 4096))
    lr = float(tcfg.get("lr", 5e-3))
    min_lr = float(tcfg.get("min_lr", 1e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    loss_name = str(tcfg.get("loss", "l1")).lower()
    log_every = max(1, int(tcfg.get("log_every", 10)))

    ckpt_path = Path(checkpoint_override or tcfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    data = dataset_override if dataset_override is not None else load_dataset(cfg["sampler"]["dataset_path"], device)
    scene_bounds = data["meta"]["scene_bounds"]

    model = build_model(cfg["model"], scene_bounds=scene_bounds, seed=seed).to(device)
    cost = model.cost_report()

    if not quiet:
        print(f"[stage 2: train] device={device} | {model.format_cost_line()}")
        print(
            f"[stage 2: train] epochs={epochs} | batch={batch_size} | lr={lr}->{min_lr} | "
            f"loss={loss_name} | target={model.target_transform}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)

    train_split = data["train"]
    val_split = data["val"]
    n_train = train_split["positions"].shape[0]
    train_targets = model.linear_rgb_to_target(train_split["rgb"])

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        seen = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            p = train_split["positions"][idx]
            nrm = train_split["normals"][idx]
            lgt = train_split["light_pos"][idx]
            tgt = train_targets[idx]

            optimizer.zero_grad(set_to_none=True)
            raw = model.forward_raw(p, nrm, lgt)
            loss = _compute_loss(raw, tgt, loss_name)
            loss.backward()
            optimizer.step()

            bs = idx.shape[0]
            running_loss += float(loss.item()) * bs
            seen += bs

        scheduler.step()
        train_loss = running_loss / max(1, seen)

        should_eval = (epoch == 1) or (epoch == epochs) or (epoch % log_every == 0)
        if should_eval:
            val_m = _eval_split(model, val_split, loss_name)
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_m["loss"],
                    "val_linear_mae": val_m["linear_mae"],
                    "val_linear_rmse": val_m["linear_rmse"],
                    "val_linear_psnr": val_m["linear_psnr"],
                }
            if not quiet:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:03d}/{epochs:03d} | lr={cur_lr:.5f} | "
                    f"train_loss={train_loss:.5f} | val_loss={val_m['loss']:.5f} | "
                    f"val_MAE={val_m['linear_mae']:.5f} | val_PSNR={val_m['linear_psnr']:.2f} dB"
                )

    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    summary = {
        "train_seconds": round(elapsed, 3),
        "samples_per_sec": round((n_train * epochs) / max(elapsed, 1e-6), 1),
        "cost": cost,
        "best": best_metrics,
    }
    save_checkpoint(
        path=ckpt_path,
        model=model,
        model_cfg=cfg["model"],
        scene_bounds=scene_bounds,
        metrics=summary,
        seed=seed,
    )
    metrics_json = ckpt_path.with_suffix(".metrics.json")
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not quiet:
        print(
            f"[stage 2: train] done in {elapsed:.2f}s ({summary['samples_per_sec']:,.0f} samples/s) | "
            f"best val_PSNR={best_metrics.get('val_linear_psnr', 0.0):.2f} dB | saved {ckpt_path}"
        )
    return model, summary, ckpt_path

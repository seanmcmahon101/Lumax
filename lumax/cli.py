"""Unified modular CLI for Lumax Neural GI Baker."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from lumax.config import apply_smoke_preset, load_config, set_nested
from lumax.dataset import generate_dataset
from lumax.evaluate import run_eval, run_sweep
from lumax.export import run_export
from lumax.train import train_model


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to YAML config (default: config.yaml)"
    )
    parser.add_argument(
        "--device", default=None, help="Override runtime.device (auto | cuda | cpu | mps)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use fast laptop/CPU smoke preset across all stages",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Dot-notation config override (repeatable), e.g. -o train.epochs=50",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumax",
        description="Lumax — Hardware-agnostic Neural Global Illumination Baker",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p_gen = sub.add_parser("generate", help="Stage 1: Generate indirect-only path-traced dataset")
    _add_common_args(p_gen)

    p_train = sub.add_parser("train", help="Stage 2: Train kilobyte-scale micro-MLP")
    _add_common_args(p_train)

    p_eval = sub.add_parser("eval", help="Stage 3a: Render unseen viewpoints/lights vs GT")
    _add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", default=None, help="Override checkpoint path to evaluate")

    p_sweep = sub.add_parser("sweep", help="Stage 3b: Automated encoder/MLP sharpness sweep")
    _add_common_args(p_sweep)

    p_exp = sub.add_parser("export", help="Stage 4: Export .lmx binary + Godot 4 .gdshaderinc")
    _add_common_args(p_exp)
    p_exp.add_argument("--checkpoint", default=None, help="Override checkpoint path to export")

    p_all = sub.add_parser("all", help="Run full pipeline: generate -> train -> eval -> export (+ optional sweep)")
    _add_common_args(p_all)
    p_all.add_argument("--with-sweep", action="store_true", help="Also run Stage 3b architecture sweep")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    if args.smoke:
        cfg = apply_smoke_preset(cfg)
    if args.device:
        set_nested(cfg, "runtime.device", args.device)

    stage = args.stage
    if stage == "generate":
        generate_dataset(cfg)
    elif stage == "train":
        train_model(cfg)
    elif stage == "eval":
        run_eval(cfg, checkpoint_path=args.checkpoint)
    elif stage == "sweep":
        run_sweep(cfg)
    elif stage == "export":
        run_export(cfg, checkpoint_path=args.checkpoint)
    elif stage == "all":
        generate_dataset(cfg)
        train_model(cfg)
        run_eval(cfg)
        if args.with_sweep:
            run_sweep(cfg)
        run_export(cfg)
    else:
        parser.error(f"Unknown stage: {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

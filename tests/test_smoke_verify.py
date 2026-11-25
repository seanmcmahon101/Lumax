"""Fast verification tests covering all 4 Lumax stages, encoders, binary parity, and GLSL structure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from lumax.config import apply_smoke_preset, load_config
from lumax.encoders import build_encoder
from lumax.export import (
    eval_lmx_numpy,
    export_godot_shader_include,
    export_lmx_binary,
    load_lmx_binary,
)
from lumax.model import build_model
from lumax.pathtracer import trace_indirect_gi
from lumax.scene import build_cornell_scene


def test_scene_and_indirect_pathtracer() -> None:
    cfg = load_config("config.yaml")
    scene = build_cornell_scene(cfg["scene"], device=torch.device("cpu"))
    assert scene.num_triangles == 22, f"Expected 10 box tris + 12 occluder tris, got {scene.num_triangles}"

    pos, nrm, _alb = scene.sample_surface_points(64)
    lgt = scene.sample_light_positions(64)
    rgb = trace_indirect_gi(scene, pos, nrm, lgt, spp=8, max_bounces=3, ray_batch_size=512)
    assert rgb.shape == (64, 3)
    assert torch.isfinite(rgb).all()
    assert (rgb >= 0.0).all()
    assert float(rgb.mean().item()) > 0.0, "Indirect bounce energy should be strictly positive"


def test_all_encoders_and_cost_accounting() -> None:
    pos = torch.randn(16, 3).clamp(-1, 1)
    nrm = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    lgt = torch.randn(16, 3).clamp(-1, 1)

    for enc_cfg in [
        {"type": "identity"},
        {"type": "positional", "num_bands_pos": 4, "num_bands_light": 2, "include_input": True},
        {"type": "fourier", "fourier_features": 16, "include_input": True},
        {"type": "chunked_grid", "chunk_count": [2, 2, 2], "chunk_resolution": [4, 4, 4], "feature_dim": 8},
    ]:
        model = build_model({"encoder": enc_cfg, "mlp": {"hidden_layers": 2, "hidden_dim": 16, "activation": "relu"}})
        out = model(pos, nrm, lgt)
        assert out.shape == (16, 3)
        assert (out >= 0.0).all()
        cost = model.cost_report()
        assert cost["runtime_params"] > 0
        assert cost["fp16_kb"] > 0.0
        assert cost["total_macs_per_pixel"] > 0


def test_binary_and_godot_glsl_export_parity() -> None:
    for enc_cfg in [
        {"type": "positional", "num_bands_pos": 3, "num_bands_light": 2, "include_input": True},
        {"type": "fourier", "fourier_features": 16, "include_input": True},
        {"type": "identity"},
    ]:
        model = build_model(
            {"encoder": enc_cfg, "mlp": {"hidden_layers": 2, "hidden_dim": 16, "activation": "relu", "target_transform": "log1p"}}
        )
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = export_lmx_binary(model, Path(tmp) / "m.lmx", quantize_fp16=True)
            inc_path = export_godot_shader_include(model, Path(tmp) / "m.gdshaderinc", quantize_fp16=True)

            lmx = load_lmx_binary(bin_path)
            assert lmx["file_bytes"] == bin_path.stat().st_size

            rng = np.random.default_rng(7)
            p = rng.uniform(-0.9, 0.9, size=(32, 3)).astype(np.float32)
            n = rng.normal(size=(32, 3)).astype(np.float32)
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
            l = rng.uniform(-0.7, 0.7, size=(32, 3)).astype(np.float32)

            with torch.no_grad():
                pt_out = model(torch.from_numpy(p), torch.from_numpy(n), torch.from_numpy(l)).numpy()
            np_out = eval_lmx_numpy(lmx, p, n, l)
            max_err = float(np.abs(pt_out - np_out).max())
            assert max_err < 1e-2, f"fp16 parity error too high for {enc_cfg['type']}: {max_err}"

            glsl = inc_path.read_text(encoding="utf-8")
            for required_token in (
                "uniform float gi_bounce_strength",
                "uniform float gi_vibrancy",
                "uniform float gi_warmth",
                "uniform float gi_shadow_lift",
                "vec3 lumax_eval_raw(",
                "vec3 lumax_grade_gi(",
                "vec3 lumax_sample_gi(",
            ):
                assert required_token in glsl, f"Missing {required_token} in generated GLSL"
            # Verify zero Tensor Core / cooperative matrix extensions
            assert "GL_NV_cooperative_matrix" not in glsl
            assert "GL_KHR_cooperative_matrix" not in glsl


if __name__ == "__main__":
    test_scene_and_indirect_pathtracer()
    test_all_encoders_and_cost_accounting()
    test_binary_and_godot_glsl_export_parity()
    print("[tests] ALL VERIFICATION CHECKS PASSED")

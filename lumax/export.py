"""Stage 4: Export trained weights to compact .lmx binary and self-contained Godot 4 GLSL include."""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lumax.config import resolve_device
from lumax.model import LOG1P_MAX_RAW, NeuralGIModel, load_checkpoint

LMX_MAGIC = b"LMX1"
LMX_VERSION = 1

ENCODER_IDS = {"identity": 0, "positional": 1, "fourier": 2}
ENCODER_NAMES = {v: k for k, v in ENCODER_IDS.items()}

ACTIVATION_IDS = {"relu": 0, "leaky_relu": 1, "silu": 2, "sin": 3}
ACTIVATION_NAMES = {v: k for k, v in ACTIVATION_IDS.items()}

FLAG_FP16 = 1 << 0
FLAG_LOG1P = 1 << 1

# Header layout (32 bytes, little-endian, Rust #[repr(C)] friendly):
#   magic[4]:       4s   "LMX1"
#   version:        u16  1
#   flags:          u16  bit0=fp16, bit1=log1p
#   encoder_id:     u8   0=identity, 1=positional, 2=fourier
#   activation_id:  u8   0=relu, 1=leaky_relu, 2=silu, 3=sin
#   num_layers:     u8   number of MLP linear layers
#   reserved:       u8   0
#   input_dim:      u16  encoder output dim
#   output_dim:     u16  3 (linear RGB)
#   runtime_params: u32  total runtime floats (buffers + weights + biases)
#   enc_blob_bytes: u32  encoder metadata payload size in bytes
#   wgt_blob_bytes: u32  layer dims + weights payload size in bytes
#   payload_crc32:  u32  CRC32 of (bounds + enc_blob + wgt_blob)
HEADER_STRUCT = struct.Struct("<4sHHBBBBHHIIII")
assert HEADER_STRUCT.size == 32


def _quantize_array(arr: np.ndarray, fp16: bool) -> np.ndarray:
    return arr.astype(np.float16 if fp16 else np.float32)


def export_lmx_binary(model: NeuralGIModel, out_path: str | Path, quantize_fp16: bool = True) -> Path:
    """Compile model weights into the tiny little-endian .lmx binary format."""
    enc_spec = model.encoder.export_spec()
    enc_type = enc_spec["type"]
    if enc_type not in ENCODER_IDS:
        raise NotImplementedError(
            f"Binary export supports {list(ENCODER_IDS.keys())}; got {enc_type!r}. "
            "Chunked 3D feature grids should be exported alongside 3D textures."
        )

    flags = (FLAG_FP16 if quantize_fp16 else 0) | (
        FLAG_LOG1P if model.target_transform == "log1p" else 0
    )

    # 1. Scene normalization bounds (6 x f32 = 24 bytes)
    aabb_min = model.aabb_min.detach().cpu().numpy().astype(np.float32)
    aabb_max = model.aabb_max.detach().cpu().numpy().astype(np.float32)
    bounds_blob = struct.pack("<6f", *aabb_min.tolist(), *aabb_max.tolist())

    # 2. Encoder payload
    if enc_type == "identity":
        enc_blob = b""
    elif enc_type == "positional":
        enc_blob = struct.pack(
            "<BBBB",
            1 if enc_spec["include_input"] else 0,
            int(enc_spec["num_bands_pos"]),
            int(enc_spec["num_bands_normal"]),
            int(enc_spec["num_bands_light"]),
        )
    elif enc_type == "fourier":
        basis = model.encoder.basis.detach().cpu().numpy()  # (F, 9)
        basis_q = _quantize_array(basis, quantize_fp16)
        enc_blob = (
            struct.pack("<BH", 1 if enc_spec["include_input"] else 0, int(enc_spec["fourier_features"]))
            + basis_q.tobytes()
        )
    else:
        raise ValueError(enc_type)

    # 3. MLP layer dimensions + row-major weights & biases
    dim_parts: list[bytes] = []
    wgt_parts: list[bytes] = []
    for layer in model.layers:
        dim_parts.append(struct.pack("<HH", int(layer.in_features), int(layer.out_features)))
        w = _quantize_array(layer.weight.detach().cpu().numpy(), quantize_fp16)  # (out, in)
        b = _quantize_array(layer.bias.detach().cpu().numpy(), quantize_fp16)    # (out,)
        wgt_parts.append(w.tobytes())
        wgt_parts.append(b.tobytes())
    wgt_blob = b"".join(dim_parts) + b"".join(wgt_parts)

    payload = bounds_blob + enc_blob + wgt_blob
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    cost = model.cost_report()
    header = HEADER_STRUCT.pack(
        LMX_MAGIC,
        LMX_VERSION,
        flags,
        ENCODER_IDS[enc_type],
        ACTIVATION_IDS[model.activation],
        len(model.layers),
        0,
        int(model.encoder.output_dim),
        3,
        int(cost["runtime_params"]),
        len(enc_blob),
        len(wgt_blob),
        crc,
    )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)
    return path


def load_lmx_binary(path: str | Path) -> dict[str, Any]:
    """Read and verify a .lmx binary file (reference loader matching the Rust binary spec)."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_STRUCT.size + 24:
        raise ValueError("Invalid .lmx file: truncated header")

    (
        magic,
        version,
        flags,
        encoder_id,
        activation_id,
        num_layers,
        _reserved,
        input_dim,
        output_dim,
        runtime_params,
        enc_blob_bytes,
        wgt_blob_bytes,
        expected_crc,
    ) = HEADER_STRUCT.unpack_from(raw, 0)

    if magic != LMX_MAGIC:
        raise ValueError(f"Invalid .lmx magic: {magic!r}")
    if version != LMX_VERSION:
        raise ValueError(f"Unsupported .lmx version: {version}")

    payload = raw[HEADER_STRUCT.size :]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"CRC32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}")

    fp16 = bool(flags & FLAG_FP16)
    log1p = bool(flags & FLAG_LOG1P)
    dtype = np.float16 if fp16 else np.float32
    itemsize = 2 if fp16 else 4

    off = 0
    bvals = struct.unpack_from("<6f", payload, off)
    off += 24
    aabb_min = np.array(bvals[:3], dtype=np.float32)
    aabb_max = np.array(bvals[3:], dtype=np.float32)

    enc_blob = payload[off : off + enc_blob_bytes]
    off += enc_blob_bytes
    wgt_blob = payload[off : off + wgt_blob_bytes]

    enc_type = ENCODER_NAMES[encoder_id]
    enc_meta: dict[str, Any] = {"type": enc_type}
    if enc_type == "positional":
        inc, bp, bn, bl = struct.unpack_from("<BBBB", enc_blob, 0)
        enc_meta.update(
            {
                "include_input": bool(inc),
                "num_bands_pos": bp,
                "num_bands_normal": bn,
                "num_bands_light": bl,
            }
        )
    elif enc_type == "fourier":
        inc, n_feat = struct.unpack_from("<BH", enc_blob, 0)
        basis = np.frombuffer(enc_blob, dtype=dtype, count=n_feat * 9, offset=3).reshape(n_feat, 9)
        enc_meta.update(
            {
                "include_input": bool(inc),
                "fourier_features": n_feat,
                "basis": basis.astype(np.float32),
            }
        )

    woff = 0
    layer_dims: list[tuple[int, int]] = []
    for _ in range(num_layers):
        in_d, out_d = struct.unpack_from("<HH", wgt_blob, woff)
        woff += 4
        layer_dims.append((in_d, out_d))

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for in_d, out_d in layer_dims:
        w_n = out_d * in_d
        w = np.frombuffer(wgt_blob, dtype=dtype, count=w_n, offset=woff).reshape(out_d, in_d).astype(np.float32)
        woff += w_n * itemsize
        b = np.frombuffer(wgt_blob, dtype=dtype, count=out_d, offset=woff).astype(np.float32)
        woff += out_d * itemsize
        layers.append((w, b))

    return {
        "version": version,
        "fp16": fp16,
        "target_transform": "log1p" if log1p else "linear",
        "encoder": enc_meta,
        "activation": ACTIVATION_NAMES[activation_id],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "runtime_params": runtime_params,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "layers": layers,
        "file_bytes": len(raw),
    }


def eval_lmx_numpy(
    lmx: dict[str, Any],
    pos: np.ndarray,
    normal: np.ndarray,
    light_pos: np.ndarray,
) -> np.ndarray:
    """Reference CPU/NumPy evaluation matching the exported GLSL shader math exactly."""
    span = np.maximum(lmx["aabb_max"] - lmx["aabb_min"], 1e-6)
    pos_n = 2.0 * (pos - lmx["aabb_min"]) / span - 1.0
    lgt_n = 2.0 * (light_pos - lmx["aabb_min"]) / span - 1.0

    enc = lmx["encoder"]
    if enc["type"] == "identity":
        h = np.concatenate([pos_n, normal, lgt_n], axis=-1)
    elif enc["type"] == "positional":
        parts: list[np.ndarray] = []
        if enc["include_input"]:
            parts.extend([pos_n, normal, lgt_n])
        for arr, n_bands in (
            (pos_n, enc["num_bands_pos"]),
            (normal, enc["num_bands_normal"]),
            (lgt_n, enc["num_bands_light"]),
        ):
            for k in range(n_bands):
                arg = arr * (math.pi * (2.0 ** k))
                parts.append(np.sin(arg))
                parts.append(np.cos(arg))
        h = np.concatenate(parts, axis=-1)
    elif enc["type"] == "fourier":
        raw = np.concatenate([pos_n, normal, lgt_n], axis=-1)
        proj = raw @ enc["basis"].T
        f_enc = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        h = np.concatenate([raw, f_enc], axis=-1) if enc["include_input"] else f_enc
    else:
        raise ValueError(enc["type"])

    act = lmx["activation"]
    layers = lmx["layers"]
    for idx, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if idx < len(layers) - 1:
            if act == "relu":
                h = np.maximum(h, 0.0)
            elif act == "leaky_relu":
                h = np.where(h >= 0.0, h, 0.05 * h)
            elif act == "silu":
                h = h / (1.0 + np.exp(-np.clip(h, -40.0, 40.0)))
            elif act == "sin":
                h = np.sin(h)

    h = np.maximum(h, 0.0)
    if lmx["target_transform"] == "log1p":
        return np.expm1(np.minimum(h, LOG1P_MAX_RAW))
    return h


def _format_float_array(name: str, values: np.ndarray, per_line: int = 8) -> str:
    flat = values.reshape(-1)
    lines = [f"const float {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(f"{float(v):.6f}" for v in flat[i : i + per_line])
        comma = "," if (i + per_line) < flat.size else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _glsl_activation_expr(var: str, act: str) -> str:
    if act == "relu":
        return f"max({var}, 0.0)"
    if act == "leaky_relu":
        return f"({var} >= 0.0 ? {var} : 0.05 * {var})"
    if act == "silu":
        return f"({var} / (1.0 + exp(-{var})))"
    if act == "sin":
        return f"sin({var})"
    raise ValueError(f"Unsupported GLSL activation: {act!r}")


def export_godot_shader_include(
    model: NeuralGIModel,
    out_path: str | Path,
    quantize_fp16: bool = True,
    uniform_defaults: dict[str, float] | None = None,
) -> Path:
    """Emit a self-contained Godot 4 .gdshaderinc using pure standard GLSL math (no Tensor Cores)."""
    enc_spec = model.encoder.export_spec()
    enc_type = enc_spec["type"]
    if enc_type not in ENCODER_IDS:
        raise NotImplementedError(f"GLSL include export supports {list(ENCODER_IDS.keys())}; got {enc_type!r}")

    defaults = uniform_defaults or {}
    d_bounce = float(defaults.get("bounce_strength", 1.0))
    d_vib = float(defaults.get("vibrancy", 1.0))
    d_warm = float(defaults.get("warmth", 0.0))
    d_lift = float(defaults.get("shadow_lift", 0.0))

    cost = model.cost_report()
    aabb_min = model.aabb_min.detach().cpu().numpy().tolist()
    aabb_max = model.aabb_max.detach().cpu().numpy().tolist()
    in_dim = int(model.encoder.output_dim)
    hidden_dim = int(model.hidden_dim)
    n_layers = len(model.layers)

    lines: list[str] = [
        "// ==========================================================================",
        "// Lumax Neural Global Illumination — Auto-Generated Godot 4 Shader Include",
        "// Pure standard GLSL math (Forward+, Mobile, Compatibility; zero ray tracing,",
        "// zero Tensor Core / cooperative-matrix extensions).",
        f"// Architecture : encoder={enc_type}(d={in_dim}), mlp={model.hidden_layers}x{hidden_dim}({model.activation})",
        f"// Footprint    : {cost['runtime_params']:,} params | {cost['fp16_kb']:.2f} KB fp16 | {cost['total_macs_per_pixel']:,} MACs/px",
        f"// Output space : {model.target_transform} -> linear RGB indirect irradiance / PI",
        "// ==========================================================================",
        "",
        "group_uniforms lumax_gi;",
        f"uniform float gi_bounce_strength : hint_range(0.0, 4.0, 0.05) = {d_bounce:.4f};",
        f"uniform float gi_vibrancy : hint_range(0.0, 2.5, 0.05) = {d_vib:.4f};",
        f"uniform float gi_warmth : hint_range(-1.0, 1.0, 0.05) = {d_warm:.4f};",
        f"uniform float gi_shadow_lift : hint_range(0.0, 0.5, 0.005) = {d_lift:.4f};",
        "group_uniforms;",
        "",
        f"const vec3 LUMAX_AABB_MIN = vec3({aabb_min[0]:.6f}, {aabb_min[1]:.6f}, {aabb_min[2]:.6f});",
        f"const vec3 LUMAX_AABB_MAX = vec3({aabb_max[0]:.6f}, {aabb_max[1]:.6f}, {aabb_max[2]:.6f});",
        f"const int LUMAX_IN_DIM = {in_dim};",
        f"const int LUMAX_HIDDEN_DIM = {hidden_dim};",
        "",
    ]

    # Emit Fourier basis matrix if needed
    if enc_type == "fourier":
        basis = _quantize_array(model.encoder.basis.detach().cpu().numpy(), quantize_fp16).astype(np.float32)
        lines.append(f"const int LUMAX_FOURIER_F = {int(enc_spec['fourier_features'])};")
        lines.append(_format_float_array("LUMAX_FOURIER_B", basis))
        lines.append("")

    # Emit MLP weights & biases (quantized to fp16 values when quantize_fp16=True for exact parity with .lmx)
    for idx, layer in enumerate(model.layers):
        w = _quantize_array(layer.weight.detach().cpu().numpy(), quantize_fp16).astype(np.float32)
        b = _quantize_array(layer.bias.detach().cpu().numpy(), quantize_fp16).astype(np.float32)
        lines.append(f"// Layer {idx}: Linear({layer.in_features} -> {layer.out_features})")
        lines.append(_format_float_array(f"LUMAX_W{idx}", w))
        lines.append(_format_float_array(f"LUMAX_B{idx}", b))
        lines.append("")

    # Helper: coordinate normalization
    lines.extend(
        [
            "vec3 lumax_normalize_coord(vec3 x) {",
            "    return 2.0 * (x - LUMAX_AABB_MIN) / max(LUMAX_AABB_MAX - LUMAX_AABB_MIN, vec3(1e-6)) - 1.0;",
            "}",
            "",
            "// Raw neural network evaluation -> linear RGB indirect irradiance / PI",
            "vec3 lumax_eval_raw(vec3 world_pos, vec3 world_normal, vec3 light_pos) {",
            "    vec3 p = lumax_normalize_coord(world_pos);",
            "    vec3 n = normalize(world_normal);",
            "    vec3 l = lumax_normalize_coord(light_pos);",
            "",
            f"    float feat[{in_dim}];",
        ]
    )

    # Emit encoder body directly inside lumax_eval_raw for universal Godot 4 backend compatibility
    if enc_type == "identity":
        lines.extend(
            [
                "    feat[0] = p.x; feat[1] = p.y; feat[2] = p.z;",
                "    feat[3] = n.x; feat[4] = n.y; feat[5] = n.z;",
                "    feat[6] = l.x; feat[7] = l.y; feat[8] = l.z;",
            ]
        )
    elif enc_type == "positional":
        cursor = 0
        if enc_spec["include_input"]:
            lines.extend(
                [
                    "    feat[0] = p.x; feat[1] = p.y; feat[2] = p.z;",
                    "    feat[3] = n.x; feat[4] = n.y; feat[5] = n.z;",
                    "    feat[6] = l.x; feat[7] = l.y; feat[8] = l.z;",
                ]
            )
            cursor = 9
        for vec_name, n_bands in (
            ("p", int(enc_spec["num_bands_pos"])),
            ("n", int(enc_spec["num_bands_normal"])),
            ("l", int(enc_spec["num_bands_light"])),
        ):
            for k in range(n_bands):
                freq = math.pi * (2.0 ** k)
                lines.append(f"    {{ vec3 a = {vec_name} * {freq:.6f};")
                lines.append(f"      vec3 sa = sin(a); vec3 ca = cos(a);")
                lines.append(
                    f"      feat[{cursor}] = sa.x; feat[{cursor+1}] = sa.y; feat[{cursor+2}] = sa.z;"
                )
                lines.append(
                    f"      feat[{cursor+3}] = ca.x; feat[{cursor+4}] = ca.y; feat[{cursor+5}] = ca.z; }}"
                )
                cursor += 6
    elif enc_type == "fourier":
        n_f = int(enc_spec["fourier_features"])
        base_off = 9 if enc_spec["include_input"] else 0
        lines.extend(
            [
                "    float raw_in[9];",
                "    raw_in[0] = p.x; raw_in[1] = p.y; raw_in[2] = p.z;",
                "    raw_in[3] = n.x; raw_in[4] = n.y; raw_in[5] = n.z;",
                "    raw_in[6] = l.x; raw_in[7] = l.y; raw_in[8] = l.z;",
            ]
        )
        if enc_spec["include_input"]:
            lines.append("    for (int i = 0; i < 9; i++) { feat[i] = raw_in[i]; }")
        lines.extend(
            [
                f"    for (int f = 0; f < {n_f}; f++) {{",
                "        float proj = 0.0;",
                "        int row = f * 9;",
                "        for (int j = 0; j < 9; j++) { proj += LUMAX_FOURIER_B[row + j] * raw_in[j]; }",
                f"        feat[{base_off} + f] = sin(proj);",
                f"        feat[{base_off + n_f} + f] = cos(proj);",
                "    }",
            ]
        )

    # Emit MLP layers using ping-pong local float buffers
    lines.extend(
        [
            "",
            f"    float h_a[{hidden_dim}];",
            f"    float h_b[{hidden_dim}];",
            "",
        ]
    )
    act_expr = _glsl_activation_expr("acc", model.activation)

    # Layer 0: feat[in_dim] -> h_a[hidden_dim]
    l0 = model.layers[0]
    lines.extend(
        [
            f"    for (int o = 0; o < {l0.out_features}; o++) {{",
            "        float acc = LUMAX_B0[o];",
            f"        int row = o * {l0.in_features};",
            f"        for (int i = 0; i < {l0.in_features}; i++) {{ acc += LUMAX_W0[row + i] * feat[i]; }}",
            f"        h_a[o] = {act_expr};",
            "    }",
        ]
    )

    # Intermediate hidden layers
    src_buf, dst_buf = "h_a", "h_b"
    for idx in range(1, n_layers - 1):
        layer = model.layers[idx]
        lines.extend(
            [
                f"    for (int o = 0; o < {layer.out_features}; o++) {{",
                f"        float acc = LUMAX_B{idx}[o];",
                f"        int row = o * {layer.in_features};",
                f"        for (int i = 0; i < {layer.in_features}; i++) {{ acc += LUMAX_W{idx}[row + i] * {src_buf}[i]; }}",
                f"        {dst_buf}[o] = {act_expr};",
                "    }",
            ]
        )
        src_buf, dst_buf = dst_buf, src_buf

    # Output layer -> vec3
    last_idx = n_layers - 1
    last_layer = model.layers[last_idx]
    lines.extend(
        [
            "    float out_rgb[3];",
            "    for (int o = 0; o < 3; o++) {",
            f"        float acc = LUMAX_B{last_idx}[o];",
            f"        int row = o * {last_layer.in_features};",
            f"        for (int i = 0; i < {last_layer.in_features}; i++) {{ acc += LUMAX_W{last_idx}[row + i] * {src_buf}[i]; }}",
            "        out_rgb[o] = max(acc, 0.0);",
            "    }",
            "    vec3 rgb = vec3(out_rgb[0], out_rgb[1], out_rgb[2]);",
        ]
    )
    if model.target_transform == "log1p":
        lines.append(f"    return exp(min(rgb, vec3({LOG1P_MAX_RAW:.1f}))) - vec3(1.0);")
    else:
        lines.append("    return rgb;")
    lines.extend(
        [
            "}",
            "",
            "// Zero-overhead artistic post-model grading in linear RGB space",
            "vec3 lumax_grade_gi(vec3 linear_gi) {",
            "    vec3 c = linear_gi * gi_bounce_strength;",
            "    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));",
            "    c = max(mix(vec3(luma), c, gi_vibrancy), vec3(0.0));",
            "    vec3 warm_tint = vec3(1.0 + 0.25 * gi_warmth, 1.0 + 0.04 * gi_warmth, 1.0 - 0.25 * gi_warmth);",
            "    c = max(c * warm_tint, vec3(0.0));",
            "    c += vec3(gi_shadow_lift) * exp(-c * 3.0);",
            "    return c;",
            "}",
            "",
            "// Main entry point: returns graded linear RGB indirect irradiance / PI",
            "// Multiply by surface ALBEDO in your spatial shader.",
            "vec3 lumax_sample_gi(vec3 world_pos, vec3 world_normal, vec3 light_pos) {",
            "    return lumax_grade_gi(lumax_eval_raw(world_pos, world_normal, light_pos));",
            "}",
            "",
        ]
    )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_godot_example_shader(out_path: str | Path, include_filename: str = "lumax_gi.gdshaderinc") -> Path:
    """Emit a ready-to-use Godot 4 spatial shader showing how to plug in Lumax GI."""
    code = f"""shader_type spatial;
render_mode diffuse_lambert, specular_schlick_ggx;

// Self-contained Lumax Neural GI include (place alongside this shader in Godot)
#include "{include_filename}"

uniform vec3 base_albedo : source_color = vec3(0.74, 0.74, 0.74);
uniform vec3 dynamic_light_world_pos = vec3(-0.5, 0.6, -0.1);
uniform bool indirect_only_preview = false;

varying vec3 v_world_pos;
varying vec3 v_world_normal;

void vertex() {{
    v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
    v_world_normal = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
}}

void fragment() {{
    vec3 gi = lumax_sample_gi(v_world_pos, normalize(v_world_normal), dynamic_light_world_pos);
    ALBEDO = indirect_only_preview ? vec3(0.0) : base_albedo;
    // Inject hardware-agnostic multi-bounce indirect radiance (zero ray tracing!)
    EMISSION = base_albedo * gi;
}}
"""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


@torch.no_grad()
def run_export(cfg: dict[str, Any], checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Run Stage 4 export: .lmx binary + Godot 4 .gdshaderinc + parity verification."""
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    ckpt = Path(checkpoint_path or cfg["train"]["checkpoint_path"])
    model, _meta = load_checkpoint(ckpt, device=device)
    model.eval()

    ecfg = cfg["export"]
    quantize_fp16 = bool(ecfg.get("quantize_fp16", True))
    bin_path = export_lmx_binary(model, ecfg["binary_path"], quantize_fp16=quantize_fp16)
    inc_path = export_godot_shader_include(
        model,
        ecfg["shader_include_path"],
        quantize_fp16=quantize_fp16,
        uniform_defaults=ecfg.get("uniform_defaults"),
    )
    example_path = export_godot_example_shader(
        ecfg["example_shader_path"], include_filename=inc_path.name
    )

    # Verify binary round-trip + fp16 quantization parity against PyTorch
    lmx = load_lmx_binary(bin_path)
    rng = np.random.default_rng(123)
    aabb_min = lmx["aabb_min"]
    aabb_max = lmx["aabb_max"]
    test_pos = rng.uniform(aabb_min, aabb_max, size=(256, 3)).astype(np.float32)
    test_nrm = rng.normal(size=(256, 3)).astype(np.float32)
    test_nrm /= np.maximum(np.linalg.norm(test_nrm, axis=-1, keepdims=True), 1e-6)
    test_lgt = rng.uniform(aabb_min * 0.7, aabb_max * 0.7, size=(256, 3)).astype(np.float32)

    pred_pt = (
        model(
            torch.from_numpy(test_pos).to(device),
            torch.from_numpy(test_nrm).to(device),
            torch.from_numpy(test_lgt).to(device),
        )
        .cpu()
        .numpy()
    )
    pred_lmx = eval_lmx_numpy(lmx, test_pos, test_nrm, test_lgt)
    max_abs_err = float(np.abs(pred_pt - pred_lmx).max())
    mean_abs_err = float(np.abs(pred_pt - pred_lmx).mean())

    bin_kb = bin_path.stat().st_size / 1024.0
    inc_kb = inc_path.stat().st_size / 1024.0
    report = {
        "binary_path": str(bin_path),
        "binary_bytes": bin_path.stat().st_size,
        "binary_kb": round(bin_kb, 3),
        "shader_include_path": str(inc_path),
        "shader_include_kb": round(inc_kb, 3),
        "example_shader_path": str(example_path),
        "quantize_fp16": quantize_fp16,
        "parity_max_abs_err": round(max_abs_err, 6),
        "parity_mean_abs_err": round(mean_abs_err, 6),
        "cost": model.cost_report(),
    }
    report_path = bin_path.with_suffix(".export.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[stage 4: export] binary={bin_path} ({bin_kb:.2f} KB, {bin_path.stat().st_size} B) | "
        f"godot_inc={inc_path} ({inc_kb:.2f} KB) | "
        f"fp16 parity max_err={max_abs_err:.6f} mean_err={mean_abs_err:.6f}"
    )
    return report

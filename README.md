# Lumax — Hardware-Agnostic Neural Global Illumination Baker

Foundational R&D codebase for baking dynamic multi-bounce indirect lighting into kilobyte-scale micro-MLPs that evaluate at runtime with **zero ray tracing**, **zero Tensor Core extensions**, and **pure standard GLSL math**.

## Architecture

All 4 modular CLI stages share a single `config.yaml` and run identically on an **8GB CUDA GPU**, **CPU**, or headlessly on **Google Colab**:

| Stage | Command | Description |
| :--- | :--- | :--- |
| **1. Generate** | `python -m lumax generate` | Procedural colored Cornell Box + interior occluder wall + movable point light, sampled via a vectorized PyTorch multi-bounce path tracer for **indirect-only** `(position, normal, light_pos) -> linear RGB (irradiance / pi)` |
| **2. Train** | `python -m lumax train` | Trains an intentionally overfitted micro-MLP (default 2–3 hidden layers, 16–32 neurons) with swappable spatial encoders (`positional`, `fourier`, `identity`, and `chunked_grid` hook) while logging exact parameter count, `fp16` KB footprint, and per-pixel MACs |
| **3. Eval & Sweep** | `python -m lumax eval` / `python -m lumax sweep` | Renders side-by-side PNG triptychs (`Path-Traced GT \| Neural GI \| Abs Error Heatmap`) on unseen viewpoints & light positions with PSNR, SSIM, and Grad-MAE sharpness scores; `sweep` outputs a Pareto table (`sweep_summary.md` & `.csv`) |
| **4. Export** | `python -m lumax export` | Compiles weights into a tiny little-endian `.lmx` binary (with CRC32 + Rust-ready layout) and a self-contained Godot 4 `.gdshaderinc` with zero-overhead linear post-model uniforms (`gi_bounce_strength`, `gi_vibrancy`, `gi_warmth`, `gi_shadow_lift`) |

## Quickstart

### Local (Laptop CPU or 8GB CUDA GPU)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate

# For an 8GB CUDA GPU, install CUDA PyTorch first:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Fast end-to-end smoke test (~seconds on CPU)
python -m lumax all --smoke --with-sweep

# Full GPU bake & sweep (uses config.yaml defaults)
python -m lumax all --with-sweep
```

### Headless Google Colab
```python
!git clone <your-repo-url> Lumax && cd Lumax && pip install -q pyyaml pillow
%cd Lumax
!python -m lumax all --with-sweep --device cuda
```

### CLI Overrides
Any key in `config.yaml` can be overridden via `-o key=val`:
```bash
python -m lumax train -o model.encoder.type=fourier -o model.mlp.hidden_dim=24 -o train.epochs=150
python -m lumax eval --checkpoint outputs/sweep/pos4_3x32/model.pt
python -m lumax export --checkpoint outputs/sweep/pos4_3x32/model.pt
```

## Future Chunked Feature Grids
`lumax/encoders.py` defines a clean `SpatialEncoder` contract and includes `ChunkedFeatureGridEncoder` (`model.encoder.type: chunked_grid`), allowing jointly trained chunked 3D feature grids to drop directly into the shared micro-MLP decoder without modifying the trainer or evaluator.

## `.lmx` Binary Weight Spec (Rust-Ready)
All fields are **little-endian**. Immediately after the 32-byte header follow:
1. `aabb_min: [f32; 3]`, `aabb_max: [f32; 3]` (24 bytes)
2. `encoder_blob` (`enc_blob_bytes` bytes: empty for `identity`; `u8 include_input, u8 bands_pos, u8 bands_nrm, u8 bands_lgt` for `positional`; `u8 include_input, u16 F` + `F * 9` `f16`/`f32` basis matrix for `fourier`)
3. `weights_blob` (`wgt_blob_bytes` bytes: `num_layers` pairs of `(u16 in_dim, u16 out_dim)` followed by row-major `W[out, in]` and `b[out]` in `f16` when `flags & 1 != 0`, else `f32`)

```rust
#[repr(C, packed)]
pub struct LmxHeader {
    pub magic: [u8; 4],       // b"LMX1"
    pub version: u16,         // 1
    pub flags: u16,           // bit 0: fp16 weights, bit 1: log1p output transform
    pub encoder_id: u8,       // 0 = identity, 1 = positional, 2 = fourier
    pub activation_id: u8,    // 0 = relu, 1 = leaky_relu, 2 = silu, 3 = sin
    pub num_layers: u8,
    pub reserved: u8,
    pub input_dim: u16,
    pub output_dim: u16,      // 3 (linear RGB)
    pub runtime_params: u32,
    pub enc_blob_bytes: u32,
    pub wgt_blob_bytes: u32,
    pub payload_crc32: u32,   // CRC32 over (bounds + enc_blob + wgt_blob)
}
```

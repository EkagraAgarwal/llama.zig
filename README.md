# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead (BDA + push constants).

## Features

- **GGUF v3** parser with architecture detection (`llama`, `granite`)
- **BF16 / F32 / Q8_0** host-side dequantization to FP32 VRAM weights
- **Full transformer forward pass**: RMSNorm, Q/K/V projections, RoPE, KV cache, scaled dot-product attention, SwiGLU FFN, LM head
- **Autoregressive generation** with temperature + top-p sampling
- **Deterministic sampler RNG lifecycle** with top-k, top-p, min-p, and typical-p filtering
- **GLSL → SPIR-V** kernels via `glslangValidator` (AMDVLK-safe)

## Build Requirements

- **Zig 0.14.0**
- **Vulkan SDK** (`glslangValidator`, loader)
- Windows or Linux with a Vulkan 1.2+ GPU (BDA required)

## Build

```bash
zig build -Doptimize=ReleaseFast
```

Shaders are compiled automatically when `glslangValidator` is on `PATH`; prebuilt `.spv` files are used as fallback.

## Run

```bash
./zig-out/bin/llama.zig --model models/granite-4.0-350m-BF16.gguf --prompt "The future of AI is" --max-tokens 32 --temperature 0.8
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--model` | Path to `.gguf` model |
| `--prompt` | Input text |
| `--max-tokens` | Tokens to generate after prompt (default 64) |
| `--temperature` | Sampling temperature (default 0.8) |

## Architecture

```
main.zig
  ├── model.zig        (GGUF metadata + model config)
  ├── sampler.zig      (temperature / top-p sampling)
  ├── tokenizer.zig    (BPE encode/decode)
  ├── weights.zig      (dequant + upload to GPU)
  ├── vulkan_backend.zig (BDA buffers, pipelines)
  ├── compute_graph.zig (DAG + dispatcher)
  └── root.zig         (root comptime struct for all ops)
```

- Fused GPU dequant matmul (avoid full FP32 weight VRAM)
- Flash-attention style tiled attention kernel
- Multi-GPU model splitting
- Chat templates for instruct models

## Optimizations

- **Native Q6_K GPU path**: The `matmul_q6_k_bda.glsl` shader provides on-the-fly 6-bit block dequantization during matrix multiplication, keeping q6_k tensors compressed at ~6.5 bits/element in VRAM. This applies to both matvec (embedding lookup, attention projections) and full matmul (LM head).
- **Compressed weight upload**: `isNativeQuantType` enables q6_k weights to upload as packed binary directly to GPU rather than dequantizing to f32 on CPU first.

## Compatibility Matrix

### Supported Model Types

Only the following model types are currently supported:
- **BF16 / F32 / F16** (native precision)
- **Q8_0** (native 8-bit quantization)
- **Q4_K** (native K-quantization)
- **Q6_K** (native K-quantization — 6-bit, ~6.5 bits/element)

All other quantization types (`q4_0`, `q5_0`, `q2_k`, `q3_k`, `q5_k`, `q8_k`) are not supported.

### Model architecture and tokenizer/sampler parity

| Capability | llama.zig | llama.cpp Vulkan reference parity | Notes |
|------|------|------|------|
| `llama` architecture | Supported | Matched | Base decoder graph path |
| `granite` architecture | Supported | Matched | BOS metadata + Granite role token handling |
| Special token passthrough (`<|...|>`) | Supported | Matched | Encoder preserves known special tokens |
| Sampler: `top_k`, `top_p`, `min_p`, `typical_p` | Supported | Matched | Stateful deterministic RNG across decode steps |

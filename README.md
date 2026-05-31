# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead (BDA + push constants).

## Features

- **GGUF v3** parser with architecture detection (`llama`, `granite`, `gemma`, `qwen`)
- **BF16 / F32 / Q8_0** host-side dequantization to FP32 VRAM weights
- **Full transformer forward pass**: RMSNorm, Q/K/V projections, RoPE, KV cache, scaled dot-product attention, SwiGLU FFN, LM head
- **Autoregressive generation** with temperature + top-p sampling
- **Deterministic sampler RNG lifecycle** with top-k, top-p, min-p, and typical-p filtering
- **GLSL → SPIR-V** kernels via `glslangValidator` (AMDVLK-safe)
- **Automatic chat template detection** for instruct models: Llama 3, Granite, Gemma, Qwen (ChatML), Llama 2 (Mistral), with automatic bypass (raw completion mode) for base models

## Build Requirements

- **Zig 0.14.0**
- **Vulkan SDK** (`glslangValidator`, loader)
- Windows or Linux with a Vulkan 1.2+ GPU (BDA required)

## Build

```bash
zig build clean
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
| `--ctx-size` | Context window size (default 8192) |
| `--top-k` | Top-k sampling (default 0 = disabled) |
| `--top-p` | Top-p nucleus sampling (default 0.9) |
| `--seed` | RNG seed for reproducibility |

**Note**: Chat template mode is enabled by default for instruct-tuned models. Base models are automatically detected (based on the absence of GGUF chat template metadata and instruct-specific control tokens) and execute in raw text autocomplete mode.

## Architecture

```
main.zig
  ├── model.zig        (GGUF metadata + model config)
  ├── chat.zig         (chat template detection + formatting)
  ├── sampler.zig      (temperature / top-p sampling)
  ├── tokenizer.zig    (BPE encode/decode, special token detection)
  ├── weights.zig      (dequant + upload to GPU)
  ├── vulkan_backend.zig (BDA buffers, pipelines)
  ├── compute_graph.zig (DAG + dispatcher)
  └── root.zig         (root comptime struct for all ops)
```

- Fused GPU dequant matmul (avoid full FP32 weight VRAM)
- Fused MLP Gate/Up and Attention QKV projection weights (dynamic type-safety fallback)
- CPU-side address offsetting for zero-cost sub-matrix query dispatches
- Flash-attention style tiled attention kernel
- Multi-GPU model splitting
- Architecture-aware chat templates for instruct models (Llama 3, Granite, Gemma, Qwen, Llama 2)

## Optimizations

- **Dynamic Weight Fusion with Type-Safety Fallback**: Concatenates constituent attention (Q/K/V) and MLP Gate/Up weights at load-time to reduce matmul dispatches. Because GGUF models can contain mixed quantization types across layers (e.g., Llama 3.2 Q4_K_M stores V as `Q6_K` but Q and K as `Q4_K`), fusions check type compatibility layer-by-layer and fall back to separate execution paths if types mismatch to prevent GPU memory corruption.
- **CPU-Side Address Offsetting**: Expanded push constants with offset parameters (`p6-p8`). Consumer nodes (`rope`, `kv_write`, `attention`, and `silu_mul`) query memory from offsets directly, eliminating the need for copy operations or shader adjustments.
- **Native Q4_K and Q6_K GPU path**: The `matmul` and `matvec` shaders provide on-the-fly dequantization during matrix multiplication. The `matvec` shaders use a thread-per-column `BlockDot` loop that is hardware-independent and correctly handles different subgroup sizes (e.g., AMD vs. NVIDIA).
- **F16 KV Cache**: The KV cache uses 16-bit floats (f16) packed into 32-bit integers, halving VRAM requirements for the context window while maintaining compatibility with all Vulkan 1.2 devices.
- **Compressed weight upload**: `isNativeQuantType` enables q6_k/q4_k weights to upload as packed binary directly to GPU rather than dequantizing to f32 on CPU first.
- **Batched GPU submits**: The decode loop combines embedding lookup and graph execution into a single command buffer submission per token, reducing CPU-GPU synchronization overhead.

### Performance

Measured on AMD Radeon RX 7700S GPU:

| Model | Quant | Generation Speed | Optimizations |
|-------|-------|-----------------|---------------|
| Llama-3.2-3B-Instruct | Q4_K_M | **~10.4 t/s** | Gate/Up Fused, QKV Fallback |
| Llama-3.2-1B | Q4_K_M | **~17.2 t/s** | Gate/Up Fused, QKV Fallback |
| Llama-3.2-1B | Q8_0 | **~5.9 t/s** | Fully Fused QKV and MLP |
| Dolphin Llama-3.1-8B | Q4_K_M | **~7.8 t/s** | Gate/Up Fused, QKV Fallback |
| Granite-4.0-350m | BF16 | ~35 t/s | Fully Fused |

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
| `gemma` architecture | Supported | Matched | Gemma 4 E2B and variants |
| `qwen` architecture | Supported | Matched | Qwen 3.5 and ChatML format |
| Chat template detection (Llama 3, Granite, Gemma, Qwen, Llama 2) | Supported | Matched | Auto-detected from GGUF metadata or architecture |
| Special token passthrough (`<|...|>`) | Supported | Matched | Encoder preserves known special tokens |
| Sampler: `top_k`, `top_p`, `min_p`, `typical_p` | Supported | Matched | Stateful deterministic RNG across decode steps |

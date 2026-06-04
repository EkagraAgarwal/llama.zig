# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead (BDA + push constants).

## Features

- **GGUF v3** parser with architecture detection (`llama`, `granite`, `gemma`, `qwen`, `qwen35`)
- **BF16 / F32 / F16** host-side dequantization; **Q8_0, Q4_K, Q5_K, Q6_K** native GPU quantized path with on-the-fly dequant
- **Full transformer forward pass**: RMSNorm, Q/K/V projections, RoPE, KV cache, scaled dot-product attention, SwiGLU/GELU FFN, LM head
- **Qwen 3.5 hybrid SSM architecture**: State Space Model (SSM) with gated delta net layers alternating with full-attention layers
- **M-RoPE (Multi-dimensional Rotary Position Embedding)** for Qwen 3.5 models
- **Autoregressive generation** with temperature, top-k, top-p, min-p, and typical-p sampling
- **Deterministic sampler RNG lifecycle** with repetition penalty (256-token rolling window)
- **GLSL → SPIR-V** kernels via `glslangValidator` (AMDVLK-safe)
- **Automatic chat template detection** for instruct models: Llama 3, Granite, Gemma, Qwen (ChatML), Llama 2 (Mistral), with automatic bypass (raw completion mode) for base models
- **Memory-mapped model loading** for fast startup and reduced memory footprint

## Build Requirements

- **Zig 0.16.0**
- **Vulkan SDK** (`glslangValidator`, loader)
- Windows with a Vulkan 1.2+ GPU (BDA required)

## Build

```bash
zig build clean
zig build -Doptimize=ReleaseFast
```

Shaders are compiled automatically when `glslangValidator` is on `PATH`; prebuilt `.spv` files are used as fallback.

## Run

```bash
./zig-out/bin/llama.zig --model models/Qwen3.5-4B-Q4_K_M.gguf --prompt "Hello" --max-tokens 32 --temperature 0.8
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--model` | Path to `.gguf` model |
| `--prompt` | Input text |
| `--max-tokens` | Tokens to generate after prompt (default 64) |
| `--temperature` | Sampling temperature (default 0.8) |
| `--ctx-size` | Context window size (default model's trained context, capped at 8192) |
| `--top-k` | Top-k sampling (default 0 = disabled) |
| `--top-p` | Top-p nucleus sampling (default 0.9) |
| `--min-p` | Min-p sampling (default 0.0 = disabled) |
| `--seed` | RNG seed for reproducibility (default 0 = random) |
| `--chat` | Enable chat mode (default true) |
| `--no-chat` | Disable chat mode (raw completion) |
| `--verbose` | Enable verbose debug output |
| `--debug-logits` | Dump top-N logits each decode step |
| `--inspect-block` | Inspect compute graph block structure |
| `--prefill-chunk` | Chunk size for batched prefill (0 = full batch) |
| `--no-gpu-embed` | Disable GPU embedding lookup (fallback to CPU path) |
| `--report-json` | Output inference metrics as JSON |

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
  ├── ssm_state.zig    (SSM state management for Qwen 3.5)
  └── root.zig         (root comptime struct for all ops)
```

## Optimizations

- **Dynamic Weight Fusion with Type-Safety Fallback**: Concatenates constituent attention (Q/K/V) and MLP Gate/Up weights at load-time to reduce matmul dispatches. Because GGUF models can contain mixed quantization types across layers (e.g., Llama 3.2 Q4_K_M stores V as `Q6_K` but Q and K as `Q4_K`), fusions check type compatibility layer-by-layer and fall back to separate execution paths if types mismatch to prevent GPU memory corruption.
- **CPU-Side Address Offsetting**: Expanded push constants with offset parameters (`p6-p8`). Consumer nodes (`rope`, `kv_write`, `attention`, and `silu_mul`) query memory from offsets directly, eliminating the need for copy operations or shader adjustments.
- **Native Q4_K, Q5_K, and Q6_K GPU path**: The `matmul` and `matvec` shaders provide on-the-fly dequantization during matrix multiplication. The `matvec` shaders use a thread-per-column `BlockDot` loop that is hardware-independent and correctly handles different subgroup sizes (e.g., AMD vs. NVIDIA).
- **F16 KV Cache**: The KV cache uses 16-bit floats (f16) packed into 32-bit integers, halving VRAM requirements for the context window while maintaining compatibility with all Vulkan 1.2 devices.
- **Compressed weight upload**: `isNativeQuantType` enables q6_k/q4_k weights to upload as packed binary directly to GPU rather than dequantizing to f32 on CPU first.
- **Batched GPU submits**: The decode loop combines embedding lookup and graph execution into a single command buffer submission per token, reducing CPU-GPU synchronization overhead.
- **Memory-mapped model loading**: Models can be loaded via mmap for faster startup and reduced RAM usage.

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

Only the following quantization types are directly supported for GPU inference:
- **F32 / F16 / BF16** (native precision, host-side dequant to f32)
- **Q8_0** (native 8-bit quantization, GPU dequant matmul)
- **Q4_K** (native K-quantization, GPU dequant matmul + matvec + embedding lookup)
- **Q5_K** (native K-quantization — 5-bit with 16-element min offset, GPU dequant matmul + matvec + embedding lookup)
- **Q6_K** (native K-quantization — 6-bit, ~6.5 bits/element, GPU dequant matmul + matvec + embedding lookup)
- **Q4_0** (falls back to f16 GPU matmul via host-side dequant)
- **Q4_1** (falls back to f32 GPU matmul via host-side dequant)

Other quantization types (`q5_0`, `q2_k`, `q3_k`, `q8_k`) are not supported.

### Model Architecture Support

| Architecture | Status | Notes |
|-------------|--------|-------|
| `llama` | Supported | Base decoder transformer |
| `granite` | Supported | BOS metadata + Granite role token handling |
| `gemma` | Supported | Gemma 4 E2B and variants |
| `qwen` | Supported | Qwen 2/2.5 series |
| `qwen35` | Supported | Qwen 3.5 with hybrid SSM architecture (M-RoPE, gated delta nets) |

### Features

| Capability | Status | Notes |
|-----------|--------|-------|
| Chat template detection (Llama 3, Granite, Gemma, Qwen, Llama 2) | Supported | Auto-detected from GGUF metadata or architecture |
| Qwen 3.5 M-RoPE | Supported | 4-dimension rotary position embedding |
| Qwen 3.5 SSM (conv1d, delta_net, gated_norm) | Supported | CPU step for decode, GPU for prefill |
| Special token passthrough (`<|...|>`) | Supported | Encoder preserves known special tokens |
| Sampler: `top_k`, `top_p`, `min_p`, `typical_p` | Supported | Stateful deterministic RNG across decode steps |
| Memory-mapped model loading | Supported | `loadModelMmap` for fast startup |
| Repetition penalty | Supported | 256-token rolling window |
| Flash attention | Supported | Tiled accumulation kernel |
| Fused MLP Gate/Up | Supported | Dynamic type-safety fallback |
| Fused QKV projections | Supported | Dynamic type-safety fallback |

## Supported Quantization Types

| Format | Elements/Block | Bytes/Block | GPU Path |
|--------|---------------|-------------|----------|
| `f32` | 1 | 4 | Uploaded as f32 |
| `f16` | 1 | 2 | Uploaded as f16 |
| `bf16` | 1 | 2 | Host dequant → f32 upload |
| `q8_0` | 32 | 34 | Native GPU (matmul + matvec) |
| `q4_0` | 32 | 18 | Host dequant → f16 upload, f16 GPU matmul/matvec |
| `q4_1` | 32 | 20 | Host dequant → f32 upload, f32 GPU matmul/matvec |
| `q4_k` | 256 | 144 | Native GPU (matmul + matvec + get_rows) |
| `q5_k` | 256 | 176 | Native GPU (matmul + matvec + get_rows) |
| `q6_k` | 256 | 210 | Native GPU (matmul + matvec + get_rows) |
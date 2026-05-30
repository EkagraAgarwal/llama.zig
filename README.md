# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead (BDA + push constants).

## Features

- **GGUF v3** parser with architecture detection (`llama`, `granite`, `gemma`, `qwen`)
- **BF16 / F32 / Q8_0** host-side dequantization to FP32 VRAM weights
- **Full transformer forward pass**: RMSNorm, Q/K/V projections, RoPE, KV cache, scaled dot-product attention, SwiGLU FFN, LM head
- **Autoregressive generation** with temperature + top-p sampling
- **Deterministic sampler RNG lifecycle** with top-k, top-p, min-p, and typical-p filtering
- **GLSL → SPIR-V** kernels via `glslangValidator` (AMDVLK-safe)
- **Automatic chat template detection** for instruct models: Llama 3, Granite, Gemma, Qwen (ChatML), Llama 2 (Mistral)

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
| `--ctx-size` | Context window size (default 8192) |
| `--top-k` | Top-k sampling (default 0 = disabled) |
| `--top-p` | Top-p nucleus sampling (default 0.9) |
| `--seed` | RNG seed for reproducibility |

**Note**: Chat mode is enabled by default — prompts are automatically formatted with the model's chat template so instruction-tuned models respond as assistants rather than raw text completions.

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
- Flash-attention style tiled attention kernel
- Multi-GPU model splitting
- Architecture-aware chat templates for instruct models (Llama 3, Granite, Gemma, Qwen, Llama 2)

## Optimizations

- **Native Q4_K and Q6_K GPU path**: The `matmul` and `matvec` shaders provide on-the-fly dequantization during matrix multiplication. The `matvec` shaders use a thread-per-column `BlockDot` loop that is hardware-independent and correctly handles different subgroup sizes (e.g., AMD vs. NVIDIA).
- **F16 KV Cache**: The KV cache uses 16-bit floats (f16) packed into 32-bit integers, halving VRAM requirements for the context window while maintaining compatibility with all Vulkan 1.2 devices.
- **Compressed weight upload**: `isNativeQuantType` enables q6_k weights to upload as packed binary directly to GPU rather than dequantizing to f32 on CPU first.
- **Batched GPU submits**: The decode loop combines embedding lookup and graph execution into a single command buffer submission per token, reducing CPU-GPU synchronization overhead.

### Performance

| Model | Quant | Generation Speed |
|-------|-------|-----------------|
| Llama-3.2-3B-Instruct | Q4_K_M | ~8.3 t/s |
| Llama-3.2-1B | Q4_K_M | ~14.3 t/s |
| Granite-4.0-350m | BF16 | ~35 t/s |

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

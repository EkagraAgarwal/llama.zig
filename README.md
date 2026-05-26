# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead.

## Project Status: COMPLETED

`llama.zig` has successfully implemented the full inference pipeline, from parsing GGUF models to generating text on the GPU.

### Key Features:
- **Pure Zig GGUF Parser**: Full support for GGUF v3, including BF16 and K-Quants.
- **Vulkan BDA Architecture**: Uses **Buffer Device Address (BDA)** and **Push Constants** to minimize dispatch latency.
- **Optimized Math Kernels**: Hand-written GLSL kernels for `MatMul`, `RMSNorm`, `RoPE`, and `Softmax`.
- **Pure Zig BPE Tokenizer**: High-performance implementation of Byte Pair Encoding (BPE) for Llama models.
- **Dynamic Compute Graph**: A DAG-based execution system that automatically manages memory and synchronization.

## Build Requirements
- **Zig 0.16.0** (Nightly)
- **Vulkan SDK** (For `glslangValidator` and validation layers)

## Quick Start
```bash
# Build the project
zig build -Doptimize=ReleaseFast

# Run inference
./zig-out/bin/llama.zig --model models/your-model.gguf --prompt "The capital of France is"
```

## Performance
`llama.zig` is designed for maximum throughput by bypassing the overhead of traditional descriptor sets. By utilizing physical GPU pointers, it achieves performance comparable to or exceeding native C++ implementations in specific workloads.

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml`.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe bindings.

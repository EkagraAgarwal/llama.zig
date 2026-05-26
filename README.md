# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead.

## Project Status: COMPLETED (Final Prototype)

`llama.zig` has successfully implemented the full inference pipeline, from parsing GGUF models to generating streaming text on the GPU.

### Key Accomplishments:
- **Zero-Overhead Dispatch**: Uses **Buffer Device Address (BDA)** and **Push Constants** to bypass the latency of traditional Vulkan descriptor sets.
- **Pure Zig Infrastructure**: GGUF v3 parser and BPE Tokenizer written from scratch in idiomatic Zig.
- **Hardware Stability**: Custom HLSL-to-SPIR-V pipeline ensures compatibility with sensitive drivers like **AMDVLK** on Windows.
- **Dynamic Compute Graph**: A flexible DAG-based system that builds Llama transformer blocks and manages GPU memory automatically.

## Build Requirements
- **Zig 0.16.0** (Nightly)
- **Vulkan SDK** (For `glslangValidator` and validation layers)

## Quick Start
```bash
# Build the project
zig build -Doptimize=ReleaseFast

# Run streaming inference
./zig-out/bin/llama.zig --model models/nomic-embed.gguf --prompt "The future of AI is"
```

## Future Work
- **4-bit Quantization**: Implement `q4_K` dequantization kernels.
- **KV Cache**: Persist key-value tensors across generation steps for faster long-context inference.
- **Multi-GPU**: Split large models across multiple Vulkan devices.

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml`.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe bindings.

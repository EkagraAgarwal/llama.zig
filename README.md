# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend with near-zero CPU overhead.

## Project Status: COMPLETED (Final Prototype)

`llama.zig` has successfully implemented the full inference pipeline, from parsing GGUF models to generating streaming text on the GPU.

### Key Accomplishments:
- **Zero-Overhead Dispatch**: Uses **Buffer Device Address (BDA)** and **Push Constants** to bypass the latency of traditional Vulkan descriptor sets.
- **Direct Vulkan Dispatch**: Bypasses abstraction layers to directly call Vulkan dispatch tables, eliminating stack-overflow segfaults on deep layer graphs.
- **Pure Zig Infrastructure**: GGUF v3 parser and BPE Tokenizer written from scratch in idiomatic Zig.
- **Hardware Stability**: Custom GLSL-to-SPIR-V pipeline ensures compatibility with sensitive drivers like **AMDVLK** on Windows, circumventing the experimental Zig SPIR-V backend.
- **Dynamic Compute Graph**: A flexible DAG-based system that builds Llama and Granite transformer blocks (RMSNorm, RoPE, SwiGLU) and manages multi-layer GPU memory automatically.

## Build Requirements
- **Zig 0.16.0**
- **Vulkan SDK** (For `glslangValidator` and validation layers)

## Quick Start
```bash
# Build the project
zig build -Doptimize=ReleaseFast

# Run streaming inference
./zig-out/bin/llama.zig --model models/granite-4.0-350m-BF16.gguf --prompt "The future of AI is"
```

## Future Work (Phase 6)
- **KV Cache**: Persist key-value tensors across generation steps for context-aware sequential inference.
- **Memory Cleanup**: Fix minor `allocPrint` string memory leaks present during graph teardown.
- **4-bit Quantization**: Implement `Q4_K_M` dequantization kernels to run larger Llama 3 models locally within 8GB VRAM.
- **Multi-GPU**: Split large models across multiple Vulkan devices.

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml`.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe bindings.

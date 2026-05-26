# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend.

## Project Status: GPU Computing (Phase 1, 2, & 3 Complete)

The project has achieved its most significant technical milestone: **Pure Zig GPU Kernels**.

### Current Capabilities:
- **GGUF Parser:** Full support for GGUF v3, including BF16 and K-Quants.
- **Vulkan Infrastructure:** 
    - Custom Windows loader for high-performance GPU access.
    - Smart GPU scoring and selection (Discrete GPU prioritized).
    - VRAM allocation and Host-to-Device memory synchronization.
- **Zig-to-SPIR-V Kernels:** 
    - Compute shaders are written directly in Zig (`src/shaders/kernels.zig`).
    - Kernels are compiled to SPIR-V and embedded into the executable during the build process.
    - Support for `VK_KHR_buffer_device_address` (64-bit GPU pointers).
- **Build System:** Fully automated Zig 0.16.0 build pipeline with zero external dependencies (Vulkan SDK required for runtime).

### Upcoming Milestones:
- [ ] Static Compute Graph dispatcher (Phase 4).
- [ ] Llama 3 model forward pass implementation.
- [ ] Tokenizer & Chat interface.

## Getting Started

### Prerequisites
- **Zig (0.16.0 or newer):** [Download Zig](https://ziglang.org/download/)
- **Vulkan SDK:** Required for the `vk.xml` registry and validation layers.

### Build
```bash
zig build
```

### Usage (Validation)
You can test the GGUF parser and Vulkan GPU compute initialization:
```bash
zig build run -- --model path/to/model.gguf
```

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml`.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe bindings.

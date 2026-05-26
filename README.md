# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend.

## Project Status: Compute Graph (Phase 4 in Progress)

The project has achieved its most significant technical milestone: **Pure Zig GPU Kernels** and a **Compute Graph Dispatcher**.

### Current Capabilities:
- **GGUF Parser:** Full support for GGUF v3, including BF16 and K-Quants.
- **Vulkan Infrastructure:** 
    - Custom Windows loader for high-performance GPU access.
    - Smart GPU scoring and selection (Discrete GPU prioritized).
    - VRAM allocation and Host-to-Device memory synchronization.
- **Compute Graph:** 
    - DAG-based execution defined in `src/compute_graph.zig`.
    - BDA (Buffer Device Address) dispatcher for high-performance dispatch.
- **Zig-to-SPIR-V Kernels:** 
    - Compute shaders are written directly in Zig (`src/shaders/kernels.zig`).
    - [!] **Note**: AMDVLK on Windows currently segfaults during pipeline creation with Zig-generated SPIR-V. See `AGENTS.md` for investigation details.

### Upcoming Milestones:
- [ ] Llama 3 model forward pass implementation (Phase 5).
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

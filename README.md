# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend.

## Project Status: Infrastructure (Phase 1 & 2 Complete)

The project has established its core infrastructure and is ready for compute kernel development.

### Current Capabilities:
- **GGUF Parser:** A pure Zig implementation capable of parsing GGUF v3 metadata, Key-Value pairs, and tensor definitions (including K-Quants).
- **Vulkan Backend:** 
    - Custom Windows loader for `vulkan-1.dll` (bypassing Zig 0.16.0 standard library limitations).
    - Smart Physical Device selection (Discrete GPU prioritization).
    - Fully initialized Logical Device with Compute Queues and Command Pools.
    - `Buffer` abstraction for VRAM management and Host-to-Device copying.
- **Build System:** Integrated with Zig 0.16.0 package manager, utilizing `vulkan-zig` and `Vulkan-Headers`.

### Upcoming Milestones:
- [ ] Zig-to-SPIR-V compute kernels for tensor math (Add, MatMul, RMSNorm).
- [ ] Static Compute Graph dispatcher.
- [ ] Llama 3 model orchestration.

## Getting Started

### Prerequisites
- **Zig (0.16.0 or newer):** [Download Zig](https://ziglang.org/download/)
- **Vulkan SDK:** Required for the `vk.xml` registry and validation layers.

### Build
```bash
zig build
```

### Usage (Validation)
You can test the GGUF parser and Vulkan initialization by pointing it to a model file:
```bash
zig build run -- --model path/to/model.gguf
```

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml` by Georgi Gerganov.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe Vulkan bindings.

# llama.zig

A high-performance port of the `llama.cpp` inference engine to Zig, featuring a **Vulkan** compute backend.

## Project Status: Foundations (Phase 1 & 2)

The project is currently in the early development phase. 

### Current Capabilities:
- **GGUF Parser:** A pure Zig implementation capable of parsing GGUF v3 metadata, Key-Value pairs, and tensor definitions.
- **Build System:** Integrated with Zig 0.16.0 package manager, utilizing `vulkan-zig` and `Vulkan-Headers` for automatic binding generation.
- **Architecture:** Core `Tensor` and `Context` structures established, mirroring `ggml` concepts.

### Upcoming Milestones:
- [ ] Vulkan Device & Queue initialization.
- [ ] Zig-to-SPIR-V compute kernels for tensor math (Add, MatMul, etc.).
- [ ] Static Compute Graph dispatcher.
- [ ] Llama 3 model orchestration.

## Getting Started

### Prerequisites
- **Zig (0.16.0 or newer):** [Download Zig](https://ziglang.org/download/)
- **Vulkan SDK:** Required for the `vk.xml` registry and runtime.

### Build
```bash
zig build
```

### Usage (Parser Validation)
You can currently test the GGUF parser by pointing it to a model file:
```bash
zig build run -- --model path/to/model.gguf
```

## Acknowledgments
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and `ggml` by Georgi Gerganov.
- Utilizing [vulkan-zig](https://github.com/Snektron/vulkan-zig) for type-safe Vulkan bindings.

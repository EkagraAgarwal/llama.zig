# llama.zig: Development Guidelines

## Project Vision
`llama.zig` is a redevelopment of the `llama.cpp` engine in idiomatic Zig. It prioritizes a **Vulkan-first** approach for cross-platform GPU acceleration on any hardware (AMD, NVIDIA, Intel, Mobile).

## Current Architecture
- `src/main.zig`: Entry point and CLI handling.
- `src/gguf.zig`: Pure Zig GGUF v3 parser (Magic, Version, KVs, Tensors).
- `src/tensor.zig`: Tensor data structures and memory layout (ne/nb strides).
- `src/vulkan_backend.zig`: Vulkan context and compute pipeline management (Setup in progress).

## Build System (Zig 0.16.0)
The project uses `build.zig.zon` for dependency management:
- `vulkan`: Binding generator.
- `vulkan_headers`: Source of the `vk.xml` registry.

## Development Conventions
- **Memory:** Always use explicit allocators (`std.mem.Allocator`).
- **Safety:** Leverage Zig's error unions (`!T`) and optional types (`?T`).
- **Formatting:** Run `zig fmt .` before committing.
- **Vulkan:** Use the generated `vulkan` module for type-safe interaction with the GPU.

## Roadmap
1. **Phase 1 (Done):** GGUF Parser & Tensor Foundations.
2. **Phase 2 (In Progress):** Vulkan Setup (Instance, Device, Queues).
3. **Phase 3:** Zig-to-SPIR-V Math Kernels.
4. **Phase 4:** Compute Graph Dispatcher.
5. **Phase 5:** Tokenizer & Llama Model Loop.

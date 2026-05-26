# llama.zig: Development Guidelines

## Project Vision
`llama.zig` is a redevelopment of the `llama.cpp` engine in idiomatic Zig. It prioritizes a **Vulkan-first** approach for cross-platform GPU acceleration on any hardware (AMD, NVIDIA, Intel, Mobile).

## Current Architecture
- `src/main.zig`: Entry point and CLI handling (Zig 0.16.0 `std.process.Init` style).
- `src/gguf.zig`: Pure Zig GGUF v3 parser (Magic, Version, KVs, Tensors).
- `src/tensor.zig`: Tensor data structures and memory layout (ne/nb strides).
- `src/vulkan_backend.zig`: Vulkan context, smart device selection, and VRAM buffer management.

## Build System (Zig 0.16.0)
The project uses `build.zig.zon` for dependency management:
- `vulkan`: Binding generator.
- `vulkan_headers`: Source of the `vk.xml` registry.

## Development Conventions
- **Memory:** Always use explicit allocators (`std.mem.Allocator`).
- **Safety:** Leverage Zig's error unions (`!T`) and optional types (`?T`).
- **Windows Loader:** Use the custom Win32 `LoadLibraryW` wrapper in `vulkan_backend.zig` until `std.DynLib` is stabilized for Windows in 0.16.0.
- **Vulkan:** Use the generated `vulkan` module for type-safe interaction with the GPU.

## Roadmap
1. **Phase 1 (Done):** GGUF Parser & Tensor Foundations.
2. **Phase 2 (Done):** Vulkan Setup (Instance, Device, Queues, Buffers).
3. **Phase 3 (Up Next):** Zig-to-SPIR-V Math Kernels.
4. **Phase 4:** Compute Graph Dispatcher.
5. **Phase 5:** Tokenizer & Llama Model Loop.

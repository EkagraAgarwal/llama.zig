# llama.zig: Development Guidelines

## Project Vision
`llama.zig` is a redevelopment of the `llama.cpp` engine in idiomatic Zig. It prioritizes a **Vulkan-first** approach for cross-platform GPU acceleration. It is one of the first projects to use the **Zig SPIR-V backend** for production-grade AI compute kernels.

## Current Architecture
- `src/main.zig`: Entry point and GPU test suite.
- `src/shaders/kernels.zig`: GPU compute kernels written in pure Zig.
- `src/gguf.zig`: Pure Zig GGUF v3 parser (supports BF16/K-Quants).
- `src/vulkan_backend.zig`: Vulkan context, smart device selection, and SPIR-V pipeline management.

## Build System (Zig 0.16.0)
The build process is complex but automated:
1. `vulkan-zig` generates bindings from `vk.xml`.
2. `src/shaders/kernels.zig` is compiled to `kernels.spv` using the Zig SPIR-V backend.
3. The `.spv` binary is embedded into a Zig module (`kernels_data`).
4. The main executable is compiled and linked with the Vulkan loader.

## Development Conventions
- **Kernels:** Add new GPU math in `src/shaders/kernels.zig`. Use `addrspace(.global)` for buffer pointers.
- **Memory:** Use `vulkan.Buffer` for VRAM management.
- **Safety:** Always use Zig's native error handling for Vulkan calls.

## Roadmap
1. **Phase 1 (Done):** GGUF Parser & Tensor Foundations.
2. **Phase 2 (Done):** Vulkan Setup (Instance, Device, Queues, Buffers).
3. **Phase 3 (Done):** Zig-to-SPIR-V Math Kernels.
4. **Phase 4 (Up Next):** Compute Graph Dispatcher.
5. **Phase 5:** Tokenizer & Llama Model Loop.

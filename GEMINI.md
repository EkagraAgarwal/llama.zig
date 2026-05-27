# llama.zig: Development Guidelines

## Project Vision
`llama.zig` is a redevelopment of the `llama.cpp` engine in idiomatic Zig. It prioritizes a **Vulkan-first** approach for cross-platform GPU acceleration. It is one of the first projects to use the **Zig SPIR-V backend** for production-grade AI compute kernels.

## Current Architecture
- `src/main.zig`: Entry point, inference loop, and model/tokenizer loader.
- `src/shaders/*.glsl`: GPU compute kernels written in GLSL, using Buffer Device Address (BDA) and Push Constants for zero-overhead dispatch.
- `src/gguf.zig`: Pure Zig GGUF v3 parser.
- `src/vulkan_backend.zig`: Vulkan context, smart device selection, and direct dispatch pipeline management (bypassing `DeviceWrapper` to prevent stack overflows).
- `src/compute_graph.zig`: Dynamic transformer graph builder and dispatcher.

## Build System (Zig 0.16.0)
The build process is automated:
1. `vulkan-zig` generates bindings from `vk.xml`.
2. `src/shaders/*.glsl` is compiled to `.spv` using `glslangValidator` (transitioned away from Zig SPIR-V backend due to AMDVLK segfaults).
3. The `.spv` binaries are embedded into a Zig module (`kernels_data.zig`).
4. The main executable is compiled and linked with the Vulkan loader.

## Development Conventions
- **Kernels:** Add new GPU math in `src/shaders/*.glsl`. Use `buffer_reference` for physical 64-bit pointers. Ensure 40-byte PushConstant headers.
- **Memory:** Use `vulkan.Buffer` for VRAM management. Heap-allocate `DeviceWrapper` to avoid 4KB+ stack copies per Vulkan call.
- **Safety:** Always use Zig's native error handling. Avoid `try` on Vulkan dispatch calls that return `vk.Result` without wrapping in a zig error union.

## Roadmap
1. **Phase 1 (Done):** GGUF Parser & Tensor Foundations.
2. **Phase 2 (Done):** Vulkan Setup (Instance, Device, Queues, Buffers).
3. **Phase 3 (Done):** Math Kernels (GLSL to SPIR-V via glslangValidator).
4. **Phase 4 (Done):** Compute Graph & BDA Dispatcher.
5. **Phase 5 (Done):** Tokenizer, Real Weight Integration, & Llama Model Loop (End-to-End Inference).
6. **Phase 6 (Up Next):** 
   - Implement Key-Value (KV) Caching for contextual generation.
   - Fix minor memory leaks in the graph builder (`allocPrint`).
   - Implement 4-bit Quantization (dequantization kernels) to run Q4_K Llama models natively.

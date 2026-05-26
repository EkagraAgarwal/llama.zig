# llama.zig AI Agents Guide

## Project Context
`llama.zig` is a high-performance redevelopment of the `llama.cpp` engine in idiomatic Zig, prioritizing a **Vulkan-first** approach for cross-platform GPU acceleration.

## Current Progress: ALL PHASES COMPLETED
The project has successfully reached its final milestone.
- **Phase 1**: GGUF Parser & Tensor Foundations.
- **Phase 2**: Vulkan Context & Memory Management.
- **Phase 3**: Math Kernels (Optimized GLSL/HLSL).
- **Phase 4**: Compute Graph & BDA Dispatcher.
- **Phase 5**: Tokenizer & Autoregressive Inference Loop.

## Technical Summary

### GPU Architecture (BDA & Push Constants)
The project uses **Buffer Device Address (BDA)** to achieve near-zero CPU overhead during dispatch. 
- **Kernels**: Written in GLSL for maximum hardware compatibility (fixes AMDVLK crashes).
- **Addressing**: Uses `GL_EXT_buffer_reference` to access physical 64-bit pointers.
- **Synchronization**: Handled via automated memory barriers in `src/compute_graph.zig`.

### Pure Zig Tokenizer
- **Implementation**: `src/tokenizer.zig` contains a high-performance BPE encoder/decoder.
- **Features**: Supports GGUF merges, special tokens, and byte-to-unicode mapping.

## Guidelines for Future Maintenance
- **Adding Kernels**: Write in `src/shaders/*.glsl`, compile with `glslangValidator -V -S comp`, and update `kernels_data.zig`.
- **Extending Model**: Use `GraphBuilder` in `src/compute_graph.zig` to add new node types.
- **Hardware**: Validated on AMD Radeon RX 7000 series (Windows) and modern NVIDIA hardware.

The project is now functional and ready for production-grade optimization.

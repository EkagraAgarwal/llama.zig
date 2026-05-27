# llama.zig AI Agents Guide

## Project Context
`llama.zig` is a high-performance redevelopment of the `llama.cpp` engine in idiomatic Zig, prioritizing a **Vulkan-first** approach for cross-platform GPU acceleration.

## Current Progress: ALL PHASES COMPLETED + VERIFIED
The project has successfully reached its final milestone and has been validated against `llama.cpp`.
- **Phase 1**: GGUF Parser & Tensor Foundations.
- **Phase 2**: Vulkan Context & Memory Management.
- **Phase 3**: Math Kernels (Optimized GLSL/HLSL).
- **Phase 4**: Compute Graph & BDA Dispatcher.
- **Phase 5**: Tokenizer & Autoregressive Inference Loop.
- **Phase 6** (completed): Bug fixes, validation, and repo hygiene.

## Key Bug Fixes Applied
- **RoPE pair convention**: `src/shaders/rope_bda.glsl` now uses consecutive pairs `(2i, 2i+1)` (LLAMA_ROPE_TYPE_NORM), not NeoX/Falcon-style offset-by-d/2 pairs.
- **BOS token insertion**: `src/tokenizer.zig` now respects `tokenizer.ggml.add_bos_token = false` from GGUF metadata. Granite 4.0 does NOT use a BOS prefix; the former unconditional BOS insert poisoned all inference.
- **Tensor quantized size**: `Tensor.size()` now correctly computes byte sizes for all quantization types.
- **Tensor.blockSize**: Enum tags fixed to lowercase (`q4_k` not `q4_K`).

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
- **Tests**: `zig build test` runs CPU reference tests in `src/ops_test.zig` covering rmsnorm, matmul, scaled_add, and silu_mul.
- **Model Validation**: Run `.\zig-out\bin\llama.zig.exe --model models/granite-4.0-350m-BF16.gguf --prompt "The capital of France is" --max-tokens 8 --temperature 0` and confirm output starts with " Paris".

The project is now functional, validated against `llama.cpp` reference output, and ready for production-grade optimization.

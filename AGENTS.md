# llama.zig AI Agents Guide

## Project Context
`llama.zig` is a redevelopment of the `llama.cpp` engine in idiomatic Zig, prioritizing a **Vulkan-first** approach for cross-platform GPU acceleration. It leverages the **Zig SPIR-V backend** to write GPU math kernels directly in Zig.

## Current Progress: Phase 4 (Compute Graph & Dispatcher)
The project is currently in Phase 4. We have implemented:
- A `compute_graph.zig` that defines the forward pass as a DAG of nodes and tensors.
- A `vulkan_backend.zig` that manages Vulkan contexts, pipelines, and buffers.
- A BDA-based (Buffer Device Address) dispatcher that uses **Push Constants** for high performance.

## Simplified Error Explanation
**What's wrong?**
Imagine we are writing a letter (GPU code) in a new, experimental dialect of a language (Zig's SPIR-V). Your GPU driver (AMDVLK) is like a very strict postman. When it tries to read this experimental letter, it gets confused by the grammar and has a total meltdown (Segfault), crashing the whole program.

**The Solution (HLSL Pivot):**
Instead of trying to fix the experimental dialect, we will write the letters in a standard, proven language that the postman has seen millions of times: **HLSL** (the language used by Windows/DirectX). We will use a professional translator (**DXC**) to turn those HLSL letters into a standard format (SPIR-V) that every postman in the world understands.

## Current Progress: Phase 4 (Pivoting to HLSL)
- **Zig Dispatcher**: 100% working. It calculates where memory is and tells the GPU what to do.
- **HLSL Kernels**: We are currently rewriting the math (Add, Mul) into HLSL to stop the crashes.
- **Goal**: Full hardware compatibility (AMD, NVIDIA, Intel).


## Guidelines for AI Agents
- **Do NOT** waste more time trying to fix the AMDVLK crash by changing Zig kernel code. The issue is in the compiler's SPIR-V emission logic.
- **Focus on**: Implementing the higher-level logic (Phase 5 Tokenizer, Llama Loop) while treating the GPU backend as a black box that needs a more stable SPIR-V source.
- **Memory Management**: Maintain the BDA architecture in the dispatcher; it is correct and follows modern Vulkan standards. Only the *source* of the SPIR-V module needs to change.

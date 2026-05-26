# Phase 4: Compute Graph & Dispatcher Implementation Plan

## Objective
Implement the "brain" of the inference engine. Phase 4 bridges the gap between raw GPU kernels (Phase 3) and a full model architecture (Phase 5). It enables chaining multiple tensor operations together with automatic memory management and optimized GPU execution.

## Core Components

### 1. Compute Graph API (`src/compute_graph.zig`)
The graph is a static list of operations that defines the model's forward pass.
*   **`Node` Struct:** Represents a single operation (e.g., MatMul). Contains:
    *   `OpType`: Enum of available kernels (ADD, MUL, MATMUL, RMS_NORM).
    *   `Inputs`: List of source tensors.
    *   `Output`: Destination tensor.
    *   `Params`: Push constants for the kernel (strides, dimensions).
*   **`GraphBuilder`:** A context-like API (similar to `ggml_context`) to append nodes sequentially.

### 2. VRAM Scratchpad Manager
Efficiently managing activations (intermediate results) to minimize VRAM usage.
*   **Static Memory Allocation:** Before execution, calculate the total size of all intermediate tensors.
*   **Offset Calculation:** Assign every tensor in the graph a specific byte offset within a single large "Scratchpad" Vulkan Buffer.
*   **Buffer Reuse (Future):** Implement a simple greedy allocator to reuse memory locations for tensors that are no longer needed in the graph.

### 3. The Dispatcher (`src/vulkan_backend.zig`)
The component that translates the Graph into Vulkan commands.
*   **Command Buffer Recording:** Iterate through the `Graph`:
    1.  Select the correct `Pipeline` (compiled SPIR-V kernel).
    2.  Bind the necessary `Descriptor Sets`.
    3.  Update `Push Constants` with tensor addresses and metadata.
    4.  Record `vkCmdDispatch`.
    5.  **Memory Barriers:** Insert `vkCmdPipelineBarrier` between nodes to ensure the GPU finishes writing to a tensor before the next operation tries to read it.

### 4. Tensor Weight Uploader
*   **GGUF to VRAM:** Logic to read weights from the GGUF file (Phase 1) and copy them into `Device Local` Vulkan Buffers via the `Staging Buffer` (Phase 2).

## Implementation Steps

### Step 1: Graph Structures
Define the `Graph` and `Node` structs in a new file `src/compute_graph.zig`. Ensure tensors can be marked as either "Weights" (persistent) or "Activations" (temporary/scratch).

### Step 2: Kernel Pipeline Registry
Update `src/vulkan_backend.zig` to manage a map of `OpType -> Pipeline`. This allows the dispatcher to instantly look up the GPU code required for a graph node.

### Step 3: Descriptor & Synchronization Logic
Implement the logic to automatically insert `vkCmdPipelineBarrier`. Since we are using `Storage Buffers`, we need `SHADER_READ` and `SHADER_WRITE` access flags to prevent race conditions.

### Step 4: Verification (The "Add-Mul" Test)
Validate the dispatcher with a compound test case:
1.  Initialize Buffer A and B with values.
2.  Graph Node 1: `C = A + B`
3.  Graph Node 2: `D = C * A`
4.  Execute Graph and verify result D on the CPU.

## Current Implementation Status: Pivoting to HLSL
The core architecture for Phase 4 is complete, but execution is blocked on AMD Windows drivers due to Zig's experimental SPIR-V backend.

### Completed:
- [x] DAG Graph logic and Tensor Role management.
- [x] BDA Dispatcher using Push Constants.
- [x] Automatic VRAM offset calculation and memory barriers.

### Issues:
- [!] **Zig SPIR-V Incompatibility**: The `amdvlk64.dll` driver crashes when loading SPIR-V code written in pure Zig. This is a known limitation of the current experimental compiler backend.

### The Fix: HLSL Compilation Pipeline
To ensure the project is stable across all GPUs (AMD, NVIDIA, Intel), we are moving to a hybrid approach:
1. **Zig Logic**: The Compute Graph and Dispatcher stay in Zig (they work perfectly).
2. **HLSL Kernels**: Math operations (add, mul, matmul) will be written in **HLSL** (High-Level Shading Language).
3. **DXC Compiler**: We will use the Microsoft `dxc` compiler to turn HLSL into "standard" SPIR-V that drivers are guaranteed to understand.

### Setup Instructions (HLSL):
1. **Source**: Kernels are now being migrated from `src/shaders/kernels.zig` to `src/shaders/kernels.hlsl`.
2. **Build Change**: Update `build.zig` to use `dxc -T cs_6_0 -E main -fspv-target-env=vulkan1.2`.
3. **Integration**: The dispatcher will continue to load the resulting `.spv` as a byte array, just as it did before.

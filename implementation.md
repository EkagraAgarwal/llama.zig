# Implementation Plan: Direct GPU Upload for q8_0 and q4_0 Quantizations

## Objective
Support native execution of `q8_0` and `q4_0` quantized models by uploading their raw block buffers directly to the GPU via Buffer Device Address (BDA) and decoding them within the matmul/GEMV shaders, bypassing the host-side decompression currently used in `weights.zig`.

## Key Files & Context
- `src/main.zig`: Handles model loading and tensor data upload.
- `src/compute_graph.zig`: Dispatches compute nodes and manages Vulkan pipelines.
- `src/shaders/get_rows_q_bda.glsl`: Shader for unpacking rows during embeddings fetch.
- `src/shaders/matmul_q4_0_bda.glsl`, `matvec_q4_0_bda.glsl`, `matmul_q8_0_bda.glsl`, `matvec_q8_0_bda.glsl` (NEW): Compute shaders to perform block-wise dequantization and matmul.

## Implementation Steps

### 1. New Shaders for `q4_0` and `q8_0`
Create 4 new shaders for block dequantization and matrix multiplication.
- `src/shaders/matmul_q4_0_bda.glsl`: Matmul shader for `q4_0` block sizes (32 elements, 18 bytes: 2 byte `d` + 16 bytes `qs`).
- `src/shaders/matvec_q4_0_bda.glsl`: GEMV (vector-matrix) shader for `q4_0`.
- `src/shaders/matmul_q8_0_bda.glsl`: Matmul shader for `q8_0` block sizes (32 elements, 34 bytes: 2 byte `d` + 32 bytes `qs`).
- `src/shaders/matvec_q8_0_bda.glsl`: GEMV shader for `q8_0`.

*Note: The dequantization logic in GLSL will mimic the `weights.zig` host decoding logic using `f16ToF32`, bitwise masking, and sub-block element extraction.*

### 2. Update `get_rows_q_bda.glsl`
Modify the existing GPU embedding fetch shader to support `q4_0` and `q8_0`.
- Add `q40At(uint row_base, uint kidx)` logic.
- Add `q80At(uint row_base, uint kidx)` logic.
- Switch on `pc.qtype == 2u` for `q4_0` and `pc.qtype == 8u` for `q8_0` in the main decode loop.

### 3. Register Pipelines and Manage Graph Dispatch
Modify `src/main.zig` and `src/compute_graph.zig` to hook up the new shaders.
- **`src/main.zig`**:
  - Add pipeline registration for `matmul_q4_0`, `matvec_q4_0`, `matmul_q8_0`, and `matvec_q8_0`.
  - In the node rewriting loop, set `node.op_type = .matmul_q` and `node.p5 = @intFromEnum(w_t.type)` when `w_t.type` is `.q4_0` or `.q8_0`.
  - During tensor upload (`if (t.type == .q4_k or t.type == .q6_k ...)`), include `.q4_0` and `.q8_0` so that `t.size()` raw bytes are directly uploaded rather than being routed through `dequantToF32`.
  - Make sure embedding size checks (`embd_row_bytes`) work correctly for `.q4_0` and `.q8_0` by dynamically retrieving the block size and bytes per block from `tensor.zig`.
- **`src/compute_graph.zig`**:
  - Update `pipelineNameForNode` to return `"matvec_q4_0"`, `"matmul_q4_0"`, `"matvec_q8_0"`, or `"matmul_q8_0"` when `qtype` is `2` or `8`.

## Verification & Testing
- Load a `.gguf` file using `q4_0` and verify inference matches expected outputs.
- Load a `.gguf` file using `q8_0` and verify inference matches expected outputs.
- Inspect logs to confirm `.matmul_q` nodes are utilized for standard model weights.

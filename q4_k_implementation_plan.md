# Implementation Plan - Q4_K Quantization Support

This plan outlines the steps to implement `Q4_K` quantization support in `llama.zig`, including CPU-side dequantization and Vulkan-accelerated inference.

## Objective
Enable loading and running GGUF models quantized with `Q4_K` (often used in `Q4_K_M` and `Q4_K_S` variants). This includes implementing the 4.5 bits-per-weight super-block structure (256 weights per block).

## Key Files & Context
- **`src/tensor.zig`**: Define `q4_k` type and its properties (block size 256, 144 bytes per block).
- **`src/weights.zig`**: Implement CPU dequantization for `q4_k`.
- **`src/gguf.zig`**: Support reading `q4_k` tensors from GGUF files.
- **`src/shaders/`**: New GLSL shaders for `q4_k` (matvec, get_rows).
- **`src/compute_graph.zig`**: Integrate `q4_k` into the Vulkan dispatcher.
- **`build.zig` & `kernels_data.zig`**: Shader compilation and embedding.

## Proposed Solution

### 1. Data Structure (Q4_K)
A `Q4_K` block (256 weights) consists of:
- `d`: `f16` super-block scale.
- `dmin`: `f16` super-block minimum.
- `scales`: 12 bytes storing 8 scales and 8 mins (6 bits each).
- `qs`: 128 bytes storing 256 4-bit weights.
Total: 144 bytes.

### 2. Implementation Steps

#### Phase 1: Core Support
1.  **Update `tensor.Type` in `src/tensor.zig`**:
    - Add `.q4_k = 12`.
    - Update `blockSize` to return 256 for `.q4_k`.
    - Update `bytesPerBlock` to return 144 for `.q4_k`.
2.  **Update `src/gguf.zig`**:
    - In `loadModel`, map tensor type ID `12` to `.q4_k`.
3.  **Update `src/weights.zig`**:
    - Implement `dequantQ4KRaw` using the 6-bit unpacking logic.
    - Update `isSupportedType`, `dequantToF32`, `readEmbeddingF32`, and `quantRowBytes`.

#### Phase 2: Vulkan Shaders
1.  **Create `src/shaders/get_rows_q4_k_bda.glsl`**:
    - Implement `q4kAt` function for BDA-based access.
    - Handle 6-bit scale/min unpacking in GLSL.
2.  **Create `src/shaders/matvec_q4_k_bda.glsl`**:
    - Implement dot product over `q4_k` blocks.
    - Follow the existing `matvec` pattern for consistency.
3.  **Update `build.zig`**:
    - Add new shaders to `compileShaders` list.
4.  **Update `kernels_data.zig`**:
    - Add `@embedFile` for the new `.spv` files.
5.  **Update `src/main.zig`**:
    - Register new pipelines in `main`.

#### Phase 3: Integration & Testing
1.  **Update `src/compute_graph.zig`**:
    - Update `pipelineNameForNode` and `quantPipelineName` to handle `.q4_k`.
2.  **Update `src/ops_test.zig`**:
    - Add unit tests for `q4_k` dequantization parity between CPU and expected values.
3.  **Verify with a real model**:
    - Test using a known `Q4_K_M` model (e.g., Llama 3.2 3B).

## Verification & Testing
- **Unit Tests**: Add tests in `weights.zig` or `ops_test.zig` to verify `dequantQ4KRaw` against a known-good pattern.
- **Parity Check**: Compare CPU dequantization results with GPU output.
- **Model Test**: Run `llama.zig` with a `Q4_K_M` model and verify it no longer produces garbled output.

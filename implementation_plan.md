# Dynamic Architecture Support for Gemma and Qwen

This plan addresses the goal of supporting Gemma and Qwen architectures (and minimizing manual intervention for future architectures) by making the `compute_graph.zig` builder data-driven. Instead of hardcoding the block structure, the graph builder will inspect the presence of specific tensors in the GGUF file to determine the correct layout.

## User Review Required

> [!IMPORTANT]
> **Activation Functions & Scaling**: While GGUF specifies model parameters, it unfortunately *does not* specify the activation function (GeLU vs SiLU) or implicit embedding scaling (like Gemma's `sqrt(n_embd)`) in standard metadata keys. We will still need to infer these two properties based on the `general.architecture` string. 
> 
> Everything else (biases, normalizations) will be 100% dynamic based on tensor presence. Is this acceptable?

## Proposed Changes

---

### `src/model.zig`

#### [MODIFY] [model.zig](file:///D:/llama.zig/src/model.zig)
- Add an `Activation` enum (`silu`, `gelu`).
- Add `activation` to `ModelConfig`.
- In `ModelConfig.init()`:
  - If `arch == .gemma`, set `activation = .gelu` and default `embedding_scale = sqrt(n_embd)` (if not overridden by metadata).
  - Otherwise, set `activation = .silu`.

---

### `src/shaders/`

#### [NEW] [gelu_mul_bda.glsl](file:///D:/llama.zig/src/shaders/gelu_mul_bda.glsl)
- Create a new compute shader for GeGLU (`gelu(gate) * up`).
- Use the tanh approximation for GeLU which is fast and standard for Gemma: `0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

#### [MODIFY] [add_bda.glsl](file:///D:/llama.zig/src/shaders/add_bda.glsl)
- Update the element-wise add kernel to support broadcasting.
- We will use `pc.p1` to define the broadcast dimension (e.g., `out_dim`).
- Logic: `b.data[pc.p1 > 0 ? (idx % pc.p1) : idx]`.
- This allows adding 1D bias vectors (like Qwen's QKV biases) to 2D batched activations during prefill.

---

### `src/compute_graph.zig`

#### [MODIFY] [compute_graph.zig](file:///D:/llama.zig/src/compute_graph.zig)
- **OpType**: Add `.gelu_mul` to the `OpType` enum.
- **Pipeline Dispatch**: Update `pipelineNameForNode` to handle `.gelu_mul`.
- **Tensor Presence Check**: Add a helper `hasTensor(name)` to check if a tensor exists in the GGUF `model_tensors` map.
- **`buildTransformerBlock`**: Rename `buildLlamaBlock` to `buildTransformerBlock` and make it dynamic:
  - **QKV Bias**: Check for `blk.{L}.attn_q.bias`, `attn_k.bias`, `attn_v.bias`. If present, insert an `add` node with broadcasting to apply the bias (supports Qwen2).
  - **QK Norm**: Check for `blk.{L}.attn_q_norm.weight` and `attn_k_norm.weight`. If present, apply `rms_norm` to Q and K before RoPE (supports Qwen3).
  - **Post-Attention Norm**: Check for `blk.{L}.attn_post_norm.weight`. If present, apply `rms_norm` to the attention output before the residual add (supports Gemma2).
  - **Activation**: Dispatch to `.gelu_mul` or `.silu_mul` based on `cfg.activation`.
  - **Post-FFN Norm**: Check for `blk.{L}.ffn_post_norm.weight`. If present, apply `rms_norm` to the FFN output before the residual add (supports Gemma2).
- **`buildLmHead`**: 
  - Check for `output.bias`. If present, apply it to the logits (supports Qwen models).

---

### Integration

#### [MODIFY] [kernels_data.zig](file:///D:/llama.zig/kernels_data.zig)
- Embed `gelu_mul_bda.spv`.

#### [MODIFY] [main.zig](file:///D:/llama.zig/src/main.zig)
- Register the `gelu_mul` pipeline.
- Call `buildTransformerBlock` instead of `buildLlamaBlock`.
- When loading required layout tensors in `validateModelLayout`, make it flexible enough to allow missing `output.weight` if it's tied.

#### [MODIFY] [build.zig](file:///D:/llama.zig/build.zig)
- Add `src/shaders/gelu_mul_bda.glsl` to the shader compilation list.

## Verification Plan

### Automated Tests
- Run `zig build test` to ensure existing CPU reference tests for existing ops pass.

### Manual Verification
- Run a Gemma model (e.g., `gemma-2b.gguf` or similar) to ensure the `gelu_mul` and `sqrt(n_embd)` scaling produce coherent text.
- Run a Qwen model (e.g., `qwen2-1.5b.gguf`) to verify QKV biases apply correctly without crashing and produce coherent text.
- Run the existing Granite model to ensure no regressions in the LLaMA/Granite path.

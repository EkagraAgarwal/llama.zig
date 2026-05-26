# Phase 5: Transformer Blocks & Full Inference

## Objective
The goal is to finish the project by implementing the full autoregressive Llama model loop. This involves building the transformer blocks using the existing compute graph and implementing the logic to sample and feed tokens back into the model.

## Remaining Tasks

### 1. Advanced Math Kernels (HLSL/GLSL)
- [ ] **RoPE (Rotary Positional Embeddings)**: Crucial for Llama's context window.
- [ ] **MatMul (Matrix Multiplication)**: Optimized BDA-based GEMM kernel for 16-bit quants.
- [ ] **RMSNorm**: Final verification of the normalization kernel.

### 2. Full Transformer Block Builder
In `src/compute_graph.zig`, add a function to build a complete Llama block:
- **Attention**: 
  - Q, K, V Projections.
  - RoPE application.
  - Scaled Dot Product Attention.
- **FFN**:
  - Up/Down/Gate Projections.
  - SwiGLU Activation.

### 3. Autoregressive Loop (`src/main.zig`)
- [ ] Implement the `while` loop that:
  - Takes the last generated token.
  - Runs the compute graph for that token.
  - Samples the next token from the output logits.
  - Appends it to the prompt and prints it.

### 4. KV Cache Management
- [ ] Allocate a large VRAM buffer to store Key and Value tensors for all layers to avoid recomputing past tokens.

## Implementation Steps

1.  **RoPE Kernel**: Write and compile `rope.glsl`.
2.  **Llama Graph**: Implement `GraphBuilder.buildLlamaBlock`.
3.  **Inference CLI**: Update `main.zig` to accept real-time prompts and generate text character by character.

## Success Criteria
- [ ] The model can generate a coherent 50-token response.
- [ ] Token generation speed is > 10 tokens/sec on an RX 7700S.

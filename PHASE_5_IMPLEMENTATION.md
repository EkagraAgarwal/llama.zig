# Phase 5: Transformer Layers & Inference Loop

## Objective
Implement the final logic layer required for text generation. This phase focuses on building the actual Llama transformer block (Attention, FFN, RMSNorm) using the completed compute graph dispatcher and the newly implemented BPE tokenizer.

## Core Components

### 1. Pure Zig Tokenizer (`src/tokenizer.zig`) - **CORE COMPLETED**
- [x] GGUF BPE Vocabulary Loader.
- [x] Greedy BPE Encoding with Merges.
- [x] Token Decoding.
- [x] Special Token Handling (BOS/EOS).

### 2. Transformer Block Builder
Extend `src/compute_graph.zig` to build a complete Llama transformer layer.
- [ ] **RMSNorm**: Add `rmsnorm` node type and link to `rmsnorm_bda.spv`.
- [ ] **Attention (Self-Attention)**: Implement:
    - [ ] Query, Key, Value projections.
    - [ ] RoPE (Rotary Positional Embeddings) kernel.
    - [ ] Scaled Dot-Product Attention (Softmax).
- [ ] **FFN (Feed-Forward Network)**: Implement:
    - [ ] SwiGLU activation kernel.
    - [ ] Up, Down, and Gate projections.
- [ ] **KV Cache**: Manage memory for past tokens.

### 3. Inference Loop (`src/main.zig`)
The main application loop that drives the model.
1. **Prompting**: Accept user input from the CLI.
2. **Tokenization**: Convert text to IDs using `src/tokenizer.zig`.
3. **Graph Execution**: 
    - Build graph for the current prompt.
    - Dispatch to GPU via `src/compute_graph.zig`.
    - Sample the next token from the output logits (Greedy or Top-P).
4. **De-tokenization**: Print the generated character to stdout.
5. **Repeat**: Feed the new token back into the model.

## Current Progress: BDA Dispatcher & Kernels
- **Dispatcher**: Supports BDA and Push Constants.
- **Kernels**: 
    - [x] `add` (GLSL)
    - [x] `mul` (GLSL)
    - [x] `rmsnorm` (HLSL/GLSL)
    - [x] `softmax` (HLSL/GLSL)

## Success Criteria
- [x] Successfully decode a GGUF vocabulary.
- [ ] Run a single forward pass of a Llama 3 transformer block without errors.
- [ ] Generate at least one coherent sentence from a prompt.

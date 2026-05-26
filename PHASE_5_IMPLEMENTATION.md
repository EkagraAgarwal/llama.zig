# Phase 5: Tokenizer & Llama Model Loop

## Objective
Implement the final logic layer required for text generation. This phase bridges the gap between the static GPU compute graph (Phase 4) and a functional chatbot. It includes the vocabulary processing (Tokenizer) and the high-level transformer block iteration (Model Loop).

## Core Components

### 1. Pure Zig Tokenizer (`src/tokenizer.zig`)
The tokenizer converts raw strings into integer token IDs and vice versa.
*   **BPE (Byte Pair Encoding)**: Implementation for Llama 3/4 style tokens.
*   **Vocabulary Loader**: Read the `tokenizer.ggml.tokens` and `tokenizer.ggml.scores` arrays from the GGUF metadata.
*   **Decoding**: Convert model output IDs back into UTF-8 strings, handling special tokens like `<|end_of_text|>`.

### 2. Transformer Block Builder
Extend `src/compute_graph.zig` to build a complete Llama transformer layer.
*   **RMSNorm**: Implement the Root Mean Square Layer Normalization kernel.
*   **Attention (Self-Attention)**: Implement the Query, Key, Value projection and Softmax.
*   **FFN (Feed-Forward Network)**: Implement the SwiGLU activation and up/down projections.
*   **KV Cache**: Manage memory for past tokens to enable fast autoregressive generation.

### 3. Inference Loop (`src/main.zig`)
The main application loop that drives the model.
1. **Prompting**: Accept user input from the CLI.
2. **Tokenization**: Convert text to IDs.
3. **Graph Execution**: 
    * Build graph for the current prompt.
    * Dispatch to GPU.
    * Sample the next token from the logits.
4. **De-tokenization**: Print the generated character to stdout.
5. **Repeat**: Feed the new token back into the model until a stop token is generated.

## Implementation Steps

### Step 1: Vocab Extraction
Update `src/gguf.zig` to specifically extract and store the tokenizer metadata (tokens, scores, types).

### Step 2: Basic Tokenizer
Implement a simple greedy BPE decoder in Zig. Test it by tokenizing a few sample strings and comparing them with `llama.cpp` outputs.

### Step 3: Math Kernels (HLSL)
Add the remaining transformer kernels in HLSL:
* `rmsnorm.glsl`
* `softmax.glsl`
* `matmul.glsl` (Optimized for 16-bit)

### Step 4: Autoregressive Loop
Connect the Tokenizer and the Compute Graph in a `while` loop in `main.zig`.

## Success Criteria
- [ ] Successfully decode a GGUF vocabulary.
- [ ] Run a single forward pass of a Llama 3 transformer block without errors.
- [ ] Generate at least one coherent sentence from a prompt.

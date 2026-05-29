# `llama.zig` Architecture Documentation

`llama.zig` is a high-performance, single-node inference engine for Large Language Models (LLMs) written entirely in Zig. It targets Vulkan 1.2+ for GPU acceleration using Compute Shaders and Buffer Device Address (BDA) to achieve memory-efficient and fast text generation. 

The primary goal of `llama.zig` is to load GGUF models, build an optimized compute graph, map operations to Vulkan shaders, and execute the generation loop.

---

## 1. High-Level Overview

The system is broken down into the following logical stages:
1. **Model Loading & Parsing:** Reads the `.gguf` file format, parses metadata, extracts model configurations (like dimensions, head counts, layer counts) and maps tensor offsets.
2. **Tokenizer Setup:** Parses the BPE/SentencePiece vocabulary and sets up special tokens.
3. **Graph Building:** A declarative `GraphBuilder` constructs a Directed Acyclic Graph (DAG) of the model's forward pass (from embedding lookup to transformer blocks to the LM head).
4. **Hardware Initialization & Memory Allocation:** The `vulkan_backend` initializes a Vulkan instance, finds a suitable physical device (preferring discrete GPUs), and allocates GPU memory using Buffer Device Address (BDA) for seamless pointer math in shaders.
5. **Execution Dispatch:** The compute graph is converted into a sequence of Vulkan Pipeline dispatches. The runtime handles batching for the prompt (prefill phase) and single-token sequential generation (decode phase).

---

## 2. Core Modules

### 2.1 Entry Point (`src/main.zig`)
The CLI frontend and orchestration layer.
- **Argument Parsing:** Handles flags like `--model`, `--prompt`, `--chat`, `--temperature`, `--max-tokens`, etc.
- **Orchestration:** 
  - Calls `gguf.loadModel` to get a file context.
  - Initializes `Tokenizer` and `ModelConfig`.
  - Instantiates `vulkan_backend.Context`.
  - Sets up the `compute_graph.GraphBuilder` and builds the DAG.
  - Converts CPU-side tensors into `vulkan.Buffer` objects, applying quantization fallbacks (e.g., dynamically converting unsupported formats to `f16` on the CPU using `weights.zig`).
- **Inference Loop:** Handles the two-phase inference (Prefill vs Decode). During Decode, it calls `token_sampler.sample` and checks for termination tokens.

### 2.2 Model Configuration (`src/model.zig`)
A translation layer that converts raw GGUF Key-Value pairs into a concrete `ModelConfig` struct.
- **Architecture Abstraction:** Normalizes names between different architectures (Llama, Granite, Gemma, Qwen).
- **Dimension Parsing:** Extracts `n_embd`, `n_layer`, `n_heads`, `n_kv_heads`, `rope_theta`, `rms_norm_eps`, and specific scaling factors (`embedding_scale`, `attention_scale`, etc.).
- **GQA Support:** Automatically infers Grouped-Query Attention (GQA) if `n_kv_heads` is omitted by inspecting the tensor shapes of the `K` weight projection.

### 2.3 GGUF Parsing (`src/gguf.zig`)
A lightweight, zero-dependency parser for the GGUF v3 format.
- Reads magic bytes (`GGUF`).
- Parses key-value pairs into a recursive `MetadataValue` tagged union.
- Reads tensor metadata (name, type, dimensions, offset).
- Exposes `readTensorData()` to seek into the file and stream raw bytes into memory.

### 2.4 Tokenizer (`src/tokenizer.zig`)
Implements BPE (Byte-Pair Encoding) and SentencePiece text tokenization.
- **Byte-to-Unicode Mapping:** Correctly parses GPT-2/Granite style vocabularies.
- **Special Tokens:** Detects architecture-specific tokens like `<|start_of_role|>` and `<|end_of_text|>`.
- **Merge Logic:** Performs greedy token merging based on rank tables (`tokenizer.ggml.merges`).
- **Decoding:** Correctly translates token IDs back into UTF-8 strings, handling leading whitespace markers (e.g., the `▁` or `_` character).

### 2.5 Sampling (`src/sampler.zig`)
Probabilistic token selection from a raw logit distribution.
- Applies standard techniques: **Temperature scaling**, **Top-K**, **Top-P (Nucleus)**, **Min-P**, and **Typical-P**.
- **Repetition Penalty:** Applies penalization vectors against recently generated tokens (maintained in a rolling window in `main.zig`).

---

## 3. GPU Backend & Execution

### 3.1 Vulkan Abstraction (`src/vulkan_backend.zig`)
Provides a highly tailored Vulkan 1.2+ environment using the `vulkan-zig` bindings.
- Automatically selects the best Physical Device (scoring discrete GPUs higher).
- Enables the critical `shader_int_64` and `buffer_device_address` features.
- Provides `PipelineRegistry` to load SPIR-V compute shaders (compiled via `glslangValidator`).
- Provides a `Buffer` abstraction representing GPU memory with explicit alignment and memory mapping capabilities.

### 3.2 Compute Graph & Dispatch (`src/compute_graph.zig`)
The heart of the inference engine.

#### `GraphBuilder`
Builds the static DAG structure. 
- Iterates over the transformer layers, adding `GraphNode` structures.
- Generates operation sequences: `rms_norm` -> `matmul` (Q/K/V) -> `rope` -> `kv_write` -> `attention` -> `matmul` (Out) -> `scaled_add` -> `rms_norm` -> `gelu_mul`/`silu_mul` -> `matmul` -> `scaled_add`.
- Computes optimal buffer offsets (scratchpad memory planning).

#### `Dispatcher`
Executes the `Graph`.
- Exposes `executePrefillBatch()` and `execute()` for step-by-step dispatch.
- **Push Constants Mapping:** Maps a generic `GraphNode` (with `p1`-`p5` parameters) into a Vulkan `PushConstants` struct consisting of Buffer Device Addresses (BDA) for `A`, `B`, and `C` tensors.
- Handles Vulkan pipeline barriers between dependent compute stages (e.g., `shader_write_bit` -> `shader_read_bit`).

### 3.3 Tensors & Quantization (`src/tensor.zig` & `src/weights.zig`)
Defines the `Tensor.Type` layout and provides host-side dequantization routines.
- Supports types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q4_k`, `q6_k`.
- **`weights.zig`**: Contains fallback math to decode complex quantized block formats. For example, `dequantQ6KRaw` dynamically unpacks 6-bit scales and sub-block quantizers into an `f32` slice.
- Enables the "Hybrid Path": If native GPU shaders for a quant type are missing, `main.zig` uses `weights.zig` to dequantize the weight to `f16` CPU-side, uploads it to the GPU, and uses `matmul_f16`.

---

## 4. Shaders (GLSL Compute)

Located in `src/shaders/*.glsl` (compiled to `.spv`).
These are headless Compute Shaders utilizing `GL_EXT_buffer_reference2` to read memory directly via pointers rather than descriptor sets.

*Key Shaders:*
- `matmul_bda.glsl` / `matvec_bda.glsl`: Standard matrix operations.
- `matmul_q4_k_bda.glsl`: Native GPU decoding for the 256-element Q4_K block format. Performs nibble-unpacking on the fly during tile multiplication.
- `attention_bda.glsl` / `flash_attn_bda.glsl`: Computes scaled dot-product attention over the KV cache.
- `rope_bda.glsl`: Applies Rotary Positional Embeddings.

---

## 5. Execution Flow Example (Decode Step)

1. `main.zig` passes the new token to the `Dispatcher`.
2. `Dispatcher.executeGetRowsQ` extracts the token's embedding from the `token_embd.weight` buffer on the GPU and stores it in the `input` tensor in the scratchpad.
3. `Dispatcher.execute()` is called. It records a Vulkan Command Buffer looping over the DAG.
4. For Layer 0:
   - `rms_norm` shader is dispatched.
   - `matmul_q4_k` is dispatched 3 times to generate Q, K, and V vectors.
   - `rope` shader applies positional rotation to Q and K.
   - `kv_write` appends K and V to the historical cache.
   - `attention` computes probabilities and blends V.
   - `matmul_q4_k` projects the output.
   - Residual additions and the Feed-Forward Network (FFN) occur.
5. LM Head: `matmul` against the vocabulary weights produces the final logits.
6. The logits buffer is transferred back to the CPU (or GPU `topk` is used).
7. `Sampler.sample()` picks the next token based on Temperature and Top-P.
8. Loop repeats.

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
- **Argument Parsing:** Delegates to `cli.zig` to handle flags like `--model`, `--prompt`, `--temperature`, etc.
- **Orchestration:** 
  - Uses `mmap.zig` and `gguf.loadModelMmap` for high-performance model loading.
  - Initializes `Tokenizer` and `ModelConfig`.
  - Instantiates the `backend.Backend` (currently defaulting to Vulkan).
  - Delegates the heavy lifting of inference to `engine.runInference`.

### 2.2 Inference Engine (`src/engine.zig`)
The high-level driver for the inference process.
- **Inference Lifecycle:** Manages the prefill (prompt processing) and decode (token-by-token generation) phases.
- **Memory Management:** Allocates the scratchpad and KV cache buffers on the GPU.
- **Embedding Loading:** Handles both CPU-side and GPU-accelerated embedding lookups.
- **Execution Loop:** Orchestrates the `compute_graph.Dispatcher` to execute the model's forward pass and applies the `Sampler` to select the next token.

### 2.3 Weight Loader & Fusion (`src/weights_loader.zig`)
Handles the complex process of loading weights from GGUF and preparing them for the GPU.
- **Dynamic Weight Fusion:** Automatically detects and concatenates constituent weights at load-time (e.g., merging Q, K, and V projections into a single `.attn_qkv.weight` tensor).
- **Fusion Type-Safety:** Validates that fused weights share compatible quantization types (e.g., prevents fusing `Q4_K` with `Q6_K` if the hardware kernels don't support mixed-type fusion).
- **Quantization Dispatch:** Identifies "native" quantization types (like `Q4_K`, `Q6_K`, `Q8_0`) that can be processed directly on the GPU, and applies fallbacks for others.
- **Upload Management:** Streams weights from the GGUF file to GPU buffers, applying dequantization on the fly where necessary.

### 2.4 Backend Abstraction (`src/backend/`)
Provides a consistent interface for hardware acceleration.
- **`interface.zig`**: Defines the `Backend` and `Buffer` types.
- **`vulkan.zig`**: The primary implementation, providing Vulkan 1.2+ acceleration using Compute Shaders and Buffer Device Address (BDA).

### 2.5 Model Configuration (`src/model.zig`)
A translation layer that converts raw GGUF Key-Value pairs into a concrete `ModelConfig` struct.
- **Architecture Abstraction:** Normalizes names between different architectures (Llama, Granite, Gemma, Qwen).
- **Dimension Parsing:** Extracts `n_embd`, `n_layer`, `n_heads`, `n_kv_heads`, `rope_theta`, `rms_norm_eps`, and specific scaling factors (`embedding_scale`, `attention_scale`, etc.).
- **GQA Support:** Automatically infers Grouped-Query Attention (GQA) if `n_kv_heads` is omitted.

### 2.6 Tokenizer (`src/tokenizer.zig`)
Implements BPE (Byte-Pair Encoding) and SentencePiece text tokenization.
- **Byte-to-Unicode Mapping:** Correctly parses GPT-2/Granite style vocabularies.
- **Special Tokens:** Detects architecture-specific tokens like `<|start_of_role|>` and `<|end_of_text|>`.
- **Merge Logic:** Performs greedy token merging based on rank tables.
- **Chat Format Detection & Base Model Bypass:** Auto-detects chat template format based on GGUF metadata or instruct-specific control tokens.

### 2.7 Sampling (`src/sampler.zig`)
Probabilistic token selection from a raw logit distribution.
- Applies standard techniques: **Temperature scaling**, **Top-K**, **Top-P (Nucleus)**, **Min-P**, and **Typical-P**.
- **Repetition Penalty:** Applies penalization vectors against recently generated tokens.

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
- **Dynamic Weight Fusion:** Concatenates weights at load-time (QKV weights into `.attn_qkv.weight` and MLP Gate/Up weights into `.ffn_gate_up.weight`) to minimize dispatches.
- **Fusion Type-Safety (`canFuseQkv` / `canFuseGateUp`):** Checks weight quantization types layer-by-layer. If types do not match (e.g. `Q6_K` value projections alongside `Q4_K` query/key projections in `Q4_K_M` GGUF models), fusions are safely disabled for that layer to prevent GPU memory layout mismatch and NaN propagation.

#### `Dispatcher`
Executes the `Graph`.
- Exposes `executePrefillBatch()` and `execute()` for step-by-step dispatch.
- **Push Constants Mapping:** Maps a generic `GraphNode` (with expanded `p1`-`p8` parameters) into a Vulkan `PushConstants` struct consisting of Buffer Device Addresses (BDA) for `A`, `B`, and `C` tensors.
- **Zero-Cost Offsetting:** Passes offset parameters (`p6-p8`) to the consumer nodes (`rope`, `kv_write`, `attention`, and `silu_mul`). These nodes apply the offsets directly to BDA pointers in shaders, bypassing copy operations.
- Handles Vulkan pipeline barriers between dependent compute stages (e.g., `shader_write_bit` -> `shader_read_bit`).

### 3.3 Tensors & Quantization (`src/tensor.zig` & `src/weights.zig`)
Defines the `Tensor.Type` layout and provides host-side dequantization routines.
- Supports types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q4_k`, `q6_k`.
- **`weights.zig`**: Contains fallback math to decode complex quantized block formats. For example, `dequantQ6KRaw` dynamically unpacks 6-bit scales and sub-block quantizers into an `f32` slice.
- Enables the "Hybrid Path": If native GPU shaders for a quant type are missing, `main.zig` uses `weights.zig` to dequantize the weight to `f16` CPU-side, uploads it to the GPU, and uses `matmul_f16`.
- **Q6_K native path**: `matmul_q6_k_bda.glsl` provides full native GPU path for q6_k — both matvec (1D, 256 workitems) and matmul (16×16 tiled). Weights stay compressed at ~6.5 bits/element in VRAM.

---

## 4. Shaders (GLSL Compute)

Located in `src/shaders/*.glsl` (compiled to `.spv`).
These are headless Compute Shaders utilizing `GL_EXT_buffer_reference2` to read memory directly via pointers rather than descriptor sets.

*Key Shaders:*
- `matmul_bda.glsl` / `matvec_bda.glsl`: Standard matrix operations.
- `matmul_q4_k_bda.glsl`: Native GPU decoding for the 256-element Q4_K block format. Performs nibble-unpacking on the fly during tile multiplication.
- `matmul_q6_k_bda.glsl`: Native GPU decoding for the 256-element Q6_K block format. Performs 6-bit nibble-unpacking and scale application on the fly during 16×16 tile multiplication.
- `matvec_q4_k_bda.glsl`: Native GPU matvec for Q4_K with on-the-fly dequantization. Uses a thread-per-column `BlockDot` loop to ensure hardware independence across AMD and Nvidia GPUs.
- `matvec_q6_k_bda.glsl`: Native GPU matvec for Q6_K with on-the-fly 6-bit dequantization. Uses a thread-per-column `BlockDot` loop to ensure hardware independence across AMD and Nvidia GPUs.
- `get_rows_q4_k_bda.glsl`: Native GPU embedding lookup for Q4_K weights.
- `get_rows_q6_k_bda.glsl`: Native GPU embedding lookup for Q6_K weights.
- `attention_bda.glsl` / `flash_attn_bda.glsl`: Computes scaled dot-product attention over the KV cache.
- `rope_bda.glsl`: Applies Rotary Positional Embeddings.

---

## 5. Token Generation Loop

**Location:** `src/main.zig` (lines 698-802)

The token generation loop is a `while` loop that produces tokens sequentially after the prefill phase:

```zig
while (generated < max_tokens) : (generated += 1) {
    // 1. Copy logits from GPU scratchpad to staging buffer
    try vk_ctx.copyBufferOffset(scratchpad, logits_offset, logits_staging, 0, ...);

    // 2. Apply optional logit softcapping (Gemma style)
    if (cfg.final_logit_softcapping > 0.0) { ... }

    // 3. GPU top-k path (temperature=0, top_k≤1) OR CPU sampling path
    if (use_gpu_topk) {
        current_token = try dispatcher.executeTopK(logits_offset, ...);
    } else {
        current_token = try token_sampler.sample(allocator, logits_persistent, ...);
    }

    // 4. Check for stop tokens (EOS, <|end_of_text|>, <|end_of_role|>)
    for (extra_stops) |stop_id| { if (current_token == stop_id) { stopped = true; } }
    if (stopped) break;

    // 5. Update repetition history (256-token rolling window)
    if (gen_history_len < 256) { gen_history[gen_history_len] = current_token; }
    else { @memcpy(gen_history[0..255], gen_history[1..256]); ... }

    // 6. Output token
    try tok.decode(&[_]tokenizer.TokenID{current_token}, writer);

    // 7. Fetch next embedding and execute graph (GPU dequant path)
    if (embd_quant_gpu) {
        // Write token_id to indices buffer
        const mapped_idx = try vk_ctx.vkd.mapMemory(...);
        @as(*u32, @ptrCast(@alignCast(mapped_idx))).* = current_token;
        vk_ctx.vkd.unmapMemory(...);

        // Batched: embed lookup + graph dispatch in ONE command buffer
        try dispatcher.ensureSubmitResources();
        _ = vk_ctx.vkd.dispatch.vkResetCommandBuffer.?(dispatcher.cmd, ...);
        _ = vk_ctx.vkd.dispatch.vkBeginCommandBuffer.?(dispatcher.cmd, ...);
        dispatcher.recordEmbedAndGraph(dispatcher.cmd, pos, embed_indices, ...);
        _ = vk_ctx.vkd.dispatch.vkEndCommandBuffer.?(dispatcher.cmd);
        try dispatcher.submitAndWait(dispatcher.cmd);
    } else { ... }  // CPU embedding path (separate calls)

    pos += 1;
}
```

### Batched GPU Path

For quantized embeddings (`embd_quant_gpu`), the embedding lookup and graph execution are combined into a single command buffer submission per token:

1. Write `token_id` to `embed_indices` buffer (host-visible,4 bytes)
2. Reset and begin command buffer
3. `recordEmbedAndGraph()` records:
   - `get_rows_q` dispatch (GPU embedding lookup)
   - Pipeline barrier (shader_write → shader_read)
   - Full graph dispatch (all transformer layers)
4. End and submit command buffer
5. Wait for completion

This reduces per-token CPU overhead from 2 submits/wait cycles to 1.

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `dispatcher.executeTopK()` | `src/compute_graph.zig:770-806` | GPU top-k selection with subgroup reduction |
| `token_sampler.sample()` | `src/sampler.zig` | CPU sampling with temperature/top-p/min-p |
| `dispatcher.executeGetRowsQ()` | `src/compute_graph.zig:721-768` | GPU embedding lookup for quantized weights |
| `dispatcher.execute()` | `src/compute_graph.zig:667-678` | Execute full DAG for single token |
| `dispatcher.recordEmbedAndGraph()` | `src/compute_graph.zig:770+` | Batched GPU embedding lookup + graph dispatch |

### Sampling Paths

**GPU Top-K Path:** Used when `temperature=0` and `top_k≤1`. The `executeTopK` shader performs parallel reduction to find the maximum logit and returns its index.

**CPU Sampling Path:** Logits are copied to a staging buffer, mapped to CPU, and processed by `token_sampler.sample()` which supports:
- Temperature scaling
- Top-K filtering
- Top-P (Nucleus) sampling
- Min-P sampling
- Typical-P sampling
- Repetition penalty (applied against 256-token rolling history)

---

## 6. Subgroup Reduction

**Location:** Shader-based implementations in `src/shaders/`

The codebase uses **workgroup-local shared memory** for reduction patterns, NOT subgroup intrinsics (no `subgroupReduce`, `subgroupShuffle`, etc.). All reductions follow a **power-of-2 tree reduction** pattern using `barrier()` between stages.

### 6.1 Softmax Reduction (`softmax_bda.glsl`)

Two-phase reduction for numerically stable softmax:

```glsl
layout(local_size_x = 256) in;
shared float shared_max[256];
shared float shared_sum[256];

// Phase 1: Local max reduction (power-of-2 halving)
for (uint s = 128; s > 0; s >>= 1) {
    if (tid < s) shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
    barrier();
}
// Phase 2: Expsum reduction
for (uint s = 128; s > 0; s >>= 1) {
    if (tid < s) shared_sum[tid] += shared_sum[tid + s];
    barrier();
}
```

**Pattern:** 256 workitems cooperatively reduce via power-of-2 halving. Each stage halves the working set, with `barrier()` ensuring all workitems sync before the next stage.

### 6.2 TopK Reduction (`topk_bda.glsl`)

Parallel reduction to find maximum value with index tracking:

```glsl
shared float s_val[256];
shared uint s_idx[256];

// Parallel reduction to find max
for (uint off = 128u; off > 0u; off >>= 1u) {
    if (lid < off) {
        if (s_val[lid + off] > s_val[lid]) {
            s_val[lid] = s_val[lid + off];
            s_idx[lid] = s_idx[lid + off];
        }
    }
    barrier();
}
```

**Pattern:** Same power-of-2 reduction, but keeps index alongside value. Final result at `s_val[0]` / `s_idx[0]` contains the maximum and its position.

### 6.3 Attention Flash (`flash_attn_bda.glsl`)

Note: Flash attention does NOT use reduction. Instead, it employs a **tiled approach** with a loop over `tile_sz` (typically 64), accumulating results in register without shared memory reduction.

### 6.4 Reduction Pattern Summary

| Shader | Workgroup Size | Reduction Type |
|--------|---------------|----------------|
| `softmax_bda.glsl` | 256 | Power-of-2 tree (max + sum) |
| `topk_bda.glsl` | 256 | Power-of-2 tree (max with index) |
| `flash_attn_bda.glsl` | 64 | Tiled accumulation (no reduction) |

All reductions use `gl_WorkgroupBarrier()` (`memory_barrier_flag` + `workgroup_barrier_flag`) to ensure correct synchronization between workitems.

---

## 7. Shader Workgroup Tiling

**Location:** All GLSL shaders in `src/shaders/`

### 7.1 Workgroup Size by Operation

| Shader | Local Size | Type |
|--------|------------|------|
| `matmul_f16`, `matmul`, `matmul_q4_0`, `matmul_q4_1`, `matmul_q4_k`, `matmul_q6_k`, `matmul_q8_0` | 16×16 | 2D tile |
| `matvec_f16`, `matvec_q4_0`, `matvec_q4_1`, `matvec_q4_k`, `matvec_q6_k`, `matvec_q8_0` | 256×1 | 1D vector |
| `softmax`, `topk` | 256×1 | 1D reduction |
| `attention`, `flash_attn`, `kv_write`, `rope`, `rmsnorm`, `silu_mul`, `gelu_mul`, `scaled_add`, `add`, `mul`, `copy` | 64×1 | 1D |

### 7.2 Tiled Matrix Multiplication (`matmul_f16_bda.glsl`)

16×16 tile-based matrix multiplication:

```glsl
layout(local_size_x = 16, local_size_y = 16) in;
shared float a_tile[16][16];
shared float b_tile[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint tiles = (pc.k + 15u) / 16u;

    for (uint t = 0u; t < tiles; ++t) {
        // Load tile A
        a_tile[ly][lx] = (row < pc.m && kx < pc.k) ? pc.a.data[row * pc.k + kx] : 0.0;
        // Load tile B
        b_tile[ly][lx] = (col < pc.n && ky < pc.k) ? f16Weight(col, ky) : 0.0;
        barrier();
        // Inner product
        for (uint kk = 0u; kk < 16u; ++kk) {
            sum += a_tile[ly][kk] * b_tile[kk][lx];
        }
        barrier();
    }
}
```

**Pattern:**
- Each workitem computes one output element `C[row][col]`
- 16×16 tile loaded into shared memory with boundary checks
- `barrier()` between load and compute phases
- Inner product accumulated across K dimension
- Final `barrier()` before next tile iteration

### 7.3 Quantized Matmul (`matmul_q4_k_bda.glsl` and `matmul_q6_k_bda.glsl`)

Same tiling structure as `matmul_f16`, but with on-the-fly dequantization during tile loading:

**Q4_K** (`matmul_q4_k_bda.glsl`):
```glsl
// Q4_K format: 256 elements per block, 144 bytes per block
// Scales and offsets packed in first 2 floats of each block
f32 q4kWeight(uint col, uint row) {
    // Bit unpacking from nibble pairs, apply scale/offset
}
```

**Q6_K** (`matmul_q6_k_bda.glsl`):
```glsl
// Q6_K format: 256 elements per block, 210 bytes per block
// 6-bit signed values (-32 to 31), per-16-element i8 scales, float16 d scale
f32 q6kWeight(uint col, uint row) {
    // 6-bit nibble unpacking + scale + d multiplication
}
```

### 7.4 Memory Access Pattern

| Operation | Buffer Layout | Access Pattern |
|-----------|---------------|----------------|
| Matrix-Vec (quantized) | Flat weight buffer | Direct BDA pointer arithmetic |
| Matrix-Mat (tiled) | (M×K) × (K×N) | Cyclic tile loading with barrier sync |
| Attention | KV cache flat buffer | Indirect access via offset indices |
| Embedding | (vocab×hidden) flat | Index-based lookup via `executeGetRowsQ` |

### 7.5 Buffer Device Address (BDA)

All GPU buffers are created with `shader_device_address_bit` enabled. Shaders use `GL_EXT_buffer_reference2` to access memory directly via pointers:

```glsl
// Push constants structure (72 bytes)
struct PushConstants {
    uint p1, p2, p3, p4, p5, p6, p7, p8;  // u32 parameters
    uint64_t a, b, c;  // Buffer Device Address pointers
};
```

No descriptor sets are used—BDA pointers are passed directly as push constants.

---

## 8. Execution Flow Example (Decode Step)

1. `main.zig` passes the new token to the `Dispatcher`.
2. `Dispatcher.executeGetRowsQ` extracts the token's embedding from the `token_embd.weight` buffer on the GPU and stores it in the `input` tensor in the scratchpad.
3. `Dispatcher.execute()` is called. It records a Vulkan Command Buffer looping over the DAG.
4. For Layer 0:
   - `rms_norm` shader is dispatched (64 workitems).
   - `matmul_q4_k` is dispatched 3 times to generate Q, K, and V vectors.
   - `rope` shader applies positional rotation to Q and K.
   - `kv_write` appends K and V to the historical cache.
   - `attention` computes probabilities and blends V (256 workitems for softmax reduction).
   - `matmul_q4_k` projects the output.
   - Residual additions and the Feed-Forward Network (FFN) occur.
5. LM Head: `matmul` against the vocabulary weights produces the final logits.
6. The logits buffer is transferred back to the CPU (or GPU `topk` is used).
7. `Sampler.sample()` picks the next token based on Temperature and Top-P.
8. Loop repeats.

---

## 9. Memory Layout

### 9.1 Scratchpad

The scratchpad stores all activations, residuals, and intermediate results. Size is computed during graph building based on `graph.scratchpad_size`.

### 9.2 KV Cache

Flat buffer of size `max_ctx * n_kv_heads * head_dim * 2 * 2 * n_layer`:
- K and V planes stored separately (using `f16` via GLSL packing)
- Indexed by position during attention (flash attention tiles over `tile_sz` chunks)

### 9.3 Quantized Weight Storage

| Format | Elements/Block | Bytes/Block | Native GPU |
|--------|---------------|-------------|------------|
| `f32` | 1 | 4 | Yes |
| `f16` | 1 | 2 | Yes |
| `q8_0` | 32 | 34 | Partial |
| `q4_0` | 32 | 18 | Partial |
| `q4_1` | 32 | 20 | Partial |
| `q4_k` | 256 | 144 | Yes |
| `q6_k` | 256 | 210 | Yes (native GPU — matvec + matmul) |

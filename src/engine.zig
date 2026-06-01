const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const backend = @import("backend/interface.zig");
const weights = @import("weights.zig");
const tensor = @import("tensor.zig");
const vk = @import("vulkan");
const weights_loader = @import("weights_loader.zig");

// Timer helper
fn nowNs() u64 {
    if (@import("builtin").os.tag == .windows) {
        var counter: std.os.windows.LARGE_INTEGER = 0;
        var freq: std.os.windows.LARGE_INTEGER = 0;
        _ = std.os.windows.ntdll.RtlQueryPerformanceCounter(&counter);
        _ = std.os.windows.ntdll.RtlQueryPerformanceFrequency(&freq);
        const c: u64 = @intCast(counter);
        const f: u64 = @intCast(freq);
        if (f == 0) return 0;
        return (c / f) * std.time.ns_per_s + ((c % f) * std.time.ns_per_s) / f;
    }
    return 0;
}

pub fn loadEmbedding(
    ctx: *gguf.GGUFContext,
    embd: *tensor.Tensor,
    tid: tokenizer.TokenID,
    n_embd: u32,
    vk_ctx: *backend.Backend,
    staging: *backend.Buffer,
    scratch: *backend.Buffer,
    graph: *compute_graph.Graph,
    scale: f32,
) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    const dst = @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd];
    try weights.readEmbeddingF32(ctx, embd, tid, dst, n_embd, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

pub fn loadEmbeddingFromTransposedCache(cache: []const f32, embd: *tensor.Tensor, tid: tokenizer.TokenID, n_embd: u32, dst: []f32, scale: f32) !void {
    const vocab_stride: usize = @intCast(embd.ne[0]);
    if (tid >= vocab_stride) return error.TokenOutOfRange;
    for (0..n_embd) |i| {
        dst[i] = cache[i * vocab_stride + tid] * scale;
    }
}

pub fn loadEmbeddingCached(
    cache: []const f32,
    embd: *tensor.Tensor,
    tid: tokenizer.TokenID,
    n_embd: u32,
    vk_ctx: *backend.Backend,
    staging: *backend.Buffer,
    scratch: *backend.Buffer,
    graph: *compute_graph.Graph,
    scale: f32,
) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    const dst = @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd];
    try loadEmbeddingFromTransposedCache(cache, embd, tid, n_embd, dst, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

pub const EngineOptions = struct {
    max_tokens: u32,
    temperature: f32,
    seed: u64,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    debug_logits: u32,
    verbose: bool,
    inspect_block: bool,
    prefill_chunk: u32,
    gpu_embed: bool,
    report_json: bool,
};

pub fn runInference(
    allocator: std.mem.Allocator,
    vk_ctx: *backend.Backend,
    registry: *backend.PipelineRegistry,
    graph: *compute_graph.Graph,
    ctx: *gguf.GGUFContext,
    cfg: *const model.ModelConfig,
    tok: *tokenizer.Tokenizer,
    token_ids: []const tokenizer.TokenID,
    options: EngineOptions,
    t_load_start: u64,
    writer: anytype,
) !void {
    // Scratchpad and cache memory allocation in GPU
    var scratchpad = try backend.Buffer.init(vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(vk_ctx);

    const uses_q4_0_f16_fallback = blk: {
        var found = false;
        for (graph.nodes.items) |node| {
            if (node.op_type == .matmul_q and node.p5 == compute_graph.q4_0_f16_fallback_qtype) {
                found = true;
                break;
            }
        }
        break :blk found;
    };

    const kv_props: vk.MemoryPropertyFlags = if (uses_q4_0_f16_fallback)
        .{ .host_visible_bit = true, .host_coherent_bit = true }
    else
        .{ .device_local_bit = true };
    var kv_cache = try backend.Buffer.init(vk_ctx, graph.kv_cache_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, kv_props);
    defer kv_cache.deinit(vk_ctx);

    var input_staging = try backend.Buffer.init(vk_ctx, model.f32Bytes(cfg.n_embd), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer input_staging.deinit(vk_ctx);

    var logits_staging = try backend.Buffer.init(vk_ctx, model.f32Bytes(cfg.vocab_size), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer logits_staging.deinit(vk_ctx);

    var topk_indices = try backend.Buffer.init(vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer topk_indices.deinit(vk_ctx);
    var topk_values = try backend.Buffer.init(vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer topk_values.deinit(vk_ctx);

    var embed_indices = try backend.Buffer.init(vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer embed_indices.deinit(vk_ctx);

    const logits_mapped_ptr = try vk_ctx.vkd.mapMemory(vk_ctx.device, logits_staging.memory, 0, model.f32Bytes(cfg.vocab_size), .{});
    defer vk_ctx.vkd.unmapMemory(vk_ctx.device, logits_staging.memory);
    const logits_persistent = @as([*]f32, @ptrCast(@alignCast(logits_mapped_ptr)))[0..cfg.vocab_size];

    var dispatcher = try backend.vulkan.Dispatcher.init(graph, vk_ctx, registry, scratchpad, kv_cache, cfg);
    defer dispatcher.deinit();

    if (options.verbose) {
        try writer.print("[verbose] graph: {} nodes, {} tensors\n", .{ graph.nodes.items.len, graph.tensors.count() });
    }

    const embd_tensor = ctx.tensors.get("token_embd.weight") orelse return error.MissingEmbeddings;
    const embd_standard_layout = embd_tensor.ne[0] == cfg.n_embd;
    const embd_transposed_layout = embd_tensor.ne[1] == cfg.n_embd;
    var embd_cache_transposed: ?[]f32 = null;
    defer if (embd_cache_transposed) |buf| allocator.free(buf);
    if (!embd_standard_layout and embd_transposed_layout) {
        const n = embd_tensor.ne[0] * embd_tensor.ne[1] * embd_tensor.ne[2] * embd_tensor.ne[3];
        const cache = try allocator.alloc(f32, n);
        try weights.dequantToF32(ctx, embd_tensor, cache);
        embd_cache_transposed = cache;
    }

    const embd_quant_gpu = options.gpu_embed and embd_standard_layout and weights_loader.isNativeQuantType(embd_tensor.type);
    var embd_gpu_buf: ?*backend.Buffer = null;
    const embd_row_bytes: u32 = @intCast(weights.quantRowBytes(embd_tensor.type, embd_tensor.ne[0]) orelse 0);
    if (embd_quant_gpu) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt| {
            if (gt.buffer) |buf| embd_gpu_buf = @ptrCast(@alignCast(buf));
        }
    }
    const embd_scale_bits: u32 = @bitCast(cfg.embedding_scale);
    const input_offset = graph.tensors.get("input").?.offset;

    if (options.verbose) {
        try writer.print("[verbose] token_ids ({d} tokens): ", .{token_ids.len});
        for (token_ids, 0..) |tid, i| {
            try writer.print("{d}", .{tid});
            if (i < token_ids.len - 1) try writer.print(", ", .{});
        }
        try writer.print("\n", .{});
    }

    const extra_stops = blk: {
        var stops: [6]tokenizer.TokenID = undefined;
        var n: usize = 0;
        if (tok.eos_token_id) |id| {
            stops[n] = id;
            n += 1;
        }
        if (tok.special.end_of_text) |id| {
            stops[n] = id;
            n += 1;
        }
        if (tok.special.end_of_role) |id| {
            stops[n] = id;
            n += 1;
        }
        if (tok.special.eot_id) |id| {
            stops[n] = id;
            n += 1;
        }
        break :blk stops[0..n];
    };

    try writer.print("\nAssistant: ", .{});

    var pos: u32 = 0;
    var generated: u32 = 0;
    const sample_cfg = sampler.SamplerConfig{
        .temperature = options.temperature,
        .top_k = options.top_k,
        .top_p = options.top_p,
        .min_p = options.min_p,
        .typical_p = 1.0,
        .seed = if (options.seed != 0) options.seed else 0xDEADBEEF,
        .repetition_window = 0,
        .repetition_penalty = 1.0,
    };
    var token_sampler = sampler.Sampler.init(sample_cfg);

    const t_prefill_start = nowNs();
    if (token_ids.len > 0) {
        const embd_bytes = model.f32Bytes(cfg.n_embd);
        const chunk = if (options.prefill_chunk > 0) options.prefill_chunk else @as(u32, @intCast(token_ids.len));
        var chunk_start: u32 = 0;
        while (chunk_start < token_ids.len) {
            const chunk_len = @min(chunk, @as(u32, @intCast(token_ids.len)) - chunk_start);
            if (embd_quant_gpu and embd_gpu_buf != null) {
                var ti: u32 = 0;
                while (ti < chunk_len) : (ti += 1) {
                    try dispatcher.executeGetRowsQ(
                        embed_indices,
                        embd_gpu_buf.?.*,
                        input_offset,
                        token_ids[chunk_start + ti],
                        cfg.n_embd,
                        @intFromEnum(embd_tensor.type),
                        embd_row_bytes,
                        embd_scale_bits,
                    );
                    try dispatcher.execute(pos);
                    pos += 1;
                }
            } else {
                const prefill_bytes = embd_bytes * chunk_len;
                var prefill_staging = try backend.Buffer.init(vk_ctx, prefill_bytes, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
                defer prefill_staging.deinit(vk_ctx);

                const mapped_prefill = try vk_ctx.vkd.mapMemory(vk_ctx.device, prefill_staging.memory, 0, prefill_bytes, .{});
                const prefill_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_prefill)))[0 .. chunk_len * cfg.n_embd];
                for (0..chunk_len) |i| {
                    const row = prefill_f32[i * cfg.n_embd .. (i + 1) * cfg.n_embd];
                    if (embd_cache_transposed) |cache| {
                        try loadEmbeddingFromTransposedCache(cache, embd_tensor, token_ids[chunk_start + i], cfg.n_embd, row, cfg.embedding_scale);
                    } else {
                        try weights.readEmbeddingF32(ctx, embd_tensor, token_ids[chunk_start + i], row, cfg.n_embd, cfg.embedding_scale);
                    }
                }
                vk_ctx.vkd.unmapMemory(vk_ctx.device, prefill_staging.memory);
                try dispatcher.executePrefillBatch(pos, chunk_len, prefill_staging, embd_bytes);
                pos += chunk_len;
            }
            chunk_start += chunk_len;
        }
    }
    const t_prefill_end = nowNs();

    var current_token: tokenizer.TokenID = if (token_ids.len > 0) token_ids[token_ids.len - 1] else 0;

    var gen_history: [256]tokenizer.TokenID = undefined;
    var gen_history_len: usize = 0;

    if (options.inspect_block) {
        const inspect_size = model.f32Bytes(cfg.n_embd);
        var inspect_staging = try backend.Buffer.init(vk_ctx, inspect_size, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        defer inspect_staging.deinit(vk_ctx);
        {
            var layer_name_buf: [32]u8 = undefined;
            for (0..cfg.n_layer) |li| {
                const lname = if (li == cfg.n_layer - 1)
                    "hidden"
                else
                    try std.fmt.bufPrint(&layer_name_buf, "blk.{d}.out", .{li});
                const gt_l = graph.tensors.get(lname) orelse continue;
                try vk_ctx.copyBufferOffset(scratchpad, gt_l.offset, inspect_staging, 0, inspect_size);
                const mapped_l = try vk_ctx.vkd.mapMemory(vk_ctx.device, inspect_staging.memory, 0, inspect_size, .{});
                const lvals = @as([*]const f32, @ptrCast(@alignCast(mapped_l)))[0..cfg.n_embd];
                var lmin: f32 = lvals[0];
                var lmax: f32 = lvals[0];
                var lsum: f32 = 0;
                var lsumsq: f32 = 0;
                for (lvals) |v| {
                    if (v < lmin) lmin = v;
                    if (v > lmax) lmax = v;
                    lsum += v;
                    lsumsq += v * v;
                }
                const lmean = lsum / @as(f32, @floatFromInt(cfg.n_embd));
                const lstd = @sqrt(lsumsq / @as(f32, @floatFromInt(cfg.n_embd)) - lmean * lmean);
                try writer.print("[layer] {d:2} {s}: min={d:.3} max={d:.3} mean={d:.3} std={d:.3}\n", .{ li, lname, lmin, lmax, lmean, lstd });
                vk_ctx.vkd.unmapMemory(vk_ctx.device, inspect_staging.memory);
            }
        }
        const names_to_dump = [_][]const u8{ "input", "blk.0.normed", "blk.0.attn", "blk.0.attn_out", "blk.0.res1", "hidden", "final.normed" };
        for (names_to_dump) |tname| {
            const gt = graph.tensors.get(tname) orelse continue;
            const sz = @min(inspect_size, gt.size);
            try vk_ctx.copyBufferOffset(scratchpad, gt.offset, inspect_staging, 0, sz);
            const mapped_i = try vk_ctx.vkd.mapMemory(vk_ctx.device, inspect_staging.memory, 0, sz, .{});
            const vals = @as([*]const f32, @ptrCast(@alignCast(mapped_i)))[0..@min(8, sz / 4)];
            try writer.print("[inspect] {s}: ", .{tname});
            for (vals) |v| try writer.print("{d:.4} ", .{v});
            try writer.print("\n", .{});
            vk_ctx.vkd.unmapMemory(vk_ctx.device, inspect_staging.memory);
        }
    }

    if (options.debug_logits > 0) {
        const t_logits = graph.tensors.get("logits").?;
        try vk_ctx.copyBufferOffset(scratchpad, t_logits.offset, logits_staging, 0, model.f32Bytes(cfg.vocab_size));
        var indices = try allocator.alloc(u32, cfg.vocab_size);
        defer allocator.free(indices);
        for (indices, 0..) |*v, i| v.* = @as(u32, @intCast(i));
        std.mem.sort(u32, indices, logits_persistent, struct {
            fn lessThan(probs: []const f32, a: u32, b: u32) bool {
                return probs[a] > probs[b];
            }
        }.lessThan);
        try writer.print("\n[top {d} logits after prefill]\n", .{options.debug_logits});
        for (indices[0..options.debug_logits]) |idx| {
            try writer.print("  id={:6}  logit={:9.4}  token=", .{ idx, logits_persistent[idx] });
            try tok.decode(&[_]tokenizer.TokenID{idx}, writer);
            try writer.print("\n", .{});
        }
    }

    var stopped: bool = false;
    const t_decode_start = nowNs();
    const logits_offset = graph.tensors.get("logits").?.offset;
    const logit_scale_bits: u32 = if (cfg.logit_scale != 1.0 and cfg.logit_scale != 0.0) @bitCast(cfg.logit_scale) else 0;
    const use_gpu_topk = options.temperature <= 0.0 and options.top_k <= 1 and cfg.final_logit_softcapping <= 0.0;
    while (generated < options.max_tokens) : (generated += 1) {
        try vk_ctx.copyBufferOffset(scratchpad, logits_offset, logits_staging, 0, model.f32Bytes(cfg.vocab_size));

        if (cfg.final_logit_softcapping > 0.0) {
            const s = cfg.final_logit_softcapping;
            for (logits_persistent) |*v| {
                v.* = std.math.tanh(v.* / s) * s;
            }
        }

        if (use_gpu_topk) {
            current_token = try dispatcher.executeTopK(logits_offset, cfg.vocab_size, topk_indices, topk_values, logit_scale_bits);
        } else {
            if (cfg.logit_scale != 1.0 and cfg.logit_scale != 0.0) {
                const inv = 1.0 / cfg.logit_scale;
                for (logits_persistent) |*v| v.* *= inv;
            }
            current_token = try token_sampler.sample(allocator, logits_persistent, gen_history[0..gen_history_len]);
        }

        if (options.verbose and options.debug_logits > 0) {
            try vk_ctx.copyBufferOffset(scratchpad, graph.tensors.get("logits").?.offset, logits_staging, 0, model.f32Bytes(cfg.vocab_size));
            const logits_s = logits_persistent;
            var indices_s = try allocator.alloc(u32, cfg.vocab_size);
            defer allocator.free(indices_s);
            for (indices_s, 0..) |*v, i| v.* = @as(u32, @intCast(i));
            std.mem.sort(u32, indices_s, logits_s, struct {
                fn lessThan(probs: []const f32, a: u32, b: u32) bool {
                    return probs[a] > probs[b];
                }
            }.lessThan);
            try writer.print("\n[step {} top {} logits]\n", .{ pos, options.debug_logits });
            for (indices_s[0..options.debug_logits]) |idx| {
                try writer.print("  id={:6}  logit={:9.4}  token=", .{ idx, logits_s[idx] });
                try tok.decode(&[_]tokenizer.TokenID{idx}, writer);
                try writer.print("\n", .{});
            }
        }

        for (extra_stops) |stop_id| {
            if (current_token == stop_id) {
                try writer.print("\n[stop token {}]\n", .{stop_id});
                stopped = true;
                break;
            }
        }
        if (stopped) break;

        if (gen_history_len < 256) {
            gen_history[gen_history_len] = current_token;
            gen_history_len += 1;
        } else {
            @memcpy(gen_history[0..255], gen_history[1..256]);
            gen_history[255] = current_token;
        }
        try tok.decode(&[_]tokenizer.TokenID{current_token}, writer);

        if (generated + 1 >= options.max_tokens) break;
        if (embd_quant_gpu and embd_gpu_buf != null) {
            const mapped_idx = try vk_ctx.vkd.mapMemory(vk_ctx.device, embed_indices.memory, 0, 4, .{});
            @as(*u32, @ptrCast(@alignCast(mapped_idx))).* = current_token;
            vk_ctx.vkd.unmapMemory(vk_ctx.device, embed_indices.memory);

            try dispatcher.ensureSubmitResources();
            const reset_flags = if ((dispatcher.submit_count & 63) == 63)
                vk.CommandBufferResetFlags{ .release_resources_bit = true }
            else
                vk.CommandBufferResetFlags{};
            _ = vk_ctx.vkd.dispatch.vkResetCommandBuffer.?(dispatcher.cmd, reset_flags);
            _ = vk_ctx.vkd.dispatch.vkBeginCommandBuffer.?(dispatcher.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });
            dispatcher.recordEmbedAndGraph(
                dispatcher.cmd,
                pos,
                embed_indices,
                embd_gpu_buf.?.*,
                input_offset,
                cfg.n_embd,
                @intFromEnum(embd_tensor.type),
                embd_row_bytes,
                embd_scale_bits,
            );
            _ = vk_ctx.vkd.dispatch.vkEndCommandBuffer.?(dispatcher.cmd);
            try dispatcher.submitAndWait(dispatcher.cmd);
        } else {
            if (embd_cache_transposed) |cache| {
                try loadEmbeddingCached(cache, embd_tensor, current_token, cfg.n_embd, vk_ctx, &input_staging, &scratchpad, graph, cfg.embedding_scale);
            } else {
                try loadEmbedding(ctx, embd_tensor, current_token, cfg.n_embd, vk_ctx, &input_staging, &scratchpad, graph, cfg.embedding_scale);
            }

            if (options.verbose) {
                const inp_t = graph.tensors.get("input").?;
                const inp_sz = @min(model.f32Bytes(8), inp_t.size);
                var dbg_staging = try backend.Buffer.init(vk_ctx, inp_sz, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
                defer dbg_staging.deinit(vk_ctx);
                try vk_ctx.copyBufferOffset(scratchpad, inp_t.offset, dbg_staging, 0, inp_sz);
                const dbg_mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, dbg_staging.memory, 0, inp_sz, .{});
                const dbg_vals = @as([*]const f32, @ptrCast(@alignCast(dbg_mapped)))[0..8];
                try writer.print("[debug] input[0..8] before execute({}): ", .{pos});
                for (dbg_vals) |v| try writer.print("{d:.4} ", .{v});
                try writer.print("(token={})\n", .{current_token});
                vk_ctx.vkd.unmapMemory(vk_ctx.device, dbg_staging.memory);
            }

            try dispatcher.execute(pos);
        }
        pos += 1;
    }
    const t_decode_end = nowNs();

    try writer.print("\n\n[Inference Complete]\n", .{});
    const load_s = @as(f64, @floatFromInt(t_prefill_start - t_load_start)) / 1e9;
    const prefill_s = @as(f64, @floatFromInt(t_prefill_end - t_prefill_start)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(t_decode_end - t_decode_start)) / 1e9;
    const prefill_tps = if (prefill_s > 0 and token_ids.len > 0) @as(f64, @floatFromInt(token_ids.len)) / prefill_s else 0.0;
    const decode_tps = if (decode_s > 0 and generated > 0) @as(f64, @floatFromInt(generated)) / decode_s else 0.0;
    try writer.print("[ Load: {d:.2}s ]\n", .{load_s});
    try writer.print("[ Prompt: {d:.1} t/s | Generation: {d:.1} t/s ]\n", .{ prefill_tps, decode_tps });
    
    if (options.report_json) {
        const graph_cost = compute_graph.estimateGraphCost(graph);
        try writer.print("{{\"load_s\":{d:.6},\"prefill_s\":{d:.6},\"decode_s\":{d:.6},\"prompt_tps\":{d:.4},\"generation_tps\":{d:.4},\"tokens_prompt\":{},\"tokens_generated\":{},\"graph_nodes\":{},\"graph_flops\":{},\"graph_bytes\":{}}}\n", .{
            load_s, prefill_s, decode_s, prefill_tps, decode_tps, token_ids.len, generated, graph_cost.total_nodes, graph_cost.approx_flops, graph_cost.approx_bytes,
        });
    }
}

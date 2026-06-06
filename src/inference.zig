const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const compute_graph = @import("compute_graph.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const weights = @import("weights.zig");
const sampler = @import("sampler.zig");
const ssm = @import("ssm_state.zig");
const cli = @import("cli.zig");

pub const InferenceContext = struct {
    allocator: std.mem.Allocator,
    vk_ctx: *vulkan.Context,
    ctx: *gguf.GGUFContext,
    cfg: *const model.ModelConfig,
    cli_cfg: *const cli.CliConfig,
    tok: *tokenizer.Tokenizer,
    graph: *compute_graph.Graph,
    dispatcher: *@import("dispatcher.zig").Dispatcher,
    
    scratchpad: vulkan.Buffer,
    ssm_conv_cache: vulkan.Buffer,
    ssm_state_cache: vulkan.Buffer,
    ssm_conv_size: u64,
    ssm_state_size: u64,
    
    input_staging: vulkan.Buffer,
    logits_staging: vulkan.Buffer,
    hidden_staging: vulkan.Buffer,
    embed_indices: vulkan.Buffer,
    logits_persistent: []f32,
    
    embd_tensor: *@import("tensor.zig").Tensor,
    embd_standard_layout: bool,
    embd_cache_transposed: ?[]f32,
    embd_quant_gpu: bool,
    embd_gpu_buf: ?vulkan.Buffer,
    embd_scale_bits: u32,
    
    pos: *u32,
    generated: *u32,
    current_token: *tokenizer.TokenID,
    gen_history: *[256]tokenizer.TokenID,
    gen_history_len: *usize,
    token_sampler: *sampler.Sampler,
};

pub fn run_prefill(ictx: InferenceContext, token_ids: []const tokenizer.TokenID, writer: anytype) !void {
    if (token_ids.len == 0) return;
    
    const input_offset = ictx.graph.tensors.get("input").?.offset;
    const chunk = if (ictx.cli_cfg.prefill_chunk > 0) ictx.cli_cfg.prefill_chunk else @as(u32, @intCast(token_ids.len));
    var chunk_start: u32 = 0;
    while (chunk_start < token_ids.len) {
        const chunk_len = @min(chunk, @as(u32, @intCast(token_ids.len)) - chunk_start);
        if (ictx.embd_quant_gpu and ictx.embd_gpu_buf != null) {
            var ti: u32 = 0;
            while (ti < chunk_len) : (ti += 1) {
                try ictx.dispatcher.execute_get_rows_q(ictx.embed_indices, ictx.embd_gpu_buf.?, input_offset, token_ids[chunk_start + ti], ictx.cfg.n_embd, @intFromEnum(ictx.embd_tensor.type), @intCast(weights.quantRowBytes(ictx.embd_tensor.type, ictx.embd_tensor.ne[0]) orelse 0), ictx.embd_scale_bits);
                try ictx.dispatcher.execute(ictx.pos.*); 
                ictx.pos.* += 1;
            }
        } else {
            const prefill_bytes = model.f32Bytes(ictx.cfg.n_embd) * chunk_len;
            var prefill_staging = try vulkan.Buffer.init(ictx.vk_ctx, prefill_bytes, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
            defer prefill_staging.deinit(ictx.vk_ctx);
            const p_mapped = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, prefill_staging.memory, 0, prefill_bytes, .{});
            const p_f32 = @as([*]f32, @ptrCast(@alignCast(p_mapped)))[0 .. chunk_len * ictx.cfg.n_embd];
            for (0..chunk_len) |i| {
                const row = p_f32[i * ictx.cfg.n_embd .. (i + 1) * ictx.cfg.n_embd];
                if (ictx.embd_cache_transposed) |cache| try loadEmbeddingFromTransposedCache(cache, ictx.embd_tensor, token_ids[chunk_start + i], ictx.cfg.n_embd, row, ictx.cfg.embedding_scale)
                else try weights.readEmbeddingF32(ictx.ctx, ictx.embd_tensor, token_ids[chunk_start + i], row, ictx.cfg.n_embd, ictx.cfg.embedding_scale);
            }
            ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, prefill_staging.memory);
            try ictx.dispatcher.execute_prefill_batch(ictx.pos.*, chunk_len, prefill_staging, model.f32Bytes(ictx.cfg.n_embd));
            ictx.pos.* += chunk_len;
        }
        chunk_start += chunk_len;
    }

    if (ictx.cli_cfg.debug_hidden) {
        // Also check input right after the prefill copies it
        const t_in = ictx.graph.tensors.get("input").?;
        try ictx.vk_ctx.copyBufferOffset(ictx.scratchpad, t_in.offset, ictx.hidden_staging, 0, model.f32Bytes(ictx.cfg.n_embd));
        const hm = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory, 0, model.f32Bytes(ictx.cfg.n_embd), .{});
        const hf = @as([*]f32, @ptrCast(@alignCast(hm)))[0..ictx.cfg.n_embd];
        var in_nan: u32 = 0;
        var in_finite: u32 = 0;
        var in_max: f32 = 0.0;
        for (hf) |v| {
            if (std.math.isNan(v)) in_nan += 1
            else if (std.math.isFinite(v)) { in_finite += 1; const av = @abs(v); if (av > in_max) in_max = av; }
        }
        ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory);
        try writer.print("\n  [debug-input] offset={d} nan={} finite={} max={d:.4}\n", .{ t_in.offset, in_nan, in_finite, in_max });
    }

    if (ictx.cli_cfg.debug_hidden) {
        // Check key tensors after prefill
        const check_names = [_][]const u8{ "input", "blk.0.res1", "blk.0.ffn_normed", "blk.0.ffn_out", "blk.0.out", "blk.3.out", "blk.4.out" };
        for (check_names) |tname| {
            const t = ictx.graph.tensors.get(tname) orelse continue;
            try writer.print("\n  [dbg] {s} offset={d} size={d}", .{ tname, t.offset, t.size });
            try ictx.vk_ctx.copyBufferOffset(ictx.scratchpad, t.offset, ictx.hidden_staging, 0, model.f32Bytes(ictx.cfg.n_embd));
            const h_mapped = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory, 0, model.f32Bytes(ictx.cfg.n_embd), .{});
            const h_f32 = @as([*]f32, @ptrCast(@alignCast(h_mapped)))[0..ictx.cfg.n_embd];
            var h_nan: u32 = 0;
            var h_finite: u32 = 0;
            var h_max_abs: f32 = 0.0;
            for (h_f32) |v| {
                if (std.math.isNan(v)) {
                    h_nan += 1;
                } else if (std.math.isFinite(v)) {
                    h_finite += 1;
                    const av = @abs(v);
                    if (av > h_max_abs) h_max_abs = av;
                }
            }
            ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory);
            try writer.print("\n  [debug-prefill] {s}: nan={} finite={} max_abs={d:.4}", .{ tname, h_nan, h_finite, h_max_abs });
        }
        try writer.print("\n", .{});
    }
    ictx.current_token.* = token_ids[token_ids.len - 1];
}

pub fn run_decode(ictx: InferenceContext, writer: anytype) !void {
    const logits_offset = ictx.graph.tensors.get("logits").?.offset;
    const input_offset = ictx.graph.tensors.get("input").?.offset;
    const hidden_offset = ictx.graph.tensors.get("final.normed").?.offset;

    while (ictx.generated.* < ictx.cli_cfg.max_tokens) : (ictx.generated.* += 1) {
        if (ictx.cli_cfg.verbose) try writer.print("[decode] step {} start\n", .{ictx.generated.*});

        try ictx.vk_ctx.copyBufferOffset(ictx.scratchpad, logits_offset, ictx.logits_staging, 0, model.f32Bytes(ictx.cfg.vocab_size));

        if (ictx.cli_cfg.debug_hidden) {
            try ictx.vk_ctx.copyBufferOffset(ictx.scratchpad, hidden_offset, ictx.hidden_staging, 0, model.f32Bytes(ictx.cfg.n_embd));
            const h_mapped = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory, 0, model.f32Bytes(ictx.cfg.n_embd), .{});
            const h_f32 = @as([*]f32, @ptrCast(@alignCast(h_mapped)))[0..ictx.cfg.n_embd];
            var h_nan: u32 = 0;
            var h_neg_inf: u32 = 0;
            var h_pos_inf: u32 = 0;
            var h_max_abs: f32 = 0.0;
            var h_finite: u32 = 0;
            for (h_f32) |v| {
                if (std.math.isNan(v)) { h_nan += 1; continue; }
                if (v == -std.math.inf(f32)) { h_neg_inf += 1; continue; }
                if (v == std.math.inf(f32)) { h_pos_inf += 1; continue; }
                h_finite += 1;
                const av = @abs(v);
                if (av > h_max_abs) h_max_abs = av;
            }
            ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.hidden_staging.memory);
            try writer.print("\n  [debug-hidden] step={} nan={} neg_inf={} pos_inf={} finite={} max_abs={d:.4}\n", .{ ictx.generated.*, h_nan, h_neg_inf, h_pos_inf, h_finite, h_max_abs });
        }

        if (ictx.cli_cfg.debug_logits > 0) {
            try writer.print("\n  [debug] Top {} logits:\n", .{ictx.cli_cfg.debug_logits});
            var top_indices = try ictx.allocator.alloc(usize, ictx.cli_cfg.debug_logits);
            defer ictx.allocator.free(top_indices);
            var top_values = try ictx.allocator.alloc(f32, ictx.cli_cfg.debug_logits);
            defer ictx.allocator.free(top_values);
            @memset(top_values, -std.math.inf(f32));
            @memset(top_indices, 0);

            for (ictx.logits_persistent, 0..) |v, i| {
                var val = v;
                if (std.math.isNan(val)) val = -std.math.inf(f32);
                var idx = i;
                for (0..ictx.cli_cfg.debug_logits) |j| {
                    if (val > top_values[j]) {
                        const tmp_v = top_values[j];
                        const tmp_i = top_indices[j];
                        top_values[j] = val;
                        top_indices[j] = idx;
                        val = tmp_v;
                        idx = tmp_i;
                    }
                }
            }

            for (0..ictx.cli_cfg.debug_logits) |j| {
                const id = top_indices[j];
                const t_str = if (id < ictx.tok.id_to_token.len) ictx.tok.id_to_token[id] else "??";
                try writer.print("    {d:4}: {d:8.4}  '{s}'\n", .{ id, top_values[j], t_str });
            }
            try writer.print("\n", .{});
        }

        ictx.current_token.* = try ictx.token_sampler.sample(ictx.allocator, ictx.logits_persistent, ictx.gen_history[0..ictx.gen_history_len.*]);

        if (ictx.tok.eos_token_id) |eos| if (ictx.current_token.* == eos) break;
        if (ictx.gen_history_len.* < 256) { ictx.gen_history[ictx.gen_history_len.*] = ictx.current_token.*; ictx.gen_history_len.* += 1; }
        else { std.mem.copyForwards(tokenizer.TokenID, ictx.gen_history[0..255], ictx.gen_history[1..256]); ictx.gen_history[255] = ictx.current_token.*; }
        
        // Use an array literal of size 1, then cast to slice.
        const c_arr = [_]tokenizer.TokenID{ictx.current_token.*};
        try ictx.tok.decode(c_arr[0..], writer);
        try writer.flush();

        if (ictx.embd_quant_gpu and ictx.embd_gpu_buf != null) {
            const m_idx = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.embed_indices.memory, 0, 4, .{});
            @as(*u32, @ptrCast(@alignCast(m_idx))).* = ictx.current_token.*;
            ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.embed_indices.memory);
            try ictx.dispatcher.execute_get_rows_q(ictx.embed_indices, ictx.embd_gpu_buf.?, input_offset, ictx.current_token.*, ictx.cfg.n_embd, @intFromEnum(ictx.embd_tensor.type), @intCast(weights.quantRowBytes(ictx.embd_tensor.type, ictx.embd_tensor.ne[0]) orelse 0), ictx.embd_scale_bits);
            try ictx.dispatcher.execute(ictx.pos.*);
        } else {
            try loadEmbedding(ictx.ctx, ictx.embd_tensor, ictx.current_token.*, ictx.cfg.n_embd, ictx.vk_ctx, ictx.input_staging, ictx.scratchpad, ictx.graph, ictx.cfg.embedding_scale);
            try ictx.dispatcher.execute(ictx.pos.*);
        }
        ictx.pos.* += 1;

        if (ictx.cfg.arch == .qwen35 and ictx.graph.ssm_cache_size > 0) {
            const head_v_dim_ssm = if (ictx.cfg.ssm_dt_rank > 0) ictx.cfg.ssm_d_inner / ictx.cfg.ssm_dt_rank else 0;
            const m_conv = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.ssm_conv_cache.memory, 0, ictx.ssm_conv_size, .{});
            const m_state = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.ssm_state_cache.memory, 0, ictx.ssm_state_size, .{});
            const m_scratch = try ictx.vk_ctx.vkd.mapMemory(ictx.vk_ctx.device, ictx.scratchpad.memory, 0, ictx.graph.scratchpad_size, .{});
            
            defer { 
                ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.ssm_conv_cache.memory); 
                ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.ssm_state_cache.memory); 
                ictx.vk_ctx.vkd.unmapMemory(ictx.vk_ctx.device, ictx.scratchpad.memory);
            }
            
            var ssm_ctx = ssm.SsmCpuContext.wrap(ictx.cfg, @as([*]f32, @ptrCast(@alignCast(m_conv)))[0..ictx.ssm_conv_size/4], @as([*]f32, @ptrCast(@alignCast(m_state)))[0..ictx.ssm_state_size/4], ictx.allocator);
            var sl: u32 = 0;
            const scratch_f32 = @as([*]f32, @ptrCast(@alignCast(m_scratch)));
            while (sl < ictx.cfg.n_layer) : (sl += 1) {
                if (ictx.cfg.isRecurrent(sl)) {
                    // Extracting the delta logic to ssm_cpu.zig is planned next, but we inline the logic here for now or call runCpuSsmDeltaForLayer which we must move too.
                    try runCpuSsmDeltaForLayer(ictx.allocator, scratch_f32, ictx.logits_persistent, &ssm_ctx, ictx.graph, sl, head_v_dim_ssm);
                }
            }
        }
    }
}

fn loadEmbedding(ctx: *gguf.GGUFContext, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, vk_ctx: *vulkan.Context, staging: vulkan.Buffer, scratch: vulkan.Buffer, graph: *compute_graph.Graph, scale: f32) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    try weights.readEmbeddingF32(ctx, embd, tid, @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd], n_embd, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging, 0, scratch, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

fn loadEmbeddingFromTransposedCache(cache: []const f32, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, dst: []f32, scale: f32) !void {
    const vocab_stride: usize = @intCast(embd.ne[0]);
    if (tid >= vocab_stride) return error.TokenOutOfRange;
    for (0..n_embd) |i| dst[i] = cache[i * vocab_stride + tid] * scale;
}

/// Run the CPU-side Gated Delta Net recurrence for one layer of one prefill
/// or decode token. The `core` output is written into the scratchpad at
/// `core_t.offset`.
pub fn runCpuSsmDeltaForLayer(
    allocator: std.mem.Allocator,
    scratch_ptr: ?[*]f32,
    logits_persistent: []f32,
    ssm_ctx: *ssm.SsmCpuContext,
    graph: *const compute_graph.Graph,
    layer: u32,
    head_v_dim: u32,
) !void {
    var ln_buf: [32]u8 = undefined;
    const ln = std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer}) catch return;

    var qn_buf: [32]u8 = undefined;  const qn_name   = std.fmt.bufPrint(&qn_buf,   "{s}.q_norm", .{ln}) catch return;
    var kn_buf: [32]u8 = undefined;  const kn_name   = std.fmt.bufPrint(&kn_buf,   "{s}.k_norm", .{ln}) catch return;
    var vn_buf: [32]u8 = undefined;  const vn_name   = std.fmt.bufPrint(&vn_buf,   "{s}.v_conv", .{ln}) catch return;
    var gate_buf: [32]u8 = undefined; const gate_name = std.fmt.bufPrint(&gate_buf, "{s}.gate",   .{ln}) catch return;
    var beta_buf: [32]u8 = undefined; const beta_name = std.fmt.bufPrint(&beta_buf, "{s}.beta",   .{ln}) catch return;
    var core_buf: [32]u8 = undefined; const core_name = std.fmt.bufPrint(&core_buf, "{s}.core",   .{ln}) catch return;

    const qn_t = graph.resolve_tensor_offset(qn_name) orelse return;
    const kn_t = graph.resolve_tensor_offset(kn_name) orelse return;
    const vn_t = graph.resolve_tensor_offset(vn_name) orelse return;
    const gate_t = graph.resolve_tensor_offset(gate_name) orelse return;
    const beta_t = graph.resolve_tensor_offset(beta_name) orelse return;
    const core_t = graph.resolve_tensor_offset(core_name) orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));

    if (scratch_ptr) |s_ptr| {
        const q_in = s_ptr[qn_t.offset / 4 ..][0 .. @as(usize, qn_t.size) / 4];
        const k_in = s_ptr[kn_t.offset / 4 ..][0 .. @as(usize, kn_t.size) / 4];
        const v_in = s_ptr[vn_t.offset / 4 ..][0 .. @as(usize, vn_t.size) / 4];
        const g_in = s_ptr[gate_t.offset / 4 ..][0 .. @as(usize, gate_t.size) / 4];
        const b_in = s_ptr[beta_t.offset / 4 ..][0 .. @as(usize, beta_t.size) / 4];
        const core_out = s_ptr[core_t.offset / 4 ..][0 .. @as(usize, core_t.size) / 4];

        try ssm_ctx.stepDeltaNet(layer, q_in, k_in, v_in, g_in, b_in, core_out, scale);
    } else {
        _ = logits_persistent;
    }
    _ = allocator;
}
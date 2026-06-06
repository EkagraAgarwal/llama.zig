const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const kernels_data = @import("kernels_data");
const compute_graph = @import("compute_graph.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const weights = @import("weights.zig");
const sampler = @import("sampler.zig");
const chat = @import("chat.zig");
const ssm = @import("ssm_state.zig");
const cli = @import("cli.zig");
const weight_uploader = @import("weight_uploader.zig");
const inference = @import("inference.zig");
const builtin = @import("builtin");
const vk = @import("vulkan");
const windows = if (builtin.os.tag == .windows) std.os.windows else struct {};

fn nowNs() u64 {
    if (builtin.os.tag == .windows) {
        var counter: windows.LARGE_INTEGER = 0;
        var freq: windows.LARGE_INTEGER = 0;
        _ = windows.ntdll.RtlQueryPerformanceCounter(&counter);
        _ = windows.ntdll.RtlQueryPerformanceFrequency(&freq);
        const c: u64 = @intCast(counter);
        const f: u64 = @intCast(freq);
        if (f == 0) return 0;
        return (c / f) * std.time.ns_per_s + ((c % f) * std.time.ns_per_s) / f;
    }
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const stdout_file = std.Io.File.stdout();
    var buffer: [4096]u8 = undefined;
    var writer_streaming = stdout_file.writerStreaming(init.io, &buffer);
    const writer = &writer_streaming.interface;

    try writer.print("llama.zig: High-Performance Vulkan Inference (v1.0.1-fix)\n", .{});

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();

    const cli_cfg = try cli.CliConfig.parse(&args_it);

    if (cli_cfg.model_path == null or cli_cfg.prompt_text == null) {
        try writer.print("Usage: llama.zig --model <path.gguf> --prompt '<text>' [--max-tokens N] [--cli_cfg.temperature T]\n", .{});
        try writer_streaming.interface.flush();
        return;
    }

    const t_load_start = nowNs();
    try writer.print("Loading model: {s}...\n", .{cli_cfg.model_path.?});
    try writer_streaming.interface.flush();
    var ctx = try gguf.loadModelMmap(allocator, cli_cfg.model_path.?);
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

    const vocab_size: u32 = @intCast(tok.id_to_token.len);
    var cfg = try model.ModelConfig.init(allocator, &ctx, vocab_size);
    defer cfg.deinit(allocator);
    if (cli_cfg.top_k > cfg.vocab_size) return error.InvalidCommandLineArguments;
    if (cli_cfg.ctx_size_override) |ctx_sz| {
        if (ctx_sz > 0) cfg.max_ctx = ctx_sz;
    } else if (cfg.max_ctx > 8192) {
        cfg.max_ctx = 8192;
    }

    try writer.print("Config: arch={s} act={s} L={} D={} H={} KV={} FF={} ctx={} rope={d:.0}\n", .{
        cfg.arch_prefix, @tagName(cfg.activation), cfg.n_layer, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads, cfg.n_ff, cfg.max_ctx, cfg.rope_theta,
    });
    try writer.print("Scales: emb={d} attn={d} res={d} logit={d}\n", .{
        cfg.embedding_scale, cfg.attention_scale, cfg.residual_scale, cfg.logit_scale,
    });
    try writer_streaming.interface.flush();
    validateModelLayout(&ctx) catch |err| {
        try writer.print("Warning: model layout check failed ({s}); continuing with dynamic graph assumptions.\n", .{@errorName(err)});
    };
    {
        var ts_buf: [64]u8 = undefined;
        const ts = std.fmt.bufPrint(&ts_buf, "[phase] post-validate t={d:.1}s\n", .{
            @as(f64, @floatFromInt(nowNs() - t_load_start)) / 1e9,
        }) catch "";
        try writer.print("{s}", .{ts});
        try writer_streaming.interface.flush();
    }

    var vk_ctx = try vulkan.Context.init(allocator);
    defer vk_ctx.deinit();

    var registry = try vulkan.PipelineRegistry.init(allocator);
    defer registry.deinit(&vk_ctx);

    try registry.register(&vk_ctx, "add", kernels_data.kernels_add_spv, "main");
    try registry.register(&vk_ctx, "mul", kernels_data.kernels_mul_spv, "main");
    try registry.register(&vk_ctx, "rms_norm", kernels_data.kernels_rmsnorm_spv, "main");
    try registry.register(&vk_ctx, "softmax", kernels_data.kernels_softmax_spv, "main");
    try registry.register(&vk_ctx, "matmul", kernels_data.kernels_matmul_spv, "main");
    try registry.register(&vk_ctx, "rope", kernels_data.kernels_rope_spv, "main");
    try registry.register(&vk_ctx, "silu_mul", kernels_data.kernels_silu_mul_spv, "main");
    try registry.register(&vk_ctx, "attention", kernels_data.kernels_attention_spv, "main");
    try registry.register(&vk_ctx, "kv_write", kernels_data.kernels_kv_write_spv, "main");
    try registry.register(&vk_ctx, "scaled_add", kernels_data.kernels_scaled_add_spv, "main");
    try registry.register(&vk_ctx, "matmul_q8_0", kernels_data.kernels_matmul_q8_0_spv, "main");
    try registry.register(&vk_ctx, "matvec_q8_0", kernels_data.kernels_matvec_q8_0_spv, "main");
    try registry.register(&vk_ctx, "matmul_f16", kernels_data.kernels_matmul_f16_spv, "main");
    try registry.register(&vk_ctx, "matvec_f16", kernels_data.kernels_matvec_f16_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q", kernels_data.kernels_get_rows_q_spv, "main");
    try registry.register(&vk_ctx, "matmul_q4_0", kernels_data.kernels_matmul_q4_0_spv, "main");
    try registry.register(&vk_ctx, "matvec_q4_0", kernels_data.kernels_matvec_q4_0_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q4_0", kernels_data.kernels_get_rows_q4_0_spv, "main");
    try registry.register(&vk_ctx, "matvec_q4_1", kernels_data.kernels_matvec_q4_1_spv, "main");
    try registry.register(&vk_ctx, "matmul_q4_1", kernels_data.kernels_matmul_q4_1_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q4_1", kernels_data.kernels_get_rows_q4_1_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q4_k", kernels_data.kernels_get_rows_q4_k_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q5_k", kernels_data.kernels_get_rows_q5_k_spv, "main");
    try registry.register(&vk_ctx, "matvec_q4_k", kernels_data.kernels_matvec_q4_k_spv, "main");
    try registry.register(&vk_ctx, "matvec_q5_k", kernels_data.kernels_matvec_q5_k_spv, "main");
    try registry.register(&vk_ctx, "matmul_q4_k", kernels_data.kernels_matmul_q4_k_spv, "main");
    try registry.register(&vk_ctx, "matmul_q5_k", kernels_data.kernels_matmul_q5_k_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q6_k", kernels_data.kernels_get_rows_q6_k_spv, "main");
    try registry.register(&vk_ctx, "matvec_q6_k", kernels_data.kernels_matvec_q6_k_spv, "main");
    try registry.register(&vk_ctx, "matmul_q6_k", kernels_data.kernels_matmul_q6_k_spv, "main");
    try registry.register(&vk_ctx, "topk", kernels_data.kernels_topk_spv, "main");
    try registry.register(&vk_ctx, "attention_flash", kernels_data.kernels_flash_attn_spv, "main");
    try registry.register(&vk_ctx, "gelu_mul", kernels_data.kernels_gelu_mul_spv, "main");
    try registry.register(&vk_ctx, "copy", kernels_data.kernels_copy_spv, "main");
    try registry.register(&vk_ctx, "softplus", kernels_data.kernels_softplus_spv, "main");
    try registry.register(&vk_ctx, "sigmoid", kernels_data.kernels_sigmoid_spv, "main");
    try registry.register(&vk_ctx, "silu", kernels_data.kernels_silu_spv, "main");
    try registry.register(&vk_ctx, "l2_norm", kernels_data.kernels_l2_norm_spv, "main");
    try registry.register(&vk_ctx, "mrope", kernels_data.kernels_mrope_spv, "main");
    try registry.register(&vk_ctx, "attn_gate_mul", kernels_data.kernels_attn_gate_mul_spv, "main");
    try registry.register(&vk_ctx, "ssm_conv1d", kernels_data.kernels_ssm_conv1d_spv, "main");
    try registry.register(&vk_ctx, "ssm_delta_net_decode", kernels_data.kernels_ssm_delta_net_decode_spv, "main");
    try registry.register(&vk_ctx, "ssm_gated_norm", kernels_data.kernels_ssm_gated_norm_spv, "main");
    try registry.register(&vk_ctx, "qwen_deinterleave", kernels_data.kernels_qwen_deinterleave_spv, "main");


    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph, &cfg, &ctx.tensors);

    try builder.add_tensor("input", model.f32Bytes(cfg.n_embd), .input);
    try builder.init_kv_caches();
    try builder.init_ssm_caches();

    var prev_out: []const u8 = "input";
    var l: u32 = 0;
    while (l < cfg.n_layer) : (l += 1) {
        const out_owned = if (l == cfg.n_layer - 1)
            try allocator.dupe(u8, "hidden")
        else
            try std.fmt.allocPrint(allocator, "blk.{d}.out", .{l});
        defer allocator.free(out_owned);
        switch (cfg.arch) {
            .qwen35 => {
                const qwen35_mod = @import("qwen35.zig");
                try qwen35_mod.buildBlockEntry(&builder, &cfg, l, prev_out, out_owned);
            },
            else => try builder.build_transformer_block(l, 0, prev_out, out_owned),
        }
        prev_out = graph.tensors.getPtr(out_owned).?.name;
    }

    const has_output = ctx.tensors.get("output.weight") != null;
    if (has_output) {
        try builder.add_tensor("output.weight", model.f32Bytes(@as(u64, cfg.vocab_size) * cfg.n_embd), .weight);
    }
    try builder.add_tensor("output_norm.weight", model.f32Bytes(cfg.n_embd), .weight);
    try builder.build_lm_head(prev_out, "logits", has_output);
    try builder.finalize();

    for (graph.nodes.items) |*node| {
        if ((node.op_type != .matmul and node.op_type != .attn_qg_matmul) or node.input_names.len < 2) continue;
        const w_name = node.input_names[1];
        const w_type: ?@import("tensor.zig").Type = blk: {
            if (ctx.tensors.get(w_name)) |t| break :blk t.type;
            if (try weight_uploader.getFusedComponentNames(allocator, w_name)) |comps| {
                defer {
                    for (comps) |c| allocator.free(c);
                    allocator.free(comps);
                }
                if (comps.len > 0) {
                    const first_t = ctx.tensors.get(comps[0]) orelse break :blk null;
                    var same_type = true;
                    for (comps[1..]) |c| {
                        if (ctx.tensors.get(c).?.type != first_t.type) {
                            same_type = false;
                            break;
                        }
                    }
                    if (same_type) break :blk first_t.type;
                }
            }
            break :blk null;
        };
        const qt = w_type orelse continue;
        if (weight_uploader.isNativeQuantType(qt)) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(qt);
        } else if (qt == .bf16) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(@import("tensor.zig").Type.f16);
        }
    }

    var uploader = weight_uploader.WeightUploader{};
    defer uploader.deinit(allocator, &vk_ctx);
    try uploader.upload(allocator, &vk_ctx, &graph, &ctx, cli_cfg.gpu_embed, cli_cfg.staging_size);

    ctx.discardMmap();

    const scratch_props: vk.MemoryPropertyFlags = if (cfg.arch == .qwen35)
        .{ .device_local_bit = true, .host_visible_bit = true, .host_coherent_bit = true }
    else
        .{ .device_local_bit = true };

    var scratchpad = try vulkan.Buffer.init(&vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, scratch_props);
    defer scratchpad.deinit(&vk_ctx);

    var kv_cache = try vulkan.Buffer.init(&vk_ctx, graph.kv_cache_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .device_local_bit = true });
    defer kv_cache.deinit(&vk_ctx);

    const ssm_props: vk.MemoryPropertyFlags = .{ .host_visible_bit = true, .host_coherent_bit = true };
    var ssm_conv_size: u64 = 0;
    var ssm_state_size: u64 = 0;
    var ssm_it = graph.tensors.iterator();
    while (ssm_it.next()) |entry| {
        if (entry.value_ptr.role == .ssm_cache) {
            if (std.mem.startsWith(u8, entry.key_ptr.*, "ssm_conv")) ssm_conv_size += entry.value_ptr.size
            else if (std.mem.startsWith(u8, entry.key_ptr.*, "ssm_state")) ssm_state_size += entry.value_ptr.size;
        }
    }
    var ssm_conv_cache = try vulkan.Buffer.init(&vk_ctx, if (ssm_conv_size > 0) ssm_conv_size else 4096, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, ssm_props);
    defer ssm_conv_cache.deinit(&vk_ctx);
    try vk_ctx.clearBuffer(ssm_conv_cache);
    var ssm_state_cache = try vulkan.Buffer.init(&vk_ctx, if (ssm_state_size > 0) ssm_state_size else 4096, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, ssm_props);
    defer ssm_state_cache.deinit(&vk_ctx);
    try vk_ctx.clearBuffer(ssm_state_cache);

    var input_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.n_embd), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer input_staging.deinit(&vk_ctx);

    var logits_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.vocab_size), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer logits_staging.deinit(&vk_ctx);

    var hidden_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.n_embd), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer hidden_staging.deinit(&vk_ctx);

    var embed_indices = try vulkan.Buffer.init(&vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer embed_indices.deinit(&vk_ctx);

    const logits_mapped_ptr = try vk_ctx.vkd.mapMemory(vk_ctx.device, logits_staging.memory, 0, model.f32Bytes(cfg.vocab_size), .{});
    defer vk_ctx.vkd.unmapMemory(vk_ctx.device, logits_staging.memory);
    const logits_persistent = @as([*]f32, @ptrCast(@alignCast(logits_mapped_ptr)))[0..cfg.vocab_size];

    var dispatcher = try @import("dispatcher.zig").Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad, kv_cache, ssm_conv_cache, ssm_state_cache, &cfg, allocator);
    dispatcher.trace_dispatch = cli_cfg.debug_trace;
    defer dispatcher.deinit();

    if (cfg.arch == .qwen35) {
        try writer.print("Qwen 3.5 SSM: d_inner={} d_state={} dt_rank={} n_group={} head_v_dim={} head_k_dim={}\n", .{
            cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_dt_rank, cfg.ssm_n_group,
            if (cfg.ssm_dt_rank > 0) cfg.ssm_d_inner / cfg.ssm_dt_rank else 0,
            cfg.ssm_d_state,
        });
        try writer.print("Weights found:\n", .{});
        var w_it = ctx.tensors.iterator();
        while (w_it.next()) |entry| {
            if (std.mem.containsAtLeast(u8, entry.key_ptr.*, 1, "blk.0")) {
                try writer.print("  {s}\n", .{entry.key_ptr.*});
            }
        }
        try writer_streaming.interface.flush();
    }

    const ssm_staging_size: u64 = if (cfg.ssm_d_inner > 0) 64 * 1024 else 0;
    var ssm_staging: vulkan.Buffer = undefined;
    if (ssm_staging_size > 0) {
        ssm_staging = try vulkan.Buffer.init(&vk_ctx, ssm_staging_size, .{ .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        dispatcher.set_ssm_staging_buffer(ssm_staging, ssm_staging_size);
    }

    const format = chat.detectChatFormat(tok.chat_template, cfg.arch, &tok.special);
    const token_ids = try chat.buildChatPrompt(&tok, format, cli_cfg.prompt_text.?, allocator);
    defer allocator.free(token_ids);

    const embd_tensor = ctx.tensors.get("token_embd.weight") orelse return error.MissingEmbeddings;
    const embd_standard_layout = embd_tensor.ne[0] == cfg.n_embd;
    var embd_cache_transposed: ?[]f32 = null;
    defer if (embd_cache_transposed) |buf| allocator.free(buf);
    if (!embd_standard_layout and embd_tensor.ne[1] == cfg.n_embd) {
        const cache = try allocator.alloc(f32, embd_tensor.ne[0] * embd_tensor.ne[1]);
        try weights.dequantToF32(&ctx, embd_tensor, cache);
        embd_cache_transposed = cache;
    }

    const embd_quant_gpu = cli_cfg.gpu_embed and embd_standard_layout and weight_uploader.isNativeQuantType(embd_tensor.type);
    var embd_gpu_buf: ?*vulkan.Buffer = null;
    if (embd_quant_gpu) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt| {
            if (gt.buffer) |buf| embd_gpu_buf = buf;
        } else {
            if (uploader.embd_gpu_buf) |*b| embd_gpu_buf = b;
        }
    }
    const embd_scale_bits: u32 = @bitCast(cfg.embedding_scale);

    try writer.print("\nAssistant: ", .{});
    try writer_streaming.interface.flush();

    var pos: u32 = 0;
    var generated: u32 = 0;
    var current_token: tokenizer.TokenID = 0;
    var token_sampler = sampler.Sampler.init(sampler.SamplerConfig{ .temperature = cli_cfg.temperature, .top_k = cli_cfg.top_k, .top_p = cli_cfg.top_p, .min_p = cli_cfg.min_p, .seed = if (cli_cfg.seed != 0) cli_cfg.seed else 0xDEADBEEF });

    var gen_history: [256]tokenizer.TokenID = undefined;
    var gen_history_len: usize = 0;

    const ictx = inference.InferenceContext{
        .allocator = allocator,
        .vk_ctx = &vk_ctx,
        .ctx = &ctx,
        .cfg = &cfg,
        .cli_cfg = &cli_cfg,
        .tok = &tok,
        .graph = &graph,
        .dispatcher = &dispatcher,
        .scratchpad = scratchpad,
        .ssm_conv_cache = ssm_conv_cache,
        .ssm_state_cache = ssm_state_cache,
        .ssm_conv_size = ssm_conv_size,
        .ssm_state_size = ssm_state_size,
        .input_staging = input_staging,
        .logits_staging = logits_staging,
        .hidden_staging = hidden_staging,
        .embed_indices = embed_indices,
        .logits_persistent = logits_persistent,
        .embd_tensor = embd_tensor,
        .embd_standard_layout = embd_standard_layout,
        .embd_cache_transposed = embd_cache_transposed,
        .embd_quant_gpu = embd_quant_gpu,
        .embd_gpu_buf = if (embd_gpu_buf) |b| b.* else null,
        .embd_scale_bits = embd_scale_bits,
        .pos = &pos,
        .generated = &generated,
        .current_token = &current_token,
        .gen_history = &gen_history,
        .gen_history_len = &gen_history_len,
        .token_sampler = &token_sampler,
    };

    try inference.run_prefill(ictx, token_ids, writer);
    try inference.run_decode(ictx, writer);
    if (ssm_staging_size > 0) ssm_staging.deinit(&vk_ctx);
}

fn validateModelLayout(ctx: *gguf.GGUFContext) !void {
    const required = [_][]const u8{ "token_embd.weight", "blk.0.attn_norm.weight", "output_norm.weight" };
    for (required) |name| if (ctx.tensors.get(name) == null) return error.UnsupportedArchitectureLayout;
}

fn loadEmbedding(ctx: *gguf.GGUFContext, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, vk_ctx: *vulkan.Context, staging: *vulkan.Buffer, scratch: *vulkan.Buffer, graph: *compute_graph.Graph, scale: f32) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    try weights.readEmbeddingF32(ctx, embd, tid, @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd], n_embd, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

fn loadEmbeddingFromTransposedCache(cache: []const f32, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, dst: []f32, scale: f32) !void {
    const vocab_stride: usize = @intCast(embd.ne[0]);
    if (tid >= vocab_stride) return error.TokenOutOfRange;
    for (0..n_embd) |i| dst[i] = cache[i * vocab_stride + tid] * scale;
}

const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const kernels_data = @import("kernels_data");
const compute_graph = @import("compute_graph.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const weights = @import("weights.zig");
const sampler = @import("sampler.zig");
const builtin = @import("builtin");
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

    try writer.print("llama.zig: High-Performance Vulkan Inference\n", .{});

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();

    var model_path: ?[]const u8 = null;
    var prompt_text: ?[]const u8 = null;
    var max_tokens: u32 = 64;
    var temperature: f32 = 0.8;
    var seed: u64 = 0;
    var top_k: u32 = 0;
    var top_p: f32 = 0.9;
    var min_p: f32 = 0.0;
    var ctx_size_override: ?u32 = null;
    var debug_logits: u32 = 0;
    var chat_mode: bool = false;
    var verbose: bool = false;
    var inspect_block: bool = false;
    var prefill_chunk: u32 = 0;
    var gpu_embed: bool = true;
    var report_json: bool = false;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt_text = args_it.next();
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            max_tokens = std.fmt.parseInt(u32, args_it.next() orelse "64", 10) catch 64;
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            temperature = std.fmt.parseFloat(f32, args_it.next() orelse "0.8") catch 0.8;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = std.fmt.parseInt(u64, args_it.next() orelse "0", 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            top_k = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            top_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.9") catch 0.9;
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            min_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.0") catch 0.0;
        } else if (std.mem.eql(u8, arg, "--ctx-size")) {
            ctx_size_override = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch null;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat_mode = true;
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "--debug-logits")) {
            debug_logits = std.fmt.parseInt(u32, args_it.next() orelse "10", 10) catch 10;
        } else if (std.mem.eql(u8, arg, "--inspect-block")) {
            inspect_block = true;
        } else if (std.mem.eql(u8, arg, "--prefill-chunk")) {
            prefill_chunk = std.fmt.parseInt(u32, args_it.next() orelse "512", 10) catch 512;
        } else if (std.mem.eql(u8, arg, "--no-gpu-embed")) {
            gpu_embed = false;
        } else if (std.mem.eql(u8, arg, "--report-json")) {
            report_json = true;
        }
    }

    if (model_path == null or prompt_text == null) {
        try writer.print("Usage: llama.zig --model <path.gguf> --prompt '<text>' [--max-tokens N] [--temperature T]\n", .{});
        try writer_streaming.interface.flush();
        return;
    }

    const t_load_start = nowNs();
    try writer.print("Loading model: {s}...\n", .{model_path.?});
    try writer_streaming.interface.flush();
    var ctx = try gguf.loadModel(allocator, model_path.?);
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

    const vocab_size: u32 = @intCast(tok.id_to_token.len);
    var cfg = try model.ModelConfig.init(allocator, &ctx, vocab_size);
    defer cfg.deinit(allocator);
    if (ctx_size_override) |ctx_sz| {
        if (ctx_sz > 0) cfg.max_ctx = ctx_sz;
    } else if (cfg.max_ctx > 8192) {
        // Keep a practical default context limit to avoid OOM on commodity GPUs.
        cfg.max_ctx = 8192;
    }

    if (cfg.arch == .unknown) {
        try writer.print(
            "Warning: unrecognized architecture '{s}'. Proceeding with shared lowering path and llama-compatible metadata fallbacks.\n",
            .{cfg.arch_prefix},
        );
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

    // Dump GGUF metadata for debugging
    if (verbose) {
        try writer.print("[verbose] GGUF KV metadata (count={}):\n", .{ctx.kvs.count()});
        var kv_it = ctx.kvs.iterator();
        while (kv_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const tag_name = @tagName(entry.value_ptr.*);
            if (entry.value_ptr.* == .array) {
                const arr = entry.value_ptr.array;
                try writer.print("  {s} [array len={}]\n", .{ key, arr.len });
                if (arr.len <= 32) {
                    for (arr, 0..) |item, ai| {
                        try writer.print("    [{d}] tag={s} ", .{ ai, @tagName(item) });
                        switch (item) {
                            .u8 => |v| try writer.print("{}\n", .{v}),
                            .u16 => |v| try writer.print("{}\n", .{v}),
                            .u32 => |v| try writer.print("{}\n", .{v}),
                            .u64 => |v| try writer.print("{}\n", .{v}),
                            .i32 => |v| try writer.print("{}\n", .{v}),
                            .i64 => |v| try writer.print("{}\n", .{v}),
                            .f32 => |v| try writer.print("{d}\n", .{v}),
                            .string => |v| try writer.print("\"{s}\"\n", .{v}),
                            else => try writer.print("?\n", .{}),
                        }
                    }
                }
            } else {
                try writer.print("  {s} [{s}] = ", .{ key, tag_name });
                switch (entry.value_ptr.*) {
                    .u8 => |v| try writer.print("{}\n", .{v}),
                    .u16 => |v| try writer.print("{}\n", .{v}),
                    .u32 => |v| try writer.print("{}\n", .{v}),
                    .u64 => |v| try writer.print("{}\n", .{v}),
                    .i8 => |v| try writer.print("{}\n", .{v}),
                    .i16 => |v| try writer.print("{}\n", .{v}),
                    .i32 => |v| try writer.print("{}\n", .{v}),
                    .i64 => |v| try writer.print("{}\n", .{v}),
                    .f32 => |v| try writer.print("{d}\n", .{v}),
                    .f64 => |v| try writer.print("{d}\n", .{v}),
                    .bool => |v| try writer.print("{}\n", .{v}),
                    .string => |v| try writer.print("\"{s}\"\n", .{v}),
                    else => try writer.print("?\n", .{}),
                }
            }
        }
        try writer_streaming.interface.flush();
    }

    if (verbose) {
        try writer.print("[verbose] GGUF Tensors (count={}):\n", .{ctx.tensors.count()});
        var t_it = ctx.tensors.iterator();
        while (t_it.next()) |entry| {
            try writer.print("  {s} type={} dims={}\n", .{ entry.key_ptr.*, entry.value_ptr.*.type, entry.value_ptr.*.n_dims });
        }
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
    try registry.register(&vk_ctx, "get_rows_q", kernels_data.kernels_get_rows_q_spv, "main");
    try registry.register(&vk_ctx, "matmul_q4_0", kernels_data.kernels_matmul_q4_0_spv, "main");
    try registry.register(&vk_ctx, "matvec_q4_0", kernels_data.kernels_matvec_q4_0_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q4_0", kernels_data.kernels_get_rows_q4_0_spv, "main");
    try registry.register(&vk_ctx, "get_rows_q6_k", kernels_data.kernels_get_rows_q6_k_spv, "main");
    try registry.register(&vk_ctx, "matvec_q6_k", kernels_data.kernels_matvec_q6_k_spv, "main");
    try registry.register(&vk_ctx, "topk", kernels_data.kernels_topk_spv, "main");
    try registry.register(&vk_ctx, "attention_flash", kernels_data.kernels_flash_attn_spv, "main");
    try registry.register(&vk_ctx, "gelu_mul", kernels_data.kernels_gelu_mul_spv, "main");
    try registry.register(&vk_ctx, "copy", kernels_data.kernels_copy_spv, "main");

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph, &cfg, &ctx.tensors);

    try builder.addTensor("input", model.f32Bytes(cfg.n_embd), .input);
    try builder.initKvCaches();

    var prev_out: []const u8 = "input";
    var l: u32 = 0;
    while (l < cfg.n_layer) : (l += 1) {
        const out_owned = if (l == cfg.n_layer - 1)
            try allocator.dupe(u8, "hidden")
        else
            try std.fmt.allocPrint(allocator, "blk.{d}.out", .{l});
        defer allocator.free(out_owned);
        try builder.buildTransformerBlock(l, 0, prev_out, out_owned);
        // Use the graph's owned copy of the name so it outlives this iteration.
        prev_out = graph.tensors.getPtr(out_owned).?.name;
    }

    const has_output = ctx.tensors.get("output.weight") != null;
    if (has_output) {
        try builder.addTensor("output.weight", model.f32Bytes(@as(u64, cfg.vocab_size) * cfg.n_embd), .weight);
    }
    try builder.addTensor("output_norm.weight", model.f32Bytes(cfg.n_embd), .weight);
    try builder.buildLmHead(prev_out, "logits", has_output);
    builder.finalize();
    try graph.verify();
    const graph_cost = compute_graph.estimateGraphCost(&graph);

    // Convert eligible matmul nodes to quantized matmul path.
    for (graph.nodes.items) |*node| {
        if (node.op_type != .matmul or node.input_names.len < 2) continue;
        const w_name = node.input_names[1];
        const w_t = ctx.tensors.get(w_name) orelse continue;
        if (isNativeQuantType(w_t.type)) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(w_t.type);
        }
    }

    // Upload weights (dequant to f32 on host)
    var weight_buffers = std.ArrayList(vulkan.Buffer).empty;
    defer {
        for (weight_buffers.items) |b| b.deinit(&vk_ctx);
        weight_buffers.deinit(allocator);
    }

    var max_staging: u64 = 0;
    var t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role == .weight) {
            if (ctx.tensors.get(entry.key_ptr.*)) |gt| {
                max_staging = @max(max_staging, if (isNativeQuantType(gt.type)) gt.size() else model.weightF32Size(gt));
            } else {
                max_staging = @max(max_staging, entry.value_ptr.size);
            }
        }
    }

    var weight_staging = try vulkan.Buffer.init(&vk_ctx, max_staging, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer weight_staging.deinit(&vk_ctx);

    t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role != .weight) continue;
        const gt = ctx.tensors.get(entry.key_ptr.*);
        if (gt == null and verbose) {
            try writer.print("[verbose] weight not in GGUF: {s}\n", .{entry.key_ptr.*});
        }
        const upload_size = if (gt) |t|
            if (isNativeQuantType(t.type)) t.size() else model.weightF32Size(t)
        else
            entry.value_ptr.size;
        const buf = try vulkan.Buffer.init(&vk_ctx, upload_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
        entry.value_ptr.buffer = try allocator.create(vulkan.Buffer);
        entry.value_ptr.buffer.?.* = buf;
        try weight_buffers.append(allocator, buf);

        if (gt) |t| {
            if (isNativeQuantType(t.type)) {
                const raw_size = t.size();
                const raw = try allocator.alloc(u8, raw_size);
                defer allocator.free(raw);
                try ctx.readTensorData(t, raw);
                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, raw_size, .{});
                @memcpy(@as([*]u8, @ptrCast(mapped))[0..raw_size], raw);
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, buf, raw_size);
            } else {
                const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                const f32_data = try allocator.alloc(f32, n);
                defer allocator.free(f32_data);
                weights.dequantToF32(&ctx, t, f32_data) catch |err| {
                    if (err == error.UnsupportedQuantType) {
                        try writer.print(
                            "Unsupported tensor quantization type '{s}' for weight '{s}'. Supported types: f32, f16, bf16, q8_0.\n",
                            .{ weights.typeName(t.type), entry.key_ptr.* },
                        );
                        try writer_streaming.interface.flush();
                    }
                    return err;
                };
                const f32_size = model.weightF32Size(t);
                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f32_size, .{});
                @memcpy(@as([*]u8, @ptrCast(mapped))[0..f32_size], std.mem.sliceAsBytes(f32_data));
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, buf, f32_size);
            }
        }
    }

    // token_embd for LM head if tied — upload quant raw when applicable
    if (!has_output) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt_entry| {
            if (ctx.tensors.get("token_embd.weight")) |t| {
                const upload_size = if (isNativeQuantType(t.type)) t.size() else model.weightF32Size(t);
                if (gt_entry.buffer == null) {
                    const buf = try vulkan.Buffer.init(&vk_ctx, upload_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
                    gt_entry.buffer = try allocator.create(vulkan.Buffer);
                    gt_entry.buffer.?.* = buf;
                    try weight_buffers.append(allocator, buf);
                }
                gt_entry.size = upload_size;
                if (isNativeQuantType(t.type)) {
                    const raw_size = t.size();
                    const raw = try allocator.alloc(u8, raw_size);
                    defer allocator.free(raw);
                    try ctx.readTensorData(t, raw);
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, raw_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..raw_size], raw);
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, gt_entry.buffer.?.*, raw_size);
                } else {
                    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    const f32_data = try allocator.alloc(f32, n);
                    defer allocator.free(f32_data);
                    try weights.dequantToF32(&ctx, t, f32_data);
                    const f32_size = model.weightF32Size(t);
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f32_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..f32_size], std.mem.sliceAsBytes(f32_data));
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, gt_entry.buffer.?.*, f32_size);
                }
            }
        }
    }

    var scratchpad = try vulkan.Buffer.init(&vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(&vk_ctx);

    var kv_cache = try vulkan.Buffer.init(&vk_ctx, graph.kv_cache_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .device_local_bit = true });
    defer kv_cache.deinit(&vk_ctx);

    var input_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.n_embd), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer input_staging.deinit(&vk_ctx);

    var logits_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.vocab_size), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer logits_staging.deinit(&vk_ctx);

    var topk_indices = try vulkan.Buffer.init(&vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer topk_indices.deinit(&vk_ctx);
    var topk_values = try vulkan.Buffer.init(&vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer topk_values.deinit(&vk_ctx);

    var embed_indices = try vulkan.Buffer.init(&vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer embed_indices.deinit(&vk_ctx);

    const logits_mapped_ptr = try vk_ctx.vkd.mapMemory(vk_ctx.device, logits_staging.memory, 0, model.f32Bytes(cfg.vocab_size), .{});
    defer vk_ctx.vkd.unmapMemory(vk_ctx.device, logits_staging.memory);
    const logits_persistent = @as([*]f32, @ptrCast(@alignCast(logits_mapped_ptr)))[0..cfg.vocab_size];

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad, kv_cache, &cfg);
    defer dispatcher.deinit();
    if (verbose) {
        try writer.print("[verbose] graph: {} nodes, {} tensors\n", .{ graph.nodes.items.len, graph.tensors.count() });
        try writer_streaming.interface.flush();
    }

    // Build prompt: apply chat template for Granite if --chat is set.
    // Format: <|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>
    var final_prompt: []const u8 = prompt_text.?;
    var final_prompt_owned: bool = false;
    if (chat_mode) {
        if (cfg.arch == .llama and tok.special.begin_of_text != null and tok.special.start_header_id != null and tok.special.end_header_id != null and tok.special.eot_id != null) {
            const bot = tok.id_to_token[tok.special.begin_of_text.?];
            const sh = tok.id_to_token[tok.special.start_header_id.?];
            const eh = tok.id_to_token[tok.special.end_header_id.?];
            const eot = tok.id_to_token[tok.special.eot_id.?];
            final_prompt = try std.fmt.allocPrint(allocator, "{s}{s}user{s}\n\n{s}{s}{s}assistant{s}\n\n", .{
                bot, sh, eh, prompt_text.?, eot, sh, eh,
            });
        } else {
            const sr_str: []const u8 = if (tok.special.start_of_role) |id| tok.id_to_token[id] else "<|start_of_role|>";
            const er_str: []const u8 = if (tok.special.end_of_role) |id| tok.id_to_token[id] else "<|end_of_role|>";
            const et_str: []const u8 = if (tok.special.end_of_text) |id| tok.id_to_token[id] else "<|end_of_text|>";
            final_prompt = try std.fmt.allocPrint(allocator, "{s}user{s}{s}{s}\n{s}assistant{s}", .{
                sr_str, er_str, et_str, prompt_text.?, sr_str, er_str,
            });
        }
        final_prompt_owned = true;
    }

    const token_ids = try tok.encode(final_prompt, allocator);
    defer allocator.free(token_ids);
    if (final_prompt_owned) {
        defer allocator.free(final_prompt);
    }

    const embd_tensor = ctx.tensors.get("token_embd.weight") orelse return error.MissingEmbeddings;
    const embd_standard_layout = embd_tensor.ne[0] == cfg.n_embd;
    const embd_transposed_layout = embd_tensor.ne[1] == cfg.n_embd;
    var embd_cache_transposed: ?[]f32 = null;
    defer if (embd_cache_transposed) |buf| allocator.free(buf);
    if (!embd_standard_layout and embd_transposed_layout) {
        const n = embd_tensor.ne[0] * embd_tensor.ne[1] * embd_tensor.ne[2] * embd_tensor.ne[3];
        const cache = try allocator.alloc(f32, n);
        try weights.dequantToF32(&ctx, embd_tensor, cache);
        embd_cache_transposed = cache;
    }

    const embd_quant_gpu = gpu_embed and embd_standard_layout and isNativeQuantType(embd_tensor.type);
    var embd_gpu_buf: ?*vulkan.Buffer = null;
    const embd_row_bytes: u32 = @intCast(weights.quantRowBytes(embd_tensor.type, embd_tensor.ne[0]) orelse 0);
    if (embd_quant_gpu) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt| {
            if (gt.buffer) |buf| embd_gpu_buf = buf;
        }
    }
    const embd_scale_bits: u32 = @bitCast(cfg.embedding_scale);
    const input_offset = graph.tensors.get("input").?.offset;

    // Verbose debug: dump token IDs
    if (verbose) {
        try writer.print("[verbose] token_ids ({d} tokens): ", .{token_ids.len});
        for (token_ids, 0..) |tid, i| {
            try writer.print("{d}", .{tid});
            if (i < token_ids.len - 1) try writer.print(", ", .{});
        }
        try writer.print("\n", .{});
        try writer_streaming.interface.flush();
    }

    // Extra stop tokens: <|end_of_text|> and <|end_of_role|> beyond the normal EOS.
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
    try writer_streaming.interface.flush();

    var pos: u32 = 0;
    var generated: u32 = 0;
    const sample_cfg = sampler.SamplerConfig{
        .temperature = temperature,
        .top_k = top_k,
        .top_p = top_p,
        .min_p = min_p,
        .typical_p = 1.0,
        .seed = if (seed != 0) seed else 0xDEADBEEF,
        .repetition_window = 0,
        .repetition_penalty = 1.0,
    };
    var token_sampler = sampler.Sampler.init(sample_cfg);

    const t_prefill_start = nowNs();
    if (token_ids.len > 0) {
        const embd_bytes = model.f32Bytes(cfg.n_embd);
        const chunk = if (prefill_chunk > 0) prefill_chunk else @as(u32, @intCast(token_ids.len));
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
                var prefill_staging = try vulkan.Buffer.init(&vk_ctx, prefill_bytes, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
                defer prefill_staging.deinit(&vk_ctx);

                const mapped_prefill = try vk_ctx.vkd.mapMemory(vk_ctx.device, prefill_staging.memory, 0, prefill_bytes, .{});
                const prefill_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_prefill)))[0 .. chunk_len * cfg.n_embd];
                for (0..chunk_len) |i| {
                    const row = prefill_f32[i * cfg.n_embd .. (i + 1) * cfg.n_embd];
                    if (embd_cache_transposed) |cache| {
                        try loadEmbeddingFromTransposedCache(cache, embd_tensor, token_ids[chunk_start + i], cfg.n_embd, row, cfg.embedding_scale);
                    } else {
                        try weights.readEmbeddingF32(&ctx, embd_tensor, token_ids[chunk_start + i], row, cfg.n_embd, cfg.embedding_scale);
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

    // Track generated token history for repetition penalty (fixed-size ring buffer).
    var gen_history: [256]tokenizer.TokenID = undefined;
    var gen_history_len: usize = 0;

    // Inspect intermediate activations after prefill.
    if (inspect_block) {
        const inspect_size = model.f32Bytes(cfg.n_embd);
        var inspect_staging = try vulkan.Buffer.init(&vk_ctx, inspect_size, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        defer inspect_staging.deinit(&vk_ctx);
        // Per-layer hidden state stats
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
        // Dump first 8 values of key tensors
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

        try writer_streaming.interface.flush();
    }

    // Debug dump of top logits after prefill.
    if (debug_logits > 0) {
        const t_logits = graph.tensors.get("logits").?;
        try vk_ctx.copyBufferOffset(scratchpad, t_logits.offset, logits_staging, 0, model.f32Bytes(cfg.vocab_size));
        const logits = logits_persistent;
        // Find top-N
        var indices = try allocator.alloc(u32, cfg.vocab_size);
        defer allocator.free(indices);
        for (indices, 0..) |*v, i| v.* = @as(u32, @intCast(i));
        std.mem.sort(u32, indices, logits, struct {
            fn lessThan(probs: []const f32, a: u32, b: u32) bool {
                return probs[a] > probs[b];
            }
        }.lessThan);
        try writer.print("\n[top {d} logits after prefill]\n", .{debug_logits});
        for (indices[0..debug_logits]) |idx| {
            try writer.print("  id={:6}  logit={:9.4}  token=", .{ idx, logits[idx] });
            try tok.decode(&[_]tokenizer.TokenID{idx}, writer);
            try writer.print("\n", .{});
        }
    }

    var stopped: bool = false;
    const t_decode_start = nowNs();
    const logits_offset = graph.tensors.get("logits").?.offset;
    const logit_scale_bits: u32 = if (cfg.logit_scale != 1.0 and cfg.logit_scale != 0.0) @bitCast(cfg.logit_scale) else 0;
    const use_gpu_topk = temperature <= 0.0 and top_k <= 1 and cfg.final_logit_softcapping <= 0.0;
    while (generated < max_tokens) : (generated += 1) {
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

        // Per-step debug logits (dump after each decode step)
        if (verbose and debug_logits > 0) {
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
            try writer.print("\n[step {} top {} logits]\n", .{ pos, debug_logits });
            for (indices_s[0..debug_logits]) |idx| {
                try writer.print("  id={:6}  logit={:9.4}  token=", .{ idx, logits_s[idx] });
                try tok.decode(&[_]tokenizer.TokenID{idx}, writer);
                try writer.print("\n", .{});
            }
        }

        // Check stop: normal EOS, plus Granite <|end_of_text|> and <|end_of_role|>.
        for (extra_stops) |stop_id| {
            if (current_token == stop_id) {
                try writer.print("\n[stop token {}]\n", .{stop_id});
                try writer_streaming.interface.flush();
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
        try writer_streaming.interface.flush();

        if (generated + 1 >= max_tokens) break;
        if (embd_quant_gpu and embd_gpu_buf != null) {
            try dispatcher.executeGetRowsQ(
                embed_indices,
                embd_gpu_buf.?.*,
                input_offset,
                current_token,
                cfg.n_embd,
                @intFromEnum(embd_tensor.type),
                embd_row_bytes,
                embd_scale_bits,
            );
        } else {
            if (embd_cache_transposed) |cache| {
                try loadEmbeddingCached(cache, embd_tensor, current_token, cfg.n_embd, &vk_ctx, &input_staging, &scratchpad, &graph, cfg.embedding_scale);
            } else {
                try loadEmbedding(&ctx, embd_tensor, current_token, cfg.n_embd, &vk_ctx, &input_staging, &scratchpad, &graph, cfg.embedding_scale);
            }
        }

        // Debug: verify input embedding actually changed
        if (verbose) {
            const inp_t = graph.tensors.get("input").?;
            const inp_sz = @min(model.f32Bytes(8), inp_t.size);
            var dbg_staging = try vulkan.Buffer.init(&vk_ctx, inp_sz, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
            defer dbg_staging.deinit(&vk_ctx);
            try vk_ctx.copyBufferOffset(scratchpad, inp_t.offset, dbg_staging, 0, inp_sz);
            const dbg_mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, dbg_staging.memory, 0, inp_sz, .{});
            const dbg_vals = @as([*]const f32, @ptrCast(@alignCast(dbg_mapped)))[0..8];
            try writer.print("[debug] input[0..8] before execute({}): ", .{pos});
            for (dbg_vals) |v| try writer.print("{d:.4} ", .{v});
            try writer.print("(token={})\n", .{current_token});
            vk_ctx.vkd.unmapMemory(vk_ctx.device, dbg_staging.memory);
            try writer_streaming.interface.flush();
        }

        try dispatcher.execute(pos);
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
    if (report_json) {
        try writer.print("{{\"load_s\":{d:.6},\"prefill_s\":{d:.6},\"decode_s\":{d:.6},\"prompt_tps\":{d:.4},\"generation_tps\":{d:.4},\"tokens_prompt\":{},\"tokens_generated\":{},\"graph_nodes\":{},\"graph_flops\":{},\"graph_bytes\":{}}}\n", .{
            load_s, prefill_s, decode_s, prefill_tps, decode_tps, token_ids.len, generated, graph_cost.total_nodes, graph_cost.approx_flops, graph_cost.approx_bytes,
        });
    }
    try writer_streaming.interface.flush();
}

fn isNativeQuantType(tt: @import("tensor.zig").Type) bool {
    return switch (tt) {
        .q8_0, .q4_0, .q6_k => true,
        else => false,
    };
}

fn validateModelLayout(ctx: *gguf.GGUFContext) !void {
    const required = [_][]const u8{
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "output_norm.weight",
    };
    for (required) |name| {
        if (ctx.tensors.get(name) == null) return error.UnsupportedArchitectureLayout;
    }
}

fn loadEmbedding(
    ctx: *gguf.GGUFContext,
    embd: *@import("tensor.zig").Tensor,
    tid: tokenizer.TokenID,
    n_embd: u32,
    vk_ctx: *vulkan.Context,
    staging: *vulkan.Buffer,
    scratch: *vulkan.Buffer,
    graph: *compute_graph.Graph,
    scale: f32,
) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    const dst = @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd];
    try weights.readEmbeddingF32(ctx, embd, tid, dst, n_embd, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

fn loadEmbeddingFromTransposedCache(cache: []const f32, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, dst: []f32, scale: f32) !void {
    const vocab_stride: usize = @intCast(embd.ne[0]);
    if (tid >= vocab_stride) return error.TokenOutOfRange;
    for (0..n_embd) |i| {
        dst[i] = cache[i * vocab_stride + tid] * scale;
    }
}

fn loadEmbeddingCached(
    cache: []const f32,
    embd: *@import("tensor.zig").Tensor,
    tid: tokenizer.TokenID,
    n_embd: u32,
    vk_ctx: *vulkan.Context,
    staging: *vulkan.Buffer,
    scratch: *vulkan.Buffer,
    graph: *compute_graph.Graph,
    scale: f32,
) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    const dst = @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd];
    try loadEmbeddingFromTransposedCache(cache, embd, tid, n_embd, dst, scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

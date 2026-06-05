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

/// Run the CPU-side Gated Delta Net recurrence for one layer of one prefill
/// or decode token. The `core` output is written into the scratchpad at
/// `core_t.offset`.
fn runCpuSsmDeltaForLayer(
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

    var name_buf: [32]u8 = undefined;
    const qn_name = std.fmt.bufPrint(&name_buf, "{s}.q_norm", .{ln}) catch return;
    @memset(name_buf[0..qn_name.len], 0);
    const kn_name = std.fmt.bufPrint(&name_buf, "{s}.k_norm", .{ln}) catch return;
    @memset(name_buf[0..kn_name.len], 0);
    const vn_name = std.fmt.bufPrint(&name_buf, "{s}.v_conv", .{ln}) catch return;
    @memset(name_buf[0..vn_name.len], 0);
    const gate_name = std.fmt.bufPrint(&name_buf, "{s}.gate", .{ln}) catch return;
    @memset(name_buf[0..gate_name.len], 0);
    const beta_name = std.fmt.bufPrint(&name_buf, "{s}.beta", .{ln}) catch return;
    @memset(name_buf[0..beta_name.len], 0);
    const core_name = std.fmt.bufPrint(&name_buf, "{s}.core", .{ln}) catch return;

    const qn_t = graph.tensors.get(qn_name) orelse return;
    const kn_t = graph.tensors.get(kn_name) orelse return;
    const vn_t = graph.tensors.get(vn_name) orelse return;
    const gate_t = graph.tensors.get(gate_name) orelse return;
    const beta_t = graph.tensors.get(beta_name) orelse return;
    const core_t = graph.tensors.get(core_name) orelse return;

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
    var debug_trace: bool = false;

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
        } else if (std.mem.eql(u8, arg, "--no-chat")) {
            chat_mode = false;
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
        } else if (std.mem.eql(u8, arg, "--debug-trace")) {
            debug_trace = true;
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
    var ctx = try gguf.loadModelMmap(allocator, model_path.?);
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

    const vocab_size: u32 = @intCast(tok.id_to_token.len);
    var cfg = try model.ModelConfig.init(allocator, &ctx, vocab_size);
    defer cfg.deinit(allocator);
    if (ctx_size_override) |ctx_sz| {
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

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph, &cfg, &ctx.tensors);

    try builder.addTensor("input", model.f32Bytes(cfg.n_embd), .input);
    try builder.initKvCaches();
    try builder.initSsmCaches();

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
                const qwen35_mod = @import("models/qwen35.zig");
                try qwen35_mod.buildBlockEntry(&builder, &cfg, l, prev_out, out_owned);
            },
            else => try builder.buildTransformerBlock(l, 0, prev_out, out_owned),
        }
        prev_out = graph.tensors.getPtr(out_owned).?.name;
    }

    const has_output = ctx.tensors.get("output.weight") != null;
    if (has_output) {
        try builder.addTensor("output.weight", model.f32Bytes(@as(u64, cfg.vocab_size) * cfg.n_embd), .weight);
    }
    try builder.addTensor("output_norm.weight", model.f32Bytes(cfg.n_embd), .weight);
    try builder.buildLmHead(prev_out, "logits", has_output);
    try builder.finalize();

    for (graph.nodes.items) |*node| {
        if ((node.op_type != .matmul and node.op_type != .attn_qg_matmul) or node.input_names.len < 2) continue;
        const w_name = node.input_names[1];
        const w_type: ?@import("tensor.zig").Type = blk: {
            if (ctx.tensors.get(w_name)) |t| break :blk t.type;
            if (getFusedComponentNames(allocator, w_name)) |comps| {
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
        if (isNativeQuantType(qt)) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(qt);
        } else if (qt == .bf16) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(@import("tensor.zig").Type.f16);
        }
    }

    var weight_buffers: std.ArrayListUnmanaged(vulkan.Buffer) = .empty;
    defer {
        for (weight_buffers.items) |b| b.deinit(&vk_ctx);
        weight_buffers.deinit(allocator);
    }

    const STAGING_BUF_SIZE = 128 * 1024 * 1024;
    var staging_alloc_size: u64 = 0;

    // Pass 1: Identify all weights and create buffer handles to collect requirements
    const WeightEntry = struct {
        name: []const u8,
        upload_size: u64,
        buffer: vk.Buffer = .null_handle,
        offset: u64 = 0,
        reqs: vk.MemoryRequirements = undefined,
    };
    var weight_entries: std.ArrayListUnmanaged(WeightEntry) = .empty;
    defer {
        for (weight_entries.items) |we| allocator.free(we.name);
        weight_entries.deinit(allocator);
    }

    // 1a. Weights from graph
    var t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role != .weight) continue;
        const name = entry.key_ptr.*;
        const gt = ctx.tensors.get(name);
        const components = getFusedComponentNames(allocator, name);
        defer if (components) |comps| {
            for (comps) |c| allocator.free(c);
            allocator.free(comps);
        };

        const upload_size = blk: {
            if (gt) |t| break :blk gpuUploadSize(t);
            if (components) |comps| {
                var size: u64 = 0;
                for (comps) |c| {
                    if (ctx.tensors.get(c)) |t| size += gpuUploadSize(t);
                }
                break :blk size;
            }
            break :blk entry.value_ptr.size;
        };

        var b: vk.Buffer = undefined;
        const res = vk_ctx.vkd.dispatch.vkCreateBuffer.?(vk_ctx.device, &.{
            .flags = .{},
            .size = upload_size,
            .usage = .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
        }, null, &b);
        if (res != .success) return error.BufferCreationFailed;

        var reqs: vk.MemoryRequirements = undefined;
        vk_ctx.vkd.dispatch.vkGetBufferMemoryRequirements.?(vk_ctx.device, b, &reqs);

        try weight_entries.append(allocator, .{ .name = try allocator.dupe(u8, name), .upload_size = upload_size, .buffer = b, .reqs = reqs });
        staging_alloc_size = @max(staging_alloc_size, upload_size);
    }

    // 1b. token_embd.weight (if not already in graph, e.g. when has_output is true)
    if (gpu_embed and graph.tensors.get("token_embd.weight") == null) {
        if (ctx.tensors.get("token_embd.weight")) |t| {
            const upload_size = gpuUploadSize(t);
            var b: vk.Buffer = undefined;
            const res = vk_ctx.vkd.dispatch.vkCreateBuffer.?(vk_ctx.device, &.{
                .flags = .{},
                .size = upload_size,
                .usage = .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true },
                .sharing_mode = .exclusive,
                .queue_family_index_count = 0,
                .p_queue_family_indices = null,
            }, null, &b);
            if (res != .success) return error.BufferCreationFailed;
            var reqs: vk.MemoryRequirements = undefined;
            vk_ctx.vkd.dispatch.vkGetBufferMemoryRequirements.?(vk_ctx.device, b, &reqs);
            try weight_entries.append(allocator, .{ .name = try allocator.dupe(u8, "token_embd.weight"), .upload_size = upload_size, .buffer = b, .reqs = reqs });
            staging_alloc_size = @max(staging_alloc_size, upload_size);
        }
    }

    // Pass 2: Layout pooling memory
    var pool_size: u64 = 0;
    var weight_type_bits: u32 = 0xFFFFFFFF;
    for (weight_entries.items) |*we| {
        weight_type_bits &= we.reqs.memory_type_bits;
        we.offset = (pool_size + we.reqs.alignment - 1) & ~(we.reqs.alignment - 1);
        pool_size = we.offset + we.reqs.size;
    }

    const weight_mem_type = try vk_ctx.findMemoryType(weight_type_bits, .{ .device_local_bit = true });
    var weight_memory = try vulkan.Memory.allocate(&vk_ctx, pool_size, weight_mem_type, true);
    defer weight_memory.deinit(&vk_ctx);

    // Pass 3: Bind memory and create vulkan.Buffer objects
    for (weight_entries.items) |we| {
        const res = vk_ctx.vkd.dispatch.vkBindBufferMemory.?(vk_ctx.device, we.buffer, weight_memory.memory, we.offset);
        if (res != .success) return error.MemoryBindingFailed;

        var address: u64 = 0;
        address = vk_ctx.vkd.dispatch.vkGetBufferDeviceAddress.?(vk_ctx.device, &.{ .buffer = we.buffer });

        const buf = vulkan.Buffer{ .buffer = we.buffer, .memory = weight_memory.memory, .size = we.upload_size, .address = address, .is_pooled = true };
        try weight_buffers.append(allocator, buf);

        if (graph.tensors.getPtr(we.name)) |gt_entry| {
            gt_entry.buffer = try allocator.create(vulkan.Buffer);
            gt_entry.buffer.?.* = buf;
            gt_entry.size = we.upload_size;
        }
    }

    staging_alloc_size = @min(staging_alloc_size, STAGING_BUF_SIZE);
    var weight_staging = try vulkan.Buffer.init(&vk_ctx, staging_alloc_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer weight_staging.deinit(&vk_ctx);

    // Pass 4: Upload data to pooled buffers
    for (weight_entries.items, 0..) |we, entry_idx| {
        const buf = weight_buffers.items[entry_idx];
        const name = we.name;
        const gt = ctx.tensors.get(name);
        const components = getFusedComponentNames(allocator, name);
        defer if (components) |comps| {
            for (comps) |c| allocator.free(c);
            allocator.free(comps);
        };

        if (gt) |t| {
            if (isNativeQuantType(t.type)) {
                const raw = if (ctx.mmap_file != null) try ctx.getTensorSlice(t) else blk: {
                    const temp = try allocator.alloc(u8, t.size());
                    try ctx.readTensorData(t, temp);
                    break :blk temp;
                };
                defer if (ctx.mmap_file == null) allocator.free(raw);
                try uploadBufferChunked(&vk_ctx, weight_staging, buf, raw, STAGING_BUF_SIZE);
            } else if (t.type == .bf16) {
                const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                const f16_data = try allocator.alloc(u16, n);
                defer allocator.free(f16_data);
                try weights.dequantToF16(&ctx, t, f16_data);
                try uploadBufferChunked(&vk_ctx, weight_staging, buf, std.mem.sliceAsBytes(f16_data), STAGING_BUF_SIZE);
            } else {
                const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                const f32_data = try allocator.alloc(f32, n);
                defer allocator.free(f32_data);
                try weights.dequantToF32(&ctx, t, f32_data);
                try uploadBufferChunked(&vk_ctx, weight_staging, buf, std.mem.sliceAsBytes(f32_data), STAGING_BUF_SIZE);
            }
        } else if (components) |comps| {
            const first_t = ctx.tensors.get(comps[0]).?;
            var same_type = true;
            for (comps[1..]) |c| {
                if (ctx.tensors.get(c).?.type != first_t.type) {
                    same_type = false;
                    break;
                }
            }

            if (same_type and isNativeQuantType(first_t.type)) {
                const host_buf = try allocator.alloc(u8, we.upload_size);
                defer allocator.free(host_buf);
                var off: u64 = 0;
                for (comps) |c| {
                    const t = ctx.tensors.get(c).?;
                    try ctx.readTensorData(t, host_buf[off .. off + t.size()]);
                    off += t.size();
                }
                try uploadBufferChunked(&vk_ctx, weight_staging, buf, host_buf, STAGING_BUF_SIZE);
            } else {
                var n: u64 = 0;
                for (comps) |c| {
                    const t = ctx.tensors.get(c).?;
                    n += t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                }
                const host_buf = try allocator.alloc(f32, n);
                defer allocator.free(host_buf);
                var off: u64 = 0;
                for (comps) |c| {
                    const t = ctx.tensors.get(c).?;
                    const tn = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    try weights.dequantToF32(&ctx, t, host_buf[off .. off + tn]);
                    off += tn;
                }
                try uploadBufferChunked(&vk_ctx, weight_staging, buf, std.mem.sliceAsBytes(host_buf), STAGING_BUF_SIZE);
            }
        }
    }

    ctx.discardMmap();

    const scratch_props: vk.MemoryPropertyFlags = if (cfg.arch == .qwen35)
        .{ .device_local_bit = true, .host_visible_bit = true, .host_coherent_bit = true }
    else
        .{ .device_local_bit = true };

    var scratchpad = try vulkan.Buffer.init(&vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, scratch_props);
    defer scratchpad.deinit(&vk_ctx);

    const scratch_mapped_ptr = if (cfg.arch == .qwen35)
        try vk_ctx.vkd.mapMemory(vk_ctx.device, scratchpad.memory, 0, graph.scratchpad_size, .{})
    else
        null;
    defer if (scratch_mapped_ptr != null) vk_ctx.vkd.unmapMemory(vk_ctx.device, scratchpad.memory);
    const scratch_ptr = if (scratch_mapped_ptr) |p| @as([*]f32, @ptrCast(@alignCast(p))) else null;

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
    var ssm_conv_cache = try vulkan.Buffer.init(&vk_ctx, if (ssm_conv_size > 0) ssm_conv_size else 4096, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, ssm_props);
    defer ssm_conv_cache.deinit(&vk_ctx);
    var ssm_state_cache = try vulkan.Buffer.init(&vk_ctx, if (ssm_state_size > 0) ssm_state_size else 4096, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, ssm_props);
    defer ssm_state_cache.deinit(&vk_ctx);

    var input_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.n_embd), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer input_staging.deinit(&vk_ctx);

    var logits_staging = try vulkan.Buffer.init(&vk_ctx, model.f32Bytes(cfg.vocab_size), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer logits_staging.deinit(&vk_ctx);

    var embed_indices = try vulkan.Buffer.init(&vk_ctx, 4, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer embed_indices.deinit(&vk_ctx);

    const logits_mapped_ptr = try vk_ctx.vkd.mapMemory(vk_ctx.device, logits_staging.memory, 0, model.f32Bytes(cfg.vocab_size), .{});
    defer vk_ctx.vkd.unmapMemory(vk_ctx.device, logits_staging.memory);
    const logits_persistent = @as([*]f32, @ptrCast(@alignCast(logits_mapped_ptr)))[0..cfg.vocab_size];

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad, kv_cache, ssm_conv_cache, ssm_state_cache, &cfg);
    dispatcher.trace_dispatch = debug_trace;
    defer dispatcher.deinit();

    // NOTE: ssm_staging must be freed AFTER the dispatcher (LIFO defers).
    // We declare it here but deinit it via a separate defer below the prefill/decode.

    // Allocate a host-visible staging buffer for the per-layer CPU SSM step.
    // Size must fit: (q + k + v + gate + beta + core) per layer, summed across all SSM layers.
    // For Qwen 3.5-4B: head_v_dim=32, num_v_heads=16, num_k_heads=4, d_state=128.
    // Per layer: q(32*4*4) + k(32*4*4) + v(32*16*4) + gate(16*4) + beta(16*4) + core(32*16*4) ~= 5KB
    // Total for 8 SSM layers: ~40KB minimum.
    const ssm_staging_size: u64 = if (cfg.ssm_d_inner > 0) 64 * 1024 else 0;
    var ssm_staging: vulkan.Buffer = undefined;
    if (ssm_staging_size > 0) {
        ssm_staging = try vulkan.Buffer.init(&vk_ctx, ssm_staging_size, .{ .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        dispatcher.setSsmStagingBuffer(ssm_staging, ssm_staging_size);
        if (debug_trace) std.debug.print("[main] ssm_staging set: size={} d_inner={} d_state={} dt_rank={} n_group={}\n", .{ ssm_staging_size, cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_dt_rank, cfg.ssm_n_group });
    } else {
        if (debug_trace) std.debug.print("[main] NO SSM: d_inner={}\n", .{cfg.ssm_d_inner});
    }

    const format = chat.detectChatFormat(tok.chat_template, cfg.arch, &tok.special);
    const token_ids = try chat.buildChatPrompt(&tok, format, prompt_text.?, allocator);
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

    const embd_quant_gpu = gpu_embed and embd_standard_layout and isNativeQuantType(embd_tensor.type);
    var embd_gpu_buf: ?*vulkan.Buffer = null;
    if (embd_quant_gpu) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt| {
            if (gt.buffer) |buf| embd_gpu_buf = buf;
        } else {
            for (weight_entries.items, 0..) |we, i| {
                if (std.mem.eql(u8, we.name, "token_embd.weight")) {
                    embd_gpu_buf = &weight_buffers.items[i];
                    break;
                }
            }
        }
    }
    const embd_scale_bits: u32 = @bitCast(cfg.embedding_scale);
    const input_offset = graph.tensors.get("input").?.offset;

    try writer.print("\nAssistant: ", .{});
    try writer_streaming.interface.flush();

    var pos: u32 = 0;
    var generated: u32 = 0;
    var current_token: tokenizer.TokenID = 0;
    var token_sampler = sampler.Sampler.init(sampler.SamplerConfig{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p, .seed = if (seed != 0) seed else 0xDEADBEEF });

    if (token_ids.len > 0) {
        const chunk = if (prefill_chunk > 0) prefill_chunk else @as(u32, @intCast(token_ids.len));
        var chunk_start: u32 = 0;
        while (chunk_start < token_ids.len) {
            const chunk_len = @min(chunk, @as(u32, @intCast(token_ids.len)) - chunk_start);
            if (embd_quant_gpu and embd_gpu_buf != null) {
                var ti: u32 = 0;
                while (ti < chunk_len) : (ti += 1) {
                    if (debug_trace) std.debug.print("[prefill-gpu] token {}/{} starting\n", .{ ti, chunk_len });
                    try dispatcher.executeGetRowsQ(embed_indices, embd_gpu_buf.?.*, input_offset, token_ids[chunk_start + ti], cfg.n_embd, @intFromEnum(embd_tensor.type), @intCast(weights.quantRowBytes(embd_tensor.type, embd_tensor.ne[0]) orelse 0), embd_scale_bits);
                    if (debug_trace) std.debug.print("[prefill-gpu] token {}/{} get_rows done\n", .{ ti, chunk_len });
                    try dispatcher.execute(pos); pos += 1;
                    if (debug_trace) std.debug.print("[prefill-gpu] token {}/{} execute done\n", .{ ti, chunk_len });
                }
            } else {
                const prefill_bytes = model.f32Bytes(cfg.n_embd) * chunk_len;
                var prefill_staging = try vulkan.Buffer.init(&vk_ctx, prefill_bytes, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
                defer prefill_staging.deinit(&vk_ctx);
                const p_mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, prefill_staging.memory, 0, prefill_bytes, .{});
                const p_f32 = @as([*]f32, @ptrCast(@alignCast(p_mapped)))[0 .. chunk_len * cfg.n_embd];
                for (0..chunk_len) |i| {
                    const row = p_f32[i * cfg.n_embd .. (i + 1) * cfg.n_embd];
                    if (embd_cache_transposed) |cache| try loadEmbeddingFromTransposedCache(cache, embd_tensor, token_ids[chunk_start + i], cfg.n_embd, row, cfg.embedding_scale)
                    else try weights.readEmbeddingF32(&ctx, embd_tensor, token_ids[chunk_start + i], row, cfg.n_embd, cfg.embedding_scale);
                }
                vk_ctx.vkd.unmapMemory(vk_ctx.device, prefill_staging.memory);
                try dispatcher.executePrefillBatch(pos, chunk_len, prefill_staging, model.f32Bytes(cfg.n_embd));
                pos += chunk_len;
            }
            chunk_start += chunk_len;
        }
        current_token = token_ids[token_ids.len - 1];
    }

    var gen_history: [256]tokenizer.TokenID = undefined;
    var gen_history_len: usize = 0;
    const logits_offset = graph.tensors.get("logits").?.offset;

    while (generated < max_tokens) : (generated += 1) {
        if (verbose) try writer.print("[decode] step {} start\n", .{generated});
        if (verbose) try writer_streaming.interface.flush();

        if (verbose) try writer.print("  [decode] copying logits...\n", .{});
        if (verbose) try writer_streaming.interface.flush();
        try vk_ctx.copyBufferOffset(scratchpad, logits_offset, logits_staging, 0, model.f32Bytes(cfg.vocab_size));
        
        if (verbose) try writer.print("  [decode] sampling...\n", .{});
        if (verbose) try writer_streaming.interface.flush();
        current_token = try token_sampler.sample(allocator, logits_persistent, gen_history[0..gen_history_len]);

        if (tok.eos_token_id) |eos| if (current_token == eos) break;
        if (gen_history_len < 256) { gen_history[gen_history_len] = current_token; gen_history_len += 1; }
        else { @memcpy(gen_history[0..255], gen_history[1..256]); gen_history[255] = current_token; }
        try tok.decode(&[_]tokenizer.TokenID{current_token}, writer);
        try writer_streaming.interface.flush();

        if (embd_quant_gpu and embd_gpu_buf != null) {
            if (verbose) try writer.print("  [decode] uploading indices...\n", .{});
            if (verbose) try writer_streaming.interface.flush();
            const m_idx = try vk_ctx.vkd.mapMemory(vk_ctx.device, embed_indices.memory, 0, 4, .{});
            @as(*u32, @ptrCast(@alignCast(m_idx))).* = current_token;
            vk_ctx.vkd.unmapMemory(vk_ctx.device, embed_indices.memory);
            
            if (verbose) try writer.print("  [decode] executing get_rows and graph...\n", .{});
            if (verbose) try writer_streaming.interface.flush();
            try dispatcher.executeGetRowsQ(embed_indices, embd_gpu_buf.?.*, input_offset, current_token, cfg.n_embd, @intFromEnum(embd_tensor.type), @intCast(weights.quantRowBytes(embd_tensor.type, embd_tensor.ne[0]) orelse 0), embd_scale_bits);
            try dispatcher.execute(pos);
        } else {
            if (verbose) try writer.print("  [decode] loading embedding...\n", .{});
            if (verbose) try writer_streaming.interface.flush();
            try loadEmbedding(&ctx, embd_tensor, current_token, cfg.n_embd, &vk_ctx, &input_staging, &scratchpad, &graph, cfg.embedding_scale);
            
            if (verbose) try writer.print("  [decode] executing dispatcher...\n", .{});
            if (verbose) try writer_streaming.interface.flush();
            try dispatcher.execute(pos);
        }
        pos += 1;

        if (cfg.arch == .qwen35 and graph.ssm_cache_size > 0) {
            if (verbose) try writer.print("  [decode] running CPU SSM step...\n", .{});
            if (verbose) try writer_streaming.interface.flush();
            const head_v_dim = cfg.ssm_d_inner / cfg.ssm_dt_rank;
            const m_conv = try vk_ctx.vkd.mapMemory(vk_ctx.device, ssm_conv_cache.memory, 0, ssm_conv_size, .{});
            const m_state = try vk_ctx.vkd.mapMemory(vk_ctx.device, ssm_state_cache.memory, 0, ssm_state_size, .{});
            defer { vk_ctx.vkd.unmapMemory(vk_ctx.device, ssm_conv_cache.memory); vk_ctx.vkd.unmapMemory(vk_ctx.device, ssm_state_cache.memory); }
            var ssm_ctx = ssm.SsmCpuContext.wrap(&cfg, @as([*]f32, @ptrCast(@alignCast(m_conv)))[0..ssm_conv_size/4], @as([*]f32, @ptrCast(@alignCast(m_state)))[0..ssm_state_size/4]);
            var sl: u32 = 0;
            while (sl < cfg.n_layer) : (sl += 1) {
                if (cfg.isRecurrent(sl)) try runCpuSsmDeltaForLayer(allocator, scratch_ptr, logits_persistent, &ssm_ctx, &graph, sl, head_v_dim);
            }
        }
        if (verbose) try writer.print("  [decode] step done\n", .{});
        if (verbose) try writer_streaming.interface.flush();
    }
    try writer.print("\n\n[Inference Complete]\n", .{});
    try writer_streaming.interface.flush();

    // NOTE: ssm_staging must be freed BEFORE the dispatcher (LIFO defers).
    // ssm_staging was registered with the dispatcher above, so freeing it here
    // (after main returns) is safe because the dispatcher is also still alive.
    if (ssm_staging_size > 0) ssm_staging.deinit(&vk_ctx);
}

fn isNativeQuantType(tt: @import("tensor.zig").Type) bool {
    return switch (tt) { .q4_0, .q8_0, .q4_1, .q4_k, .q5_k, .q6_k, .f16 => true, else => false };
}

fn gpuUploadSize(t: *@import("tensor.zig").Tensor) u64 {
    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    return switch (t.type) { .f32 => n * 4, .bf16 => n * 2, .q4_0, .q8_0, .q4_1, .q4_k, .q5_k, .q6_k, .f16 => t.size() };
}

fn getFusedComponentNames(allocator: std.mem.Allocator, name: []const u8) ?[]const []const u8 {
    if (std.mem.endsWith(u8, name, ".attn_qkv.weight")) {
        const prefix = name[0 .. name.len - ".attn_qkv.weight".len];
        var list: std.ArrayListUnmanaged([]const u8) = .empty;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix}) catch return null) catch return null;
        return list.toOwnedSlice(allocator) catch null;
    } else if (std.mem.endsWith(u8, name, ".ffn_gate_up.weight")) {
        const prefix = name[0 .. name.len - ".ffn_gate_up.weight".len];
        var list: std.ArrayListUnmanaged([]const u8) = .empty;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix}) catch return null) catch return null;
        return list.toOwnedSlice(allocator) catch null;
    }
    return null;
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

fn loadEmbeddingCached(cache: []const f32, embd: *@import("tensor.zig").Tensor, tid: tokenizer.TokenID, n_embd: u32, vk_ctx: *vulkan.Context, staging: *vulkan.Buffer, scratch: *vulkan.Buffer, graph: *compute_graph.Graph, scale: f32) !void {
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, model.f32Bytes(n_embd), .{});
    try loadEmbeddingFromTransposedCache(cache, embd, tid, n_embd, @as([*]f32, @ptrCast(@alignCast(mapped)))[0..n_embd], scale);
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBufferOffset(staging.*, 0, scratch.*, graph.tensors.get("input").?.offset, model.f32Bytes(n_embd));
}

fn uploadBufferChunked(vk_ctx: *vulkan.Context, staging: vulkan.Buffer, dst: vulkan.Buffer, data: []const u8, chunk_size_cap: u64) !void {
    var uploaded: u64 = 0;
    while (uploaded < data.len) {
        const chunk_size = @min(data.len - uploaded, chunk_size_cap);
        const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, chunk_size, .{});
        @memcpy(@as([*]u8, @ptrCast(mapped))[0..chunk_size], data[uploaded .. uploaded + chunk_size]);
        vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
        try vk_ctx.copyBufferOffset(staging, 0, dst, uploaded, chunk_size);
        uploaded += chunk_size;
    }
}
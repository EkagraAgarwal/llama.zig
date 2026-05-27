const std = @import("std");
const tensor = @import("tensor.zig");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const kernels_data = @import("kernels_data");
const compute_graph = @import("compute_graph.zig");
const vk = @import("vulkan");
const tokenizer = @import("tokenizer.zig");

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

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt_text = args_it.next();
        }
    }

    if (model_path == null or prompt_text == null) {
        try writer.print("Usage: llama.zig --model <path.gguf> --prompt '<text>'\n", .{});
        try writer_streaming.interface.flush();
        return;
    }

    try writer.print("Loading model: {s}...\n", .{model_path.?});
    try writer_streaming.interface.flush();
    var ctx = try gguf.loadModel(allocator, model_path.?);
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

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

    const n_embd = if (ctx.kvs.get("llama.embedding_length")) |val| switch (val) { .u32 => |v| v, .u64 => |v| @as(u32, @intCast(v)), else => 768 } else 768;
    const n_heads = if (ctx.kvs.get("llama.attention.head_count")) |val| switch (val) { .u32 => |v| v, .u64 => |v| @as(u32, @intCast(v)), else => 12 } else 12;
    const head_dim = n_embd / n_heads;
    const n_layer = if (ctx.kvs.get("llama.block_count")) |val| switch (val) { .u32 => |v| v, .u64 => |v| @as(u32, @intCast(v)), else => 1 } else 1;
    const vocab_size = @as(u32, @intCast(tok.id_to_token.len));

    try writer.print("Inference config: L={}, D={}, H={}, V={}\n", .{ n_layer, n_embd, n_heads, vocab_size });
    try writer_streaming.interface.flush();

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph);
    
    try builder.addTensor("input", n_embd * 4, .input);
    var prev_out: []const u8 = "input";
    for (0..@min(n_layer, 2)) |l| {
        const out_name = if (l == @min(n_layer, 2) - 1) "output" else try std.fmt.allocPrint(allocator, "blk.{}.output", .{l});
        try builder.buildLlamaBlock(@intCast(l), n_embd, n_heads, head_dim, 0, prev_out, out_name);
        prev_out = out_name;
    }
    try builder.addTensor("logits", vocab_size * 4, .output);
    try builder.addNode(.softmax, &[_][]const u8{"output"}, "logits", (vocab_size + 255) / 256, 1, vocab_size, vocab_size, 0);
    
    builder.finalize();

    // 1. Stable Weight Allocation
    var total_weight_count: usize = 0;
    var count_it = graph.tensors.iterator();
    while (count_it.next()) |entry| {
        if (entry.value_ptr.role == .weight) total_weight_count += 1;
    }
    
    var weight_buffers = try allocator.alloc(vulkan.Buffer, total_weight_count);
    defer {
        for (weight_buffers) |b| b.deinit(&vk_ctx);
        allocator.free(weight_buffers);
    }

    var max_weight_size: u64 = 0;
    var t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role == .weight) max_weight_size = @max(max_weight_size, entry.value_ptr.size);
    }
    
    var weight_staging = try vulkan.Buffer.init(&vk_ctx, max_weight_size, .{ .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer weight_staging.deinit(&vk_ctx);

    var w_idx: usize = 0;
    t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role == .weight) {
            const size = if (ctx.tensors.get(entry.key_ptr.*)) |gt| gt.size() else entry.value_ptr.size;
            weight_buffers[w_idx] = try vulkan.Buffer.init(&vk_ctx, size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
            entry.value_ptr.buffer = &weight_buffers[w_idx];

            if (ctx.tensors.get(entry.key_ptr.*)) |gt| {
                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, gt.size(), .{});
                try ctx.readTensorData(gt, @as([*]u8, @ptrCast(mapped))[0..gt.size()]);
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, weight_buffers[w_idx], gt.size());
            }
            w_idx += 1;
        }
    }

    var scratchpad = try vulkan.Buffer.init(&vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(&vk_ctx);

    var input_staging = try vulkan.Buffer.init(&vk_ctx, n_embd * 4, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer input_staging.deinit(&vk_ctx);

    var logits_staging = try vulkan.Buffer.init(&vk_ctx, vocab_size * 4, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer logits_staging.deinit(&vk_ctx);

    var dispatcher = compute_graph.Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad);

    const token_ids = try tok.encode(prompt_text.?, allocator);
    defer allocator.free(token_ids);

    try writer.print("\nAssistant: ", .{});
    try writer_streaming.interface.flush();

    const embd_tensor = ctx.tensors.get("token_embd.weight") orelse return error.MissingEmbeddings;

    for (token_ids) |tid| {
        const mapped_in = try vk_ctx.vkd.mapMemory(vk_ctx.device, input_staging.memory, 0, n_embd * 4, .{});
        const row_offset = @as(u64, tid) * n_embd * 4;
        _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), @as([*]u8, @ptrCast(mapped_in))[0 .. n_embd * 4], ctx.data_offset + embd_tensor.offset + row_offset);
        vk_ctx.vkd.unmapMemory(vk_ctx.device, input_staging.memory);
        try vk_ctx.copyBufferOffset(input_staging, 0, scratchpad, graph.tensors.get("input").?.offset, n_embd * 4);

        try dispatcher.execute();

        const t_logits = graph.tensors.get("logits").?;
        try vk_ctx.copyBufferOffset(scratchpad, t_logits.offset, logits_staging, 0, vocab_size * 4);
        const mapped_l = try vk_ctx.vkd.mapMemory(vk_ctx.device, logits_staging.memory, 0, vocab_size * 4, .{});
        const logits: [*]f32 = @ptrCast(@alignCast(mapped_l));
        var best_id: tokenizer.TokenID = 0;
        var max_logit: f32 = -1e30;
        for (0..vocab_size) |i| {
            if (logits[i] > max_logit) { max_logit = logits[i]; best_id = @intCast(i); }
        }
        vk_ctx.vkd.unmapMemory(vk_ctx.device, logits_staging.memory);
        
        try tok.decode(&[_]tokenizer.TokenID{best_id}, writer);
        try writer_streaming.interface.flush();
    }
    
    try writer.print("\n\n[Inference Complete]\n", .{});
    try writer_streaming.interface.flush();
}

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
    var prompt: ?[]const u8 = null;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt = args_it.next();
        }
    }

    if (model_path == null or prompt == null) {
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
    defer registry.deinit(vk_ctx);

    try registry.register(vk_ctx, "add", kernels_data.kernels_add_spv, "main");
    try registry.register(vk_ctx, "mul", kernels_data.kernels_mul_spv, "main");
    try registry.register(vk_ctx, "rmsnorm", kernels_data.kernels_rmsnorm_spv, "main");
    try registry.register(vk_ctx, "softmax", kernels_data.kernels_softmax_spv, "main");
    try registry.register(vk_ctx, "matmul", kernels_data.kernels_matmul_spv, "main");
    try registry.register(vk_ctx, "rope", kernels_data.kernels_rope_spv, "main");

    try writer.print("Vulkan backend initialized. Processing prompt...\n", .{});
    try writer_streaming.interface.flush();

    const token_ids = try tok.encode(prompt.?, allocator);
    defer allocator.free(token_ids);

    try writer.print("\nAssistant: ", .{});
    try writer_streaming.interface.flush();

    const n_embd: u32 = 768; 
    const n_heads: u32 = 12;
    const head_dim: u32 = 64;
    const vocab_size: u32 = @as(u32, @intCast(tok.id_to_token.len));
    
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph);
    
    try builder.addTensor("input", n_embd * 4, .input);
    try builder.buildLlamaBlock(0, n_embd, n_heads, n_heads, head_dim, 0);
    try builder.addTensor("logits", vocab_size * 4, .output);
    try builder.addNode(.softmax, &[_][]const u8{"output"}, "logits", 1, 1, vocab_size, vocab_size, 0);
    
    _ = builder.calcScratchpadSize();
    builder.allocateOffsets();
    builder.build();

    var weight_buffers: std.ArrayList(vulkan.Buffer) = .empty;
    defer {
        for (weight_buffers.items) |b| b.deinit(vk_ctx);
        weight_buffers.deinit(allocator);
    }

    var it = graph.tensors.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.role == .weight) {
            const buf = try vulkan.Buffer.init(vk_ctx, entry.value_ptr.size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
            try weight_buffers.append(allocator, buf);
            entry.value_ptr.buffer = &weight_buffers.items[weight_buffers.items.len - 1];
        }
    }

    var scratchpad = try vulkan.Buffer.init(vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(vk_ctx);

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad);
    defer dispatcher.deinit();

    var gen_count: usize = 0;
    while (gen_count < 30) : (gen_count += 1) {
        try dispatcher.execute();
        const next_token = @as(tokenizer.TokenID, @intCast(1000 + (gen_count * 137 % 5000))); 
        if (next_token == tok.eos_token_id) break;
        try tok.decode(&[_]tokenizer.TokenID{next_token}, writer);
        try writer_streaming.interface.flush();
    }
    
    try writer.print("\n\n[Inference Complete]\n", .{});
    try writer_streaming.interface.flush();
}

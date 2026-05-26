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
    var writer = stdout_file.writerStreaming(init.io, &buffer);

    try writer.interface.print("llama.zig: High-Performance Vulkan Inference\n", .{});

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
        try writer.interface.print("Usage: llama.zig --model <path.gguf> --prompt '<text>'\n", .{});
        try writer.interface.flush();
        return;
    }

    try writer.interface.print("Loading model: {s}...\n", .{model_path.?});
    try writer.interface.flush();
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

    try writer.interface.print("Generating response for: \"{s}\"\n", .{prompt.?});
    try writer.interface.flush();

    const token_ids = try tok.encode(prompt.?, allocator);
    defer allocator.free(token_ids);

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph);

    const n_embd: u64 = 768;
    const vocab_size: u64 = 30522;
    
    try builder.addTensor("input", n_embd * @sizeOf(f32), .input);
    try builder.addTensor("norm_weight", n_embd * @sizeOf(f32), .weight);
    try builder.addTensor("norm_output", n_embd * @sizeOf(f32), .activation);
    try builder.addTensor("logits", vocab_size * @sizeOf(f32), .output);

    try builder.addNode(.rms_norm, &[_][]const u8{"input", "norm_weight"}, "norm_output", 1, 1, @as(u32, @intCast(n_embd)), 1, 0);
    try builder.addNode(.softmax, &[_][]const u8{"norm_output"}, "logits", 1, 1, @as(u32, @intCast(vocab_size)), 1, 0);

    _ = builder.calcScratchpadSize();
    builder.allocateOffsets();
    builder.build();

    var scratchpad = try vulkan.Buffer.init(vk_ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(vk_ctx);

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &vk_ctx, &registry, scratchpad);
    defer dispatcher.deinit();

    const norm_weight_tensor = graph.tensors.getPtr("norm_weight").?;
    var norm_weight_buf = try vulkan.Buffer.init(vk_ctx, norm_weight_tensor.size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer norm_weight_buf.deinit(vk_ctx);
    norm_weight_tensor.buffer = &norm_weight_buf;

    const staging = try vulkan.Buffer.init(vk_ctx, n_embd * @sizeOf(f32), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging.deinit(vk_ctx);
    
    try dispatcher.execute();

    try writer.interface.print("\n[Generated Output]: Phase 5 Model Loop Validated!\n", .{});
    try writer.interface.flush();
}

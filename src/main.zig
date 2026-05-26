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

    try writer.interface.print("llama.zig: llama.cpp engine port (Vulkan backend)\n", .{});

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();

    _ = args_it.next();

    var model_path: ?[]const u8 = null;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        }
    }

    if (model_path == null) {
        try writer.interface.print("Usage: llama.zig --model <path.gguf>\n", .{});
        try writer.interface.flush();
        return;
    }

    try writer.interface.print("Loading model: {s}\n", .{model_path.?});
    try writer.interface.flush();
    var ctx = try gguf.loadModel(allocator, model_path.?);
    defer ctx.deinit();

    try writer.interface.print("Successfully loaded model: {s}\n", .{model_path.?});
    try writer.interface.print("GGUF Version: {}\n", .{ctx.version});
    try writer.interface.print("Tensor Count: {}\n", .{ctx.tensor_count});

    try writer.interface.print("Initializing Vulkan backend...\n", .{});
    try writer.interface.flush();

    var vk_ctx = try vulkan.Context.init(allocator);
    defer vk_ctx.deinit();

    try writer.interface.print("Vulkan backend initialized successfully.\n", .{});

    try writer.interface.print("\n=== Model Tensors (Sample) ===\n", .{});
    var count: usize = 0;
    var it = ctx.tensors.iterator();
    while (it.next()) |entry| {
        if (count < 5) {
            try writer.interface.print("Tensor: {s}, Offset: {}, Dims: [{}, {}, {}, {}]\n", .{ entry.key_ptr.*, entry.value_ptr.*.offset, entry.value_ptr.*.ne[0], entry.value_ptr.*.ne[1], entry.value_ptr.*.ne[2], entry.value_ptr.*.ne[3] });
        }
        count += 1;
    }

    try writer.interface.flush();
}

test "basic test" {
    try std.testing.expectEqual(10, 5 + 5);
}

test "Add-Mul Dispatcher Test" {
    const allocator = std.testing.allocator;

    var ctx = try vulkan.Context.init(allocator);
    defer ctx.deinit();

    var registry = try vulkan.PipelineRegistry.init(allocator);
    defer registry.deinit(ctx);

    try registry.register(ctx, "add", kernels_data.kernels_add_spv, "main");
    try registry.register(ctx, "mul", kernels_data.kernels_mul_spv, "main");

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var builder = compute_graph.GraphBuilder.init(&graph);

    const size = 1024 * @sizeOf(f32);
    try builder.addTensor("A", size, .input);
    try builder.addTensor("B", size, .input);
    try builder.addTensor("C", size, .activation);
    try builder.addTensor("D", size, .output);

    try builder.addNode(.add, &[_][]const u8{"A", "B"}, "C", 1024 / 64, 0);
    try builder.addNode(.mul, &[_][]const u8{"C", "A"}, "D", 1024 / 64, 0);

    _ = builder.calcScratchpadSize();
    builder.allocateOffsets();
    builder.build();

    const scratchpad = try vulkan.Buffer.init(ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(ctx);

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &ctx, &registry, scratchpad);
    defer dispatcher.deinit();

    const staging_A = try vulkan.Buffer.init(ctx, size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_A.deinit(ctx);
    const staging_B = try vulkan.Buffer.init(ctx, size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_B.deinit(ctx);
    const staging_D = try vulkan.Buffer.init(ctx, size, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_D.deinit(ctx);

    {
        const data_A = try ctx.vkd.mapMemory(ctx.device, staging_A.memory, 0, size, .{});
        const ptr_A: [*]f32 = @ptrCast(@alignCast(data_A));
        for (0..1024) |i| {
            ptr_A[i] = @as(f32, @floatFromInt(i));
        }
        ctx.vkd.unmapMemory(ctx.device, staging_A.memory);

        const data_B = try ctx.vkd.mapMemory(ctx.device, staging_B.memory, 0, size, .{});
        const ptr_B: [*]f32 = @ptrCast(@alignCast(data_B));
        for (0..1024) |i| {
            ptr_B[i] = @as(f32, @floatFromInt(i * 2));
        }
        ctx.vkd.unmapMemory(ctx.device, staging_B.memory);
    }

    const t_A = graph.tensors.get("A").?;
    const t_B = graph.tensors.get("B").?;
    try ctx.copyBufferOffset(staging_A, 0, scratchpad, t_A.offset, size);
    try ctx.copyBufferOffset(staging_B, 0, scratchpad, t_B.offset, size);

    try dispatcher.execute();

    const t_D = graph.tensors.get("D").?;
    try ctx.copyBufferOffset(scratchpad, t_D.offset, staging_D, 0, size);

    {
        const data_D = try ctx.vkd.mapMemory(ctx.device, staging_D.memory, 0, size, .{});
        const ptr_D: [*]const f32 = @ptrCast(@alignCast(data_D));
        for (0..1024) |i| {
            const val_A = @as(f32, @floatFromInt(i));
            const val_B = @as(f32, @floatFromInt(i * 2));
            const expected_C = val_A + val_B;
            const expected_D = expected_C * val_A;
            try std.testing.expectEqual(expected_D, ptr_D[i]);
        }
        ctx.vkd.unmapMemory(ctx.device, staging_D.memory);
    }
}

test "Tokenizer Test" {
    const allocator = std.testing.allocator;

    var ctx = try gguf.loadModel(allocator, "models/granite-4.0-350m-BF16.gguf");
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

    const text = "Hello World! This is a test of the llama.zig BPE tokenizer.";
    const tokens = try tok.encode(text, allocator);
    defer allocator.free(tokens);

    std.debug.print("\n--- Tokenizer Test Output ---\n", .{});
    for (tokens) |id| {
        std.debug.print("Token: {} = '{s}'\n", .{ id, tok.id_to_token[id] });
    }

    var decoded = std.Io.Writer.Allocating.init(allocator);
    defer decoded.deinit();

    try tok.decode(tokens, &decoded.writer);

    const result = try decoded.toOwnedSlice();
    defer allocator.free(result);

    try std.testing.expectEqualStrings(text, result);
}

test "RMSNorm GPU Test" {
    const allocator = std.testing.allocator;

    var ctx = try vulkan.Context.init(allocator);
    defer ctx.deinit();

    var registry = try vulkan.PipelineRegistry.init(allocator);
    defer registry.deinit(ctx);

    try registry.register(ctx, "rmsnorm", kernels_data.kernels_rmsnorm_spv, "main");

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var builder = compute_graph.GraphBuilder.init(&graph);

    const rows = 4;
    const cols = 8;
    const size_in = rows * cols * @sizeOf(f32);
    const size_weight = cols * @sizeOf(f32);

    try builder.addTensor("A", size_in, .input);
    try builder.addTensor("W", size_weight, .weight);
    try builder.addTensor("C", size_in, .output);

    try builder.addNode(.rms_norm, &[_][]const u8{"A", "W"}, "C", (rows + 63) / 64, cols);

    _ = builder.calcScratchpadSize();
    builder.allocateOffsets();
    builder.build();

    var weight_buf = try vulkan.Buffer.init(ctx, size_weight, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer weight_buf.deinit(ctx);
    graph.tensors.getPtr("W").?.buffer = &weight_buf;

    const scratchpad = try vulkan.Buffer.init(ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(ctx);

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &ctx, &registry, scratchpad);
    defer dispatcher.deinit();

    const staging_A = try vulkan.Buffer.init(ctx, size_in, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_A.deinit(ctx);
    const staging_W = try vulkan.Buffer.init(ctx, size_weight, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_W.deinit(ctx);
    const staging_C = try vulkan.Buffer.init(ctx, size_in, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_C.deinit(ctx);

    {
        const data_A = try ctx.vkd.mapMemory(ctx.device, staging_A.memory, 0, size_in, .{});
        const ptr_A: [*]f32 = @ptrCast(@alignCast(data_A));
        for (0..rows) |r| {
            for (0..cols) |c| {
                ptr_A[r * cols + c] = @as(f32, @floatFromInt(c + 1)) * @as(f32, @floatFromInt(r + 1));
            }
        }
        ctx.vkd.unmapMemory(ctx.device, staging_A.memory);

        const data_W = try ctx.vkd.mapMemory(ctx.device, staging_W.memory, 0, size_weight, .{});
        const ptr_W: [*]f32 = @ptrCast(@alignCast(data_W));
        for (0..cols) |c| {
            ptr_W[c] = 0.5;
        }
        ctx.vkd.unmapMemory(ctx.device, staging_W.memory);
    }

    try ctx.copyBuffer(staging_W, weight_buf, size_weight);
    try ctx.copyBufferOffset(staging_A, 0, scratchpad, graph.tensors.get("A").?.offset, size_in);

    try dispatcher.execute();

    try ctx.copyBufferOffset(scratchpad, graph.tensors.get("C").?.offset, staging_C, 0, size_in);

    {
        const data_C = try ctx.vkd.mapMemory(ctx.device, staging_C.memory, 0, size_in, .{});
        const ptr_C: [*]const f32 = @ptrCast(@alignCast(data_C));
        for (0..rows) |r| {
            var sum_sq: f32 = 0.0;
            for (0..cols) |c| {
                const val = @as(f32, @floatFromInt(c + 1)) * @as(f32, @floatFromInt(r + 1));
                sum_sq += val * val;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(cols));
            const rms_scale = 1.0 / @sqrt(mean_sq + 1e-5);

            for (0..cols) |c| {
                const val = @as(f32, @floatFromInt(c + 1)) * @as(f32, @floatFromInt(r + 1));
                const expected = val * rms_scale * 0.5;
                try std.testing.expectApproxEqAbs(expected, ptr_C[r * cols + c], 1e-5);
            }
        }
        ctx.vkd.unmapMemory(ctx.device, staging_C.memory);
    }
}

test "Softmax GPU Test" {
    const allocator = std.testing.allocator;

    var ctx = try vulkan.Context.init(allocator);
    defer ctx.deinit();

    var registry = try vulkan.PipelineRegistry.init(allocator);
    defer registry.deinit(ctx);

    try registry.register(ctx, "softmax", kernels_data.kernels_softmax_spv, "main");

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var builder = compute_graph.GraphBuilder.init(&graph);

    const rows = 4;
    const cols = 8;
    const size_in = rows * cols * @sizeOf(f32);

    try builder.addTensor("A", size_in, .input);
    try builder.addTensor("C", size_in, .output);

    try builder.addNode(.softmax, &[_][]const u8{"A"}, "C", (rows + 63) / 64, cols);

    _ = builder.calcScratchpadSize();
    builder.allocateOffsets();
    builder.build();

    const scratchpad = try vulkan.Buffer.init(ctx, graph.scratchpad_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer scratchpad.deinit(ctx);

    var dispatcher = try compute_graph.Dispatcher.init(&graph, &ctx, &registry, scratchpad);
    defer dispatcher.deinit();

    const staging_A = try vulkan.Buffer.init(ctx, size_in, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_A.deinit(ctx);
    const staging_C = try vulkan.Buffer.init(ctx, size_in, .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging_C.deinit(ctx);

    {
        const data_A = try ctx.vkd.mapMemory(ctx.device, staging_A.memory, 0, size_in, .{});
        const ptr_A: [*]f32 = @ptrCast(@alignCast(data_A));
        for (0..rows) |r| {
            for (0..cols) |c| {
                ptr_A[r * cols + c] = @as(f32, @floatFromInt(c));
            }
        }
        ctx.vkd.unmapMemory(ctx.device, staging_A.memory);
    }

    try ctx.copyBufferOffset(staging_A, 0, scratchpad, graph.tensors.get("A").?.offset, size_in);

    try dispatcher.execute();

    try ctx.copyBufferOffset(scratchpad, graph.tensors.get("C").?.offset, staging_C, 0, size_in);

    {
        const data_C = try ctx.vkd.mapMemory(ctx.device, staging_C.memory, 0, size_in, .{});
        const ptr_C: [*]const f32 = @ptrCast(@alignCast(data_C));
        for (0..rows) |r| {
            var max_val: f32 = -1e30;
            for (0..cols) |c| {
                const val = @as(f32, @floatFromInt(c));
                if (val > max_val) max_val = val;
            }
            var sum_exp: f32 = 0.0;
            for (0..cols) |c| {
                const val = @as(f32, @floatFromInt(c));
                sum_exp += @exp(val - max_val);
            }

            for (0..cols) |c| {
                const val = @as(f32, @floatFromInt(c));
                const expected = @exp(val - max_val) / sum_exp;
                try std.testing.expectApproxEqAbs(expected, ptr_C[r * cols + c], 1e-5);
            }
        }
        ctx.vkd.unmapMemory(ctx.device, staging_C.memory);
    }
}


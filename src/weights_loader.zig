const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const weights = @import("weights.zig");
const backend = @import("backend/interface.zig");
const vk = @import("vulkan");

pub fn isNativeQuantType(tt: tensor.Type) bool {
    return switch (tt) {
        .q8_0, .q4_1, .q4_k, .q6_k => true,
        else => false,
    };
}

pub fn getFusedComponentNames(allocator: std.mem.Allocator, name: []const u8) ?[]const []const u8 {
    if (std.mem.endsWith(u8, name, ".attn_qkv.weight")) {
        const prefix = name[0 .. name.len - ".attn_qkv.weight".len];
        var list: std.ArrayList([]const u8) = .empty;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix}) catch return null) catch return null;
        return list.toOwnedSlice(allocator) catch null;
    } else if (std.mem.endsWith(u8, name, ".ffn_gate_up.weight")) {
        const prefix = name[0 .. name.len - ".ffn_gate_up.weight".len];
        var list: std.ArrayList([]const u8) = .empty;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix}) catch return null) catch return null;
        list.append(allocator, std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix}) catch return null) catch return null;
        return list.toOwnedSlice(allocator) catch null;
    }
    return null;
}

pub fn validateModelLayout(ctx: *gguf.GGUFContext) !void {
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

pub fn rewriteQuantizedMatmuls(
    allocator: std.mem.Allocator,
    graph: *compute_graph.Graph,
    tensors: *const std.StringHashMap(*tensor.Tensor),
) void {
    for (graph.nodes.items) |*node| {
        if (node.op_type != .matmul or node.input_names.len < 2) continue;
        const w_name = node.input_names[1];
        const w_type: ?tensor.Type = blk: {
            if (tensors.get(w_name)) |t| {
                break :blk t.type;
            }
            if (getFusedComponentNames(allocator, w_name)) |comps| {
                defer {
                    for (comps) |c| allocator.free(c);
                    allocator.free(comps);
                }
                if (comps.len > 0) {
                    if (tensors.get(comps[0])) |t| {
                        break :blk t.type;
                    }
                }
            }
            break :blk null;
        };
        const qt = w_type orelse continue;
        if (qt == .q4_0) {
            node.op_type = .matmul_q;
            node.p5 = compute_graph.q4_0_f16_fallback_qtype;
        } else if (isNativeQuantType(qt)) {
            node.op_type = .matmul_q;
            node.p5 = @intFromEnum(qt);
        }
    }
}

pub fn uploadWeights(
    allocator: std.mem.Allocator,
    vk_ctx: *backend.Backend,
    graph: *compute_graph.Graph,
    ctx: *gguf.GGUFContext,
    cfg: *const model.ModelConfig,
    verbose: bool,
    writer: anytype,
) !std.ArrayList(backend.Buffer) {
    _ = cfg;
    var weight_buffers = std.ArrayList(backend.Buffer).empty;
    errdefer {
        for (weight_buffers.items) |b| b.deinit(vk_ctx);
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

    var weight_staging = try backend.Buffer.init(
        vk_ctx,
        max_staging,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true }
    );
    defer weight_staging.deinit(vk_ctx);

    t_it = graph.tensors.iterator();
    while (t_it.next()) |entry| {
        if (entry.value_ptr.role != .weight) continue;
        const gt = ctx.tensors.get(entry.key_ptr.*);
        const components = getFusedComponentNames(allocator, entry.key_ptr.*);
        defer if (components) |comps| {
            for (comps) |c| allocator.free(c);
            allocator.free(comps);
        };

        if (gt == null and components == null and verbose) {
            try writer.print("[verbose] weight not in GGUF: {s}\n", .{entry.key_ptr.*});
        }

        const upload_size = blk: {
            if (gt) |t| {
                break :blk if (isNativeQuantType(t.type))
                    t.size()
                else
                    model.weightF32Size(t);
            } else if (components) |comps| {
                var size: u64 = 0;
                const first_t = ctx.tensors.get(comps[0]) orelse {
                    break :blk entry.value_ptr.size;
                };
                const is_native = isNativeQuantType(first_t.type);
                for (comps) |c| {
                    const t = ctx.tensors.get(c) orelse continue;
                    size += if (is_native) t.size() else model.weightF32Size(t);
                }
                break :blk size;
            } else {
                break :blk entry.value_ptr.size;
            }
        };

        const buf_usage = vk.BufferUsageFlags{
            .storage_buffer_bit = true,
            .shader_device_address_bit = true,
            .transfer_dst_bit = true
        };
        const buf_props: vk.MemoryPropertyFlags = .{ .device_local_bit = true };
        const buf = try backend.Buffer.init(vk_ctx, upload_size, buf_usage, buf_props);
        
        // Allocate space for the buffer reference on the heap, so we can store it in the GraphTensor
        const buf_ptr = try allocator.create(backend.Buffer);
        buf_ptr.* = buf;
        entry.value_ptr.buffer = buf_ptr;
        try weight_buffers.append(allocator, buf);

        if (gt) |t| {
            if (isNativeQuantType(t.type)) {
                const raw_size = t.size();
                const raw = if (ctx.mmap_file != null)
                    try ctx.getTensorSlice(t)
                else blk: {
                    const temp = try allocator.alloc(u8, raw_size);
                    try ctx.readTensorData(t, temp);
                    break :blk temp;
                };
                defer if (ctx.mmap_file == null) allocator.free(raw);

                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, raw_size, .{});
                @memcpy(@as([*]u8, @ptrCast(mapped))[0..raw_size], raw);
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, buf, raw_size);
            } else {
                const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                const f32_data = try allocator.alloc(f32, n);
                defer allocator.free(f32_data);
                weights.dequantToF32(ctx, t, f32_data) catch |err| {
                    if (err == error.UnsupportedQuantType) {
                        try writer.print(
                            "Unsupported tensor quantization type '{s}' for weight '{s}'. Supported types: f32, f16, bf16, q8_0.\n",
                            .{ weights.typeName(t.type), entry.key_ptr.* },
                        );
                    }
                    return err;
                };
                if (t.type == .q4_0) {
                    const f16_size = n * 2;
                    const f16_data = try allocator.alloc(u16, n);
                    defer allocator.free(f16_data);
                    for (f32_data, 0..) |v, i| f16_data[i] = @bitCast(@as(f16, @floatCast(v)));
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f16_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..f16_size], std.mem.sliceAsBytes(f16_data));
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, buf, f16_size);
                } else {
                    const f32_size = model.weightF32Size(t);
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f32_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..f32_size], std.mem.sliceAsBytes(f32_data));
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, buf, f32_size);
                }
            }
        } else if (components) |comps| {
            const first_t = ctx.tensors.get(comps[0]).?;
            if (isNativeQuantType(first_t.type)) {
                var offset: u64 = 0;
                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, upload_size, .{});
                for (comps) |c| {
                    const t = ctx.tensors.get(c).?;
                    const raw_size = t.size();
                    const raw = if (ctx.mmap_file != null)
                        try ctx.getTensorSlice(t)
                    else blk: {
                        const temp = try allocator.alloc(u8, raw_size);
                        try ctx.readTensorData(t, temp);
                        break :blk temp;
                    };
                    defer if (ctx.mmap_file == null) allocator.free(raw);

                    @memcpy(@as([*]u8, @ptrCast(mapped))[offset .. offset + raw_size], raw);
                    offset += raw_size;
                }
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, buf, upload_size);
            } else {
                var offset: u64 = 0;
                const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, upload_size, .{});
                for (comps) |c| {
                    const t = ctx.tensors.get(c).?;
                    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    const f32_data = try allocator.alloc(f32, n);
                    defer allocator.free(f32_data);
                    weights.dequantToF32(ctx, t, f32_data) catch |err| {
                        vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                        return err;
                    };

                    const comp_size = n * 4;
                    @memcpy(@as([*]u8, @ptrCast(mapped))[offset .. offset + comp_size], std.mem.sliceAsBytes(f32_data));
                    offset += comp_size;
                }
                vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                try vk_ctx.copyBuffer(weight_staging, buf, upload_size);
            }
        }
    }

    // token_embd for LM head if tied — upload quant raw when applicable
    if (ctx.tensors.get("output.weight") == null) {
        if (graph.tensors.getPtr("token_embd.weight")) |gt_entry| {
            if (ctx.tensors.get("token_embd.weight")) |t| {
                const upload_size = if (isNativeQuantType(t.type))
                    t.size()
                else if (t.type == .q4_0 or t.type == .q4_k)
                    (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]) * 2
                else
                    model.weightF32Size(t);
                if (gt_entry.buffer == null) {
                    const buf = try backend.Buffer.init(
                        vk_ctx,
                        upload_size,
                        .{ .storage_buffer_bit = true, .shader_device_address_bit = true, .transfer_dst_bit = true },
                        .{ .device_local_bit = true }
                    );
                    const buf_ptr = try allocator.create(backend.Buffer);
                    buf_ptr.* = buf;
                    gt_entry.buffer = buf_ptr;
                    try weight_buffers.append(allocator, buf);
                }
                gt_entry.size = upload_size;
                if (isNativeQuantType(t.type)) {
                    const raw_size = t.size();
                    const raw = if (ctx.mmap_file != null)
                        try ctx.getTensorSlice(t)
                    else blk: {
                        const temp = try allocator.alloc(u8, raw_size);
                        try ctx.readTensorData(t, temp);
                        break :blk temp;
                    };
                    defer if (ctx.mmap_file == null) allocator.free(raw);

                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, raw_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..raw_size], raw);
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, @as(*backend.Buffer, @ptrCast(@alignCast(gt_entry.buffer.?))).*, raw_size);
                } else if (t.type == .q4_0 or t.type == .q4_k) {
                    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    const f32_data = try allocator.alloc(f32, n);
                    defer allocator.free(f32_data);
                    try weights.dequantToF32(ctx, t, f32_data);
                    const f16_size = n * 2;
                    const f16_data = try allocator.alloc(u16, n);
                    defer allocator.free(f16_data);
                    for (f32_data, 0..) |v, i| f16_data[i] = @bitCast(@as(f16, @floatCast(v)));
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f16_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..f16_size], std.mem.sliceAsBytes(f16_data));
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, @as(*backend.Buffer, @ptrCast(@alignCast(gt_entry.buffer.?))).*, f16_size);
                } else {
                    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    const f32_data = try allocator.alloc(f32, n);
                    defer allocator.free(f32_data);
                    try weights.dequantToF32(ctx, t, f32_data);
                    const f32_size = model.weightF32Size(t);
                    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, weight_staging.memory, 0, f32_size, .{});
                    @memcpy(@as([*]u8, @ptrCast(mapped))[0..f32_size], std.mem.sliceAsBytes(f32_data));
                    vk_ctx.vkd.unmapMemory(vk_ctx.device, weight_staging.memory);
                    try vk_ctx.copyBuffer(weight_staging, @as(*backend.Buffer, @ptrCast(@alignCast(gt_entry.buffer.?))).*, f32_size);
                }
            }
        }
    }

    return weight_buffers;
}

test "rewriteQuantizedMatmuls selects native and fallback quant paths" {
    const allocator = std.testing.allocator;

    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const w_native = try tensor.Tensor.init(allocator, "w_native", .q4_k, &.{ 2, 2 });
    defer w_native.deinit(allocator);
    try tensors.put("w_native", w_native);

    const w_fallback = try tensor.Tensor.init(allocator, "w_fallback", .q4_0, &.{ 2, 2 });
    defer w_fallback.deinit(allocator);
    try tensors.put("w_fallback", w_fallback);

    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph, undefined, &tensors);
    try builder.addTensor("x", 8, .input);
    try builder.addTensor("w_native", 8, .weight);
    try builder.addTensor("w_fallback", 8, .weight);
    try builder.addTensor("y_native", 8, .activation);
    try builder.addTensor("y_fallback", 8, .activation);
    try builder.addNode(.matmul, &.{ "x", "w_native" }, "y_native", 1, 1, 1, 2, 2, 0);
    try builder.addNode(.matmul, &.{ "x", "w_fallback" }, "y_fallback", 1, 1, 1, 2, 2, 0);

    rewriteQuantizedMatmuls(allocator, &graph, &tensors);

    try std.testing.expectEqual(compute_graph.OpType.matmul_q, graph.nodes.items[0].op_type);
    try std.testing.expectEqual(@intFromEnum(tensor.Type.q4_k), graph.nodes.items[0].p5);
    try std.testing.expectEqual(compute_graph.OpType.matmul_q, graph.nodes.items[1].op_type);
    try std.testing.expectEqual(compute_graph.q4_0_f16_fallback_qtype, graph.nodes.items[1].p5);
}

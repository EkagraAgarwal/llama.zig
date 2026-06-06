const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const compute_graph = @import("compute_graph.zig");
const weights = @import("weights.zig");
const tensor = @import("tensor.zig");
const vk = @import("vulkan");

pub const WeightUploader = struct {
    weight_memory: ?vulkan.Memory = null,
    weight_buffers: std.ArrayListUnmanaged(vulkan.Buffer) = .empty,
    embd_gpu_buf: ?vulkan.Buffer = null,

    pub fn deinit(self: *WeightUploader, allocator: std.mem.Allocator, vk_ctx: *vulkan.Context) void {
        for (self.weight_buffers.items) |b| b.deinit(vk_ctx);
        self.weight_buffers.deinit(allocator);
        if (self.weight_memory) |*mem| mem.deinit(vk_ctx);
    }

    pub fn upload(self: *WeightUploader, allocator: std.mem.Allocator, vk_ctx: *vulkan.Context, graph: *compute_graph.Graph, ctx: *gguf.GGUFContext, gpu_embed: bool, staging_buf_size: u64) !void {
        var staging_alloc_size: u64 = 0;

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

        // Pass 1: Identify all weights and create buffer handles to collect requirements
        var t_it = graph.tensors.iterator();
        while (t_it.next()) |entry| {
            if (entry.value_ptr.role != .weight) continue;
            const name = entry.key_ptr.*;
            const gt = ctx.tensors.get(name);
            const components = try getFusedComponentNames(allocator, name);
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

        var pool_size: u64 = 0;
        var weight_type_bits: u32 = 0xFFFFFFFF;
        for (weight_entries.items) |*we| {
            weight_type_bits &= we.reqs.memory_type_bits;
            we.offset = (pool_size + we.reqs.alignment - 1) & ~(we.reqs.alignment - 1);
            pool_size = we.offset + we.reqs.size;
        }

        const weight_mem_type = try vk_ctx.findMemoryType(weight_type_bits, .{ .device_local_bit = true });
        self.weight_memory = try vulkan.Memory.allocate(vk_ctx, pool_size, weight_mem_type, true);

        for (weight_entries.items) |we| {
            const res = vk_ctx.vkd.dispatch.vkBindBufferMemory.?(vk_ctx.device, we.buffer, self.weight_memory.?.memory, we.offset);
            if (res != .success) return error.MemoryBindingFailed;

            var address: u64 = 0;
            address = vk_ctx.vkd.dispatch.vkGetBufferDeviceAddress.?(vk_ctx.device, &.{ .buffer = we.buffer });

            const buf = vulkan.Buffer{ .buffer = we.buffer, .memory = self.weight_memory.?.memory, .size = we.upload_size, .address = address, .is_pooled = true };
            try self.weight_buffers.append(allocator, buf);

            if (graph.tensors.getPtr(we.name)) |gt_entry| {
                gt_entry.buffer = try allocator.create(vulkan.Buffer);
                gt_entry.buffer.?.* = buf;
                gt_entry.size = we.upload_size;
            } else if (std.mem.eql(u8, we.name, "token_embd.weight")) {
                self.embd_gpu_buf = buf;
            }
        }

        staging_alloc_size = @min(staging_alloc_size, staging_buf_size);
        var weight_staging = try vulkan.Buffer.init(vk_ctx, staging_alloc_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        defer weight_staging.deinit(vk_ctx);

        for (weight_entries.items, 0..) |we, entry_idx| {
            const buf = self.weight_buffers.items[entry_idx];
            const name = we.name;
            const gt = ctx.tensors.get(name);
            const components = try getFusedComponentNames(allocator, name);
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
                    try uploadBufferChunked(vk_ctx, weight_staging, buf, raw, staging_buf_size);
                } else {
                    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
                    const f32_data = try allocator.alloc(f32, n);
                    defer allocator.free(f32_data);
                    try weights.dequantToF32(ctx, t, f32_data);
                    try uploadBufferChunked(vk_ctx, weight_staging, buf, std.mem.sliceAsBytes(f32_data), staging_buf_size);
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
                    try uploadBufferChunked(vk_ctx, weight_staging, buf, host_buf, staging_buf_size);
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
                        try weights.dequantToF32(ctx, t, host_buf[off .. off + tn]);
                        off += tn;
                    }
                    try uploadBufferChunked(vk_ctx, weight_staging, buf, std.mem.sliceAsBytes(host_buf), staging_buf_size);
                }
            } else if (graph.synthetic_weights.contains(name)) {
                const host_buf = try allocator.alloc(u8, we.upload_size);
                defer allocator.free(host_buf);
                @memset(host_buf, 0);
                try uploadBufferChunked(vk_ctx, weight_staging, buf, host_buf, staging_buf_size);
            } else {
                std.log.warn("WARNING: Weight tensor {s} not found in GGUF and not synthetic! Uploading uninitialized memory.", .{name});
            }
        }
    }
};

pub fn isNativeQuantType(tt: tensor.Type) bool {
    return switch (tt) { .q4_0, .q8_0, .q4_1, .q4_k, .q5_k, .q6_k, .f16 => true, else => false };
}

pub fn gpuUploadSize(t: *tensor.Tensor) u64 {
    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    return switch (t.type) { 
        .f32, .bf16 => n * 4, 
        .q4_0, .q8_0, .q4_1, .q4_k, .q5_k, .q6_k, .f16 => t.size() 
    };
}

pub fn getFusedComponentNames(allocator: std.mem.Allocator, name: []const u8) !?[]const []const u8 {
    if (std.mem.endsWith(u8, name, ".attn_qkv.weight")) {
        const prefix = name[0 .. name.len - ".attn_qkv.weight".len];
        var list: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (list.items) |item| allocator.free(item);
            list.deinit(allocator);
        }
        try list.append(allocator, try std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix}));
        try list.append(allocator, try std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix}));
        try list.append(allocator, try std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix}));
        return try list.toOwnedSlice(allocator);
    } else if (std.mem.endsWith(u8, name, ".ffn_gate_up.weight")) {
        const prefix = name[0 .. name.len - ".ffn_gate_up.weight".len];
        var list: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (list.items) |item| allocator.free(item);
            list.deinit(allocator);
        }
        try list.append(allocator, try std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix}));
        try list.append(allocator, try std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix}));
        return try list.toOwnedSlice(allocator);
    }
    return null;
}

pub fn uploadBufferChunked(vk_ctx: *vulkan.Context, staging: vulkan.Buffer, dst: vulkan.Buffer, data: []const u8, chunk_size_cap: u64) !void {
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

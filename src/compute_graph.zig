const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const vk = @import("vulkan");

pub const OpType = enum(u32) {
    add = 0,
    mul = 1,
    matmul = 2,
    rms_norm = 3,
    softmax = 4,
    rope = 5,
    silu_mul = 6,
};

pub const TensorRole = enum(u2) {
    weight = 0,
    activation = 1,
    input = 2,
    output = 3,
};

pub const GraphTensor = struct {
    name: []const u8,
    size: u64,
    offset: u64 = 0,
    role: TensorRole,
    buffer: ?*vulkan.Buffer = null,
};

pub const GraphNode = struct {
    op_type: OpType,
    input_names: [][]const u8,
    output_name: []const u8,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    p1: u32 = 0,
    p2: u32 = 0,
    p3: u32 = 0,
    p4: u32 = 0,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(GraphNode),
    tensors: std.StringHashMap(GraphTensor),
    scratchpad_size: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Graph {
        return Graph{
            .allocator = allocator,
            .nodes = .empty,
            .tensors = std.StringHashMap(GraphTensor).init(allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        for (self.nodes.items) |node| {
            for (node.input_names) |name| self.allocator.free(name);
            self.allocator.free(node.input_names);
            self.allocator.free(node.output_name);
        }
        self.nodes.deinit(self.allocator);
        var it = self.tensors.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.tensors.deinit();
    }
};

pub const GraphBuilder = struct {
    graph: *Graph,

    pub fn init(graph: *Graph) GraphBuilder {
        return GraphBuilder{ .graph = graph };
    }

    pub fn addTensor(self: *GraphBuilder, name: []const u8, size: u64, role: TensorRole) !void {
        if (self.graph.tensors.contains(name)) return;
        const owned = try self.graph.allocator.dupe(u8, name);
        try self.graph.tensors.put(owned, .{ .name = owned, .size = size, .role = role });
    }

    pub fn addNode(self: *GraphBuilder, op: OpType, inputs: []const []const u8, output: []const u8, dx: u32, dy: u32, p1: u32, p2: u32, p3: u32) !void {
        const owned_in = try self.graph.allocator.alloc([]const u8, inputs.len);
        for (inputs, 0..) |in, i| owned_in[i] = try self.graph.allocator.dupe(u8, in);
        try self.graph.nodes.append(self.graph.allocator, .{ .op_type = op, .input_names = owned_in, .output_name = try self.graph.allocator.dupe(u8, output), .dispatch_x = dx, .dispatch_y = dy, .dispatch_z = 1, .p1 = p1, .p2 = p2, .p3 = p3 });
    }

    pub fn buildLlamaBlock(self: *GraphBuilder, layer: u32, n_embd: u32, n_heads: u32, head_dim: u32, pos: u32, in_name: []const u8, out_name: []const u8) !void {
        const ln = try std.fmt.allocPrint(self.graph.allocator, "blk.{}", .{layer});
        defer self.graph.allocator.free(ln);

        // --- ATTENTION ---
        // 1. RMSNorm
        const nw = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_norm.weight", .{ln});
        const normed_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.normed", .{ln});
        defer self.graph.allocator.free(nw);
        try self.addTensor(nw, n_embd * 4, .weight);
        try self.addTensor(normed_name, n_embd * 4, .activation);
        try self.addNode(.rms_norm, &[_][]const u8{in_name, nw}, normed_name, (n_embd + 63) / 64, 1, n_embd, n_embd, 0);

        // 2. Q Proj & RoPE
        const qw = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_q.weight", .{ln});
        const q_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.q", .{ln});
        defer self.graph.allocator.free(qw);
        try self.addTensor(qw, n_embd * n_embd * 4, .weight);
        try self.addTensor(q_name, n_embd * 4, .activation);
        try self.addNode(.matmul, &[_][]const u8{normed_name, qw}, q_name, (n_embd + 15) / 16, 1, n_embd, n_embd, n_embd);
        try self.addNode(.rope, &[_][]const u8{q_name}, q_name, (n_embd + 63) / 64, 1, n_heads, head_dim, pos);

        // 3. Out Proj
        const ow = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_output.weight", .{ln});
        const attn_out_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_out", .{ln});
        defer self.graph.allocator.free(ow);
        try self.addTensor(ow, n_embd * n_embd * 4, .weight);
        try self.addTensor(attn_out_name, n_embd * 4, .activation);
        try self.addNode(.matmul, &[_][]const u8{q_name, ow}, attn_out_name, (n_embd + 15) / 16, 1, n_embd, n_embd, n_embd);

        // 4. Residual Add
        const res1_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.res1", .{ln});
        try self.addTensor(res1_name, n_embd * 4, .activation);
        try self.addNode(.add, &[_][]const u8{in_name, attn_out_name}, res1_name, (n_embd + 63) / 64, 1, n_embd, 0, 0);

        // --- FFN ---
        // 5. FFN Norm
        const fnw = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_norm.weight", .{ln});
        const ffn_normed_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_normed", .{ln});
        defer self.graph.allocator.free(fnw);
        try self.addTensor(fnw, n_embd * 4, .weight);
        try self.addTensor(ffn_normed_name, n_embd * 4, .activation);
        try self.addNode(.rms_norm, &[_][]const u8{res1_name, fnw}, ffn_normed_name, (n_embd + 63) / 64, 1, n_embd, n_embd, 0);

        // 6. FFN Gate & Up MatMul
        const n_ff = (n_embd * 8) / 3; // SwiGLU hidden dim
        const gw = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_gate.weight", .{ln});
        const uw = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_up.weight", .{ln});
        const gate_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.gate", .{ln});
        const up_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.up", .{ln});
        defer self.graph.allocator.free(gw);
        defer self.graph.allocator.free(uw);
        try self.addTensor(gw, n_embd * n_ff * 4, .weight);
        try self.addTensor(uw, n_embd * n_ff * 4, .weight);
        try self.addTensor(gate_name, n_ff * 4, .activation);
        try self.addTensor(up_name, n_ff * 4, .activation);
        try self.addNode(.matmul, &[_][]const u8{ffn_normed_name, gw}, gate_name, (n_ff + 15) / 16, 1, 1, n_ff, n_embd);
        try self.addNode(.matmul, &[_][]const u8{ffn_normed_name, uw}, up_name, (n_ff + 15) / 16, 1, 1, n_ff, n_embd);

        // 7. SiLU * Up
        try self.addNode(.silu_mul, &[_][]const u8{gate_name, up_name}, gate_name, (n_ff + 63) / 64, 1, n_ff, 0, 0);

        // 8. FFN Down
        const dw = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_down.weight", .{ln});
        const ffn_out_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_out", .{ln});
        defer self.graph.allocator.free(dw);
        try self.addTensor(dw, n_ff * n_embd * 4, .weight);
        try self.addTensor(ffn_out_name, n_embd * 4, .activation);
        try self.addNode(.matmul, &[_][]const u8{gate_name, dw}, ffn_out_name, (n_embd + 15) / 16, 1, 1, n_embd, n_ff);

        // 9. Final Residual Add
        try self.addTensor(out_name, n_embd * 4, .activation);
        try self.addNode(.add, &[_][]const u8{res1_name, ffn_out_name}, out_name, (n_embd + 63) / 64, 1, n_embd, 0, 0);
    }

    pub fn finalize(self: *GraphBuilder) void {
        var it = self.graph.tensors.iterator();
        var off: u64 = 0;
        while (it.next()) |entry| {
            if (entry.value_ptr.role != .weight) {
                entry.value_ptr.offset = off;
                off += (entry.value_ptr.size + 255) & ~@as(u64, 255);
            }
        }
        self.graph.scratchpad_size = off;
    }
};

pub const Dispatcher = struct {
    graph: *Graph,
    ctx: *vulkan.Context,
    registry: *vulkan.PipelineRegistry,
    scratchpad: vulkan.Buffer,

    pub fn init(graph: *Graph, ctx: *vulkan.Context, registry: *vulkan.PipelineRegistry, scratch: vulkan.Buffer) Dispatcher {
        return .{ .graph = graph, .ctx = ctx, .registry = registry, .scratchpad = scratch };
    }

    pub fn execute(self: *Dispatcher) !void {
        var cmd: vk.CommandBuffer = undefined;
        _ = self.ctx.vkd.dispatch.vkAllocateCommandBuffers.?(self.ctx.device, &.{ .command_pool = self.ctx.cmd_pool, .level = .primary, .command_buffer_count = 1 }, (&cmd)[0..1]);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        for (self.graph.nodes.items) |node| {
            const pipe = self.registry.get(@tagName(node.op_type)) orelse continue;
            var pc = vulkan.PushConstants{ .p1 = node.p1, .p2 = node.p2, .p3 = node.p3, .p4 = node.p4, .a = 0, .b = 0, .c = 0 };
            if (node.input_names.len >= 1) {
                const t = self.graph.tensors.get(node.input_names[0]).?;
                pc.a = if (t.role == .weight) t.buffer.?.address + t.offset else self.scratchpad.address + t.offset;
            }
            if (node.input_names.len >= 2) {
                const t = self.graph.tensors.get(node.input_names[1]).?;
                pc.b = if (t.role == .weight) t.buffer.?.address + t.offset else self.scratchpad.address + t.offset;
            }
            const ot = self.graph.tensors.get(node.output_name).?;
            pc.c = self.scratchpad.address + ot.offset;

            self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
            self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
            self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, node.dispatch_x, node.dispatch_y, node.dispatch_z);
            
            const barrier = vk.MemoryBarrier{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true } };
            self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(cmd, .{ .compute_shader_bit = true }, .{ .compute_shader_bit = true }, .{}, 1, (&barrier)[0..1], 0, null, 0, null);
        }

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(cmd);
        const submit_info = vk.SubmitInfo{ .wait_semaphore_count = 0, .p_wait_semaphores = null, .p_wait_dst_stage_mask = null, .command_buffer_count = 1, .p_command_buffers = (&cmd)[0..1], .signal_semaphore_count = 0, .p_signal_semaphores = null };
        _ = self.ctx.vkd.dispatch.vkQueueSubmit.?(self.ctx.compute_queue, 1, (&submit_info)[0..1], .null_handle);
        _ = self.ctx.vkd.dispatch.vkQueueWaitIdle.?(self.ctx.compute_queue);
        self.ctx.vkd.dispatch.vkFreeCommandBuffers.?(self.ctx.device, self.ctx.cmd_pool, 1, (&cmd)[0..1]);
    }
};

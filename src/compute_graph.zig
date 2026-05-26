const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const vk = @import("vulkan");

pub const OpType = enum(u32) {
    add = 0,
    mul = 1,
    matmul = 2,
    rms_norm = 3,
    softmax = 4,
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
    is_static: bool = false,
};

pub const GraphNode = struct {
    op_type: OpType,
    input_names: [][]const u8,
    output_name: []const u8,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    param: u32 = 0,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: []GraphNode,
    tensors: std.StringHashMap(GraphTensor),
    scratchpad_size: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Graph {
        return Graph{
            .allocator = allocator,
            .nodes = &[_]GraphNode{},
            .tensors = std.StringHashMap(GraphTensor).init(allocator),
            .scratchpad_size = 0,
        };
    }

    pub fn deinit(self: *Graph) void {
        for (self.nodes) |node| {
            for (node.input_names) |name| {
                self.allocator.free(name);
            }
            self.allocator.free(node.input_names);
            self.allocator.free(node.output_name);
        }
        self.allocator.free(self.nodes);
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.tensors.deinit();
    }
};

pub const GraphBuilder = struct {
    graph: *Graph,
    node_list: []GraphNode,
    node_count: usize = 0,

    pub fn init(graph: *Graph) GraphBuilder {
        return GraphBuilder{
            .graph = graph,
            .node_list = &[_]GraphNode{},
            .node_count = 0,
        };
    }

    pub fn addTensor(self: *GraphBuilder, name: []const u8, size: u64, role: TensorRole) !void {
        const owned_name = try self.graph.allocator.dupe(u8, name);
        try self.graph.tensors.put(owned_name, .{
            .name = owned_name,
            .size = size,
            .role = role,
            .is_static = role == .weight,
        });
    }

    pub fn addNode(self: *GraphBuilder, op_type: OpType, input_names: []const []const u8, output_name: []const u8, dispatch_x: u32, param: u32) !void {
        const owned_inputs = try self.graph.allocator.alloc([]const u8, input_names.len);
        for (input_names, 0..) |name, i| {
            owned_inputs[i] = try self.graph.allocator.dupe(u8, name);
        }
        const owned_output = try self.graph.allocator.dupe(u8, output_name);

        const node = GraphNode{
            .op_type = op_type,
            .input_names = owned_inputs,
            .output_name = owned_output,
            .dispatch_x = dispatch_x,
            .dispatch_y = 1,
            .dispatch_z = 1,
            .param = param,
        };

        self.node_list = try self.graph.allocator.realloc(self.node_list, self.node_count + 1);
        self.node_list[self.node_count] = node;
        self.node_count += 1;
    }

    pub fn calcScratchpadSize(self: *GraphBuilder) u64 {
        var total: u64 = 0;
        var it = self.graph.tensors.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.role != .weight) {
                total += entry.value_ptr.size;
            }
        }
        self.graph.scratchpad_size = total;
        return total;
    }

    pub fn allocateOffsets(self: *GraphBuilder) void {
        var it = self.graph.tensors.iterator();
        var offset: u64 = 0;
        while (it.next()) |entry| {
            var tensor = entry.value_ptr.*;
            if (tensor.role == .weight) {
                tensor.offset = 0;
                tensor.is_static = true;
            } else {
                tensor.offset = offset;
                offset += tensor.size;
            }
            entry.value_ptr.* = tensor;
        }
    }

    pub fn build(self: *GraphBuilder) void {
        self.graph.nodes = self.node_list;
    }
};

pub const PushConstants = vulkan.PushConstants;

pub const Dispatcher = struct {
    graph: *Graph,
    ctx: *vulkan.Context,
    pipeline_registry: *vulkan.PipelineRegistry,
    scratchpad: vulkan.Buffer,

    pub fn init(graph: *Graph, ctx: *vulkan.Context, pipeline_registry: *vulkan.PipelineRegistry, scratchpad: vulkan.Buffer) !Dispatcher {
        return Dispatcher{
            .graph = graph,
            .ctx = ctx,
            .pipeline_registry = pipeline_registry,
            .scratchpad = scratchpad,
        };
    }

    pub fn deinit(self: *Dispatcher) void {
        _ = self;
    }

    pub fn execute(self: *Dispatcher) !void {
        var cmd_buf: vk.CommandBuffer = undefined;
        try self.ctx.vkd.allocateCommandBuffers(self.ctx.device, &.{
            .command_pool = self.ctx.cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, (&cmd_buf)[0..1]);
        defer self.ctx.vkd.freeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, (&cmd_buf)[0..1]);

        try self.ctx.vkd.beginCommandBuffer(cmd_buf, &.{ .flags = .{ .one_time_submit_bit = true } });

        for (self.graph.nodes, 0..) |node, node_idx| {
            try self.recordNode(&cmd_buf, &node, node_idx);
        }

        try self.ctx.vkd.endCommandBuffer(cmd_buf);

        const submit_info = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = (&cmd_buf)[0..1],
        };
        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, (&submit_info)[0..1], .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.compute_queue);
        try self.ctx.vkd.deviceWaitIdle(self.ctx.device);
        }
    fn recordNode(self: *Dispatcher, cmd_buf: *vk.CommandBuffer, node: *const GraphNode, node_idx: usize) !void {
        const pipeline_ref = switch (node.op_type) {
            .add => &self.pipeline_registry.add_pipeline,
            .mul => &self.pipeline_registry.mul_pipeline,
            .rms_norm => &self.pipeline_registry.rmsnorm_pipeline,
            .softmax => &self.pipeline_registry.softmax_pipeline,
            else => return error.UnsupportedOpType,
        };

        var pc = PushConstants{ .n = 0, .d = node.param, .a = 0, .b = 0, .c = 0 };

        if (node.input_names.len >= 1) {
            const t = self.graph.tensors.get(node.input_names[0]) orelse return error.TensorNotFound;
            if (t.role == .weight) {
                pc.a = t.buffer.?.address + t.offset;
            } else {
                pc.a = self.scratchpad.address + t.offset;
            }
            pc.n = @as(u32, @intCast(t.size / 4));
        }
        if (node.input_names.len >= 2) {
            const t = self.graph.tensors.get(node.input_names[1]) orelse return error.TensorNotFound;
            if (t.role == .weight) {
                pc.b = t.buffer.?.address + t.offset;
            } else {
                pc.b = self.scratchpad.address + t.offset;
            }
        }

        const ot = self.graph.tensors.get(node.output_name) orelse return error.TensorNotFound;
        pc.c = self.scratchpad.address + ot.offset;

        std.debug.print("Dispatching {s}: a=0x{x}, b=0x{x}, c=0x{x}, n={}\n", .{ @tagName(node.op_type), pc.a, pc.b, pc.c, pc.n });

        self.ctx.vkd.cmdBindPipeline(cmd_buf.*, .compute, pipeline_ref.pipeline);
        self.ctx.vkd.cmdPushConstants(cmd_buf.*, pipeline_ref.layout, .{ .compute_bit = true }, 0, @sizeOf(PushConstants), &pc);
        self.ctx.vkd.cmdDispatch(cmd_buf.*, node.dispatch_x, node.dispatch_y, node.dispatch_z);

        if (node_idx < self.graph.nodes.len - 1) {
            const mem_barrier = vk.MemoryBarrier{
                .src_access_mask = .{ .shader_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
            };
            self.ctx.vkd.cmdPipelineBarrier(cmd_buf.*, .{ .compute_shader_bit = true }, .{ .compute_shader_bit = true }, .{}, (&mem_barrier)[0..1], &[_]vk.BufferMemoryBarrier{}, &[_]vk.ImageMemoryBarrier{});
        }
    }
};

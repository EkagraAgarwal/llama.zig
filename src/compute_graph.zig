const std = @import("std");
const vulkan = @import("vulkan_backend.zig");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const vk = @import("vulkan");

pub const OpType = enum(u32) {
    add = 0,
    mul = 1,
    matmul = 2,
    rms_norm = 3,
    softmax = 4,
    rope = 5,
    silu_mul = 6,
    attention = 7,
    kv_write = 8,
    scaled_add = 9,
    matmul_q = 10,
    get_rows_q = 11,
    topk = 12,
    attention_flash = 13,
};

pub const TensorRole = enum(u8) {
    weight = 0,
    activation = 1,
    input = 2,
    output = 3,
    kv_cache = 4,
};

pub const GraphTensor = struct {
    name: []const u8,
    size: u64,
    offset: u64 = 0,
    role: TensorRole,
    buffer: ?*vulkan.Buffer = null,
    layer: u32 = 0,
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
    p5: u32 = 0,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(GraphNode),
    tensors: std.StringHashMap(GraphTensor),
    scratchpad_size: u64 = 0,
    kv_cache_size: u64 = 0,

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
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            if (entry.value_ptr.buffer) |buf| {
                self.allocator.destroy(buf);
            }
        }
        self.tensors.deinit();
    }
};

pub const GraphBuilder = struct {
    graph: *Graph,
    cfg: *const model.ModelConfig,

    pub fn init(graph: *Graph, cfg: *const model.ModelConfig) GraphBuilder {
        return GraphBuilder{ .graph = graph, .cfg = cfg };
    }

    fn f32Size(n: u32) u64 {
        return @as(u64, n) * 4;
    }

    pub fn addTensor(self: *GraphBuilder, name: []const u8, size: u64, role: TensorRole) !void {
        if (self.graph.tensors.contains(name)) return;
        const owned = try self.graph.allocator.dupe(u8, name);
        try self.graph.tensors.put(owned, .{ .name = owned, .size = size, .role = role });
    }

    pub fn addNode(self: *GraphBuilder, op: OpType, inputs: []const []const u8, output: []const u8, dx: u32, dy: u32, p1: u32, p2: u32, p3: u32, p4: u32) !void {
        try self.addNodeP(op, inputs, output, dx, dy, p1, p2, p3, p4, 0);
    }

    pub fn addNodeP(self: *GraphBuilder, op: OpType, inputs: []const []const u8, output: []const u8, dx: u32, dy: u32, p1: u32, p2: u32, p3: u32, p4: u32, p5: u32) !void {
        const owned_in = try self.graph.allocator.alloc([]const u8, inputs.len);
        for (inputs, 0..) |in, i| owned_in[i] = try self.graph.allocator.dupe(u8, in);
        try self.graph.nodes.append(self.graph.allocator, .{
            .op_type = op,
            .input_names = owned_in,
            .output_name = try self.graph.allocator.dupe(u8, output),
            .dispatch_x = dx,
            .dispatch_y = dy,
            .dispatch_z = 1,
            .p1 = p1,
            .p2 = p2,
            .p3 = p3,
            .p4 = p4,
            .p5 = p5,
        });
    }

    pub fn initKvCaches(self: *GraphBuilder) !void {
        const kv_per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 4 * 2;
        self.graph.kv_cache_size = kv_per_layer * self.cfg.n_layer;
        var l: u32 = 0;
        while (l < self.cfg.n_layer) : (l += 1) {
            var name_buf: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "kv.{d}", .{l});
            try self.addTensor(name, kv_per_layer, .kv_cache);
            if (self.graph.tensors.getPtr(name)) |t| t.layer = l;
        }
    }

    pub fn buildLlamaBlock(self: *GraphBuilder, layer: u32, pos: u32, in_name: []const u8, out_name: []const u8) !void {
        const cfg = self.cfg;
        const n_embd = cfg.n_embd;
        const n_heads = cfg.n_heads;
        const n_kv = cfg.n_kv_heads;
        const head_dim = cfg.head_dim;
        const n_ff = cfg.n_ff;
        const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);
        const rope_bits: u32 = @bitCast(cfg.rope_theta);

        var ln_buf: [32]u8 = undefined;
        const ln = try std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer});

        var nw_buf: [64]u8 = undefined;
        const nw = try std.fmt.bufPrint(&nw_buf, "{s}.attn_norm.weight", .{ln});
        var normed_buf: [64]u8 = undefined;
        const normed = try std.fmt.bufPrint(&normed_buf, "{s}.normed", .{ln});
        try self.addTensor(nw, f32Size(n_embd), .weight);
        try self.addTensor(normed, f32Size(n_embd), .activation);
        try self.addNode(.rms_norm, &.{ in_name, nw }, normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

        var qw_buf: [64]u8 = undefined;
        const qw = try std.fmt.bufPrint(&qw_buf, "{s}.attn_q.weight", .{ln});
        var kw_buf: [64]u8 = undefined;
        const kw = try std.fmt.bufPrint(&kw_buf, "{s}.attn_k.weight", .{ln});
        var vw_buf: [64]u8 = undefined;
        const vw = try std.fmt.bufPrint(&vw_buf, "{s}.attn_v.weight", .{ln});
        var qn_buf: [64]u8 = undefined;
        const qn = try std.fmt.bufPrint(&qn_buf, "{s}.q", .{ln});
        var kn_buf: [64]u8 = undefined;
        const kn = try std.fmt.bufPrint(&kn_buf, "{s}.k", .{ln});
        var vn_buf: [64]u8 = undefined;
        const vn = try std.fmt.bufPrint(&vn_buf, "{s}.v", .{ln});
        var attn_buf: [64]u8 = undefined;
        const attn = try std.fmt.bufPrint(&attn_buf, "{s}.attn", .{ln});

        const qkv_w = n_embd * n_embd;
        try self.addTensor(qw, f32Size(@intCast(qkv_w)), .weight);
        try self.addTensor(kw, f32Size(@intCast(n_embd * n_kv * head_dim)), .weight);
        try self.addTensor(vw, f32Size(@intCast(n_embd * n_kv * head_dim)), .weight);
        try self.addTensor(qn, f32Size(n_embd), .activation);
        try self.addTensor(kn, f32Size(n_kv * head_dim), .activation);
        try self.addTensor(vn, f32Size(n_kv * head_dim), .activation);
        try self.addTensor(attn, f32Size(n_embd), .activation);

        const mat_dx = (n_embd + 15) / 16;
        try self.addNode(.matmul, &.{ normed, qw }, qn, mat_dx, 1, 1, n_embd, n_embd, 0);
        try self.addNode(.matmul, &.{ normed, kw }, kn, ((n_kv * head_dim) + 15) / 16, 1, 1, n_kv * head_dim, n_embd, 0);
        try self.addNode(.matmul, &.{ normed, vw }, vn, ((n_kv * head_dim) + 15) / 16, 1, 1, n_kv * head_dim, n_embd, 0);

        try self.addNode(.rope, &.{qn}, qn, (n_embd + 63) / 64, 1, n_heads, head_dim, pos, rope_bits);
        try self.addNode(.rope, &.{kn}, kn, (n_kv * head_dim + 63) / 64, 1, n_kv, head_dim, pos, rope_bits);

        var kv_name_buf: [16]u8 = undefined;
        const kv_name = try std.fmt.bufPrint(&kv_name_buf, "kv.{d}", .{layer});
        try self.addNode(.kv_write, &.{ kn, vn, kv_name }, kn, (n_kv * head_dim + 63) / 64, 1, n_kv, head_dim, cfg.max_ctx, pos);
        const attn_p2 = head_dim | (n_kv << 16);
        const attn_scale_bits: u32 = @bitCast(cfg.attention_scale);
        try self.addNodeP(.attention, &.{ qn, kv_name }, attn, n_heads, 1, n_heads, attn_p2, cfg.max_ctx, pos, attn_scale_bits);

        var ow_buf: [64]u8 = undefined;
        const ow = try std.fmt.bufPrint(&ow_buf, "{s}.attn_output.weight", .{ln});
        var attn_out_buf: [64]u8 = undefined;
        const attn_out = try std.fmt.bufPrint(&attn_out_buf, "{s}.attn_out", .{ln});
        try self.addTensor(ow, f32Size(@intCast(qkv_w)), .weight);
        try self.addTensor(attn_out, f32Size(n_embd), .activation);
        try self.addNode(.matmul, &.{ attn, ow }, attn_out, mat_dx, 1, 1, n_embd, n_embd, 0);

        var res1_buf: [64]u8 = undefined;
        const res1 = try std.fmt.bufPrint(&res1_buf, "{s}.res1", .{ln});
        try self.addTensor(res1, f32Size(n_embd), .activation);
        const res_scale_bits: u32 = @bitCast(cfg.residual_scale);
        if (cfg.residual_scale != 1.0) {
            try self.addNode(.scaled_add, &.{ in_name, attn_out }, res1, (n_embd + 63) / 64, 1, n_embd, res_scale_bits, 0, 0);
        } else {
            try self.addNode(.add, &.{ in_name, attn_out }, res1, (n_embd + 63) / 64, 1, n_embd, 0, 0, 0);
        }

        var fnw_buf: [64]u8 = undefined;
        const fnw = try std.fmt.bufPrint(&fnw_buf, "{s}.ffn_norm.weight", .{ln});
        var ffn_normed_buf: [64]u8 = undefined;
        const ffn_normed = try std.fmt.bufPrint(&ffn_normed_buf, "{s}.ffn_normed", .{ln});
        try self.addTensor(fnw, f32Size(n_embd), .weight);
        try self.addTensor(ffn_normed, f32Size(n_embd), .activation);
        try self.addNode(.rms_norm, &.{ res1, fnw }, ffn_normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

        var gw_buf: [64]u8 = undefined;
        const gw = try std.fmt.bufPrint(&gw_buf, "{s}.ffn_gate.weight", .{ln});
        var uw_buf: [64]u8 = undefined;
        const uw = try std.fmt.bufPrint(&uw_buf, "{s}.ffn_up.weight", .{ln});
        var gate_buf: [64]u8 = undefined;
        const gate = try std.fmt.bufPrint(&gate_buf, "{s}.gate", .{ln});
        var up_buf: [64]u8 = undefined;
        const up = try std.fmt.bufPrint(&up_buf, "{s}.up", .{ln});
        try self.addTensor(gw, f32Size(n_embd * n_ff), .weight);
        try self.addTensor(uw, f32Size(n_embd * n_ff), .weight);
        try self.addTensor(gate, f32Size(n_ff), .activation);
        try self.addTensor(up, f32Size(n_ff), .activation);
        const ff_dx = (n_ff + 15) / 16;
        try self.addNode(.matmul, &.{ ffn_normed, gw }, gate, ff_dx, 1, 1, n_ff, n_embd, 0);
        try self.addNode(.matmul, &.{ ffn_normed, uw }, up, ff_dx, 1, 1, n_ff, n_embd, 0);
        try self.addNode(.silu_mul, &.{ gate, up }, gate, (n_ff + 63) / 64, 1, n_ff, 0, 0, 0);

        var dw_buf: [64]u8 = undefined;
        const dw = try std.fmt.bufPrint(&dw_buf, "{s}.ffn_down.weight", .{ln});
        var ffn_out_buf: [64]u8 = undefined;
        const ffn_out = try std.fmt.bufPrint(&ffn_out_buf, "{s}.ffn_out", .{ln});
        try self.addTensor(dw, f32Size(n_ff * n_embd), .weight);
        try self.addTensor(ffn_out, f32Size(n_embd), .activation);
        try self.addNode(.matmul, &.{ gate, dw }, ffn_out, (n_embd + 15) / 16, 1, 1, n_embd, n_ff, 0);

        try self.addTensor(out_name, f32Size(n_embd), .activation);
        if (cfg.residual_scale != 1.0) {
            try self.addNode(.scaled_add, &.{ res1, ffn_out }, out_name, (n_embd + 63) / 64, 1, n_embd, res_scale_bits, 0, 0);
        } else {
            try self.addNode(.add, &.{ res1, ffn_out }, out_name, (n_embd + 63) / 64, 1, n_embd, 0, 0, 0);
        }
    }

    pub fn buildLmHead(self: *GraphBuilder, in_name: []const u8, logits_name: []const u8, has_output_weight: bool) !void {
        const cfg = self.cfg;
        const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);

        const norm_w = "output_norm.weight";
        const normed = "final.normed";
        try self.addTensor(normed, f32Size(cfg.n_embd), .activation);
        try self.addNode(.rms_norm, &.{ in_name, norm_w }, normed, (cfg.n_embd + 63) / 64, 1, cfg.n_embd, cfg.n_embd, eps_bits, 0);

        const out_w = if (has_output_weight) "output.weight" else "token_embd.weight";
        if (!has_output_weight) {
            try self.addTensor("token_embd.weight", f32Size(cfg.n_embd) * cfg.vocab_size, .weight);
        }
        try self.addTensor(logits_name, f32Size(cfg.vocab_size), .output);
        const mat_dx = (cfg.vocab_size + 15) / 16;
        try self.addNode(.matmul, &.{ normed, out_w }, logits_name, mat_dx, 1, 1, cfg.vocab_size, cfg.n_embd, 0);
    }

    pub fn finalize(self: *GraphBuilder) void {
        var it = self.graph.tensors.iterator();
        var off: u64 = 0;
        while (it.next()) |entry| {
            if (entry.value_ptr.role == .activation or entry.value_ptr.role == .input or entry.value_ptr.role == .output) {
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
    kv_cache: vulkan.Buffer,
    cfg: *const model.ModelConfig,
    cmd: vk.CommandBuffer = .null_handle,
    fence: vk.Fence = .null_handle,
    flash_attn_threshold: u32 = 64,

    pub fn init(graph: *Graph, ctx: *vulkan.Context, registry: *vulkan.PipelineRegistry, scratch: vulkan.Buffer, kv: vulkan.Buffer, cfg: *const model.ModelConfig) !Dispatcher {
        var self = Dispatcher{
            .graph = graph,
            .ctx = ctx,
            .registry = registry,
            .scratchpad = scratch,
            .kv_cache = kv,
            .cfg = cfg,
        };
        try self.ensureSubmitResources();
        return self;
    }

    pub fn deinit(self: *Dispatcher) void {
        if (self.fence != .null_handle) {
            self.ctx.vkd.dispatch.vkDestroyFence.?(self.ctx.device, self.fence, null);
            self.fence = .null_handle;
        }
        if (self.cmd != .null_handle) {
            self.ctx.vkd.dispatch.vkFreeCommandBuffers.?(self.ctx.device, self.ctx.cmd_pool, 1, (&self.cmd)[0..1]);
            self.cmd = .null_handle;
        }
    }

    fn ensureSubmitResources(self: *Dispatcher) !void {
        if (self.fence == .null_handle) {
            _ = self.ctx.vkd.dispatch.vkCreateFence.?(self.ctx.device, &.{ .flags = .{} }, null, &self.fence);
        }
        if (self.cmd == .null_handle) {
            _ = self.ctx.vkd.dispatch.vkAllocateCommandBuffers.?(
                self.ctx.device,
                &.{ .command_pool = self.ctx.cmd_pool, .level = .primary, .command_buffer_count = 1 },
                (&self.cmd)[0..1],
            );
        }
    }

    fn tensorAddr(self: *Dispatcher, name: []const u8) u64 {
        const t = self.graph.tensors.get(name) orelse return 0;
        return switch (t.role) {
            .weight => t.buffer.?.address,
            .kv_cache => self.kvCacheLayerOffset(t.layer),
            .input, .activation, .output => self.scratchpad.address + t.offset,
        };
    }

    fn kvCacheLayerOffset(self: *Dispatcher, layer: u32) u64 {
        const per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 4 * 2;
        return self.kv_cache.address + per_layer * layer;
    }

    fn nodeLayerKey(node: GraphNode) u32 {
        if (std.mem.startsWith(u8, node.output_name, "blk.")) {
            var i: usize = 4;
            var layer: u32 = 0;
            while (i < node.output_name.len and node.output_name[i] >= '0' and node.output_name[i] <= '9') : (i += 1) {
                layer = layer * 10 + (node.output_name[i] - '0');
            }
            return layer + 1;
        }
        if (std.mem.eql(u8, node.output_name, "hidden")) return 0xFFFFFFFE;
        if (std.mem.eql(u8, node.output_name, "logits")) return 0xFFFFFFFF;
        return 0;
    }

    fn emitComputeBarrier(self: *Dispatcher, cmd: vk.CommandBuffer) void {
        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(
            cmd,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            1,
            (&barrier)[0..1],
            0,
            null,
            0,
            null,
        );
    }

    fn pipelineNameForNode(self: *Dispatcher, node: GraphNode) ?[]const u8 {
        return switch (node.op_type) {
            .matmul_q => blk: {
                const qtype = node.p5;
                if (node.p1 <= 1) {
                    if (qtype == @intFromEnum(tensor.Type.q6_k)) break :blk "matvec_q6_k";
                    break :blk "matvec_q4_k";
                }
                if (qtype == @intFromEnum(tensor.Type.q6_k)) break :blk "matmul_q6_k";
                break :blk "matmul_q4_k";
            },
            .attention => blk: {
                if (node.p4 + 1 >= self.flash_attn_threshold) break :blk "attention_flash";
                break :blk "attention";
            },
            else => @tagName(node.op_type),
        };
    }

    fn dispatchNode(self: *Dispatcher, cmd: vk.CommandBuffer, node: GraphNode, pos: u32) void {
        const pipe_name = self.pipelineNameForNode(node) orelse return;
        const pipe = self.registry.get(pipe_name) orelse return;

        var pc = vulkan.PushConstants{
            .p1 = node.p1,
            .p2 = node.p2,
            .p3 = node.p3,
            .p4 = node.p4,
            .p5 = node.p5,
            .a = 0,
            .b = 0,
            .c = 0,
        };

        if (node.op_type == .matmul_q and node.p1 <= 1) {
            pc.p1 = node.p2;
            pc.p2 = node.p3;
            pc.p3 = 0;
        }

        switch (node.op_type) {
            .kv_write => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.input_names[2]);
                pc.p4 = pos;
            },
            .attention => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
                pc.p4 = pos;
                if (pos + 1 >= self.flash_attn_threshold) pc.p6 = 64;
            },
            .rope => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]);
                pc.c = self.tensorAddr(node.output_name);
                pc.p3 = pos;
            },
            .get_rows_q => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
            .topk => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
            else => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]);
                if (node.input_names.len >= 2) pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
        }

        var dx = node.dispatch_x;
        var dy = node.dispatch_y;
        if (node.op_type == .matmul_q and node.p1 <= 1) {
            dx = (node.p2 + 255) / 256;
            dy = 1;
        }

        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, dx, dy, node.dispatch_z);
    }

    fn submitAndWait(self: *Dispatcher, cmd: vk.CommandBuffer) !void {
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetFences.?(self.ctx.device, 1, (&self.fence)[0..1]);
        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = null,
            .p_wait_dst_stage_mask = null,
            .command_buffer_count = 1,
            .p_command_buffers = (&cmd)[0..1],
            .signal_semaphore_count = 0,
            .p_signal_semaphores = null,
        };
        _ = self.ctx.vkd.dispatch.vkQueueSubmit.?(self.ctx.compute_queue, 1, (&submit_info)[0..1], self.fence);
        _ = self.ctx.vkd.dispatch.vkWaitForFences.?(self.ctx.device, 1, (&self.fence)[0..1], @enumFromInt(1), std.math.maxInt(u64));
    }

    fn recordGraph(self: *Dispatcher, cmd: vk.CommandBuffer, pos: u32) void {
        for (self.graph.nodes.items) |node| {
            self.dispatchNode(cmd, node, pos);
            self.emitComputeBarrier(cmd);
        }
    }

    pub fn execute(self: *Dispatcher, pos: u32) !void {
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{ .release_resources_bit = true });
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });
        self.recordGraph(self.cmd, pos);
        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executePrefillBatch(self: *Dispatcher, pos_start: u32, n_tokens: u32, input_batch: vulkan.Buffer, input_stride: u64) !void {
        const input_tensor = self.graph.tensors.get("input") orelse return error.MissingInputTensor;
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{ .release_resources_bit = true });
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        var i: u32 = 0;
        while (i < n_tokens) : (i += 1) {
            const copy_region = vk.BufferCopy{
                .src_offset = @as(u64, i) * input_stride,
                .dst_offset = input_tensor.offset,
                .size = input_stride,
            };
            self.ctx.vkd.dispatch.vkCmdCopyBuffer.?(self.cmd, input_batch.buffer, self.scratchpad.buffer, 1, (&copy_region)[0..1]);
            const copy_barrier = vk.MemoryBarrier{
                .src_access_mask = .{ .transfer_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
            };
            self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(
                self.cmd,
                .{ .transfer_bit = true },
                .{ .compute_shader_bit = true },
                .{},
                1,
                (&copy_barrier)[0..1],
                0,
                null,
                0,
                null,
            );
            self.recordGraph(self.cmd, pos_start + i);
        }

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executeGetRowsQ(
        self: *Dispatcher,
        indices_buf: vulkan.Buffer,
        weights_buf: vulkan.Buffer,
        out_offset: u64,
        token_id: u32,
        n_embd: u32,
        qtype: u32,
        row_bytes: u32,
        scale_bits: u32,
    ) !void {
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{ .release_resources_bit = true });
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, indices_buf.memory, 0, 4, .{});
        @as(*u32, @ptrCast(@alignCast(mapped))).* = token_id;
        self.ctx.vkd.unmapMemory(self.ctx.device, indices_buf.memory);

        const pipe = self.registry.get("get_rows_q") orelse return error.MissingPipeline;
        var pc = vulkan.PushConstants{
            .p1 = n_embd,
            .p2 = 1,
            .p3 = qtype,
            .p4 = scale_bits,
            .p5 = row_bytes,
            .a = indices_buf.address,
            .b = weights_buf.address,
            .c = self.scratchpad.address + out_offset,
        };
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(self.cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(self.cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(self.cmd, (n_embd + 255) / 256, 1, 1);

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executeTopK(
        self: *Dispatcher,
        logits_offset: u64,
        vocab_size: u32,
        out_indices_buf: vulkan.Buffer,
        out_values_buf: vulkan.Buffer,
        logit_scale_bits: u32,
    ) !u32 {
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{ .release_resources_bit = true });
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        const pipe = self.registry.get("topk") orelse return error.MissingPipeline;
        var pc = vulkan.PushConstants{
            .p1 = vocab_size,
            .p2 = 1,
            .p3 = logit_scale_bits,
            .a = self.scratchpad.address + logits_offset,
            .b = out_indices_buf.address,
            .c = out_values_buf.address,
        };
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(self.cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(self.cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(self.cmd, 1, 1, 1);

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, out_indices_buf.memory, 0, 4, .{});
        const id: u32 = @as(*u32, @ptrCast(@alignCast(mapped))).*;
        self.ctx.vkd.unmapMemory(self.ctx.device, out_indices_buf.memory);
        return id;
    }
};

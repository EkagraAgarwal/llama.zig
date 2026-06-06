const std = @import("std");
const vk = @import("vulkan");
const vulkan = @import("vulkan_backend.zig");
const compute_graph = @import("compute_graph.zig");
const Graph = compute_graph.Graph;
const GraphNode = compute_graph.GraphNode;
const tensor = @import("tensor.zig");
const weights = @import("weights.zig");
const model = @import("model.zig");

pub fn vkCall(res: vk.Result, call_name: []const u8) !void {
    if (res != .success) {
        std.log.err("Vulkan error in {s}: {}", .{ call_name, res });
        return error.VulkanError;
    }
}

pub const PushConstantPacker = struct {
    bytes: [128]u8 = undefined,
    offset: usize = 0,

    pub fn init() PushConstantPacker {
        return .{};
    }

    pub fn push(self: *PushConstantPacker, val: anytype) void {
        const T = @TypeOf(val);
        const size = @sizeOf(T);
        if (self.offset + size > 128) @panic("Push constants overflow");
        std.mem.copyForwards(u8, self.bytes[self.offset .. self.offset + size], std.mem.asBytes(&val));
        self.offset += size;
    }

    pub fn get(self: *const PushConstantPacker) []const u8 {
        return self.bytes[0..self.offset];
    }
};

pub const Dispatcher = struct {
    graph: *Graph,
    ctx: *vulkan.Context,
    registry: *vulkan.PipelineRegistry,
    scratchpad: vulkan.Buffer,
    kv_cache: vulkan.Buffer,
    allocator: std.mem.Allocator,
    /// Host-visible staging buffer used by the per-token CPU SSM step.
    /// Set by `set_ssm_staging_buffer` before prefill/decode.
    ssm_staging: ?vulkan.Buffer = null,
    ssm_staging_size: u64 = 0,
    /// SSM conv1d rolling-window buffer (host-visible; CPU reads/writes
    /// per decode step). May be a zero-sized placeholder when not in use.
    ssm_conv_buf: vulkan.Buffer,
    /// SSM per-head recurrent state matrix (host-visible). May be zero-sized.
    ssm_state_buf: vulkan.Buffer,
    cfg: *const model.ModelConfig,
    cmd: vk.CommandBuffer = .null_handle,
    fence: vk.Fence = .null_handle,
    flash_attn_threshold: u32 = 1,
    submit_count: u32 = 0,
    reported_graph: bool = false,
    trace_dispatch: bool = false,
    check_nans: bool = false,

    pub fn init(
        graph: *Graph,
        ctx: *vulkan.Context,
        registry: *vulkan.PipelineRegistry,
        scratch: vulkan.Buffer,
        kv: vulkan.Buffer,
        ssm_conv: vulkan.Buffer,
        ssm_state: vulkan.Buffer,
        cfg: *const model.ModelConfig,
        allocator: std.mem.Allocator,
    ) !Dispatcher {
        var self = Dispatcher{
            .graph = graph,
            .ctx = ctx,
            .registry = registry,
            .scratchpad = scratch,
            .kv_cache = kv,
            .ssm_conv_buf = ssm_conv,
            .ssm_state_buf = ssm_state,
            .cfg = cfg,
            .allocator = allocator,
        };
        try self.ensure_submit_resources();
        return self;
    }

    /// Set the host-visible staging buffer used by the per-token CPU SSM
    /// step. Call this once before prefill/decode.
    pub fn set_ssm_staging_buffer(self: *Dispatcher, staging: vulkan.Buffer, size: u64) void {
        self.ssm_staging = staging;
        self.ssm_staging_size = size;
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

    pub fn ensure_submit_resources(self: *Dispatcher) !void {
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

    fn tensor_addr(self: *Dispatcher, name: []const u8) u64 {
        // Slices resolve to parent + offset (no separate storage).
        if (self.graph.slices.get(name)) |slice| {
            const parent_addr = self.tensor_addr(slice.parent);
            return parent_addr + slice.offset;
        }
        const t = self.graph.tensors.get(name) orelse return 0;
        return switch (t.role) {
            .weight => t.buffer.?.address,
            .kv_cache => self.kv_cache_layer_offset(t.layer),
            .ssm_cache => self.ssm_cache_layer_offset(t.layer, t.size),
            .input, .activation, .output => self.scratchpad.address + t.offset,
        };
    }

    fn kv_cache_layer_offset(self: *Dispatcher, layer: u32) u64 {
        const per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 2 * 2;
        return self.kv_cache.address + per_layer * layer;
    }

    /// Returns the base BDA of a per-layer SSM cache tensor. SSM caches are
    /// split into two ranges (conv state + recurrent state) within the
    /// `ssm_conv_buf` and `ssm_state_buf` buffers. The tensor's `size` field
    /// tells us which one it is (conv = smaller, state = larger).
    fn ssm_cache_layer_offset(self: *Dispatcher, layer: u32, size_bytes: u64) u64 {
        const cfg = self.cfg;
        if (cfg.ssm_d_inner == 0 or cfg.ssm_d_state == 0 or cfg.ssm_n_group == 0) return 0;
        const d_conv = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
        const conv_channels = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const head_v_dim = cfg.ssm_d_inner / if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1;
        const head_k_dim = cfg.ssm_d_state;
        const rec_per_layer = head_v_dim * head_k_dim * cfg.ssm_dt_rank;
        const conv_per_layer = @as(u64, d_conv - 1) * conv_channels;
        const per_layer_bytes_conv = conv_per_layer * 4;
        if (size_bytes <= per_layer_bytes_conv + 8) {
            // conv cache: stored in ssm_conv_buf
            return self.ssm_conv_buf.address + per_layer_bytes_conv * layer;
        }
        // recurrent state cache: stored in ssm_state_buf
        return self.ssm_state_buf.address + (rec_per_layer * 4) * layer;
    }

    fn emit_compute_barrier(self: *Dispatcher, cmd: vk.CommandBuffer) void {
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

    pub fn pipeline_name_for_node(self: *Dispatcher, node: GraphNode) ?[]const u8 {
        return switch (node.op_type) {
            .matmul_q => blk: {
                break :blk quant_pipeline_name(node.p5, node.p1 <= 1);
            },
            .get_rows_q => blk: {
                const qtype = @as(u32, node.p5);
                const qt: tensor.Type = @enumFromInt(qtype);
                break :blk switch (qt) {
                    .q4_0 => "get_rows_q4_0",
                    .q4_1 => "get_rows_q4_1",
                    .q4_k => "get_rows_q4_k",
                    .q5_k => "get_rows_q5_k",
                    .q6_k => "get_rows_q6_k",
                    else => "get_rows_q",
                };
            },
            .attention => blk: {
                if (node.p4 + 1 >= self.flash_attn_threshold) break :blk "attention_flash";
                break :blk "attention";
            },
            .gelu_mul => "gelu_mul",
            // The joint Q+gate matmul is a regular matmul (the Q/gate split
            // happens via slice offsets). The Qwen 3.5 graph emits it as
            // .attn_qg_matmul to distinguish intent; the dispatcher treats
            // it identically to .matmul.
            .attn_qg_matmul => "matmul",
            // SSM conv1d and gated_norm have dedicated pipelines; the
            // .ssm_delta_net_decode shader is only used for prefill (N>1).
            // For decode (N=1) the CPU step in main.zig replaces the
            // dispatch; if the op is still in the graph with N=1 the
            // shader will run on a 1-token input and may produce NaN/garbage
            // (a real fix would be to add a "cpu_step" op-type; for now
            // we map to the same shader so the graph compiles).
            .ssm_conv1d => "ssm_conv1d",
            .ssm_gated_norm => "ssm_gated_norm",
            .ssm_delta_net_decode => "ssm_delta_net_decode",
            .softplus => "softplus",
            .sigmoid => "sigmoid",
            .silu => "silu",
            .l2_norm => "l2_norm",
            .rope_multi => "mrope",
            .attn_gate_mul => "attn_gate_mul",
            .matmul => "matmul",
            else => @tagName(node.op_type),
        };
    }

    pub fn quant_pipeline_name(qtype: u32, is_matvec: bool) []const u8 {
        if (qtype == compute_graph.q4_0_f16_fallback_qtype) {
            return "matmul_f16";
        }
        const qt: tensor.Type = @enumFromInt(qtype);
        return if (is_matvec)
            switch (qt) { .q4_0 => "matvec_q4_0", .q4_1 => "matvec_q4_1", .q4_k => "matvec_q4_k", .q5_k => "matvec_q5_k", .q6_k => "matvec_q6_k", .f16 => "matvec_f16", else => "matvec_q8_0" }
        else
            switch (qt) { .q4_0 => "matmul_q4_0", .q4_1 => "matmul_q4_1", .q4_k => "matmul_q4_k", .q5_k => "matmul_q5_k", .q6_k => "matmul_q6_k", .f16 => "matmul_f16", else => "matmul_q8_0" };
    }

    fn dispatch_node(self: *Dispatcher, cmd: vk.CommandBuffer, node: GraphNode, pos: u32) !void {
        const pipe_name = self.pipeline_name_for_node(node) orelse {
            std.log.err("Missing pipeline name for node: {s}", .{@tagName(node.op_type)});
            return error.MissingPipeline;
        };
        const pipe = self.registry.get(pipe_name) orelse {
            std.log.err("Pipeline '{s}' not found in registry", .{pipe_name});
            return error.MissingPipeline;
        };

        var pc = vulkan.PushConstants{
            .p1 = node.p1,
            .p2 = node.p2,
            .p3 = node.p3,
            .p4 = node.p4,
            .p5 = node.p5,
            .p6 = node.p6,
            .p7 = node.p7,
            .p8 = node.p8,
            .a = 0,
            .b = 0,
            .c = 0,
        };

        switch (node.op_type) {
            .kv_write => {
                pc.a = self.tensor_addr(node.input_names[0]) + node.p5;
                pc.b = self.tensor_addr(node.input_names[1]) + node.p6;
                pc.c = self.tensor_addr(node.input_names[2]);
                pc.p4 = pos;
            },
            .attention => {
                pc.a = self.tensor_addr(node.input_names[0]) + node.p6;
                pc.b = self.tensor_addr(node.input_names[1]);
                pc.c = self.tensor_addr(node.output_name);
                pc.p4 = pos;
                if (pos + 1 >= self.flash_attn_threshold) pc.p6 = 64;
            },
            .rope => {
                if (node.input_names.len >= 1) pc.a = self.tensor_addr(node.input_names[0]) + node.p5;
                pc.c = self.tensor_addr(node.output_name) + node.p5;
                pc.p3 = pos;
            },
            .rope_multi => {
                if (node.input_names.len >= 1) pc.a = self.tensor_addr(node.input_names[0]) + node.p5;
                pc.c = self.tensor_addr(node.output_name) + node.p5;
                pc.p3 = pos;
                // sec0/sec1/sec2/sec3 are already in p5..p8 from the builder; the
                // shader reads them as the divisor groups for MRoPE.
            },
            .get_rows_q => {
                pc.a = self.tensor_addr(node.input_names[0]);
                pc.b = self.tensor_addr(node.input_names[1]);
                pc.c = self.tensor_addr(node.output_name);
            },
            .copy => {
                pc.a = self.tensor_addr(node.input_names[0]);
                pc.c = self.tensor_addr(node.output_name);
            },
            .silu_mul, .gelu_mul => {
                pc.a = self.tensor_addr(node.input_names[0]) + node.p5;
                pc.b = self.tensor_addr(node.input_names[1]) + node.p6;
                pc.c = self.tensor_addr(node.output_name) + node.p7;
            },
            // SSM conv1d shader takes 4 buffers: state (in/out), new_chunk,
            // kernel, output.
            .ssm_conv1d => {
                pc.a = self.tensor_addr(node.input_names[0]); // state (in/out)
                pc.b = self.tensor_addr(node.input_names[1]); // new chunk
                pc.c = self.tensor_addr(node.input_names[2]); // kernel
                pc.d = self.tensor_addr(node.output_name);    // output
            },
            // SSM delta-net decode takes 6 inputs + 1 output: state, q, k, v, g,
            // beta → output. Mapped to push constant slots a..g.
            .ssm_delta_net_decode => {
                pc.a = self.tensor_addr(node.input_names[0]); // state
                pc.b = self.tensor_addr(node.input_names[1]); // q
                pc.c = self.tensor_addr(node.input_names[2]); // k
                pc.d = self.tensor_addr(node.input_names[3]); // v
                pc.e = self.tensor_addr(node.input_names[4]); // g
                pc.f = self.tensor_addr(node.input_names[5]); // beta
                pc.g = self.tensor_addr(node.output_name);    // output
            },
            // SSM gated norm: core (in), z (in), rms_norm_weight (in), out.
            .ssm_gated_norm => {
                pc.a = self.tensor_addr(node.input_names[0]);
                pc.b = self.tensor_addr(node.input_names[1]);
                pc.c = self.tensor_addr(node.input_names[2]);
                pc.d = self.tensor_addr(node.output_name);
            },
            .qwen_deinterleave => {
                pc.a = self.tensor_addr(node.input_names[0]); // qg
                pc.b = self.tensor_addr(node.output_name);    // q
                pc.c = self.tensor_addr(node.input_names[1]); // gate
            },
            else => {
                if (node.input_names.len >= 1) pc.a = self.tensor_addr(node.input_names[0]);
                if (node.input_names.len >= 2) pc.b = self.tensor_addr(node.input_names[1]);
                pc.c = self.tensor_addr(node.output_name);
            },
        }

        var dx = node.dispatch_x;
        var dy = node.dispatch_y;
        if (node.op_type == .matmul_q and node.p1 <= 1 and !std.mem.eql(u8, pipe_name, "matmul_f16")) {
            dx = (node.p2 + 7) / 8;
            dy = 1;
        }

        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
        if (self.trace_dispatch) std.debug.print("[dispatch] n={} i={} op={s} dx={} dy={} dz={}\n", .{ self.submit_count, pos, @tagName(node.op_type), dx, dy, node.dispatch_z });
        self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, dx, dy, node.dispatch_z);
    }

    pub fn submit_and_wait(self: *Dispatcher, cmd: vk.CommandBuffer) !void {
        try self.ensure_submit_resources();
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
        self.submit_count += 1;
    }

    fn tensors_alias(self: *const Dispatcher, name1: []const u8, name2: []const u8) bool {
        if (std.mem.eql(u8, name1, name2)) return true;

        const s1 = self.graph.slices.get(name1);
        const s2 = self.graph.slices.get(name2);

        const p1 = if (s1) |s| s.parent else name1;
        const p2 = if (s2) |s| s.parent else name2;

        if (std.mem.eql(u8, p1, p2)) {
            const off1 = if (s1) |s| s.offset else 0;
            const sz1 = if (s1) |s| s.size else (self.graph.tensors.get(name1) orelse return false).size;

            const off2 = if (s2) |s| s.offset else 0;
            const sz2 = if (s2) |s| s.size else (self.graph.tensors.get(name2) orelse return false).size;

            return (off1 < off2 + sz2) and (off2 < off1 + sz1);
        }

        return false;
    }

    fn has_dependency(self: *const Dispatcher, node1: GraphNode, node2: GraphNode) bool {
        for (node1.input_names) |in_name| {
            if (self.tensors_alias(in_name, node2.output_name)) return true;
        }
        for (node2.input_names) |in_name| {
            if (self.tensors_alias(node1.output_name, in_name)) return true;
        }
        if (self.tensors_alias(node1.output_name, node2.output_name)) return true;

        if (node2.op_type == .qwen_deinterleave and node2.input_names.len >= 2) {
            for (node1.input_names) |in_name| {
                if (self.tensors_alias(in_name, node2.input_names[1])) return true;
            }
        }
        if (node1.op_type == .qwen_deinterleave and node1.input_names.len >= 2) {
            for (node2.input_names) |in_name| {
                if (self.tensors_alias(node1.input_names[1], in_name)) return true;
            }
        }
        return false;
    }

    pub fn record_graph(self: *Dispatcher, cmd: vk.CommandBuffer, pos: u32) !void {
        const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;
        var last_barrier_idx: usize = 0;
        const nodes = self.graph.nodes.items;

        for (nodes, 0..) |node, i| {
            if (use_cpu_ssm and node.op_type == .ssm_delta_net_decode) continue;

            var need_barrier = false;
            if (i > 0) {
                var j = i - 1;
                while (true) {
                    if (self.has_dependency(node, nodes[j])) {
                        need_barrier = true;
                        break;
                    }
                    if (j == last_barrier_idx) break;
                    j -= 1;
                }
            }

            if (need_barrier) {
                self.emit_compute_barrier(cmd);
                last_barrier_idx = i;
            }
            try self.dispatch_node(cmd, node, pos);
        }
    }

    pub fn execute(self: *Dispatcher, pos: u32) !void {
        try self.ensure_submit_resources();
        const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;
        const nodes = self.graph.nodes.items;

        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true } });

        var last_barrier_idx: usize = 0;
        for (nodes, 0..) |node, i| {
            var need_barrier = false;
            if (i > 0) {
                var j = i - 1;
                while (true) {
                    if (self.has_dependency(node, nodes[j])) {
                        need_barrier = true;
                        break;
                    }
                    if (j == last_barrier_idx) break;
                    j -= 1;
                }
            }

            if (need_barrier) {
                self.emit_compute_barrier(self.cmd);
                last_barrier_idx = i;
            }

            if (use_cpu_ssm and node.op_type == .ssm_delta_net_decode) {
                // Interleave CPU step: finish current batch, wait, run CPU, restart GPU
                _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
                try self.submit_and_wait(self.cmd);
                try self.run_ssm_cpu_step(pos, node);

                _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{});
                _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
                last_barrier_idx = 0;
            } else {
                try self.dispatch_node(self.cmd, node, pos);
            }
        }

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submit_and_wait(self.cmd);
    }

    pub fn execute_prefill_batch(self: *Dispatcher, pos_start: u32, n_tokens: u32, input_batch: vulkan.Buffer, input_stride: u64) !void {
        const input_tensor = self.graph.tensors.get("input") orelse return error.MissingInputTensor;
        try self.ensure_submit_resources();

        // const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;
        const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;

        var i: u32 = 0;
        while (i < n_tokens) : (i += 1) {
            const copy_region = vk.BufferCopy{
                .src_offset = @as(u64, i) * input_stride,
                .dst_offset = input_tensor.offset,
                .size = input_stride,
            };

            _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{});
            _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
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

            const nodes = self.graph.nodes.items;
            var last_barrier_idx: usize = 0;
            for (nodes, 0..) |node, idx| {
                var need_barrier = false;
                if (idx > 0) {
                    var j = idx - 1;
                    while (true) {
                        if (self.has_dependency(node, nodes[j])) {
                            need_barrier = true;
                            break;
                        }
                        if (j == last_barrier_idx) break;
                        j -= 1;
                    }
                }
                if (need_barrier) {
                    self.emit_compute_barrier(self.cmd);
                    last_barrier_idx = idx;
                }

                if (use_cpu_ssm and node.op_type == .ssm_delta_net_decode) {
                    // End the current command buffer, submit, wait, then run
                    // the CPU SSM step for this layer. The next command
                    // buffer (for the rest of the graph) will see the CPU
                    // output as if it came from a transfer write.
                    std.debug.print("[prefill] using CPU SSM step for layer {s} token {}\n", .{ node.output_name, pos_start + i });
                    _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
                    try self.submit_and_wait(self.cmd);
                    try self.run_ssm_cpu_step(pos_start + i, node);

                    _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, .{});
                    _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
                    // Barrier: transfer_write (from CPU copy) -> shader_read
                    // for the next GPU segment.
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
                    last_barrier_idx = 0; // reset for fresh command buffer
                } else {
                    try self.dispatch_node(self.cmd, node, pos_start + i);
                }
            }
            _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
            try self.submit_and_wait(self.cmd);
        }
    }

    /// Run the CPU-side Gated Delta Net step for a single ssm_delta_net_decode
    /// node during prefill. Reads Q/K/V/gate/beta from the scratchpad, calls
    /// the CPU recurrence, and writes the `core` output back to the scratchpad.
    fn run_ssm_cpu_step(self: *Dispatcher, pos: u32, node: GraphNode) !void {
        _ = pos;
        const staging = self.ssm_staging orelse return;
        const staging_size = self.ssm_staging_size;

        // Output name is "blk.{layer}.core"; parse the layer index.
        var layer: u32 = 0;
        const prefix = "blk.";
        const suffix = ".core";
        if (std.mem.startsWith(u8, node.output_name, prefix)) {
            const rest = node.output_name[prefix.len..];
            if (std.mem.endsWith(u8, rest, suffix)) {
                const mid = rest[0 .. rest.len - suffix.len];
                layer = std.fmt.parseInt(u32, mid, 10) catch return;
            }
        }

        const head_v_dim: u32 = if (self.cfg.ssm_dt_rank > 0) self.cfg.ssm_d_inner / self.cfg.ssm_dt_rank else return;

        const conv_bytes: u64 = @as(u64, self.cfg.ssm_d_inner + 2 * self.cfg.ssm_n_group * self.cfg.ssm_d_state) *
            @as(u64, (if (self.cfg.ssm_d_conv > 0) self.cfg.ssm_d_conv else 4) - 1);
        const rec_bytes: u64 = @as(u64, head_v_dim) * @as(u64, self.cfg.ssm_d_state) * @as(u64, self.cfg.ssm_dt_rank);
        const n_main: u32 = self.cfg.n_layer -| self.cfg.nextn_predict_layers;
        const mapped_conv = try self.ctx.vkd.mapMemory(self.ctx.device, self.ssm_conv_buf.memory, 0, conv_bytes * n_main * @sizeOf(f32), .{});
        const mapped_state = try self.ctx.vkd.mapMemory(self.ctx.device, self.ssm_state_buf.memory, 0, rec_bytes * n_main * @sizeOf(f32), .{});
        defer {
            self.ctx.vkd.unmapMemory(self.ctx.device, self.ssm_conv_buf.memory);
            self.ctx.vkd.unmapMemory(self.ctx.device, self.ssm_state_buf.memory);
        }
        const conv_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_conv)))[0..@as(usize, conv_bytes) * @as(usize, n_main)];
        const state_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_state)))[0..@as(usize, rec_bytes) * @as(usize, n_main)];
        var ssm_ctx = @import("ssm_state.zig").SsmCpuContext.wrap(self.cfg, conv_f32, state_f32, self.allocator);

        var ln_buf: [32]u8 = undefined;
        const ln = std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer}) catch return;
        var qn_buf: [32]u8 = undefined;  const qn_name   = std.fmt.bufPrint(&qn_buf,   "{s}.q_norm", .{ln}) catch return;
        var kn_buf: [32]u8 = undefined;  const kn_name   = std.fmt.bufPrint(&kn_buf,   "{s}.k_norm", .{ln}) catch return;
        var vn_buf: [32]u8 = undefined;  const vn_name   = std.fmt.bufPrint(&vn_buf,   "{s}.v_conv", .{ln}) catch return;
        var gate_buf: [32]u8 = undefined; const gate_name = std.fmt.bufPrint(&gate_buf, "{s}.gate",   .{ln}) catch return;
        var beta_buf: [32]u8 = undefined; const beta_name = std.fmt.bufPrint(&beta_buf, "{s}.beta",   .{ln}) catch return;
        var core_buf: [32]u8 = undefined; const core_name = std.fmt.bufPrint(&core_buf, "{s}.core",   .{ln}) catch return;

        const qn_t = self.graph.resolve_tensor_offset(qn_name) orelse return;
        const kn_t = self.graph.resolve_tensor_offset(kn_name) orelse return;
        const vn_t = self.graph.resolve_tensor_offset(vn_name) orelse return;
        const gate_t = self.graph.resolve_tensor_offset(gate_name) orelse return;
        const beta_t = self.graph.resolve_tensor_offset(beta_name) orelse return;
        const core_t = self.graph.resolve_tensor_offset(core_name) orelse return;

        const total_input: u64 = qn_t.size + kn_t.size + vn_t.size + gate_t.size + beta_t.size;
        const q_off: u64 = 0;
        const k_off: u64 = qn_t.size;
        const v_off: u64 = k_off + kn_t.size;
        const g_off: u64 = v_off + vn_t.size;
        const b_off: u64 = g_off + gate_t.size;
        const core_off: u64 = staging_size - core_t.size;
        if (core_off < total_input) {
            if (self.trace_dispatch) std.debug.print("[ssm-cpu] FATAL: staging overflow layer={} total_input={} core_off={} staging={}\n", .{ layer, total_input, core_off, staging_size });
            return error.SsmStagingOverflow;
        }

        try self.ctx.copyBufferOffset(self.scratchpad, qn_t.offset, staging, q_off, qn_t.size);
        try self.ctx.copyBufferOffset(self.scratchpad, kn_t.offset, staging, k_off, kn_t.size);
        try self.ctx.copyBufferOffset(self.scratchpad, vn_t.offset, staging, v_off, vn_t.size);
        try self.ctx.copyBufferOffset(self.scratchpad, gate_t.offset, staging, g_off, gate_t.size);
        try self.ctx.copyBufferOffset(self.scratchpad, beta_t.offset, staging, b_off, beta_t.size);

        const mapped_ssm = try self.ctx.vkd.mapMemory(self.ctx.device, staging.memory, 0, staging_size, .{});
        defer self.ctx.vkd.unmapMemory(self.ctx.device, staging.memory);



        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));
        try ssm_ctx.stepDeltaNet(
            layer,
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[q_off / 4 ..][0 .. @as(usize, qn_t.size) / 4],
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[k_off / 4 ..][0 .. @as(usize, kn_t.size) / 4],
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[v_off / 4 ..][0 .. @as(usize, vn_t.size) / 4],
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[g_off / 4 ..][0 .. @as(usize, gate_t.size) / 4],
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[b_off / 4 ..][0 .. @as(usize, beta_t.size) / 4],
            @as([*]f32, @ptrCast(@alignCast(mapped_ssm)))[core_off / 4 ..][0 .. @as(usize, core_t.size) / 4],
            scale,
        );

        try self.ctx.copyBufferOffset(staging, core_off, self.scratchpad, core_t.offset, core_t.size);


    }

    pub fn execute_get_rows_q(
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
        try self.ensure_submit_resources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, indices_buf.memory, 0, 4, .{});
        @as(*u32, @ptrCast(@alignCast(mapped))).* = token_id;
        self.ctx.vkd.unmapMemory(self.ctx.device, indices_buf.memory);

                const pipe_name = switch (qtype) {
            @intFromEnum(tensor.Type.q4_0) => "get_rows_q4_0",
            @intFromEnum(tensor.Type.q4_1) => "get_rows_q4_1",
            @intFromEnum(tensor.Type.q4_k) => "get_rows_q4_k",
            @intFromEnum(tensor.Type.q6_k) => "get_rows_q6_k",
            else => "get_rows_q",
        };
        const pipe = self.registry.get(pipe_name) orelse return error.MissingPipeline;
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
        try self.submit_and_wait(self.cmd);
    }

    pub fn record_embed_and_graph(
        self: *Dispatcher,
        cmd: vk.CommandBuffer,
        pos: u32,
        indices_buf: vulkan.Buffer,
        weights_buf: vulkan.Buffer,
        out_offset: u64,
        n_embd: u32,
        qtype: u32,
        row_bytes: u32,
        scale_bits: u32,
    ) !void {
        const pipe_name = switch (qtype) {
            @intFromEnum(tensor.Type.q4_0) => "get_rows_q4_0",
            @intFromEnum(tensor.Type.q4_1) => "get_rows_q4_1",
            @intFromEnum(tensor.Type.q4_k) => "get_rows_q4_k",
            @intFromEnum(tensor.Type.q6_k) => "get_rows_q6_k",
            else => "get_rows_q",
        };
        const pipe = self.registry.get(pipe_name) orelse return;
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
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, (n_embd + 255) / 256, 1, 1);

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

        try self.record_graph(cmd, pos);
    }

    pub fn execute_top_k(
        self: *Dispatcher,
        logits_offset: u64,
        vocab_size: u32,
        out_indices_buf: vulkan.Buffer,
        out_values_buf: vulkan.Buffer,
        logit_scale_bits: u32,
    ) !u32 {
        try self.ensure_submit_resources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
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
        try self.submit_and_wait(self.cmd);

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, out_indices_buf.memory, 0, 4, .{});
        const id: u32 = @as(*u32, @ptrCast(@alignCast(mapped))).*;
        self.ctx.vkd.unmapMemory(self.ctx.device, out_indices_buf.memory);
        return id;
    }
};


const std = @import("std");
const dispatcher_mod = @import("dispatcher.zig");
const Dispatcher = dispatcher_mod.Dispatcher;
const ssm_state = @import("ssm_state.zig");
pub const SsmCpuContext = ssm_state.SsmCpuContext;
const graph_data = @import("graph_data.zig");
const Graph = graph_data.Graph;
const GraphNode = graph_data.GraphNode;

pub fn run_cpu_ssm_delta_for_layer(
    allocator: std.mem.Allocator,
    scratch_ptr: ?[*]f32,
    logits_persistent: []f32,
    ssm_ctx: *SsmCpuContext,
    graph: *const Graph,
    layer: u32,
    head_v_dim: u32,
) !void {
    var ln_buf: [32]u8 = undefined;
    const ln = std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer}) catch return;

    var qn_buf: [32]u8 = undefined;  const qn_name   = std.fmt.bufPrint(&qn_buf,   "{s}.q_norm", .{ln}) catch return;
    var kn_buf: [32]u8 = undefined;  const kn_name   = std.fmt.bufPrint(&kn_buf,   "{s}.k_norm", .{ln}) catch return;
    var vn_buf: [32]u8 = undefined;  const vn_name   = std.fmt.bufPrint(&vn_buf,   "{s}.v_conv", .{ln}) catch return;
    var gate_buf: [32]u8 = undefined; const gate_name = std.fmt.bufPrint(&gate_buf, "{s}.gate",   .{ln}) catch return;
    var beta_buf: [32]u8 = undefined; const beta_name = std.fmt.bufPrint(&beta_buf, "{s}.beta",   .{ln}) catch return;
    var core_buf: [32]u8 = undefined; const core_name = std.fmt.bufPrint(&core_buf, "{s}.core",   .{ln}) catch return;

    const qn_t = graph.resolve_tensor_offset(qn_name) orelse return;
    const kn_t = graph.resolve_tensor_offset(kn_name) orelse return;
    const vn_t = graph.resolve_tensor_offset(vn_name) orelse return;
    const gate_t = graph.resolve_tensor_offset(gate_name) orelse return;
    const beta_t = graph.resolve_tensor_offset(beta_name) orelse return;
    const core_t = graph.resolve_tensor_offset(core_name) orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));

    if (scratch_ptr) |s_ptr| {
        const q_in = s_ptr[qn_t.offset / 4 ..][0 .. @as(usize, qn_t.size) / 4];
        const k_in = s_ptr[kn_t.offset / 4 ..][0 .. @as(usize, kn_t.size) / 4];
        const v_in = s_ptr[vn_t.offset / 4 ..][0 .. @as(usize, vn_t.size) / 4];
        const g_in = s_ptr[gate_t.offset / 4 ..][0 .. @as(usize, gate_t.size) / 4];
        const b_in = s_ptr[beta_t.offset / 4 ..][0 .. @as(usize, beta_t.size) / 4];
        const core_out = s_ptr[core_t.offset / 4 ..][0 .. @as(usize, core_t.size) / 4];

        try ssm_ctx.stepDeltaNet(layer, q_in, k_in, v_in, g_in, b_in, core_out, scale);
    } else {
        _ = logits_persistent;
    }
    _ = allocator;
}

pub fn run_ssm_cpu_step(self: *Dispatcher, allocator: std.mem.Allocator) !void {
    const node = self.current_node orelse return error.MissingSsmNode;
    
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
    var ssm_ctx = SsmCpuContext.wrap(self.cfg, conv_f32, state_f32, allocator);

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

    const staging = self.ssm_staging orelse return;
    const staging_size = self.ssm_staging_size;

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

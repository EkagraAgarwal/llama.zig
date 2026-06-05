const std = @import("std");
const builtin = @import("builtin");

const windows = if (builtin.os.tag == .windows) struct {
    pub const HANDLE = *anyopaque;
    pub const LARGE_INTEGER = i64;
    pub const ntdll = struct {
        pub extern "ntdll" fn RtlQueryPerformanceCounter(Counter: *LARGE_INTEGER) callconv(.winapi) std.os.windows.BOOL;
        pub extern "ntdll" fn RtlQueryPerformanceFrequency(LinkTime: *LARGE_INTEGER) callconv(.winapi) std.os.windows.BOOL;
    };
} else struct {};

fn nowNs() u64 {
    if (builtin.os.tag == .windows) {
        var counter: windows.LARGE_INTEGER = 0;
        var freq: windows.LARGE_INTEGER = 0;
        _ = windows.ntdll.RtlQueryPerformanceCounter(&counter);
        _ = windows.ntdll.RtlQueryPerformanceFrequency(&freq);
        const c: u64 = @intCast(counter);
        const f: u64 = @intCast(freq);
        if (f == 0) return 0;
        return (c / f) * std.time.ns_per_s + ((c % f) * std.time.ns_per_s) / f;
    }
    return 0;
}
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
    gelu_mul = 14,
    copy = 15,
    // Qwen 3.5 / hybrid SSM ops (additive — values 16+).
    softplus = 16,
    sigmoid = 17,
    silu = 18,
    l2_norm = 19,
    ssm_conv1d = 20,
    ssm_delta_net_decode = 21,
    ssm_gated_norm = 22,
    rope_multi = 23,
    attn_qg_matmul = 24,
    attn_gate_mul = 25,
};

pub const TensorRole = enum(u8) {
    weight = 0,
    activation = 1,
    input = 2,
    output = 3,
    kv_cache = 4,
    ssm_cache = 5,
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
    p6: u32 = 0,
    p7: u32 = 0,
    p8: u32 = 0,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(GraphNode),
    tensors: std.StringHashMap(GraphTensor),
    scratchpad_size: u64 = 0,
    kv_cache_size: u64 = 0,
    ssm_cache_size: u64 = 0,
    /// Virtual slice tensors that point into a parent tensor at a byte offset.
    /// Resolved by Dispatcher.tensorAddr as `parent_addr + offset`.
    slices: std.StringHashMap(SliceRef),
    /// Weight tensor names whose contents are all-zero and should be uploaded
    /// as zeros by the weight upload loop (no GGUF backing tensor).
    synthetic_weights: std.StringHashMap(void),

    pub const SliceRef = struct {
        parent: []const u8,
        offset: u64,
        size: u64,
    };

    pub fn init(allocator: std.mem.Allocator) Graph {
        return Graph{
            .allocator = allocator,
            .nodes = .empty,
            .tensors = std.StringHashMap(GraphTensor).init(allocator),
            .slices = std.StringHashMap(SliceRef).init(allocator),
            .synthetic_weights = std.StringHashMap(void).init(allocator),
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
        var sit = self.slices.iterator();
        while (sit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.parent);
        }
        self.slices.deinit();
        var wit = self.synthetic_weights.iterator();
        while (wit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.synthetic_weights.deinit();
    }

    pub fn verify(self: *const Graph) !void {
        for (self.nodes.items) |node| {
            if (!self.tensors.contains(node.output_name) and !self.slices.contains(node.output_name)) {
                std.debug.print("Error: Missing output tensor '{s}' for op {s}\n", .{ node.output_name, @tagName(node.op_type) });
                return error.MissingOutputTensor;
            }
            for (node.input_names) |input_name| {
                if (!self.tensors.contains(input_name) and !self.slices.contains(input_name)) {
                    std.debug.print("Error: Missing input tensor '{s}' for op {s}\n", .{ input_name, @tagName(node.op_type) });
                    return error.MissingInputTensor;
                }
            }
        }
    }
};

pub const GraphCostSummary = struct {
    total_nodes: u32 = 0,
    approx_flops: u64 = 0,
    approx_bytes: u64 = 0,
    matmul_nodes: u32 = 0,
    attention_nodes: u32 = 0,
    other_nodes: u32 = 0,
};

pub const q4_0_f16_fallback_qtype: u32 = 0xffff_ff00;

pub fn estimateGraphCost(graph: *const Graph) GraphCostSummary {
    var out = GraphCostSummary{
        .total_nodes = @intCast(graph.nodes.items.len),
    };
    for (graph.nodes.items) |node| {
        switch (node.op_type) {
            .matmul, .matmul_q => {
                out.matmul_nodes += 1;
                out.approx_flops += @as(u64, node.p1) * @as(u64, node.p2) * @as(u64, node.p3) * 2;
                out.approx_bytes += (@as(u64, node.p1) * @as(u64, node.p3) + @as(u64, node.p2) * @as(u64, node.p3) + @as(u64, node.p1) * @as(u64, node.p2)) * 4;
            },
            .attention, .attention_flash => {
                out.attention_nodes += 1;
                out.approx_flops += @as(u64, node.p1) * @as(u64, node.p2) * 4;
                out.approx_bytes += @as(u64, node.p1) * @as(u64, node.p2) * 4;
            },
            else => {
                out.other_nodes += 1;
                out.approx_bytes += (@as(u64, node.p1) + @as(u64, node.p2) + @as(u64, node.p3)) * 4;
            },
        }
    }
    return out;
}

pub const GraphBuilder = struct {
    graph: *Graph,
    cfg: *const model.ModelConfig,
    model_tensors: ?*const std.StringHashMap(*tensor.Tensor) = null,

    pub fn init(graph: *Graph, cfg: *const model.ModelConfig, model_tensors: ?*const std.StringHashMap(*tensor.Tensor)) GraphBuilder {
        return GraphBuilder{ .graph = graph, .cfg = cfg, .model_tensors = model_tensors };
    }

    fn f32Size(n: u32) u64 {
        return @as(u64, n) * 4;
    }

    pub fn hasTensor(self: *GraphBuilder, name: []const u8) bool {
        if (self.model_tensors) |tbl| return tbl.contains(name);
        return false;
    }

    pub fn matmulDims(self: *GraphBuilder, weight_name: []const u8, fallback_out: u32, fallback_in: u32) struct { out: u32, in: u32 } {
        if (self.model_tensors) |tbl| {
            if (tbl.get(weight_name)) |wt| {
                return .{ .out = @intCast(wt.ne[1]), .in = @intCast(wt.ne[0]) };
            }
        }
        return .{ .out = fallback_out, .in = fallback_in };
    }

    fn canFuseQkv(self: *GraphBuilder, qw: []const u8, kw: []const u8, vw: []const u8) bool {
        if (self.model_tensors) |tbl| {
            const qt = tbl.get(qw) orelse return false;
            const kt = tbl.get(kw) orelse return false;
            const vt = tbl.get(vw) orelse return false;
            return qt.type == kt.type and kt.type == vt.type;
        }
        return false;
    }

    fn canFuseGateUp(self: *GraphBuilder, gw: []const u8, uw: []const u8) bool {
        if (self.model_tensors) |tbl| {
            const gt = tbl.get(gw) orelse return false;
            const ut = tbl.get(uw) orelse return false;
            return gt.type == ut.type;
        }
        return false;
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
        try self.addNodeP8(op, inputs, output, dx, dy, p1, p2, p3, p4, p5, 0, 0, 0);
    }

    pub fn addNodeP8(self: *GraphBuilder, op: OpType, inputs: []const []const u8, output: []const u8, dx: u32, dy: u32, p1: u32, p2: u32, p3: u32, p4: u32, p5: u32, p6: u32, p7: u32, p8: u32) !void {
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
            .p6 = p6,
            .p7 = p7,
            .p8 = p8,
        });
    }

    /// Register a synthetic weight (zero-filled) with no GGUF backing tensor.
    /// The weight upload loop in main.zig will check graph.synthetic_weights
    /// and upload zeros instead of reading from the GGUF.
    pub fn addSyntheticWeight(self: *GraphBuilder, name: []const u8, f32_count: u32) !void {
        std.debug.print("WARNING: Using synthetic (zero) weight for {s}\n", .{name});
        try self.addTensor(name, f32Size(f32_count), .weight);
        if (self.graph.synthetic_weights.contains(name)) return;
        const owned = try self.graph.allocator.dupe(u8, name);
        try self.graph.synthetic_weights.put(owned, {});
    }

    /// Register a virtual slice tensor that points into `parent_name` at the
    /// given byte offset. The Dispatcher resolves `tensorAddr(slice_name)` to
    /// `parent_addr + offset` (no separate storage).
    pub fn addSlice(self: *GraphBuilder, name: []const u8, parent_name: []const u8, byte_offset: u64, size_bytes: u64) ![]const u8 {
        if (self.graph.slices.getEntry(name)) |entry| {
            if (std.mem.eql(u8, entry.value_ptr.parent, parent_name) and entry.value_ptr.offset == byte_offset and entry.value_ptr.size == size_bytes) {
                return entry.key_ptr.*;
            }
            // Redefinition: free old parent and dupe new one.
            const new_parent = try self.graph.allocator.dupe(u8, parent_name);
            self.graph.allocator.free(entry.value_ptr.parent);
            entry.value_ptr.* = .{
                .parent = new_parent,
                .offset = byte_offset,
                .size = size_bytes,
            };
            return entry.key_ptr.*;
        }
        const owned_name = try self.graph.allocator.dupe(u8, name);
        errdefer self.graph.allocator.free(owned_name);
        const owned_parent = try self.graph.allocator.dupe(u8, parent_name);
        errdefer self.graph.allocator.free(owned_parent);

        try self.graph.slices.put(owned_name, .{
            .parent = owned_parent,
            .offset = byte_offset,
            .size = size_bytes,
        });
        return owned_name;
    }

    /// Register a virtual slice tensor that points into the parent at a given
    /// FLOAT offset and FLOAT count (helper for the common case).
    pub fn addSliceF32(self: *GraphBuilder, name: []const u8, parent_name: []const u8, float_offset: u32, float_count: u32) ![]const u8 {
        return self.addSlice(name, parent_name, @as(u64, float_offset) * 4, @as(u64, float_count) * 4);
    }

    pub const SsmQkvzLayout = enum {
        /// Modern Qwen 3.5: separate `attn_qkv` (Q+K+V) and `attn_gate` (Z) weights.
        fused_gate,
        /// Legacy path: single `ssm_in` weight packing [Q, K, V, Z].
        legacy_ssm_in,
        /// Neither weight present; caller will synthesize a zero-weight equivalent.
        synthetic,
    };

    pub fn resolveSsmQkvLayout(self: *GraphBuilder, layer_prefix: []const u8) SsmQkvzLayout {
        var qkv_buf: [64]u8 = undefined;
        const qkv_name = std.fmt.bufPrint(&qkv_buf, "{s}.attn_qkv.weight", .{layer_prefix}) catch return .synthetic;
        var qw_buf: [64]u8 = undefined;
        const qw_name = std.fmt.bufPrint(&qw_buf, "{s}.attn_q.weight", .{layer_prefix}) catch return .synthetic;
        if (self.hasTensor(qkv_name) or self.hasTensor(qw_name)) {
            var gate_buf: [64]u8 = undefined;
            const gate_name = std.fmt.bufPrint(&gate_buf, "{s}.attn_gate.weight", .{layer_prefix}) catch return .synthetic;
            if (self.hasTensor(gate_name)) return .fused_gate;
        }

        var sin_buf: [64]u8 = undefined;
        const sin_name = std.fmt.bufPrint(&sin_buf, "{s}.ssm_in.weight", .{layer_prefix}) catch return .synthetic;
        if (self.hasTensor(sin_name)) return .legacy_ssm_in;

        return .synthetic;
    }

    /// Register SSM caches (conv1d rolling window + per-head recurrent state
    /// matrix) for every main layer. MTP/NextN layers do not get SSM caches
    /// because they are dense attention blocks.
    pub fn initSsmCaches(self: *GraphBuilder) !void {
        const cfg = self.cfg;
        if (cfg.ssm_d_inner == 0 or cfg.ssm_d_state == 0 or cfg.ssm_n_group == 0) {
            return; // not a Qwen 3.5 model — leave ssm_cache_size at 0
        }
        const d_conv = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
        const conv_channels = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const head_v_dim = cfg.ssm_d_inner / if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1;
        const head_k_dim = cfg.ssm_d_state;
        const rec_state_per_layer = head_v_dim * head_k_dim * cfg.ssm_dt_rank;
        const conv_state_per_layer = @as(u64, d_conv - 1) * conv_channels;
        const per_layer = conv_state_per_layer + rec_state_per_layer;
        const n_main = cfg.n_layer -| cfg.nextn_predict_layers;
        const total = per_layer * n_main;
        self.graph.ssm_cache_size = total * 4; // f32 bytes

        var l: u32 = 0;
        while (l < n_main) : (l += 1) {
            var conv_name_buf: [24]u8 = undefined;
            const conv_name = try std.fmt.bufPrint(&conv_name_buf, "ssm_conv.{d}", .{l});
            try self.addTensor(conv_name, conv_state_per_layer * 4, .ssm_cache);
            if (self.graph.tensors.getPtr(conv_name)) |t| t.layer = l;

            var rec_name_buf: [24]u8 = undefined;
            const rec_name = try std.fmt.bufPrint(&rec_name_buf, "ssm_state.{d}", .{l});
            try self.addTensor(rec_name, rec_state_per_layer * 4, .ssm_cache);
            if (self.graph.tensors.getPtr(rec_name)) |t| t.layer = l;
        }
    }

    pub fn initKvCaches(self: *GraphBuilder) !void {
        const kv_per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 2 * 2;
        self.graph.kv_cache_size = kv_per_layer * self.cfg.n_layer;
        var l: u32 = 0;
        while (l < self.cfg.n_layer) : (l += 1) {
            var name_buf: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "kv.{d}", .{l});
            try self.addTensor(name, kv_per_layer, .kv_cache);
            if (self.graph.tensors.getPtr(name)) |t| t.layer = l;
        }
    }

    pub fn buildTransformerBlock(self: *GraphBuilder, layer: u32, pos: u32, in_name: []const u8, out_name: []const u8) !void {
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

        // 1. Attention Norm
        var nw_buf: [64]u8 = undefined;
        const nw = try std.fmt.bufPrint(&nw_buf, "{s}.attn_norm.weight", .{ln});
        var normed_buf: [64]u8 = undefined;
        const normed = try std.fmt.bufPrint(&normed_buf, "{s}.normed", .{ln});
        try self.addTensor(nw, f32Size(n_embd), .weight);
        try self.addTensor(normed, f32Size(n_embd), .activation);
        try self.addNode(.rms_norm, &.{ in_name, nw }, normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

        // 2. QKV Projections
        var qw_buf: [64]u8 = undefined;
        const qw = try std.fmt.bufPrint(&qw_buf, "{s}.attn_q.weight", .{ln});
        var kw_buf: [64]u8 = undefined;
        const kw = try std.fmt.bufPrint(&kw_buf, "{s}.attn_k.weight", .{ln});
        var vw_buf: [64]u8 = undefined;
        const vw = try std.fmt.bufPrint(&vw_buf, "{s}.attn_v.weight", .{ln});
        var qkvw_buf: [64]u8 = undefined;
        const qkvw = try std.fmt.bufPrint(&qkvw_buf, "{s}.attn_qkv.weight", .{ln});

        var qn_buf: [64]u8 = undefined;
        var kn_buf: [64]u8 = undefined;
        var vn_buf: [64]u8 = undefined;
        var qkvn_buf: [64]u8 = undefined;

        var qn: []const u8 = undefined;
        var kn: []const u8 = undefined;
        var vn: []const u8 = undefined;
        var q_offset: u32 = 0;
        var k_offset: u32 = 0;
        var v_offset: u32 = 0;

        const q_out = n_heads * head_dim;
        const kv_out = n_kv * head_dim;

        const use_fused_qkv = self.canFuseQkv(qw, kw, vw);

        if (use_fused_qkv) {
            const qkvn = try std.fmt.bufPrint(&qkvn_buf, "{s}.qkv", .{ln});
            const qkv_dims = struct { out: u32, in: u32 }{ .out = q_out + 2 * kv_out, .in = n_embd };
            try self.addTensor(qkvw, f32Size(qkv_dims.out * qkv_dims.in), .weight);
            try self.addTensor(qkvn, f32Size(qkv_dims.out), .activation);
            try self.addNode(.matmul, &.{ normed, qkvw }, qkvn, (qkv_dims.out + 15) / 16, 1, 1, qkv_dims.out, qkv_dims.in, 0);

            qn = qkvn;
            kn = qkvn;
            vn = qkvn;
            q_offset = 0;
            k_offset = q_out * 4;
            v_offset = (q_out + kv_out) * 4;
        } else {
            qn = try std.fmt.bufPrint(&qn_buf, "{s}.q", .{ln});
            kn = try std.fmt.bufPrint(&kn_buf, "{s}.k", .{ln});
            vn = try std.fmt.bufPrint(&vn_buf, "{s}.v", .{ln});

            const q_dims = self.matmulDims(qw, q_out, n_embd);
            const k_dims = self.matmulDims(kw, kv_out, n_embd);
            const v_dims = self.matmulDims(vw, kv_out, n_embd);

            try self.addTensor(qw, f32Size(q_dims.out * q_dims.in), .weight);
            try self.addTensor(kw, f32Size(k_dims.out * k_dims.in), .weight);
            try self.addTensor(vw, f32Size(v_dims.out * v_dims.in), .weight);
            try self.addTensor(qn, f32Size(q_dims.out), .activation);
            try self.addTensor(kn, f32Size(k_dims.out), .activation);
            try self.addTensor(vn, f32Size(v_dims.out), .activation);

            try self.addNodeP(.matmul, &.{ normed, qw }, qn, (q_out + 15) / 16, 1, 1, q_out, n_embd, 0, 0);
            try self.addNodeP(.matmul, &.{ normed, kw }, kn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
            try self.addNodeP(.matmul, &.{ normed, vw }, vn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
        }

        // 3. Optional QKV Biases (Qwen2)
        const q_bias_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_q.bias", .{ln});
        defer self.graph.allocator.free(q_bias_name);
        if (self.hasTensor(q_bias_name)) {
            try self.addTensor(q_bias_name, f32Size(q_out), .weight);
            try self.addNode(.add, &.{ qn, q_bias_name }, qn, (q_out + 63) / 64, 1, q_out, q_out, 0, 0);
        }
        const k_bias_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_k.bias", .{ln});
        defer self.graph.allocator.free(k_bias_name);
        if (self.hasTensor(k_bias_name)) {
            try self.addTensor(k_bias_name, f32Size(kv_out), .weight);
            try self.addNode(.add, &.{ kn, k_bias_name }, kn, (kv_out + 63) / 64, 1, kv_out, kv_out, 0, 0);
        }
        const v_bias_name = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_v.bias", .{ln});
        defer self.graph.allocator.free(v_bias_name);
        if (self.hasTensor(v_bias_name)) {
            try self.addTensor(v_bias_name, f32Size(kv_out), .weight);
            try self.addNode(.add, &.{ vn, v_bias_name }, vn, (kv_out + 63) / 64, 1, kv_out, kv_out, 0, 0);
        }

        // 4. Optional QK Norm (Qwen3)
        const q_norm_w = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_q_norm.weight", .{ln});
        defer self.graph.allocator.free(q_norm_w);
        if (self.hasTensor(q_norm_w)) {
            try self.addTensor(q_norm_w, f32Size(q_out), .weight);
            try self.addNode(.rms_norm, &.{ qn, q_norm_w }, qn, (q_out + 63) / 64, 1, q_out, q_out, eps_bits, 0);
        }
        const k_norm_w = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_k_norm.weight", .{ln});
        defer self.graph.allocator.free(k_norm_w);
        if (self.hasTensor(k_norm_w)) {
            try self.addTensor(k_norm_w, f32Size(kv_out), .weight);
            try self.addNode(.rms_norm, &.{ kn, k_norm_w }, kn, (kv_out + 63) / 64, 1, kv_out, kv_out, eps_bits, 0);
        }

        // 5. RoPE
        const q_head_dim = if (n_heads > 0) q_out / n_heads else head_dim;
        const k_head_dim = if (n_kv > 0) kv_out / n_kv else head_dim;
        try self.addNodeP8(.rope, &.{qn}, qn, (q_out + 63) / 64, 1, n_heads, q_head_dim, pos, rope_bits, q_offset, 0, 0, 0);
        try self.addNodeP8(.rope, &.{kn}, kn, (kv_out + 63) / 64, 1, n_kv, k_head_dim, pos, rope_bits, k_offset, 0, 0, 0);

        // 6. Attention
        var attn_buf: [64]u8 = undefined;
        const attn = try std.fmt.bufPrint(&attn_buf, "{s}.attn", .{ln});
        try self.addTensor(attn, f32Size(q_out), .activation);

        var kv_name_buf: [16]u8 = undefined;
        const kv_name = try std.fmt.bufPrint(&kv_name_buf, "kv.{d}", .{layer});
        try self.addNodeP8(.kv_write, &.{ kn, vn, kv_name }, kn, ((kv_out / 2) + 63) / 64, 1, n_kv, k_head_dim, cfg.max_ctx, pos, k_offset, v_offset, 0, 0);
        const attn_p2 = k_head_dim | (n_kv << 16);
        const attn_scale_bits: u32 = @bitCast(cfg.attention_scale);
        try self.addNodeP8(.attention, &.{ qn, kv_name }, attn, n_heads, 1, n_heads, attn_p2, cfg.max_ctx, pos, attn_scale_bits, q_offset, 0, 0);

        // 7. Output Projection
        var ow_buf: [64]u8 = undefined;
        var ow = try std.fmt.bufPrint(&ow_buf, "{s}.attn_output.weight", .{ln});
        if (!self.hasTensor(ow)) ow = try std.fmt.bufPrint(&ow_buf, "{s}.proj.weight", .{ln});

        var attn_out_buf: [64]u8 = undefined;
        const attn_out = try std.fmt.bufPrint(&attn_out_buf, "{s}.attn_out", .{ln});
        const o_dims = self.matmulDims(ow, n_embd, q_out);
        try self.addTensor(ow, f32Size(o_dims.out * o_dims.in), .weight);
        try self.addTensor(attn_out, f32Size(o_dims.out), .activation);
        try self.addNodeP(.matmul, &.{ attn, ow }, attn_out, (o_dims.out + 15) / 16, 1, 1, o_dims.out, o_dims.in, 0, 0);

        // 8. Optional Post-Attention Norm (Gemma2)
        const attn_post_norm_w = try std.fmt.allocPrint(self.graph.allocator, "{s}.attn_post_norm.weight", .{ln});
        defer self.graph.allocator.free(attn_post_norm_w);
        if (self.hasTensor(attn_post_norm_w)) {
            try self.addTensor(attn_post_norm_w, f32Size(o_dims.out), .weight);
            try self.addNode(.rms_norm, &.{ attn_out, attn_post_norm_w }, attn_out, (o_dims.out + 63) / 64, 1, o_dims.out, o_dims.out, eps_bits, 0);
        }

        // 9. Residual Add 1
        var res1_buf: [64]u8 = undefined;
        const res1 = try std.fmt.bufPrint(&res1_buf, "{s}.res1", .{ln});
        try self.addTensor(res1, f32Size(o_dims.out), .activation);
        const res_scale_bits: u32 = @bitCast(cfg.residual_scale);
        if (cfg.residual_scale != 1.0) {
            try self.addNode(.scaled_add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, res_scale_bits, 0, 0);
        } else {
            try self.addNode(.add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, 0, 0, 0);
        }

        // 10. FFN Norm
        var fnw_buf: [64]u8 = undefined;
        const fnw = try std.fmt.bufPrint(&fnw_buf, "{s}.ffn_norm.weight", .{ln});
        var ffn_normed_buf: [64]u8 = undefined;
        const ffn_normed = try std.fmt.bufPrint(&ffn_normed_buf, "{s}.ffn_normed", .{ln});
        try self.addTensor(fnw, f32Size(n_embd), .weight);
        try self.addTensor(ffn_normed, f32Size(o_dims.out), .activation);
        try self.addNode(.rms_norm, &.{ res1, fnw }, ffn_normed, (o_dims.out + 63) / 64, 1, o_dims.out, o_dims.out, eps_bits, 0);

        // 11. Feed Forward
        var gw_buf: [64]u8 = undefined;
        const gw = try std.fmt.bufPrint(&gw_buf, "{s}.ffn_gate.weight", .{ln});
        var uw_buf: [64]u8 = undefined;
        const uw = try std.fmt.bufPrint(&uw_buf, "{s}.ffn_up.weight", .{ln});
        var gate_up_w_buf: [64]u8 = undefined;
        const gate_up_w = try std.fmt.bufPrint(&gate_up_w_buf, "{s}.ffn_gate_up.weight", .{ln});
        var gate_up_n_buf: [64]u8 = undefined;
        const gate_up_n = try std.fmt.bufPrint(&gate_up_n_buf, "{s}.gate_up", .{ln});

        var gate: []const u8 = undefined;
        var up: []const u8 = undefined;
        var gate_offset: u32 = 0;
        var up_offset: u32 = 0;

        const g_dims = self.matmulDims(gw, n_ff, o_dims.out);
        const use_fused_gate_up = self.canFuseGateUp(gw, uw);

        if (use_fused_gate_up) {
            const gate_up_dims = struct { out: u32, in: u32 }{ .out = g_dims.out * 2, .in = g_dims.in };
            try self.addTensor(gate_up_w, f32Size(gate_up_dims.out * gate_up_dims.in), .weight);
            try self.addTensor(gate_up_n, f32Size(gate_up_dims.out), .activation);
            try self.addNode(.matmul, &.{ ffn_normed, gate_up_w }, gate_up_n, (gate_up_dims.out + 15) / 16, 1, 1, gate_up_dims.out, gate_up_dims.in, 0);

            gate = gate_up_n;
            up = gate_up_n;
            gate_offset = 0;
            up_offset = g_dims.out * 4;
        } else {
            var gate_buf: [64]u8 = undefined;
            gate = try self.graph.allocator.dupe(u8, try std.fmt.bufPrint(&gate_buf, "{s}.gate", .{ln}));
            var up_buf: [64]u8 = undefined;
            up = try self.graph.allocator.dupe(u8, try std.fmt.bufPrint(&up_buf, "{s}.up", .{ln}));

            const u_dims = self.matmulDims(uw, n_ff, o_dims.out);
            try self.addTensor(gw, f32Size(g_dims.out * g_dims.in), .weight);
            try self.addTensor(uw, f32Size(u_dims.out * u_dims.in), .weight);
            try self.addTensor(gate, f32Size(g_dims.out), .activation);
            try self.addTensor(up, f32Size(u_dims.out), .activation);
            try self.addNodeP(.matmul, &.{ ffn_normed, gw }, gate, (g_dims.out + 15) / 16, 1, 1, g_dims.out, g_dims.in, 0, 0);
            try self.addNodeP(.matmul, &.{ ffn_normed, uw }, up, (u_dims.out + 15) / 16, 1, 1, u_dims.out, u_dims.in, 0, 0);
        }

        const activation_op: OpType = if (cfg.activation == .gelu) .gelu_mul else .silu_mul;
        try self.addNodeP8(activation_op, &.{ gate, up }, gate, (g_dims.out + 63) / 64, 1, g_dims.out, 0, 0, 0, gate_offset, up_offset, gate_offset, 0);

        var dw_buf: [64]u8 = undefined;
        const dw = try std.fmt.bufPrint(&dw_buf, "{s}.ffn_down.weight", .{ln});
        var ffn_out_buf: [64]u8 = undefined;
        const ffn_out = try std.fmt.bufPrint(&ffn_out_buf, "{s}.ffn_out", .{ln});
        const d_dims = self.matmulDims(dw, o_dims.out, g_dims.out);
        try self.addTensor(dw, f32Size(d_dims.out * d_dims.in), .weight);
        try self.addTensor(ffn_out, f32Size(d_dims.out), .activation);
        try self.addNodeP(.matmul, &.{ gate, dw }, ffn_out, (d_dims.out + 15) / 16, 1, 1, d_dims.out, d_dims.in, 0, 0);

        // 12. Optional Post-FFN Norm (Gemma2)
        const ffn_post_norm_w = try std.fmt.allocPrint(self.graph.allocator, "{s}.ffn_post_norm.weight", .{ln});
        defer self.graph.allocator.free(ffn_post_norm_w);
        if (self.hasTensor(ffn_post_norm_w)) {
            try self.addTensor(ffn_post_norm_w, f32Size(d_dims.out), .weight);
            try self.addNode(.rms_norm, &.{ ffn_out, ffn_post_norm_w }, ffn_out, (d_dims.out + 63) / 64, 1, d_dims.out, d_dims.out, eps_bits, 0);
        }

        // 13. Residual Add 2
        try self.addTensor(out_name, f32Size(d_dims.out), .activation);
        if (cfg.residual_scale != 1.0) {
            try self.addNode(.scaled_add, &.{ res1, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, res_scale_bits, 0, 0);
        } else {
            try self.addNode(.add, &.{ res1, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, 0, 0, 0);
        }
    }

    pub fn buildLmHead(self: *GraphBuilder, in_name: []const u8, logits_name: []const u8, has_output_weight: bool) !void {
        const cfg = self.cfg;
        const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);

        const norm_w = "output_norm.weight";
        const normed = "final.normed";
        try self.addTensor(norm_w, f32Size(cfg.n_embd), .weight);
        try self.addTensor(normed, f32Size(cfg.n_embd), .activation);
        try self.addNode(.rms_norm, &.{ in_name, norm_w }, normed, (cfg.n_embd + 63) / 64, 1, cfg.n_embd, cfg.n_embd, eps_bits, 0);

        const out_w = if (has_output_weight) "output.weight" else "token_embd.weight";
        const out_dims = self.matmulDims(out_w, cfg.vocab_size, cfg.n_embd);
        try self.addTensor(out_w, f32Size(out_dims.out * out_dims.in), .weight);

        try self.addTensor(logits_name, f32Size(out_dims.out), .activation);
        try self.addNodeP(.matmul, &.{ normed, out_w }, logits_name, (out_dims.out + 15) / 16, 1, 1, out_dims.out, out_dims.in, 0, 0);

        // Optional Output Bias (Qwen)
        const out_bias = "output.bias";
        if (self.hasTensor(out_bias)) {
            try self.addTensor(out_bias, f32Size(out_dims.out), .weight);
            try self.addNode(.add, &.{ logits_name, out_bias }, logits_name, (out_dims.out + 63) / 64, 1, out_dims.out, out_dims.out, 0, 0);
        }
    }

    /// Emit a fused-or-split QKV projection. Mirrors the logic in llama.zig.
    /// Returns {q, k, v, q_offset, k_offset, v_offset} where the offsets are
    /// in BYTES into the `qkv` activation tensor (for the fused case) and zero
    /// for the split case.
    pub fn emitQkv(
        self: *GraphBuilder,
        layer_prefix: []const u8,
        in_name: []const u8,
        q_out: u32,
        kv_out: u32,
        n_embd: u32,
    ) !struct { q: []const u8, k: []const u8, v: []const u8, q_off: u32, k_off: u32, v_off: u32 } {
        var qw_buf: [64]u8 = undefined;
        const qw = try std.fmt.bufPrint(&qw_buf, "{s}.attn_q.weight", .{layer_prefix});
        var kw_buf: [64]u8 = undefined;
        const kw = try std.fmt.bufPrint(&kw_buf, "{s}.attn_k.weight", .{layer_prefix});
        var vw_buf: [64]u8 = undefined;
        const vw = try std.fmt.bufPrint(&vw_buf, "{s}.attn_v.weight", .{layer_prefix});

        var qkvw_buf: [64]u8 = undefined;
        const qkvw = try std.fmt.bufPrint(&qkvw_buf, "{s}.attn_qkv.weight", .{layer_prefix});

        if (self.canFuseQkv(qw, kw, vw)) {
            var qkvn_buf: [64]u8 = undefined;
            const qkvn = try std.fmt.bufPrint(&qkvn_buf, "{s}.qkv", .{layer_prefix});
            const qkv_dims = struct { out: u32, in: u32 }{ .out = q_out + 2 * kv_out, .in = n_embd };
            try self.addTensor(qkvw, f32Size(qkv_dims.out * qkv_dims.in), .weight);
            try self.addTensor(qkvn, f32Size(qkv_dims.out), .activation);
            try self.addNode(.matmul, &.{ in_name, qkvw }, qkvn, (qkv_dims.out + 15) / 16, 1, 1, qkv_dims.out, qkv_dims.in, 0);
            return .{
                .q = qkvn,
                .k = qkvn,
                .v = qkvn,
                .q_off = 0,
                .k_off = q_out * 4,
                .v_off = (q_out + kv_out) * 4,
            };
        }

        var qn_buf: [64]u8 = undefined;
        const qn = try std.fmt.bufPrint(&qn_buf, "{s}.q", .{layer_prefix});
        var kn_buf: [64]u8 = undefined;
        const kn = try std.fmt.bufPrint(&kn_buf, "{s}.k", .{layer_prefix});
        var vn_buf: [64]u8 = undefined;
        const vn = try std.fmt.bufPrint(&vn_buf, "{s}.v", .{layer_prefix});
        const q_dims = self.matmulDims(qw, q_out, n_embd);
        const k_dims = self.matmulDims(kw, kv_out, n_embd);
        const v_dims = self.matmulDims(vw, kv_out, n_embd);
        try self.addTensor(qw, f32Size(q_dims.out * q_dims.in), .weight);
        try self.addTensor(kw, f32Size(k_dims.out * k_dims.in), .weight);
        try self.addTensor(vw, f32Size(v_dims.out * v_dims.in), .weight);
        try self.addTensor(qn, f32Size(q_dims.out), .activation);
        try self.addTensor(kn, f32Size(k_dims.out), .activation);
        try self.addTensor(vn, f32Size(v_dims.out), .activation);
        try self.addNodeP(.matmul, &.{ in_name, qw }, qn, (q_out + 15) / 16, 1, 1, q_out, n_embd, 0, 0);
        try self.addNodeP(.matmul, &.{ in_name, kw }, kn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
        try self.addNodeP(.matmul, &.{ in_name, vw }, vn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
        return .{
            .q = qn,
            .k = kn,
            .v = vn,
            .q_off = 0,
            .k_off = 0,
            .v_off = 0,
        };
    }

    /// Emit a fused-or-split gate/up projection for the FFN. Returns the gate
    /// and up tensor names plus the byte offsets (zero in the split case).
    pub fn emitGateUp(
        self: *GraphBuilder,
        layer_prefix: []const u8,
        in_name: []const u8,
        ff_out: u32,
        ff_in: u32,
    ) !struct { gate: []const u8, up: []const u8, gate_off: u32, up_off: u32, out: u32 } {
        var gw_buf: [64]u8 = undefined;
        const gw = try std.fmt.bufPrint(&gw_buf, "{s}.ffn_gate.weight", .{layer_prefix});
        var uw_buf: [64]u8 = undefined;
        const uw = try std.fmt.bufPrint(&uw_buf, "{s}.ffn_up.weight", .{layer_prefix});
        var gu_buf: [64]u8 = undefined;
        const gate_up_w = try std.fmt.bufPrint(&gu_buf, "{s}.ffn_gate_up.weight", .{layer_prefix});

        if (self.canFuseGateUp(gw, uw)) {
            var gn_buf: [64]u8 = undefined;
            const gate_up_n = try std.fmt.bufPrint(&gn_buf, "{s}.gate_up", .{layer_prefix});
            const total = ff_out * 2;
            try self.addTensor(gate_up_w, f32Size(total * ff_in), .weight);
            try self.addTensor(gate_up_n, f32Size(total), .activation);
            try self.addNode(.matmul, &.{ in_name, gate_up_w }, gate_up_n, (total + 15) / 16, 1, 1, total, ff_in, 0);
            // Dupe the name into heap storage so it survives this function's
            // return (caller may pass it to subsequent addNode calls).
            const gate_up_owned = try self.graph.allocator.dupe(u8, gate_up_n);
            return .{ .gate = gate_up_owned, .up = gate_up_owned, .gate_off = 0, .up_off = ff_out * 4, .out = ff_out };
        }

        var g_buf: [64]u8 = undefined;
        const gate = try std.fmt.bufPrint(&g_buf, "{s}.gate", .{layer_prefix});
        var u_buf: [64]u8 = undefined;
        const up = try std.fmt.bufPrint(&u_buf, "{s}.up", .{layer_prefix});
        const g_dims = self.matmulDims(gw, ff_out, ff_in);
        const u_dims = self.matmulDims(uw, ff_out, ff_in);
        try self.addTensor(gw, f32Size(g_dims.out * g_dims.in), .weight);
        try self.addTensor(uw, f32Size(u_dims.out * u_dims.in), .weight);
        try self.addTensor(gate, f32Size(g_dims.out), .activation);
        try self.addTensor(up, f32Size(u_dims.out), .activation);
        try self.addNodeP(.matmul, &.{ in_name, gw }, gate, (ff_out + 15) / 16, 1, 1, ff_out, ff_in, 0, 0);
        try self.addNodeP(.matmul, &.{ in_name, uw }, up, (ff_out + 15) / 16, 1, 1, ff_out, ff_in, 0, 0);
        // Dupe the names into heap storage so they survive this function's
        // return (caller passes them to subsequent addNode calls).
        return .{
            .gate = try self.graph.allocator.dupe(u8, gate),
            .up = try self.graph.allocator.dupe(u8, up),
            .gate_off = 0,
            .up_off = 0,
            .out = ff_out,
        };
    }

    /// Emit a multi-axis RoPE (MRoPE) op. If all `rope_sections` are zero, this
    /// falls back to the standard `.rope` op. `q_offset`/`k_offset` are in
    /// BYTES into the q/k activation tensors.
    pub fn emitRoPEMulti(
        self: *GraphBuilder,
        q: []const u8,
        k: []const u8,
        q_out: u32,
        kv_out: u32,
        q_head_dim: u32,
        k_head_dim: u32,
        pos: u32,
        rope_bits: u32,
        q_offset: u32,
        k_offset: u32,
    ) !void {
        const sec = self.cfg.rope_sections;
        const use_mrope = sec[0] != 0 or sec[1] != 0 or sec[2] != 0 or sec[3] != 0;
        if (!use_mrope) {
            // Fall back to standard rope (the shader applies per-pair rotation
            // with a single divisor equal to head_dim).
            try self.addNodeP8(.rope, &.{q}, q, (q_out + 63) / 64, 1, q_out / q_head_dim, q_head_dim, pos, rope_bits, q_offset, 0, 0, 0);
            try self.addNodeP8(.rope, &.{k}, k, (kv_out + 63) / 64, 1, kv_out / k_head_dim, k_head_dim, pos, rope_bits, k_offset, 0, 0, 0);
            return;
        }
        const q_n_heads = q_out / q_head_dim;
        const k_n_heads = kv_out / k_head_dim;
        // Push-constant layout for rope_multi:
        //   p1=n_heads, p2=head_dim, p3=pos, p4=rope_theta, p5=byte_offset, p6=sec0, p7=sec1, p8=sec2
        try self.addNodeP8(.rope_multi, &.{q}, q, (q_out + 63) / 64, 1, q_n_heads, q_head_dim, pos, rope_bits, q_offset, sec[0], sec[1], sec[2]);
        try self.addNodeP8(.rope_multi, &.{k}, k, (kv_out + 63) / 64, 1, k_n_heads, k_head_dim, pos, rope_bits, k_offset, sec[0], sec[1], sec[2]);
    }

    pub fn finalize(self: *GraphBuilder) !void {
        const allocator = self.graph.allocator;
        var first_use = std.StringHashMap(usize).init(allocator);
        defer first_use.deinit();
        var last_use = std.StringHashMap(usize).init(allocator);
        defer last_use.deinit();

        // Pass 1: Identify lifetimes of activations, inputs, and outputs.
        var t_it = self.graph.tensors.iterator();
        while (t_it.next()) |entry| {
            const t = entry.value_ptr;
            if (t.role == .input) {
                try first_use.put(entry.key_ptr.*, 0);
                try last_use.put(entry.key_ptr.*, 0);
            }
        }

        for (self.graph.nodes.items, 0..) |node, i| {
            if (!first_use.contains(node.output_name)) {
                try first_use.put(node.output_name, i);
            }
            try last_use.put(node.output_name, i);

            for (node.input_names) |in_name| {
                try last_use.put(in_name, i);
            }
        }

        // Tensors with role .output must persist until the end of the graph execution.
        t_it = self.graph.tensors.iterator();
        while (t_it.next()) |entry| {
            if (entry.value_ptr.role == .output) {
                try last_use.put(entry.key_ptr.*, self.graph.nodes.items.len);
            }
        }

        // Pass 2: Greedy block allocation based on non-overlapping lifetimes.
        var to_alloc: std.ArrayListUnmanaged(*GraphTensor) = .empty;
        defer to_alloc.deinit(allocator);
        t_it = self.graph.tensors.iterator();
        while (t_it.next()) |entry| {
            const t = entry.value_ptr;
            if (t.role == .activation or t.role == .input or t.role == .output) {
                try to_alloc.append(allocator, t);
            }
        }

        const SortContext = struct {
            first_use: *std.StringHashMap(usize),
            pub fn lessThan(ctx: @This(), a: *GraphTensor, b: *GraphTensor) bool {
                const fa = ctx.first_use.get(a.name) orelse 0;
                const fb = ctx.first_use.get(b.name) orelse 0;
                if (fa != fb) return fa < fb;
                return a.size > b.size; // Larger first for better packing
            }
        };
        std.sort.heap(*GraphTensor, to_alloc.items, SortContext{ .first_use = &first_use }, SortContext.lessThan);

        var free_blocks: std.ArrayListUnmanaged(struct { offset: u64, size: u64 }) = .empty;
        defer free_blocks.deinit(allocator);

        var current_high_water: u64 = 0;
        var live: std.ArrayListUnmanaged(*GraphTensor) = .empty;
        defer live.deinit(allocator);

        for (to_alloc.items) |t| {
            const t_start = first_use.get(t.name) orelse 0;

            // Retire tensors that are no longer used by the time this tensor is born.
            var j: usize = 0;
            while (j < live.items.len) {
                const lt = live.items[j];
                const lt_end = last_use.get(lt.name) orelse 0;
                if (lt_end < t_start) {
                    try free_blocks.append(allocator, .{ .offset = lt.offset, .size = (lt.size + 255) & ~@as(u64, 255) });
                    _ = live.swapRemove(j);
                } else {
                    j += 1;
                }
            }

            const aligned_size = (t.size + 255) & ~@as(u64, 255);
            var best_idx: ?usize = null;
            var min_waste: u64 = std.math.maxInt(u64);
            for (free_blocks.items, 0..) |block, idx| {
                if (block.size >= aligned_size) {
                    const waste = block.size - aligned_size;
                    if (waste < min_waste) {
                        min_waste = waste;
                        best_idx = idx;
                    }
                }
            }

            if (best_idx) |idx| {
                const block = free_blocks.swapRemove(idx);
                t.offset = block.offset;
                if (block.size > aligned_size) {
                    try free_blocks.append(allocator, .{ .offset = block.offset + aligned_size, .size = block.size - aligned_size });
                }
            } else {
                t.offset = current_high_water;
                current_high_water += aligned_size;
            }

            try live.append(allocator, t);
        }

        var max_scratch: u64 = 0;
        for (to_alloc.items) |t| {
            max_scratch = @max(max_scratch, t.offset + ((t.size + 255) & ~@as(u64, 255)));
        }
        self.graph.scratchpad_size = max_scratch;
    }
};

pub const Dispatcher = struct {
    graph: *Graph,
    ctx: *vulkan.Context,
    registry: *vulkan.PipelineRegistry,
    scratchpad: vulkan.Buffer,
    kv_cache: vulkan.Buffer,
    /// Host-visible staging buffer used by the per-token CPU SSM step.
    /// Set by `setSsmStagingBuffer` before prefill/decode.
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
        };
        try self.ensureSubmitResources();
        return self;
    }

    /// Set the host-visible staging buffer used by the per-token CPU SSM
    /// step. Call this once before prefill/decode.
    pub fn setSsmStagingBuffer(self: *Dispatcher, staging: vulkan.Buffer, size: u64) void {
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

    pub fn ensureSubmitResources(self: *Dispatcher) !void {
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
        // Slices resolve to parent + offset (no separate storage).
        if (self.graph.slices.get(name)) |slice| {
            const parent_addr = self.tensorAddr(slice.parent);
            return parent_addr + slice.offset;
        }
        const t = self.graph.tensors.get(name) orelse return 0;
        return switch (t.role) {
            .weight => t.buffer.?.address,
            .kv_cache => self.kvCacheLayerOffset(t.layer),
            .ssm_cache => self.ssmCacheLayerOffset(t.layer, t.size),
            .input, .activation, .output => self.scratchpad.address + t.offset,
        };
    }

    fn kvCacheLayerOffset(self: *Dispatcher, layer: u32) u64 {
        const per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 2 * 2;
        return self.kv_cache.address + per_layer * layer;
    }

    /// Returns the base BDA of a per-layer SSM cache tensor. SSM caches are
    /// split into two ranges (conv state + recurrent state) within the
    /// `ssm_conv_buf` and `ssm_state_buf` buffers. The tensor's `size` field
    /// tells us which one it is (conv = smaller, state = larger).
    fn ssmCacheLayerOffset(self: *Dispatcher, layer: u32, size_bytes: u64) u64 {
        const cfg = self.cfg;
        if (cfg.ssm_d_inner == 0 or cfg.ssm_d_state == 0 or cfg.ssm_n_group == 0) return 0;
        const d_conv = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
        const conv_channels = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const head_v_dim = cfg.ssm_d_inner / if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1;
        const rec_per_layer = head_v_dim * head_v_dim * cfg.ssm_dt_rank;
        const conv_per_layer = @as(u64, d_conv - 1) * conv_channels;
        const per_layer_bytes_conv = conv_per_layer * 4;
        if (size_bytes <= per_layer_bytes_conv + 8) {
            // conv cache: stored in ssm_conv_buf
            return self.ssm_conv_buf.address + per_layer_bytes_conv * layer;
        }
        // recurrent state cache: stored in ssm_state_buf
        return self.ssm_state_buf.address + (rec_per_layer * 4) * layer;
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
                break :blk quantPipelineName(node.p5, node.p1 <= 1);
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

    fn quantPipelineName(qtype: u32, is_matvec: bool) []const u8 {
        if (qtype == q4_0_f16_fallback_qtype) {
            return "matmul_f16";
        }
        const qt: tensor.Type = @enumFromInt(qtype);
        return if (is_matvec)
            switch (qt) { .q4_0 => "matvec_q4_0", .q4_1 => "matvec_q4_1", .q4_k => "matvec_q4_k", .q5_k => "matvec_q5_k", .q6_k => "matvec_q6_k", .f16 => "matvec_f16", else => "matvec_q8_0" }
        else
            switch (qt) { .q4_0 => "matmul_q4_0", .q4_1 => "matmul_q4_1", .q4_k => "matmul_q4_k", .q5_k => "matmul_q5_k", .q6_k => "matmul_q6_k", .f16 => "matmul_f16", else => "matmul_q8_0" };
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
            .p6 = node.p6,
            .p7 = node.p7,
            .p8 = node.p8,
            .a = 0,
            .b = 0,
            .c = 0,
        };

        switch (node.op_type) {
            .kv_write => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.b = self.tensorAddr(node.input_names[1]) + node.p6;
                pc.c = self.tensorAddr(node.input_names[2]);
                pc.p4 = pos;
            },
            .attention => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p6;
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
                pc.p4 = pos;
                if (pos + 1 >= self.flash_attn_threshold) pc.p6 = 64;
            },
            .rope => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.c = self.tensorAddr(node.output_name) + node.p5;
                pc.p3 = pos;
            },
            .rope_multi => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.c = self.tensorAddr(node.output_name) + node.p5;
                pc.p3 = pos;
                // sec0/sec1/sec2/sec3 are already in p5..p8 from the builder; the
                // shader reads them as the divisor groups for MRoPE.
            },
            .get_rows_q => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
            .copy => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.c = self.tensorAddr(node.output_name);
            },
            .silu_mul, .gelu_mul => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.b = self.tensorAddr(node.input_names[1]) + node.p6;
                pc.c = self.tensorAddr(node.output_name) + node.p7;
            },
            // SSM conv1d shader takes 4 buffers: state (in/out), new_chunk,
            // kernel, output.
            .ssm_conv1d => {
                pc.a = self.tensorAddr(node.input_names[0]); // state (in/out)
                pc.b = self.tensorAddr(node.input_names[1]); // new chunk
                pc.c = self.tensorAddr(node.input_names[2]); // kernel
                pc.d = self.tensorAddr(node.output_name);    // output
            },
            // SSM delta-net decode takes 6 inputs + 1 output: state, q, k, v, g,
            // beta → output. Mapped to push constant slots a..g.
            .ssm_delta_net_decode => {
                pc.a = self.tensorAddr(node.input_names[0]); // state
                pc.b = self.tensorAddr(node.input_names[1]); // q
                pc.c = self.tensorAddr(node.input_names[2]); // k
                pc.d = self.tensorAddr(node.input_names[3]); // v
                pc.e = self.tensorAddr(node.input_names[4]); // g
                pc.f = self.tensorAddr(node.input_names[5]); // beta
                pc.g = self.tensorAddr(node.output_name);    // output
            },
            // SSM gated norm: core (in), z (in), rms_norm_weight (in), out.
            .ssm_gated_norm => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.input_names[2]);
                pc.d = self.tensorAddr(node.output_name);
            },
            else => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]);
                if (node.input_names.len >= 2) pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
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

    pub fn submitAndWait(self: *Dispatcher, cmd: vk.CommandBuffer) !void {
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
        self.submit_count += 1;
    }

    fn tensorsAlias(self: *const Dispatcher, name1: []const u8, name2: []const u8) bool {
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

    fn hasDependency(self: *const Dispatcher, node1: GraphNode, node2: GraphNode) bool {
        for (node1.input_names) |in_name| {
            if (self.tensorsAlias(in_name, node2.output_name)) return true;
        }
        for (node2.input_names) |in_name| {
            if (self.tensorsAlias(node1.output_name, in_name)) return true;
        }
        if (self.tensorsAlias(node1.output_name, node2.output_name)) return true;
        return false;
    }

    pub fn recordGraph(self: *Dispatcher, cmd: vk.CommandBuffer, pos: u32) void {
        // const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;
        var last_barrier_idx: usize = 0;
        const nodes = self.graph.nodes.items;

        for (nodes, 0..) |node, i| {
            // Skip SSM delta-net dispatch when using CPU path
            // if (use_cpu_ssm and node.op_type == .ssm_delta_net_decode) continue;

            var need_barrier = false;
            if (i > 0) {
                var j = i - 1;
                while (true) {
                    if (self.hasDependency(node, nodes[j])) {
                        need_barrier = true;
                        break;
                    }
                    if (j == last_barrier_idx) break;
                    j -= 1;
                }
            }

            if (need_barrier) {
                self.emitComputeBarrier(cmd);
                last_barrier_idx = i;
            }
            self.dispatchNode(cmd, node, pos);
        }
    }

    pub fn execute(self: *Dispatcher, pos: u32) !void {
        try self.ensureSubmitResources();
        if (self.trace_dispatch) std.debug.print("[execute] pos={} use_cpu_ssm={} ssm_staging_set={} d_inner={}\n", .{ pos, self.ssm_staging != null and self.cfg.ssm_d_inner > 0, self.ssm_staging != null, self.cfg.ssm_d_inner });
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });
        
        // Optimizing: only record what's strictly necessary.
        // For decoding, the graph structure is static, only 'pos' changes.
        self.recordGraph(self.cmd, pos);
        
        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executePrefillBatch(self: *Dispatcher, pos_start: u32, n_tokens: u32, input_batch: vulkan.Buffer, input_stride: u64) !void {
        const input_tensor = self.graph.tensors.get("input") orelse return error.MissingInputTensor;
        try self.ensureSubmitResources();

        // const use_cpu_ssm = self.ssm_staging != null and self.cfg.ssm_d_inner > 0;

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
                        if (self.hasDependency(node, nodes[j])) {
                            need_barrier = true;
                            break;
                        }
                        if (j == last_barrier_idx) break;
                        j -= 1;
                    }
                }
                if (need_barrier) {
                    self.emitComputeBarrier(self.cmd);
                    last_barrier_idx = idx;
                }

                if (false and node.op_type == .ssm_delta_net_decode) {
                    // End the current command buffer, submit, wait, then run
                    // the CPU SSM step for this layer. The next command
                    // buffer (for the rest of the graph) will see the CPU
                    // output as if it came from a transfer write.
                    _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
                    try self.submitAndWait(self.cmd);
                    try self.runSsmCpuStep(pos_start + i, node);

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
                    self.dispatchNode(self.cmd, node, pos_start + i);
                }
            }
            _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
            try self.submitAndWait(self.cmd);
        }
    }

    /// Run the CPU-side Gated Delta Net step for a single ssm_delta_net_decode
    /// node during prefill. Reads Q/K/V/gate/beta from the scratchpad, calls
    /// the CPU recurrence, and writes the `core` output back to the scratchpad.
    fn runSsmCpuStep(self: *Dispatcher, pos: u32, node: GraphNode) !void {
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
        const rec_bytes: u64 = @as(u64, head_v_dim) * @as(u64, head_v_dim) * @as(u64, self.cfg.ssm_dt_rank);
        const n_main: u32 = self.cfg.n_layer -| self.cfg.nextn_predict_layers;
        const mapped_conv = try self.ctx.vkd.mapMemory(self.ctx.device, self.ssm_conv_buf.memory, 0, conv_bytes * n_main, .{});
        const mapped_state = try self.ctx.vkd.mapMemory(self.ctx.device, self.ssm_state_buf.memory, 0, rec_bytes * n_main, .{});
        defer {
            self.ctx.vkd.unmapMemory(self.ctx.device, self.ssm_conv_buf.memory);
            self.ctx.vkd.unmapMemory(self.ctx.device, self.ssm_state_buf.memory);
        }
        const conv_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_conv)))[0..@as(usize, conv_bytes) * @as(usize, n_main)];
        const state_f32 = @as([*]f32, @ptrCast(@alignCast(mapped_state)))[0..@as(usize, rec_bytes) * @as(usize, n_main)];
        var ssm_ctx = @import("ssm_state.zig").SsmCpuContext.wrap(self.cfg, conv_f32, state_f32);

        var ln_buf: [32]u8 = undefined;
        const ln = std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer}) catch return;
        var name_buf: [32]u8 = undefined;
        const qn_name = std.fmt.bufPrint(&name_buf, "{s}.q_norm", .{ln}) catch return;
        @memset(name_buf[0..qn_name.len], 0);
        const kn_name = std.fmt.bufPrint(&name_buf, "{s}.k_norm", .{ln}) catch return;
        @memset(name_buf[0..kn_name.len], 0);
        const vn_name = std.fmt.bufPrint(&name_buf, "{s}.v_conv", .{ln}) catch return;
        @memset(name_buf[0..vn_name.len], 0);
        const gate_name = std.fmt.bufPrint(&name_buf, "{s}.gate", .{ln}) catch return;
        @memset(name_buf[0..gate_name.len], 0);
        const beta_name = std.fmt.bufPrint(&name_buf, "{s}.beta", .{ln}) catch return;
        @memset(name_buf[0..beta_name.len], 0);
        const core_name = std.fmt.bufPrint(&name_buf, "{s}.core", .{ln}) catch return;

        const qn_t = self.graph.tensors.get(qn_name) orelse return;
        const kn_t = self.graph.tensors.get(kn_name) orelse return;
        const vn_t = self.graph.tensors.get(vn_name) orelse return;
        const gate_t = self.graph.tensors.get(gate_name) orelse return;
        const beta_t = self.graph.tensors.get(beta_name) orelse return;
        const core_t = self.graph.tensors.get(core_name) orelse return;

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
        try self.submitAndWait(self.cmd);
    }

    pub fn recordEmbedAndGraph(
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
    ) void {
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

        self.recordGraph(cmd, pos);
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
        try self.submitAndWait(self.cmd);

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, out_indices_buf.memory, 0, 4, .{});
        const id: u32 = @as(*u32, @ptrCast(@alignCast(mapped))).*;
        self.ctx.vkd.unmapMemory(self.ctx.device, out_indices_buf.memory);
        return id;
    }
};

test "quantized kernel selection is architecture independent" {
    const t = std.testing;
    try t.expectEqualStrings("matvec_q8_0", Dispatcher.quantPipelineName(@intFromEnum(tensor.Type.q8_0), true));
    try t.expectEqualStrings("matmul_q8_0", Dispatcher.quantPipelineName(@intFromEnum(tensor.Type.q8_0), false));
    try t.expectEqualStrings("matmul_f16", Dispatcher.quantPipelineName(q4_0_f16_fallback_qtype, true));
    try t.expectEqualStrings("matmul_f16", Dispatcher.quantPipelineName(q4_0_f16_fallback_qtype, false));
}

test "initSsmCaches populates per-layer conv + state tensors" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 2560,
        .n_layer = 8,
        .n_heads = 16,
        .n_kv_heads = 4,
        .n_ff = 9216,
        .head_dim = 160,
        .vocab_size = 100,
        .max_ctx = 128,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .ssm_d_conv = 4,
        .ssm_d_inner = 1024,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 4,
        .full_attn_interval = 4,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try builder.initSsmCaches();

    // For n_layer=8, interval=4: n_main=8, but every 4th layer is attention,
    // so SSM caches are still allocated for all 8 layers (the cache lives for
    // the lifetime of the graph, not just SSM layers).
    try t.expect(graph.tensors.contains("ssm_conv.0"));
    try t.expect(graph.tensors.contains("ssm_conv.7"));
    try t.expect(graph.tensors.contains("ssm_state.0"));
    try t.expect(graph.tensors.contains("ssm_state.7"));
    try t.expect(graph.ssm_cache_size > 0);

    const conv0 = graph.tensors.get("ssm_conv.0").?;
    try t.expectEqual(@as(u8, @intFromEnum(TensorRole.ssm_cache)), @as(u8, @intFromEnum(conv0.role)));
}

test "initSsmCaches is a no-op when SSM dims are zero" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const cfg = model.ModelConfig{
        .arch = .llama, .arch_prefix = "llama", .activation = .silu,
        .n_embd = 256, .n_layer = 2, .n_heads = 4, .n_kv_heads = 4, .n_ff = 1024,
        .head_dim = 64, .vocab_size = 100, .max_ctx = 128, .rope_theta = 1.0e4,
        .rms_norm_eps = 1.0e-5, .wtype = .f32, .embedding_scale = 1.0,
        .attention_scale = 0.0, .residual_scale = 1.0, .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try builder.initSsmCaches();
    try t.expectEqual(@as(u64, 0), graph.ssm_cache_size);
    try t.expect(!graph.tensors.contains("ssm_conv.0"));
}

test "addSlice registers parent-relative view" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const cfg = model.ModelConfig{
        .arch = .llama, .arch_prefix = "llama", .activation = .silu,
        .n_embd = 256, .n_layer = 1, .n_heads = 4, .n_kv_heads = 4, .n_ff = 1024,
        .head_dim = 64, .vocab_size = 100, .max_ctx = 128, .rope_theta = 1.0e4,
        .rms_norm_eps = 1.0e-5, .wtype = .f32, .embedding_scale = 1.0,
        .attention_scale = 0.0, .residual_scale = 1.0, .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try builder.addTensor("parent", 1024, .activation);
    const slice = try builder.addSliceF32("view", "parent", 8, 16);
    try t.expectEqualStrings("view", slice);

    const ref = graph.slices.get("view").?;
    try t.expectEqualStrings("parent", ref.parent);
    try t.expectEqual(@as(u64, 32), ref.offset); // 8 floats * 4 bytes
    try t.expectEqual(@as(u64, 64), ref.size);   // 16 floats * 4 bytes
}

test "addSyntheticWeight marks tensor as zero-upload" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const cfg = model.ModelConfig{
        .arch = .qwen35, .arch_prefix = "qwen35", .activation = .silu,
        .n_embd = 256, .n_layer = 1, .n_heads = 4, .n_kv_heads = 2, .n_ff = 1024,
        .head_dim = 64, .vocab_size = 100, .max_ctx = 128, .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6, .wtype = .q4_k, .embedding_scale = 1.0,
        .attention_scale = 0.0, .residual_scale = 1.0, .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try builder.addSyntheticWeight("synth_zero", 100);
    try t.expect(graph.tensors.contains("synth_zero"));
    try t.expect(graph.synthetic_weights.contains("synth_zero"));
    const t_entry = graph.tensors.get("synth_zero").?;
    try t.expectEqual(@as(u64, 400), t_entry.size);
}

test "resolveSsmQkvLayout picks fused_gate when both weights present" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const qkv = try tensor.Tensor.init(allocator, "blk.0.attn_qkv.weight", .q4_k, &.{ 256, 2560 });
    defer qkv.deinit(allocator);
    try tensors.put("blk.0.attn_qkv.weight", qkv);

    const gate = try tensor.Tensor.init(allocator, "blk.0.attn_gate.weight", .q4_k, &.{ 256, 2560 });
    defer gate.deinit(allocator);
    try tensors.put("blk.0.attn_gate.weight", gate);

    const cfg = model.ModelConfig{
        .arch = .qwen35, .arch_prefix = "qwen35", .activation = .silu,
        .n_embd = 2560, .n_layer = 1, .n_heads = 16, .n_kv_heads = 4, .n_ff = 9216,
        .head_dim = 160, .vocab_size = 100, .max_ctx = 128, .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6, .wtype = .q4_k, .embedding_scale = 1.0,
        .attention_scale = 0.0, .residual_scale = 1.0, .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try t.expectEqual(GraphBuilder.SsmQkvzLayout.fused_gate, builder.resolveSsmQkvLayout("blk.0"));
}

test "resolveSsmQkvLayout returns synthetic when no SSM weights present" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const cfg = model.ModelConfig{
        .arch = .qwen35, .arch_prefix = "qwen35", .activation = .silu,
        .n_embd = 2560, .n_layer = 1, .n_heads = 16, .n_kv_heads = 4, .n_ff = 9216,
        .head_dim = 160, .vocab_size = 100, .max_ctx = 128, .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6, .wtype = .q4_k, .embedding_scale = 1.0,
        .attention_scale = 0.0, .residual_scale = 1.0, .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
    };
    var builder = GraphBuilder.init(&graph, &cfg, &tensors);
    try t.expectEqual(GraphBuilder.SsmQkvzLayout.synthetic, builder.resolveSsmQkvLayout("blk.0"));
}

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

    pub fn verify(self: *const Graph) !void {
        for (self.nodes.items) |node| {
            if (!self.tensors.contains(node.output_name)) return error.MissingOutputTensor;
            for (node.input_names) |input_name| {
                if (!self.tensors.contains(input_name)) {
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

    pub fn canFuseQkv(self: *GraphBuilder, qw: []const u8, kw: []const u8, vw: []const u8) bool {
        if (self.model_tensors) |tbl| {
            const qt = tbl.get(qw) orelse return false;
            const kt = tbl.get(kw) orelse return false;
            const vt = tbl.get(vw) orelse return false;
            return qt.type == kt.type and kt.type == vt.type;
        }
        return false;
    }

    pub fn canFuseGateUp(self: *GraphBuilder, gw: []const u8, uw: []const u8) bool {
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
    flash_attn_threshold: u32 = 1,
    submit_count: u32 = 0,
    reported_graph: bool = false,

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
        const t = self.graph.tensors.get(name) orelse return 0;
        return switch (t.role) {
            .weight => t.buffer.?.address,
            .kv_cache => self.kvCacheLayerOffset(t.layer),
            .input, .activation, .output => self.scratchpad.address + t.offset,
        };
    }

    fn kvCacheLayerOffset(self: *Dispatcher, layer: u32) u64 {
        const per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 2 * 2;
        return self.kv_cache.address + per_layer * layer;
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
                    .q6_k => "get_rows_q6_k",
                    else => "get_rows_q",
                };
            },
            .attention => blk: {
                if (node.p4 + 1 >= self.flash_attn_threshold) break :blk "attention_flash";
                break :blk "attention";
            },
            .gelu_mul => "gelu_mul",
            else => @tagName(node.op_type),
        };
    }

    fn quantPipelineName(qtype: u32, is_matvec: bool) []const u8 {
        if (qtype == q4_0_f16_fallback_qtype) {
            return if (is_matvec) "matvec_f16" else "matmul_f16";
        }
        const qt: tensor.Type = @enumFromInt(qtype);
        return if (is_matvec)
            switch (qt) { .q4_0 => "matvec_q4_0", .q4_1 => "matvec_q4_1", .q4_k => "matvec_q4_k", .q6_k => "matvec_q6_k", else => "matvec_q8_0" }
        else
            switch (qt) { .q4_0 => "matmul_q4_0", .q4_1 => "matmul_q4_1", .q4_k => "matmul_q4_k", .q6_k => "matmul_q6_k", else => "matmul_q8_0" };
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
            else => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]);
                if (node.input_names.len >= 2) pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
        }

        var dx = node.dispatch_x;
        var dy = node.dispatch_y;
        if (node.op_type == .matmul_q and node.p1 <= 1) {
            dx = (node.p2 + 7) / 8;
            dy = 1;
        }

        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(vulkan.PushConstants), @ptrCast(&pc));
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

    fn hasDependency(node1: GraphNode, node2: GraphNode) bool {
        for (node1.input_names) |in_name| {
            if (std.mem.eql(u8, in_name, node2.output_name)) return true;
        }
        for (node2.input_names) |in_name| {
            if (std.mem.eql(u8, node1.output_name, in_name)) return true;
        }
        if (std.mem.eql(u8, node1.output_name, node2.output_name)) return true;
        return false;
    }

    pub fn recordGraph(self: *Dispatcher, cmd: vk.CommandBuffer, pos: u32) void {
        var last_barrier_idx: usize = 0;
        const nodes = self.graph.nodes.items;

        for (nodes, 0..) |node, i| {
            var need_barrier = false;
            if (i > 0) {
                var j = i - 1;
                while (true) {
                    if (hasDependency(node, nodes[j])) {
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
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
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
    try t.expectEqualStrings("matvec_f16", Dispatcher.quantPipelineName(q4_0_f16_fallback_qtype, true));
    try t.expectEqualStrings("matmul_f16", Dispatcher.quantPipelineName(q4_0_f16_fallback_qtype, false));
}

const std = @import("std");
const vulkan_backend = @import("vulkan_backend.zig");

pub const TENSOR_ALIGN: u64 = 256;
pub const TENSOR_ALIGN_MASK: u64 = 255;

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
    qwen_deinterleave = 26,
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
    buffer: ?*vulkan_backend.Buffer = null,
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
    /// Resolved by Dispatcher.tensor_addr as `parent_addr + offset`.
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
                std.log.err("Missing output tensor '{s}' for op {s}", .{ node.output_name, @tagName(node.op_type) });
                return error.MissingOutputTensor;
            }
            for (node.input_names) |input_name| {
                if (!self.tensors.contains(input_name) and !self.slices.contains(input_name)) {
                    std.log.err("Missing input tensor '{s}' for op {s}", .{ input_name, @tagName(node.op_type) });
                    return error.MissingInputTensor;
                }
            }
        }
    }

    /// Resolve a tensor or slice name to an absolute byte offset and size.
    /// For slices, the parent's offset is resolved recursively (up to 8 levels).
    /// Returns null if the name is not found or the parent chain is broken.
    pub fn resolve_tensor_offset(self: *const Graph, name: []const u8) ?struct { offset: u64, size: u64 } {
        if (self.tensors.get(name)) |t| {
            return .{ .offset = t.offset, .size = t.size };
        }
        if (self.slices.get(name)) |slice| {
            var visited: [8][]const u8 = undefined;
            var depth: usize = 0;
            var current_parent: []const u8 = slice.parent;
            var current_offset: u64 = slice.offset;
            while (true) {
                for (visited[0..depth]) |v| {
                    if (std.mem.eql(u8, v, current_parent)) return null;
                }
                if (depth >= visited.len) return null;
                visited[depth] = current_parent;
                depth += 1;
                if (self.tensors.get(current_parent)) |t| {
                    return .{ .offset = t.offset + current_offset, .size = slice.size };
                }
                if (self.slices.get(current_parent)) |parent_slice| {
                    current_offset = parent_slice.offset + current_offset;
                    current_parent = parent_slice.parent;
                    continue;
                }
                return null;
            }
        }
        return null;
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

pub fn estimate_graph_cost(graph: *const Graph) GraphCostSummary {
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

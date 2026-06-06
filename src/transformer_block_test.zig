const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const model = @import("model.zig");
const tensor = @import("tensor.zig");

test "transformer_block graph creation" {
    const allocator = std.testing.allocator;
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    
    var tensors = std.StringArrayHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();
    
    // Mock the weights needed for a Llama block
    const weights = [_][]const u8{
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };
    for (weights) |name| {
        var t = try allocator.create(tensor.Tensor);
        t.* = .{ .name = name, .type = .f32, .ne = [_]u64{ 128, 128, 1, 1 }, .offset = 0 };
        try tensors.put(name, t);
    }
    defer {
        for (tensors.values()) |t| allocator.destroy(t);
    }

    const cfg = model.ModelConfig{
        .arch = .llama,
        .n_embd = 128,
        .n_layer = 1,
        .n_heads = 4,
        .n_kv_heads = 4,
        .n_ff = 256,
        .max_ctx = 128,
        .vocab_size = 100,
        .head_dim = 32,
    };
    
    var builder = compute_graph.GraphBuilder.init(&graph, &cfg, &tensors);
    try builder.add_tensor("input", 128 * 4, .input);
    try builder.build_transformer_block(0, 0, "input", "output");
    
    try std.testing.expect(graph.nodes.items.len > 0);
}

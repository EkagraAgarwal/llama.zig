const std = @import("std");
const gguf = @import("gguf.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const vulkan = @import("vulkan_backend.zig");
const compute_graph = @import("compute_graph.zig");
const dispatcher_mod = @import("dispatcher.zig");
const inference = @import("inference.zig");
const sampler = @import("sampler.zig");
const weight_uploader = @import("weight_uploader.zig");
const cli = @import("cli.zig");

test "Llama-3.2-1B regression (no-crash)" {
    const allocator = std.testing.allocator;
    const model_path = "models/Llama-3.2-1B.Q4_K_M.gguf";
    
    // Check if model exists
    std.fs.cwd().access(model_path, .{}) catch return;

    var ctx = try gguf.loadModelMmap(allocator, model_path);
    defer ctx.deinit();

    var tok = try tokenizer.Tokenizer.init(allocator, &ctx);
    defer tok.deinit();

    const vocab_size: u32 = @intCast(tok.id_to_token.len);
    var cfg = try model.ModelConfig.init(allocator, &ctx, vocab_size);
    defer cfg.deinit(allocator);

    var vk_ctx = try vulkan.Context.init(allocator);
    defer vk_ctx.deinit();

    // Registry needs actual kernels, which might be missing in test build 
    // unless kernels_data is linked.
    // For regression test, we just want to ensure the graph builder and 
    // loader logic works with actual GGUF.
    
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();
    var builder = compute_graph.GraphBuilder.init(&graph, &cfg, &ctx.tensors);

    try builder.add_tensor("input", model.f32Bytes(cfg.n_embd), .input);
    try builder.init_kv_caches();
    
    var prev_out: []const u8 = "input";
    for (0..cfg.n_layer) |l| {
        const out_name = try std.fmt.allocPrint(allocator, "blk.{d}.out", .{l});
        defer allocator.free(out_name);
        try builder.build_transformer_block(@intCast(l), 0, prev_out, out_name);
        prev_out = graph.tensors.getPtr(out_name).?.name;
    }
    try builder.build_lm_head(prev_out, "logits", ctx.tensors.get("output.weight") != null);
    try builder.finalize();

    try std.testing.expect(graph.nodes.items.len > 0);
}

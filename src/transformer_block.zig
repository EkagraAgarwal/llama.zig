const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const GraphBuilder = compute_graph.GraphBuilder;
const ModelConfig = @import("model.zig").ModelConfig;
const OpType = compute_graph.OpType;

pub fn build_llama_block(builder: *GraphBuilder, cfg: *const ModelConfig, layer: u32, pos: u32, in_name: []const u8, out_name: []const u8) !void {
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
    try builder.add_tensor(nw, @as(u64, n_embd) * 4, .weight);
    try builder.add_tensor(normed, @as(u64, n_embd) * 4, .activation);
    try builder.add_node(.rms_norm, &.{ in_name, nw }, normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

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

    const use_fused_qkv = builder.can_fuse_qkv(qw, kw, vw);

    if (use_fused_qkv) {
        const qkvn = try std.fmt.bufPrint(&qkvn_buf, "{s}.qkv", .{ln});
        const qkv_dims = struct { out: u32, in: u32 }{ .out = q_out + 2 * kv_out, .in = n_embd };
        try builder.add_tensor(qkvw, @as(u64, qkv_dims.out * qkv_dims.in) * 4, .weight);
        try builder.add_tensor(qkvn, @as(u64, qkv_dims.out) * 4, .activation);
        try builder.add_node(.matmul, &.{ normed, qkvw }, qkvn, (qkv_dims.out + 15) / 16, 1, 1, qkv_dims.out, qkv_dims.in, 0);

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

        const q_dims = builder.matmul_dims(qw, q_out, n_embd);
        const k_dims = builder.matmul_dims(kw, kv_out, n_embd);
        const v_dims = builder.matmul_dims(vw, kv_out, n_embd);

        try builder.add_tensor(qw, @as(u64, q_dims.out * q_dims.in) * 4, .weight);
        try builder.add_tensor(kw, @as(u64, k_dims.out * k_dims.in) * 4, .weight);
        try builder.add_tensor(vw, @as(u64, v_dims.out * v_dims.in) * 4, .weight);
        try builder.add_tensor(qn, @as(u64, q_dims.out) * 4, .activation);
        try builder.add_tensor(kn, @as(u64, k_dims.out) * 4, .activation);
        try builder.add_tensor(vn, @as(u64, v_dims.out) * 4, .activation);

        try builder.add_node_p(.matmul, &.{ normed, qw }, qn, (q_out + 15) / 16, 1, 1, q_out, n_embd, 0, 0);
        try builder.add_node_p(.matmul, &.{ normed, kw }, kn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
        try builder.add_node_p(.matmul, &.{ normed, vw }, vn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
    }

    // 3. Optional QKV Biases (Qwen2)
    const q_bias_name = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_q.bias", .{ln});
    defer builder.graph.allocator.free(q_bias_name);
    if (builder.has_tensor(q_bias_name)) {
        try builder.add_tensor(q_bias_name, @as(u64, q_out) * 4, .weight);
        try builder.add_node(.add, &.{ qn, q_bias_name }, qn, (q_out + 63) / 64, 1, q_out, q_out, 0, 0);
    }
    const k_bias_name = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_k.bias", .{ln});
    defer builder.graph.allocator.free(k_bias_name);
    if (builder.has_tensor(k_bias_name)) {
        try builder.add_tensor(k_bias_name, @as(u64, kv_out) * 4, .weight);
        try builder.add_node(.add, &.{ kn, k_bias_name }, kn, (kv_out + 63) / 64, 1, kv_out, kv_out, 0, 0);
    }
    const v_bias_name = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_v.bias", .{ln});
    defer builder.graph.allocator.free(v_bias_name);
    if (builder.has_tensor(v_bias_name)) {
        try builder.add_tensor(v_bias_name, @as(u64, kv_out) * 4, .weight);
        try builder.add_node(.add, &.{ vn, v_bias_name }, vn, (kv_out + 63) / 64, 1, kv_out, kv_out, 0, 0);
    }

    // 4. Optional QK Norm (Qwen3)
    const q_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_q_norm.weight", .{ln});
    defer builder.graph.allocator.free(q_norm_w);
    if (builder.has_tensor(q_norm_w)) {
        try builder.add_tensor(q_norm_w, @as(u64, q_out) * 4, .weight);
        try builder.add_node(.rms_norm, &.{ qn, q_norm_w }, qn, (q_out + 63) / 64, 1, q_out, q_out, eps_bits, 0);
    }
    const k_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_k_norm.weight", .{ln});
    defer builder.graph.allocator.free(k_norm_w);
    if (builder.has_tensor(k_norm_w)) {
        try builder.add_tensor(k_norm_w, @as(u64, kv_out) * 4, .weight);
        try builder.add_node(.rms_norm, &.{ kn, k_norm_w }, kn, (kv_out + 63) / 64, 1, kv_out, kv_out, eps_bits, 0);
    }

    // 5. RoPE
    const q_head_dim = if (n_heads > 0) q_out / n_heads else head_dim;
    const k_head_dim = if (n_kv > 0) kv_out / n_kv else head_dim;
    try builder.add_node_p8(.rope, &.{qn}, qn, (q_out + 63) / 64, 1, n_heads, q_head_dim, pos, rope_bits, q_offset, 0, 0, 0);
    try builder.add_node_p8(.rope, &.{kn}, kn, (kv_out + 63) / 64, 1, n_kv, k_head_dim, pos, rope_bits, k_offset, 0, 0, 0);

    // 6. Attention
    var attn_buf: [64]u8 = undefined;
    const attn = try std.fmt.bufPrint(&attn_buf, "{s}.attn", .{ln});
    try builder.add_tensor(attn, @as(u64, q_out) * 4, .activation);

    var kv_name_buf: [16]u8 = undefined;
    const kv_name = try std.fmt.bufPrint(&kv_name_buf, "kv.{d}", .{layer});
    try builder.add_node_p8(.kv_write, &.{ kn, vn, kv_name }, kn, ((kv_out / 2) + 63) / 64, 1, n_kv, k_head_dim, cfg.max_ctx, pos, k_offset, v_offset, 0, 0);
    const attn_p2 = k_head_dim | (n_kv << 16);
    const attn_scale_bits: u32 = @bitCast(cfg.attention_scale);
    try builder.add_node_p8(.attention, &.{ qn, kv_name }, attn, n_heads, 1, n_heads, attn_p2, cfg.max_ctx, pos, attn_scale_bits, q_offset, 0, 0);

    // 7. Output Projection
    var ow_buf: [64]u8 = undefined;
    var ow = try std.fmt.bufPrint(&ow_buf, "{s}.attn_output.weight", .{ln});
    if (!builder.has_tensor(ow)) ow = try std.fmt.bufPrint(&ow_buf, "{s}.proj.weight", .{ln});

    var attn_out_buf: [64]u8 = undefined;
    const attn_out = try std.fmt.bufPrint(&attn_out_buf, "{s}.attn_out", .{ln});
    const o_dims = builder.matmul_dims(ow, n_embd, q_out);
    try builder.add_tensor(ow, @as(u64, o_dims.out * o_dims.in) * 4, .weight);
    try builder.add_tensor(attn_out, @as(u64, o_dims.out) * 4, .activation);
    try builder.add_node_p(.matmul, &.{ attn, ow }, attn_out, (o_dims.out + 15) / 16, 1, 1, o_dims.out, o_dims.in, 0, 0);

    // 8. Optional Post-Attention Norm (Gemma2)
    const attn_post_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_post_norm.weight", .{ln});
    defer builder.graph.allocator.free(attn_post_norm_w);
    if (builder.has_tensor(attn_post_norm_w)) {
        try builder.add_tensor(attn_post_norm_w, @as(u64, o_dims.out) * 4, .weight);
        try builder.add_node(.rms_norm, &.{ attn_out, attn_post_norm_w }, attn_out, (o_dims.out + 63) / 64, 1, o_dims.out, o_dims.out, eps_bits, 0);
    }

    // 9. Residual Add 1
    var res1_buf: [64]u8 = undefined;
    const res1 = try std.fmt.bufPrint(&res1_buf, "{s}.res1", .{ln});
    try builder.add_tensor(res1, @as(u64, o_dims.out) * 4, .activation);
    const res_scale_bits: u32 = @bitCast(cfg.residual_scale);
    if (cfg.residual_scale != 1.0) {
        try builder.add_node(.scaled_add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, res_scale_bits, 0, 0);
    } else {
        try builder.add_node(.add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, 0, 0, 0);
    }

    // 10. FFN Norm
    var fnw_buf: [64]u8 = undefined;
    const fnw = try std.fmt.bufPrint(&fnw_buf, "{s}.ffn_norm.weight", .{ln});
    var ffn_normed_buf: [64]u8 = undefined;
    const ffn_normed = try std.fmt.bufPrint(&ffn_normed_buf, "{s}.ffn_normed", .{ln});
    try builder.add_tensor(fnw, @as(u64, n_embd) * 4, .weight);
    try builder.add_tensor(ffn_normed, @as(u64, o_dims.out) * 4, .activation);
    try builder.add_node(.rms_norm, &.{ res1, fnw }, ffn_normed, (o_dims.out + 63) / 64, 1, o_dims.out, o_dims.out, eps_bits, 0);

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

    const g_dims = builder.matmul_dims(gw, n_ff, o_dims.out);
    const use_fused_gate_up = builder.can_fuse_gate_up(gw, uw);

    if (use_fused_gate_up) {
        const gate_up_dims = struct { out: u32, in: u32 }{ .out = g_dims.out * 2, .in = g_dims.in };
        try builder.add_tensor(gate_up_w, @as(u64, gate_up_dims.out * gate_up_dims.in) * 4, .weight);
        try builder.add_tensor(gate_up_n, @as(u64, gate_up_dims.out) * 4, .activation);
        try builder.add_node(.matmul, &.{ ffn_normed, gate_up_w }, gate_up_n, (gate_up_dims.out + 15) / 16, 1, 1, gate_up_dims.out, gate_up_dims.in, 0);

        gate = gate_up_n;
        up = gate_up_n;
        gate_offset = 0;
        up_offset = g_dims.out * 4;
    } else {
        var gate_path_buf: [64]u8 = undefined;
        gate = try builder.graph.allocator.dupe(u8, try std.fmt.bufPrint(&gate_path_buf, "{s}.gate", .{ln}));
        var up_path_buf: [64]u8 = undefined;
        up = try builder.graph.allocator.dupe(u8, try std.fmt.bufPrint(&up_path_buf, "{s}.up", .{ln}));

        const u_dims = builder.matmul_dims(uw, n_ff, o_dims.out);
        try builder.add_tensor(gw, @as(u64, g_dims.out * g_dims.in) * 4, .weight);
        try builder.add_tensor(uw, @as(u64, u_dims.out * u_dims.in) * 4, .weight);
        try builder.add_tensor(gate, @as(u64, g_dims.out) * 4, .activation);
        try builder.add_tensor(up, @as(u64, u_dims.out) * 4, .activation);
        try builder.add_node_p(.matmul, &.{ ffn_normed, gw }, gate, (g_dims.out + 15) / 16, 1, 1, g_dims.out, g_dims.in, 0, 0);
        try builder.add_node_p(.matmul, &.{ ffn_normed, uw }, up, (u_dims.out + 15) / 16, 1, 1, u_dims.out, u_dims.in, 0, 0);
    }

    const activation_op: OpType = if (cfg.activation == .gelu) .gelu_mul else .silu_mul;
    try builder.add_node_p8(activation_op, &.{ gate, up }, gate, (g_dims.out + 63) / 64, 1, g_dims.out, 0, 0, 0, gate_offset, up_offset, gate_offset, 0);

    var dw_buf: [64]u8 = undefined;
    const dw = try std.fmt.bufPrint(&dw_buf, "{s}.ffn_down.weight", .{ln});
    var ffn_out_buf: [64]u8 = undefined;
    const ffn_out = try std.fmt.bufPrint(&ffn_out_buf, "{s}.ffn_out", .{ln});
    const d_dims = builder.matmul_dims(dw, o_dims.out, g_dims.out);
    try builder.add_tensor(dw, @as(u64, d_dims.out * d_dims.in) * 4, .weight);
    try builder.add_tensor(ffn_out, @as(u64, d_dims.out) * 4, .activation);
    try builder.add_node_p(.matmul, &.{ gate, dw }, ffn_out, (d_dims.out + 15) / 16, 1, 1, d_dims.out, d_dims.in, 0, 0);

    // 12. Optional Post-FFN Norm (Gemma2)
    const ffn_post_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.ffn_post_norm.weight", .{ln});
    defer builder.graph.allocator.free(ffn_post_norm_w);
    if (builder.has_tensor(ffn_post_norm_w)) {
        try builder.add_tensor(ffn_post_norm_w, @as(u64, d_dims.out) * 4, .weight);
        try builder.add_node(.rms_norm, &.{ ffn_out, ffn_post_norm_w }, ffn_out, (d_dims.out + 63) / 64, 1, d_dims.out, d_dims.out, eps_bits, 0);
    }

    // 13. Residual Add 2
    try builder.add_tensor(out_name, @as(u64, d_dims.out) * 4, .activation);
    if (cfg.residual_scale != 1.0) {
        try builder.add_node(.scaled_add, &.{ res1, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, res_scale_bits, 0, 0);
    } else {
        try builder.add_node(.add, &.{ res1, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, 0, 0, 0);
    }
}
